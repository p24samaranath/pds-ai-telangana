"""
Facebook Prophet Demand Forecasting Model
==========================================
Handles seasonality: festival seasons, harvest periods, monthly ration cycles.

Key fixes vs previous version
------------------------------
* INDIAN_HOLIDAYS: dates are now generated *dynamically* for any year range
  instead of being hardcoded to 2024.  Variable-date festivals (Holi, Diwali,
  Eid) are pre-computed for 2024-2027 and auto-extended for adjacent years.
* _build_prophet_df: gracefully accepts both "quantity_lifted" (monthly agg)
  and "quantity_kg" (raw column name) as the target variable.
* EnsembleForecaster.update_weights: now properly exposed; added _track_mape()
  helper so callers can easily update weights from actual vs predicted arrays.
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not installed — will use statistical fallback")


# ── Dynamic Indian Holiday Generation ─────────────────────────────────────────

# Fixed-date holidays (month, day) — same every year
_FIXED_HOLIDAYS = {
    "Republic Day":       (1, 26),
    "Independence Day":   (8, 15),
    "Christmas":          (12, 25),
    "Sankranti / Pongal": (1, 15),
    "Gandhi Jayanti":     (10, 2),
}

# Variable holidays per year (month, day, name)
# Derived from official Indian government calendars
_VARIABLE_HOLIDAYS: Dict[int, List] = {
    2023: [
        (3, 8,  "Holi"),
        (4, 21, "Eid ul-Fitr"),
        (6, 28, "Eid ul-Adha"),
        (9, 19, "Ganesh Chaturthi"),
        (10, 24, "Dussehra"),
        (11, 12, "Diwali"),
    ],
    2024: [
        (3, 25, "Holi"),
        (4, 10, "Eid ul-Fitr"),
        (4, 17, "Ram Navami"),
        (6, 17, "Eid ul-Adha"),
        (9, 7,  "Ganesh Chaturthi"),
        (10, 12, "Dussehra"),
        (11, 1, "Diwali"),
    ],
    2025: [
        (3, 14, "Holi"),
        (3, 30, "Eid ul-Fitr"),
        (4, 6,  "Ram Navami"),
        (6, 7,  "Eid ul-Adha"),
        (8, 27, "Ganesh Chaturthi"),
        (10, 2, "Dussehra"),
        (10, 20, "Diwali"),
    ],
    2026: [
        (3, 3,  "Holi"),
        (3, 20, "Eid ul-Fitr"),
        (3, 26, "Ram Navami"),
        (5, 27, "Eid ul-Adha"),
        (8, 16, "Ganesh Chaturthi"),
        (9, 21, "Dussehra"),
        (11, 8, "Diwali"),
    ],
    2027: [
        (3, 22, "Holi"),
        (3, 9,  "Eid ul-Fitr"),
        (4, 14, "Ram Navami"),
        (5, 17, "Eid ul-Adha"),
        (9, 4,  "Ganesh Chaturthi"),
        (10, 11, "Dussehra"),
        (10, 29, "Diwali"),
    ],
}


def _get_indian_holidays(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Generate Indian public holidays DataFrame for a range of years.
    Prophet requires columns: holiday, ds, lower_window, upper_window.
    """
    rows = []

    for year in range(start_year, end_year + 1):
        # Fixed-date holidays
        for name, (month, day) in _FIXED_HOLIDAYS.items():
            try:
                rows.append({
                    "holiday":      name,
                    "ds":           pd.Timestamp(year, month, day),
                    "lower_window": -1,
                    "upper_window":  1,
                })
            except ValueError:
                pass   # e.g. Feb 30 edge cases

        # Variable-date holidays — use nearest known year if exact year missing
        closest_year = min(
            _VARIABLE_HOLIDAYS.keys(),
            key=lambda y: abs(y - year),
        )
        for month, day, name in _VARIABLE_HOLIDAYS[closest_year]:
            try:
                rows.append({
                    "holiday":      name,
                    "ds":           pd.Timestamp(year, month, day),
                    "lower_window": -2,   # demand peaks ~2 days before festivals
                    "upper_window":  1,
                })
            except ValueError:
                pass

    df = pd.DataFrame(rows)
    df["ds"] = pd.to_datetime(df["ds"])
    return df.drop_duplicates(subset=["holiday", "ds"]).reset_index(drop=True)


class ProphetForecaster:
    """
    Wraps Facebook Prophet for PDS commodity demand forecasting.
    Incorporates dynamic Indian holidays, festival seasons, and harvest cycles.
    """

    def __init__(
        self,
        seasonality_mode: str = "multiplicative",
        changepoint_prior_scale: float = 0.05,
        interval_width: float = 0.90,
    ):
        self.seasonality_mode         = seasonality_mode
        self.changepoint_prior_scale  = changepoint_prior_scale
        self.interval_width           = interval_width
        self.model: Optional[object]  = None
        self._last_ds: Optional[pd.Timestamp] = None

    def _build_prophet_df(self, monthly_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert monthly aggregated data into Prophet ds/y format.
        Accepts either:
            - "month" + "quantity_lifted"  (output of _prepare_monthly_data)
            - "transaction_date" + "quantity_kg"  (raw normalised format)
        """
        df = monthly_df.copy()

        # ds column
        if "month" in df.columns:
            df["ds"] = pd.to_datetime(df["month"], errors="coerce")
        elif "transaction_date" in df.columns:
            df["ds"] = pd.to_datetime(df["transaction_date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
        else:
            raise ValueError("monthly_df must contain 'month' or 'transaction_date'")

        # y column
        if "quantity_lifted" in df.columns:
            df["y"] = pd.to_numeric(df["quantity_lifted"], errors="coerce")
        elif "quantity_kg" in df.columns:
            df["y"] = pd.to_numeric(df["quantity_kg"], errors="coerce")
        else:
            raise ValueError("monthly_df must contain 'quantity_lifted' or 'quantity_kg'")

        return df[["ds", "y"]].dropna().sort_values("ds").reset_index(drop=True)

    def train(self, monthly_df: pd.DataFrame) -> Dict:
        if not PROPHET_AVAILABLE:
            return {"status": "skipped", "reason": "prophet_unavailable"}

        try:
            prophet_df = self._build_prophet_df(monthly_df)
        except ValueError as e:
            return {"status": "error", "reason": str(e)}

        if len(prophet_df) < 3:
            return {"status": "insufficient_data", "rows": len(prophet_df)}

        # Generate holidays for the data range + 1 year forecast
        start_year = int(prophet_df["ds"].dt.year.min())
        end_year   = int(prophet_df["ds"].dt.year.max()) + 1
        holidays   = _get_indian_holidays(start_year, end_year)

        self.model = Prophet(
            holidays=holidays,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            interval_width=self.interval_width,
            weekly_seasonality=False,
            daily_seasonality=False,
            yearly_seasonality=True,
        )
        self.model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
        self.model.fit(prophet_df)
        self._last_ds = prophet_df["ds"].max()

        logger.info(f"Prophet trained on {len(prophet_df)} months (holidays: {len(holidays)} events)")
        return {"status": "success", "data_points": len(prophet_df)}

    def predict(self, months_ahead: int = 3) -> List[Dict]:
        if not PROPHET_AVAILABLE or self.model is None:
            return self._fallback_predict(months_ahead)

        future   = self.model.make_future_dataframe(periods=months_ahead, freq="MS")
        forecast = self.model.predict(future)
        recent   = forecast.tail(months_ahead)

        results = []
        for i, (_, row) in enumerate(recent.iterrows()):
            results.append({
                "month_offset":          i + 1,
                "predicted_quantity_kg": round(max(0.0, row["yhat"]),       2),
                "confidence_lower":      round(max(0.0, row["yhat_lower"]), 2),
                "confidence_upper":      round(max(0.0, row["yhat_upper"]), 2),
                "model":                 "prophet",
                "trend":                 round(row.get("trend", 0),    2),
                "seasonality":           round(row.get("yearly", 0),   2),
                "holidays_effect":       round(row.get("holidays", 0), 2),
            })
        return results

    def _fallback_predict(self, months_ahead: int) -> List[Dict]:
        """Linear-trend fallback when Prophet is unavailable or untrained."""
        base = 500.0
        return [
            {
                "month_offset":          m + 1,
                "predicted_quantity_kg": round(base * (1 + 0.01 * m), 2),
                "confidence_lower":      round(base * 0.85, 2),
                "confidence_upper":      round(base * 1.15, 2),
                "model":                 "linear_fallback",
            }
            for m in range(months_ahead)
        ]

    def evaluate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        mask = actual != 0
        if not mask.any():
            return 0.0
        return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])))


class EnsembleForecaster:
    """
    Combines LSTM + Prophet predictions with adaptive weight average.
    Weights are updated via update_weights() after actual data arrives.
    """

    def __init__(self, lstm_weight: float = 0.4, prophet_weight: float = 0.6):
        # Start with prophet-dominant: more reliable than LSTM on sparse PDS data
        self.lstm_weight    = lstm_weight
        self.prophet_weight = prophet_weight
        self._lstm_mape     = 0.20
        self._prophet_mape  = 0.15

    def combine(
        self,
        lstm_preds: List[Dict],
        prophet_preds: List[Dict],
    ) -> List[Dict]:
        results = []
        for lstm, proph in zip(lstm_preds, prophet_preds):
            blended_qty   = (
                self.lstm_weight    * lstm["predicted_quantity_kg"]
                + self.prophet_weight * proph["predicted_quantity_kg"]
            )
            blended_lower = min(lstm["confidence_lower"],  proph["confidence_lower"])
            blended_upper = max(lstm["confidence_upper"],  proph["confidence_upper"])

            risk_flag = None
            if blended_qty < blended_lower * 0.70:
                risk_flag = "understock_risk"
            elif blended_qty > blended_upper * 1.30:
                risk_flag = "overstock_risk"

            results.append({
                "month_offset":          lstm["month_offset"],
                "predicted_quantity_kg": round(blended_qty,   2),
                "confidence_lower":      round(blended_lower, 2),
                "confidence_upper":      round(blended_upper, 2),
                "model":                 "ensemble",
                "risk_flag":             risk_flag,
                "lstm_prediction":       lstm["predicted_quantity_kg"],
                "prophet_prediction":    proph["predicted_quantity_kg"],
                "weights": {
                    "lstm":    round(self.lstm_weight,    3),
                    "prophet": round(self.prophet_weight, 3),
                },
            })
        return results

    def update_weights(self, lstm_mape: float, prophet_mape: float) -> None:
        """
        Bayesian weight update: model with lower MAPE gets higher weight.
        Called after actual observations arrive to update the ensemble.
        """
        total = lstm_mape + prophet_mape
        if total < 1e-8:
            return
        self._lstm_mape    = lstm_mape
        self._prophet_mape = prophet_mape
        self.lstm_weight    = prophet_mape / total   # lower error → higher weight
        self.prophet_weight = lstm_mape    / total
        logger.info(
            f"Ensemble weights updated — LSTM: {self.lstm_weight:.3f} "
            f"(MAPE={lstm_mape:.2%}), Prophet: {self.prophet_weight:.3f} "
            f"(MAPE={prophet_mape:.2%})"
        )

    def track_and_update(
        self,
        actual: np.ndarray,
        lstm_predicted: np.ndarray,
        prophet_predicted: np.ndarray,
    ) -> Dict:
        """Helper: compute MAPEs and update weights in one call."""
        def _mape(a, p):
            mask = a != 0
            if not mask.any():
                return 0.20
            return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])))

        lstm_mape    = _mape(actual, lstm_predicted)
        prophet_mape = _mape(actual, prophet_predicted)
        self.update_weights(lstm_mape, prophet_mape)
        return {"lstm_mape": lstm_mape, "prophet_mape": prophet_mape}
