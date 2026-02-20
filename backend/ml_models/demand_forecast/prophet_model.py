"""
Facebook Prophet Demand Forecasting Model
Handles seasonality: festival seasons, harvest periods, monthly ration cycles.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not installed — will use statistical fallback")


class ProphetForecaster:
    """
    Wraps Facebook Prophet for PDS commodity demand forecasting.
    Incorporates Indian holidays, festival seasons, and harvest cycles.
    """

    # Key Indian holidays / events affecting ration demand
    INDIAN_HOLIDAYS = pd.DataFrame({
        "holiday": [
            "Republic Day", "Holi", "Ram Navami", "Eid ul-Fitr",
            "Independence Day", "Ganesh Chaturthi", "Dussehra", "Diwali",
            "Eid ul-Adha", "Christmas", "Sankranti / Pongal",
        ],
        "ds": pd.to_datetime([
            "2024-01-26", "2024-03-25", "2024-04-17", "2024-04-10",
            "2024-08-15", "2024-09-07", "2024-10-12", "2024-11-01",
            "2024-06-17", "2024-12-25", "2024-01-15",
        ]),
        "lower_window": [-1] * 11,
        "upper_window": [1] * 11,
    })

    def __init__(self, seasonality_mode: str = "multiplicative",
                 changepoint_prior_scale: float = 0.05,
                 interval_width: float = 0.90):
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.interval_width = interval_width
        self.model: Optional[object] = None

    def _build_prophet_df(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Convert transaction data into Prophet-expected ds/y format."""
        df = transactions_df.copy()
        df["ds"] = pd.to_datetime(df["month"])
        df["y"] = df["quantity_lifted"].astype(float)
        return df[["ds", "y"]].dropna().sort_values("ds")

    def train(self, transactions_df: pd.DataFrame) -> Dict:
        if not PROPHET_AVAILABLE:
            logger.info("Prophet unavailable — skipping training")
            return {"status": "skipped", "reason": "prophet_unavailable"}

        prophet_df = self._build_prophet_df(transactions_df)
        if len(prophet_df) < 3:
            return {"status": "insufficient_data", "rows": len(prophet_df)}

        self.model = Prophet(
            holidays=self.INDIAN_HOLIDAYS,
            seasonality_mode=self.seasonality_mode,
            changepoint_prior_scale=self.changepoint_prior_scale,
            interval_width=self.interval_width,
            weekly_seasonality=False,
            daily_seasonality=False,
            yearly_seasonality=True,
        )
        # Add monthly seasonality for ration cycles
        self.model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

        self.model.fit(prophet_df)
        logger.info(f"Prophet trained on {len(prophet_df)} months of data")
        return {"status": "success", "data_points": len(prophet_df)}

    def predict(self, months_ahead: int = 3) -> List[Dict]:
        if not PROPHET_AVAILABLE or self.model is None:
            return self._fallback_predict(months_ahead)

        future = self.model.make_future_dataframe(periods=months_ahead, freq="MS")
        forecast = self.model.predict(future)
        recent = forecast.tail(months_ahead)

        results = []
        for i, (_, row) in enumerate(recent.iterrows()):
            results.append({
                "month_offset": i + 1,
                "predicted_quantity_kg": round(max(0, row["yhat"]), 2),
                "confidence_lower": round(max(0, row["yhat_lower"]), 2),
                "confidence_upper": round(max(0, row["yhat_upper"]), 2),
                "model": "prophet",
                "trend": round(row.get("trend", 0), 2),
                "seasonality": round(row.get("yearly", 0), 2),
            })
        return results

    def _fallback_predict(self, months_ahead: int) -> List[Dict]:
        """Linear-trend fallback when Prophet is unavailable."""
        base = 500.0
        results = []
        for m in range(months_ahead):
            results.append({
                "month_offset": m + 1,
                "predicted_quantity_kg": round(base * (1 + 0.01 * m), 2),
                "confidence_lower": round(base * 0.85, 2),
                "confidence_upper": round(base * 1.15, 2),
                "model": "linear_fallback",
            })
        return results

    def evaluate_mape(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        mask = actual != 0
        return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])))


class EnsembleForecaster:
    """
    Combines LSTM + Prophet predictions with weighted average.
    Weights are adjusted based on recent MAPE performance.
    """

    def __init__(self, lstm_weight: float = 0.5, prophet_weight: float = 0.5):
        self.lstm_weight = lstm_weight
        self.prophet_weight = prophet_weight

    def combine(self, lstm_preds: List[Dict], prophet_preds: List[Dict]) -> List[Dict]:
        results = []
        for lstm, prophet in zip(lstm_preds, prophet_preds):
            blended_qty = (
                self.lstm_weight * lstm["predicted_quantity_kg"]
                + self.prophet_weight * prophet["predicted_quantity_kg"]
            )
            blended_lower = min(lstm["confidence_lower"], prophet["confidence_lower"])
            blended_upper = max(lstm["confidence_upper"], prophet["confidence_upper"])

            # Detect risk flags
            risk_flag = None
            if blended_qty < blended_lower * 0.7:
                risk_flag = "understock_risk"
            elif blended_qty > blended_upper * 1.3:
                risk_flag = "overstock_risk"

            results.append({
                "month_offset": lstm["month_offset"],
                "predicted_quantity_kg": round(blended_qty, 2),
                "confidence_lower": round(blended_lower, 2),
                "confidence_upper": round(blended_upper, 2),
                "model": "ensemble",
                "risk_flag": risk_flag,
                "lstm_prediction": lstm["predicted_quantity_kg"],
                "prophet_prediction": prophet["predicted_quantity_kg"],
            })
        return results

    def update_weights(self, lstm_mape: float, prophet_mape: float):
        """Bayesian weight update: better model gets higher weight."""
        total = lstm_mape + prophet_mape
        if total == 0:
            return
        self.lstm_weight = prophet_mape / total      # lower error → higher weight
        self.prophet_weight = lstm_mape / total
        logger.info(
            f"Updated weights — LSTM: {self.lstm_weight:.2f}, Prophet: {self.prophet_weight:.2f}"
        )
