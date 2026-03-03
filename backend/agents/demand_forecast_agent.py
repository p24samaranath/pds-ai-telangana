"""
Demand Forecast Agent
Predicts monthly commodity demand per FPS shop using LSTM + Prophet ensemble.
Autonomously retrains when MAPE exceeds threshold.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, date
import logging

from ml_models.demand_forecast.lstm_model import LSTMForecaster
from ml_models.demand_forecast.prophet_model import ProphetForecaster, EnsembleForecaster
from ml_models.demand_forecast.entitlement_model import EntitlementDemandModel
from app.constants import CommodityType
from app.config import settings

logger = logging.getLogger(__name__)


class DemandForecastAgent:
    """
    Autonomous demand forecasting agent.
    - Maintains per-shop, per-commodity forecasters
    - Auto-retrains when prediction error exceeds threshold
    - Returns structured forecast results with confidence intervals and risk flags
    """

    def __init__(self):
        self.lstm_forecasters: Dict[str, LSTMForecaster] = {}
        self.prophet_forecasters: Dict[str, ProphetForecaster] = {}
        self.ensemble = EnsembleForecaster(lstm_weight=0.4, prophet_weight=0.6)
        self.entitlement_model = EntitlementDemandModel()
        self.last_mape: Dict[str, float] = {}
        self.retrain_threshold = settings.MODEL_RETRAIN_THRESHOLD_MAPE

    def _model_key(self, shop_id: str, commodity: str) -> str:
        return f"{shop_id}_{commodity}"

    # ── Data Preparation ──────────────────────────────────────────────────────

    def _prepare_monthly_data(
        self, transactions_df: pd.DataFrame, shop_id: str, commodity: str
    ) -> pd.DataFrame:
        """Aggregate transactions to monthly totals for a shop/commodity pair."""
        mask = (
            (transactions_df["fps_shop_id"].astype(str) == str(shop_id))
            & (transactions_df["commodity"] == commodity)
        )
        shop_df = transactions_df[mask].copy()
        shop_df["month"] = pd.to_datetime(shop_df["transaction_date"]).dt.to_period("M").astype(str)

        monthly = shop_df.groupby("month").agg(
            quantity_lifted=("quantity_kg", "sum"),
            active_cards=("card_id", "nunique"),
        ).reset_index()
        return monthly.sort_values("month")

    # ── Forecasting ───────────────────────────────────────────────────────────

    def _district_monthly_data(
        self, transactions_df: pd.DataFrame, district: str, commodity: str
    ) -> pd.DataFrame:
        """
        Aggregate commodity quantities across all shops in a district per month.
        Used as a fallback when individual shop history is too short.
        """
        if "district" not in transactions_df.columns:
            return pd.DataFrame()
        mask = (
            (transactions_df["district"] == district)
            & (transactions_df["commodity"] == commodity)
        )
        dist_df = transactions_df[mask].copy()
        dist_df["month"] = pd.to_datetime(dist_df["transaction_date"]).dt.to_period("M").astype(str)
        monthly = (
            dist_df.groupby("month")
            .agg(quantity_lifted=("quantity_kg", "mean"),  # per-shop average
                 active_cards=("card_id", "nunique"))
            .reset_index()
            .sort_values("month")
        )
        return monthly

    def _growth_adjusted_forecast(
        self, base_qty: float, months_ahead: int,
        last_month_str: str, monthly_growth: float = 0.02
    ) -> List[Dict]:
        """
        Simple month-over-month growth model used when ML cannot train.
        Assumes monthly_growth fractional change.
        """
        try:
            last_month = pd.to_datetime(last_month_str)
        except Exception:
            last_month = datetime.utcnow()

        results = []
        qty = base_qty
        for m in range(months_ahead):
            qty = qty * (1 + monthly_growth)
            fm = (last_month + pd.DateOffset(months=m + 1)).strftime("%Y-%m")
            results.append({
                "month_offset": m + 1,
                "predicted_quantity_kg": round(max(qty, 0), 2),
                "confidence_lower":      round(max(qty * 0.85, 0), 2),
                "confidence_upper":      round(qty * 1.15, 2),
                "model": "growth_adjusted_heuristic",
            })
        return results

    async def forecast_shop(
        self,
        shop_id: str,
        shop_name: str,
        district: str,
        transactions_df: pd.DataFrame,
        commodity: CommodityType,
        months_ahead: int = 3,
        force_retrain: bool = False,
    ) -> Dict[str, Any]:
        """Generate demand forecast for a single shop/commodity combination."""
        key = self._model_key(shop_id, commodity.value)
        monthly_df = self._prepare_monthly_data(transactions_df, shop_id, commodity.value)

        # ── Insufficient shop-level history: try district aggregate, then heuristic ──
        if len(monthly_df) < 3:
            dist_monthly = self._district_monthly_data(transactions_df, district, commodity.value)
            base_qty = 0.0
            last_month_str = datetime.utcnow().strftime("%Y-%m")

            if len(monthly_df) >= 1:
                # Use shop's actual value as base
                base_qty = float(monthly_df["quantity_lifted"].iloc[-1])
                last_month_str = monthly_df["month"].iloc[-1]
            elif len(dist_monthly) >= 1:
                # Use district per-shop average
                base_qty = float(dist_monthly["quantity_lifted"].iloc[-1])
                last_month_str = dist_monthly["month"].iloc[-1]

            if base_qty == 0.0:
                return {
                    "shop_id": shop_id, "shop_name": shop_name,
                    "district": district, "commodity": commodity.value,
                    "error": "no_data", "months_of_data": 0, "forecasts": [],
                }

            preds = self._growth_adjusted_forecast(base_qty, months_ahead, last_month_str)
            results = []
            for pred in preds:
                try:
                    fm = (pd.to_datetime(last_month_str) + pd.DateOffset(months=pred["month_offset"])).strftime("%Y-%m")
                except Exception:
                    fm = pred.get("month_offset", "")
                results.append({
                    "fps_shop_id": shop_id, "shop_name": shop_name,
                    "district": district, "commodity": commodity.value,
                    "forecast_month": fm,
                    "predicted_quantity_kg": pred["predicted_quantity_kg"],
                    "confidence_lower": pred["confidence_lower"],
                    "confidence_upper": pred["confidence_upper"],
                    "model_used": pred["model"],
                    "risk_flag": None,
                })
            return {
                "shop_id": shop_id, "shop_name": shop_name,
                "district": district, "commodity": commodity.value,
                "forecasts": results, "months_of_history": len(monthly_df),
                "retrained": False, "fallback": "growth_heuristic",
            }

        # Decide whether to retrain
        needs_retrain = (
            force_retrain
            or key not in self.lstm_forecasters
            or self.last_mape.get(key, 1.0) > self.retrain_threshold
        )

        if needs_retrain:
            lstm = LSTMForecaster()
            lstm.train(monthly_df, epochs=30)
            self.lstm_forecasters[key] = lstm

            prophet = ProphetForecaster()
            prophet.train(monthly_df)
            self.prophet_forecasters[key] = prophet

            logger.info(f"Retrained models for {key}")
        else:
            lstm = self.lstm_forecasters[key]
            prophet = self.prophet_forecasters[key]

        lstm_preds = lstm.predict(monthly_df, months_ahead=months_ahead)
        prophet_preds = prophet.predict(months_ahead=months_ahead)
        ensemble_preds = self.ensemble.combine(lstm_preds, prophet_preds)

        # Build month labels
        last_month = pd.to_datetime(monthly_df["month"].iloc[-1])
        results = []
        for pred in ensemble_preds:
            forecast_month = last_month + pd.DateOffset(months=pred["month_offset"])
            results.append({
                "fps_shop_id": shop_id,
                "shop_name": shop_name,
                "district": district,
                "commodity": commodity.value,
                "forecast_month": forecast_month.strftime("%Y-%m"),
                "predicted_quantity_kg": pred["predicted_quantity_kg"],
                "confidence_lower": pred["confidence_lower"],
                "confidence_upper": pred["confidence_upper"],
                "model_used": pred["model"],
                "risk_flag": pred.get("risk_flag"),
            })

        return {
            "shop_id": shop_id,
            "shop_name": shop_name,
            "district": district,
            "commodity": commodity.value,
            "forecasts": results,
            "months_of_history": len(monthly_df),
            "retrained": needs_retrain,
        }

    async def run(
        self,
        shops: List[Dict],
        transactions_df: pd.DataFrame,
        commodities: Optional[List[CommodityType]] = None,
        months_ahead: int = 3,
        beneficiaries_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Run demand forecasting for all shops and commodities.

        Produces two parallel outputs:
          1. Prophet/LSTM time-series forecasts (historical trend-based)
          2. Entitlement-physics forecasts (from AAY/PHH card counts)
          3. Supply gap analysis: actual vs entitlement

        Args:
            shops           : List of shop dicts (from shops_df.to_dict("records"))
            transactions_df : Normalised long-format transactions
            commodities     : Commodities to forecast (default: RICE, WHEAT)
            months_ahead    : Number of months to forecast
            beneficiaries_df: Optional — provides AAY/PHH card counts for
                              accurate entitlement model; falls back to 10/90 split
        """
        if commodities is None:
            commodities = [CommodityType.RICE, CommodityType.WHEAT]

        all_forecasts = []
        risk_flags    = []
        errors        = []

        # ── Time-series forecasts (Prophet/LSTM) ─────────────────────────────
        for shop in shops:
            for commodity in commodities:
                try:
                    result = await self.forecast_shop(
                        shop_id=str(shop.get("shop_id", shop.get("fps_shop_id", ""))),
                        shop_name=shop.get("shop_name", "Unknown"),
                        district=shop.get("district", "Unknown"),
                        transactions_df=transactions_df,
                        commodity=commodity,
                        months_ahead=months_ahead,
                    )
                    all_forecasts.extend(result.get("forecasts", []))
                    for f in result.get("forecasts", []):
                        if f.get("risk_flag"):
                            risk_flags.append(f)
                except Exception as e:
                    logger.error(f"Forecast error for shop {shop.get('shop_id')}: {e}")
                    errors.append({"shop_id": shop.get("shop_id"), "error": str(e)})

        # ── Entitlement supply gap analysis ──────────────────────────────────
        supply_gap_results = {}
        district_supply_summary = []
        try:
            # Build a shop-level aggregate from transactions_df
            shops_df_tmp = pd.DataFrame(shops)
            if "shop_id" in shops_df_tmp.columns and "fps_shop_id" not in shops_df_tmp.columns:
                shops_df_tmp = shops_df_tmp.rename(columns={"shop_id": "fps_shop_id"})

            # Aggregate rice totals per shop from transactions
            rice_txns = transactions_df[transactions_df["commodity"] == "rice"] if not transactions_df.empty else pd.DataFrame()
            if not rice_txns.empty and "fps_shop_id" in rice_txns.columns:
                rice_agg = (
                    rice_txns.groupby("fps_shop_id")["quantity_kg"]
                    .sum()
                    .rename("rice_total_kg")
                    .reset_index()
                )
                if "fps_shop_id" in shops_df_tmp.columns:
                    shop_for_gap = shops_df_tmp.merge(rice_agg, on="fps_shop_id", how="left")
                    shop_for_gap["rice_total_kg"] = shop_for_gap["rice_total_kg"].fillna(0)
                    if "total_cards" not in shop_for_gap.columns:
                        shop_for_gap["total_cards"] = 100  # fallback
                    shop_for_gap["fps_shop_id"] = shop_for_gap["fps_shop_id"].astype(str)

                    gap_df = self.entitlement_model.compute_supply_gap(
                        shop_df=shop_for_gap,
                        beneficiaries_df=beneficiaries_df,
                    )

                    under_supplied = gap_df[gap_df.get("supply_status", pd.Series()) == "under_supplied"]
                    over_supplied  = gap_df[gap_df.get("supply_status", pd.Series()) == "over_supplied"]

                    supply_gap_results = {
                        "n_shops_analysed":        len(gap_df),
                        "n_under_supplied":        int(len(under_supplied)),
                        "n_over_supplied":         int(len(over_supplied)),
                        "n_critically_under":      int((gap_df.get("supply_status", pd.Series()) == "critically_under_supplied").sum()),
                        "total_expected_rice_kg":  round(float(gap_df["expected_rice_kg"].sum()), 1) if "expected_rice_kg" in gap_df.columns else 0,
                        "total_actual_rice_kg":    round(float(gap_df["rice_total_kg"].sum()), 1) if "rice_total_kg" in gap_df.columns else 0,
                        "over_supplied_shop_ids":  over_supplied["fps_shop_id"].tolist()[:20] if "fps_shop_id" in over_supplied.columns else [],
                    }

                    district_df = self.entitlement_model.district_supply_summary(gap_df)
                    district_supply_summary = district_df.to_dict("records") if not district_df.empty else []

                    # Flag over-supplied shops as demand risk flags
                    for _, row in over_supplied.iterrows():
                        risk_flags.append({
                            "fps_shop_id": str(row.get("fps_shop_id", "")),
                            "district":    str(row.get("district", "")),
                            "risk_flag":   "over_supplied",
                            "rice_gap_pct": float(row.get("rice_gap_pct", 0)),
                            "model_used":  "entitlement_physics",
                        })

        except Exception as e:
            logger.error(f"Entitlement supply gap analysis failed: {e}")
            supply_gap_results = {"error": str(e)}

        logger.info(
            f"Demand Forecast Agent complete: {len(all_forecasts)} ts-forecasts, "
            f"{len(risk_flags)} risk flags, {len(errors)} errors, "
            f"supply_gap shops: {supply_gap_results.get('n_shops_analysed', 0)}"
        )

        return {
            "agent":                  "demand_forecast",
            "forecasts":              all_forecasts,
            "risk_flags":             risk_flags,
            "total_forecasts":        len(all_forecasts),
            "total_risk_flags":       len(risk_flags),
            "errors":                 errors,
            "supply_gap":             supply_gap_results,
            "district_supply_summary": district_supply_summary,
            "generated_at":           datetime.utcnow().isoformat(),
        }

    def update_mape(self, shop_id: str, commodity: str,
                    actual: np.ndarray, predicted: np.ndarray):
        """Called by the orchestrator after actual data arrives to update error tracking."""
        key = self._model_key(shop_id, commodity)
        mask = actual != 0
        if mask.sum() == 0:
            return
        mape = float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])))
        self.last_mape[key] = mape
        if mape > self.retrain_threshold:
            logger.warning(
                f"MAPE {mape:.2%} exceeds threshold {self.retrain_threshold:.2%} "
                f"for {key} — will retrain on next run"
            )
