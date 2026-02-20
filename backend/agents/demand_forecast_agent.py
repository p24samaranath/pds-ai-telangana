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
        self.ensemble = EnsembleForecaster(lstm_weight=0.5, prophet_weight=0.5)
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
    ) -> Dict[str, Any]:
        """Run demand forecasting for all shops and commodities."""
        if commodities is None:
            commodities = [CommodityType.RICE, CommodityType.WHEAT]

        all_forecasts = []
        risk_flags = []
        errors = []

        for shop in shops:
            for commodity in commodities:
                try:
                    result = await self.forecast_shop(
                        shop_id=str(shop["shop_id"]),
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

        logger.info(
            f"Demand Forecast Agent complete: {len(all_forecasts)} forecasts, "
            f"{len(risk_flags)} risk flags, {len(errors)} errors"
        )

        return {
            "agent": "demand_forecast",
            "forecasts": all_forecasts,
            "risk_flags": risk_flags,
            "total_forecasts": len(all_forecasts),
            "total_risk_flags": len(risk_flags),
            "errors": errors,
            "generated_at": datetime.utcnow().isoformat(),
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
