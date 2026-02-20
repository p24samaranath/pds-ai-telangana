"""Demand forecast API routes."""
from fastapi import APIRouter, Query
from typing import Optional
from datetime import datetime

from agents.demand_forecast_agent import DemandForecastAgent
from database.schemas import ForecastRequest
from services.data_ingestion import DataIngestionService
from app.constants import CommodityType

router = APIRouter(prefix="/api/v1/forecasts", tags=["Demand Forecasts"])
_forecast_agent = DemandForecastAgent()
_data_service = DataIngestionService()


@router.post("/")
async def get_forecasts(request: ForecastRequest):
    """Generate demand forecasts for shops/districts using LSTM + Prophet ensemble."""
    shops_df, _, txn_df = _data_service.get_all_data()

    # Filter by shop or district
    if request.shop_id:
        shops_df = shops_df[shops_df["shop_id"].astype(str) == str(request.shop_id)]
    elif request.district:
        shops_df = shops_df[shops_df["district"] == request.district]

    if len(shops_df) == 0:
        return {
            "forecasts": [],
            "generated_at": datetime.utcnow().isoformat(),
            "total_shops": 0,
            "message": "No shops matched the filter criteria.",
        }

    # Resolve commodity — may be a CommodityType enum or raw string
    if request.commodity:
        commodity = request.commodity if isinstance(request.commodity, CommodityType) else CommodityType(request.commodity)
        commodities = [commodity]
    else:
        commodities = [CommodityType.RICE, CommodityType.WHEAT]

    shops_list = shops_df.head(20).to_dict("records")   # Limit for response time

    result = await _forecast_agent.run(
        shops=shops_list,
        transactions_df=txn_df,
        commodities=commodities,
        months_ahead=request.months_ahead,
    )

    return {
        "forecasts": result["forecasts"],
        "risk_flags": result.get("risk_flags", []),
        "errors": result.get("errors", []),
        "generated_at": datetime.utcnow().isoformat(),
        "total_shops": len(shops_list),
        "total_forecasts": result.get("total_forecasts", 0),
    }


@router.get("/risk-flags")
async def get_risk_flags(
    district: Optional[str] = Query(None),
    commodity: Optional[str] = Query(None),
):
    """Get current overstock/understock risk flags."""
    shops_df, _, txn_df = _data_service.get_all_data()
    if district:
        shops_df = shops_df[shops_df["district"] == district]

    shops_list = shops_df.head(30).to_dict("records")
    commodities = [CommodityType.RICE, CommodityType.WHEAT]

    result = await _forecast_agent.run(
        shops=shops_list,
        transactions_df=txn_df,
        commodities=commodities,
        months_ahead=1,
    )
    return {
        "risk_flags": result["risk_flags"],
        "total_risk_flags": len(result["risk_flags"]),
        "generated_at": datetime.utcnow().isoformat(),
    }
