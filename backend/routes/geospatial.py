"""Geospatial analysis API routes."""
import asyncio
from fastapi import APIRouter, Query
from typing import Optional
from datetime import datetime, timedelta

from agents.geospatial_agent import GeospatialAgent
from services.data_ingestion import DataIngestionService

router = APIRouter(prefix="/api/v1/geo", tags=["Geospatial"])
_geo_agent = GeospatialAgent()
_data_service = DataIngestionService()

# ── Route-level result cache (avoids 4× parallel re-runs on geo map page load) ─
_geo_cache: dict = {"result": None, "at": None}
_geo_lock = asyncio.Lock()
_GEO_CACHE_TTL = timedelta(seconds=120)   # re-run at most once every 2 minutes


async def _get_geo_result(**kwargs) -> dict:
    """Return cached geo result if fresh; otherwise run the full pipeline once."""
    async with _geo_lock:
        now = datetime.utcnow()
        if (
            _geo_cache["result"] is not None
            and _geo_cache["at"] is not None
            and now - _geo_cache["at"] < _GEO_CACHE_TTL
        ):
            return _geo_cache["result"]
        shops_df, bene_df, txn_df = _data_service.get_all_data()
        result = await _geo_agent.run(
            shops_df=shops_df,
            beneficiaries_df=bene_df,
            transactions_df=txn_df,
            **kwargs,
        )
        _geo_cache["result"] = result
        _geo_cache["at"] = now
        return result


@router.get("/analysis")
async def get_geospatial_analysis(
    district: Optional[str] = Query(None),
    n_recommendations: int = Query(5, ge=1, le=20),
):
    """Full geospatial coverage analysis with underserved zones and recommendations."""
    result = await _get_geo_result(n_new_shop_recommendations=n_recommendations)
    if district:
        result = dict(result)
        result["underserved_zones"] = [
            z for z in result.get("underserved_zones", [])
            if z.get("district", "").lower() == district.lower()
        ]
    return result


@router.get("/underserved-zones")
async def get_underserved_zones(district: Optional[str] = Query(None)):
    """Get areas where beneficiaries are >5 km from nearest FPS shop."""
    result = await _get_geo_result()
    zones = result.get("underserved_zones", [])
    if district:
        zones = [z for z in zones if z.get("district", "").lower() == district.lower()]
    return {"underserved_zones": zones, "count": len(zones), "generated_at": datetime.utcnow().isoformat()}


@router.get("/recommendations")
async def get_new_shop_recommendations(n: int = Query(5, ge=1, le=20)):
    """Get ranked recommendations for new FPS shop locations."""
    result = await _get_geo_result(n_new_shop_recommendations=n)
    return {
        "recommendations": result.get("new_location_recommendations", []),
        "generated_at": datetime.utcnow().isoformat(),
    }


@router.get("/shops")
async def get_shops_geojson(district: Optional[str] = Query(None)):
    """Return all FPS shops as a GeoJSON FeatureCollection for map display."""
    shops_df, _, _ = _data_service.get_all_data()
    if district:
        shops_df = shops_df[shops_df["district"] == district]

    features = []
    for _, row in shops_df.dropna(subset=["latitude", "longitude"]).iterrows():
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row["longitude"]), float(row["latitude"])],
            },
            "properties": {
                "shop_id": str(row.get("shop_id", "")),
                "shop_name": str(row.get("shop_name", "")),
                "district": str(row.get("district", "")),
                "is_active": bool(row.get("is_active", True)),
                "total_cards": int(row.get("total_cards", 0)),
            },
        })

    return {
        "type": "FeatureCollection",
        "features": features,
        "total": len(features),
    }


@router.get("/accessibility-scores")
async def get_district_accessibility_scores():
    """Get 0-1 accessibility score per district."""
    result = await _get_geo_result()
    return {
        "district_scores": result.get("district_accessibility_scores", {}),
        "overall_score": result.get("overall_accessibility_score"),
        "generated_at": datetime.utcnow().isoformat(),
    }
