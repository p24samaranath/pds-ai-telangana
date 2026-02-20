"""Fraud detection and alert management API routes."""
from fastapi import APIRouter, Query
from typing import Optional
from datetime import datetime

from agents.fraud_detection_agent import FraudDetectionAgent
from database.schemas import FraudAlertResponse
from services.data_ingestion import DataIngestionService
from app.constants import FraudSeverity

router = APIRouter(prefix="/api/v1/fraud", tags=["Fraud Detection"])
_fraud_agent = FraudDetectionAgent()
_data_service = DataIngestionService()


@router.get("/alerts")
async def get_fraud_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity: Critical, High, Medium, Low"),
    district: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
):
    """Run fraud detection and return alerts."""
    shops_df, _, txn_df = _data_service.get_all_data()

    if district:
        shop_ids = shops_df[shops_df["district"] == district]["shop_id"].astype(str).tolist()
        if "fps_shop_id" in txn_df.columns:
            txn_df = txn_df[txn_df["fps_shop_id"].astype(str).isin(shop_ids)]

    result = await _fraud_agent.run(txn_df)
    alerts = result.get("alerts", [])

    if severity:
        alerts = [a for a in alerts if a.get("severity", "").lower() == severity.lower()]

    return {
        "alerts": alerts[:limit],
        "summary": result.get("summary", {}),
        "generated_at": datetime.utcnow().isoformat(),
    }


@router.get("/alerts/critical")
async def get_critical_alerts():
    """Get only Critical severity fraud alerts for immediate action."""
    _, _, txn_df = _data_service.get_all_data()
    result = await _fraud_agent.run(txn_df)
    critical = [a for a in result.get("alerts", []) if a.get("severity") == "Critical"]
    return {"critical_alerts": critical, "count": len(critical), "generated_at": datetime.utcnow().isoformat()}


@router.post("/score-transaction")
async def score_single_transaction(transaction: dict):
    """Score a single transaction for real-time fraud detection."""
    result = await _fraud_agent.score_transaction(transaction)
    return result


@router.get("/summary")
async def get_fraud_summary():
    """Get fraud detection summary statistics."""
    _, _, txn_df = _data_service.get_all_data()
    result = await _fraud_agent.run(txn_df)
    return {
        "summary": result.get("summary", {}),
        "generated_at": datetime.utcnow().isoformat(),
    }
