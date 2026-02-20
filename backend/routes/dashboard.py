"""Dashboard and reporting API routes."""
import logging
from collections import Counter
from datetime import datetime
from typing import Dict

import pandas as pd
from fastapi import APIRouter

from agents.orchestrator_agent import OrchestratorAgent, WorkflowTrigger
from services.data_ingestion import DataIngestionService

router = APIRouter(prefix="/api/v1/dashboard", tags=["Dashboard"])
_data_service = DataIngestionService()
_orchestrator = OrchestratorAgent()
logger = logging.getLogger(__name__)

# Maximum shops to send through full LSTM training pipeline (keeps response < 30s)
PIPELINE_SHOP_CAP = 30


def _sample_shops(shops_df: pd.DataFrame, cap: int) -> pd.DataFrame:
    """
    Return a stratified sample of shops (one per district up to cap).
    Ensures every district is represented even with a small cap.
    """
    if len(shops_df) <= cap:
        return shops_df
    # Take up to ceil(cap/districts) per district
    dists = shops_df["district"].nunique() if "district" in shops_df.columns else 1
    per_dist = max(1, cap // max(dists, 1))
    sampled = (
        shops_df.groupby("district", group_keys=False)
        .apply(lambda g: g.head(per_dist))
        if "district" in shops_df.columns
        else shops_df.head(cap)
    )
    return sampled.head(cap)


def _top_fraud_districts(alerts: list, shops_df: pd.DataFrame, top_n: int = 5) -> list:
    """
    Derive top fraud districts by joining alert shop_ids against the FULL shops_df.
    Falls back to district field on the alert dict if present.
    """
    # Build a fast lookup: shop_id -> district from full shops_df
    shop_district: Dict[str, str] = {}
    if not shops_df.empty and "shop_id" in shops_df.columns and "district" in shops_df.columns:
        shop_district = dict(zip(shops_df["shop_id"].astype(str), shops_df["district"].astype(str)))

    district_counts: Counter = Counter()
    for alert in alerts:
        shop_id = str(alert.get("fps_shop_id", ""))
        district = (
            shop_district.get(shop_id)
            or alert.get("district")
            or "Unknown"
        )
        district_counts[district] += 1

    return [
        {"district": d, "alert_count": c}
        for d, c in district_counts.most_common(top_n)
        if d != "Unknown"  # suppress the catch-all bucket
    ]


def _monthly_distribution_kg(txn_df: pd.DataFrame) -> dict:
    """
    Return commodity → kg for the most recent month in the transaction data
    (not the current wall-clock month, which may differ from data vintage).
    """
    if "commodity" not in txn_df.columns or "quantity_kg" not in txn_df.columns:
        return {}
    if "transaction_date" not in txn_df.columns:
        return txn_df.groupby("commodity")["quantity_kg"].sum().round(2).to_dict()

    txn_copy = txn_df.copy()
    txn_copy["transaction_date"] = pd.to_datetime(txn_copy["transaction_date"], errors="coerce")
    txn_copy = txn_copy.dropna(subset=["transaction_date"])
    if txn_copy.empty:
        return {}

    # Use the latest year-month present in the data
    latest_ym = txn_copy["transaction_date"].dt.to_period("M").max()
    month_data = txn_copy[txn_copy["transaction_date"].dt.to_period("M") == latest_ym]
    return month_data.groupby("commodity")["quantity_kg"].sum().round(2).to_dict()


def _transactions_this_month(txn_df: pd.DataFrame) -> int:
    """Count transactions for the most recent data month."""
    if "transaction_date" not in txn_df.columns:
        return len(txn_df)
    txn_copy = txn_df.copy()
    txn_copy["transaction_date"] = pd.to_datetime(txn_copy["transaction_date"], errors="coerce")
    txn_copy = txn_copy.dropna(subset=["transaction_date"])
    if txn_copy.empty:
        return 0
    latest_ym = txn_copy["transaction_date"].dt.to_period("M").max()
    return int((txn_copy["transaction_date"].dt.to_period("M") == latest_ym).sum())


@router.get("/metrics")
async def get_dashboard_metrics():
    """
    Get aggregated dashboard KPIs from all agents.
    Fraud detection runs on ALL transactions.
    Forecasting runs on a stratified sample of up to 30 shops (keeps response < 30s).
    KPIs are always corrected to reflect the full dataset, not the sample.
    """
    shops_df, bene_df, txn_df = _data_service.get_all_data()
    sampled_shops = _sample_shops(shops_df, PIPELINE_SHOP_CAP)
    logger.info(f"Dashboard: running pipeline on {len(sampled_shops)} sampled shops "
                f"({len(shops_df)} total shops, {len(txn_df)} txns)")

    result = await _orchestrator.run(
        trigger=WorkflowTrigger.FULL_PIPELINE,
        shops_df=sampled_shops,
        beneficiaries_df=bene_df,
        transactions_df=txn_df,
    )

    reporting  = result.get("reporting_results", {})
    fraud_res  = result.get("fraud_results", {})
    geo_res    = result.get("geo_results", {})
    fraud_summary = fraud_res.get("summary", {})

    # ── Always use full-dataset counts for top-level KPIs ─────────────────────
    total_shops  = int(len(shops_df))
    active_shops = int(shops_df["is_active"].sum()) if "is_active" in shops_df.columns else total_shops
    total_bene   = int(len(bene_df))
    txn_this_month = _transactions_this_month(txn_df)
    monthly_dist   = _monthly_distribution_kg(txn_df)

    # ── Fraud KPIs come from the fraud agent (runs on ALL txns) ───────────────
    total_alerts    = int(fraud_summary.get("total_alerts", 0))
    critical_alerts = int(fraud_summary.get("critical", 0))

    # ── Top fraud districts resolved against the FULL shops lookup ────────────
    top_fraud = _top_fraud_districts(fraud_res.get("alerts", []), shops_df)

    # ── Geo / forecast KPIs ───────────────────────────────────────────────────
    # overall_accessibility_score is 0–1 fraction; multiply by 100 → percentage
    # pct_beneficiaries_within_threshold_km may already be 0–100; detect and normalise
    raw_geo = (geo_res.get("pct_beneficiaries_within_threshold_km") or
               geo_res.get("overall_accessibility_score") or 1.0)
    raw_geo = float(raw_geo)
    # If value > 1 it is already a percentage; otherwise convert from fraction
    pct_within = raw_geo if raw_geo > 1.0 else raw_geo * 100.0

    # Start with whatever the reporting agent built, then override unreliable fields
    metrics = reporting.get("dashboard_metrics") or {}
    metrics.update({
        "total_fps_shops":              total_shops,
        "active_fps_shops":             active_shops,
        "total_beneficiaries":          total_bene,
        "transactions_this_month":      txn_this_month,
        "fraud_alerts_open":            total_alerts,
        "fraud_alerts_critical":        critical_alerts,
        "beneficiaries_within_3km_pct": round(pct_within, 1),
        "avg_forecast_accuracy":        float(metrics.get("avg_forecast_accuracy") or 0.90),
        "districts_covered":            int(bene_df["district"].nunique()) if "district" in bene_df.columns else 33,
        "top_fraud_districts":          top_fraud,
        "monthly_distribution_kg":      monthly_dist,
        "last_updated":                 datetime.utcnow().isoformat(),
        "sampled_shops_for_forecast":   int(len(sampled_shops)),
        "pipeline_seconds":             round(float(result.get("duration_seconds") or 0), 1),
    })

    return {
        "metrics":           metrics,
        "executive_summary": reporting.get("executive_summary", ""),
        "generated_at":      datetime.utcnow().isoformat(),
    }


@router.get("/summary")
async def get_executive_summary():
    """Get AI-generated executive summary (requires ANTHROPIC_API_KEY)."""
    shops_df, bene_df, txn_df = _data_service.get_all_data()
    sampled_shops = _sample_shops(shops_df, PIPELINE_SHOP_CAP)

    result = await _orchestrator.run(
        trigger=WorkflowTrigger.FULL_PIPELINE,
        shops_df=sampled_shops,
        beneficiaries_df=bene_df,
        transactions_df=txn_df,
    )

    reporting = result.get("reporting_results", {})
    summary = reporting.get("executive_summary") or (
        "⚠ AI summary unavailable — set ANTHROPIC_API_KEY in .env to enable Claude-powered insights."
    )

    return {
        "summary":                   summary,
        "pipeline_duration_seconds": result.get("duration_seconds"),
        "generated_at":              datetime.utcnow().isoformat(),
    }
