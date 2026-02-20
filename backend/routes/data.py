"""
Data Management API Routes
===========================
GET  /api/v1/data/status          — what data is loaded and its date range
GET  /api/v1/data/manifest        — list every downloadable file on Telangana Open Data
POST /api/v1/data/fetch/latest    — download the most recent month for all 3 datasets
POST /api/v1/data/fetch/range     — download a specific year-month range
POST /api/v1/data/fetch/all       — download everything (slow, background task)
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Query
from pydantic import BaseModel

from services.data_ingestion import DataIngestionService
from services.telangana_fetcher import TelanganaFetcher

router = APIRouter(prefix="/api/v1/data", tags=["Data Management"])
logger = logging.getLogger(__name__)

_data_svc   = DataIngestionService()
_fetcher    = TelanganaFetcher()

# Track background job state
_job_status: dict = {"running": False, "last_run": None, "last_result": None}


# ── Request / Response models ──────────────────────────────────────────────────

class FetchRangeRequest(BaseModel):
    from_year:  int = 2024
    from_month: int = 1
    to_year:    int = 2025
    to_month:   int = 12


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.get("/status")
async def data_status():
    """
    Returns the current data source for each dataset (real CSV vs synthetic)
    plus row counts and date ranges.
    """
    status = _data_svc.data_status()
    status["checked_at"] = datetime.utcnow().isoformat()
    return status


@router.get("/manifest")
async def get_manifest():
    """
    Calls the Telangana Open Data API and returns a full manifest of every
    available CSV file across all three datasets. Useful for seeing what
    date ranges exist before triggering a download.
    """
    manifest = _fetcher.discover()
    summary = {
        name: {
            "total_files": len(files),
            "earliest": _earliest(files),
            "latest":   _latest(files),
            "files":    files[:5],          # show first 5 for brevity
            "note":     f"...and {max(0, len(files)-5)} more" if len(files) > 5 else "",
        }
        for name, files in manifest.items()
    }
    return {"manifest": summary, "fetched_at": datetime.utcnow().isoformat()}


@router.post("/fetch/latest")
async def fetch_latest():
    """
    Download the single most-recent CSV for each dataset and write master files
    (fps_shops.csv, transactions.csv, beneficiaries.csv) to data/raw/.
    Fast — typically 3 HTTP calls.
    """
    if _job_status["running"]:
        return {"error": "A download job is already running. Try again shortly."}

    _job_status["running"] = True
    try:
        result = _fetcher.fetch_and_save(mode="latest")
        _job_status["last_result"] = result
        _job_status["last_run"]    = datetime.utcnow().isoformat()
        return {
            "status":  "success",
            "rows_downloaded": result,
            "message": "Master files updated in data/raw/. Restart the API or call any agent endpoint to use new data.",
            "completed_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"fetch/latest failed: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}
    finally:
        _job_status["running"] = False


@router.post("/fetch/range")
async def fetch_range(req: FetchRangeRequest):
    """
    Download all files within a year-month window and concatenate into master files.

    Example body:
        {"from_year": 2024, "from_month": 1, "to_year": 2024, "to_month": 12}
    """
    if _job_status["running"]:
        return {"error": "A download job is already running."}

    _job_status["running"] = True
    try:
        result = _fetcher.fetch_and_save(
            mode="range",
            from_year=req.from_year, from_month=req.from_month,
            to_year=req.to_year,     to_month=req.to_month,
        )
        _job_status["last_result"] = result
        _job_status["last_run"]    = datetime.utcnow().isoformat()
        return {
            "status":  "success",
            "range":   f"{req.from_year}-{req.from_month:02d} → {req.to_year}-{req.to_month:02d}",
            "rows_downloaded": result,
            "completed_at": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"fetch/range failed: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}
    finally:
        _job_status["running"] = False


@router.post("/fetch/all")
async def fetch_all(background_tasks: BackgroundTasks):
    """
    Download every file from 2018 to present (background task — can take minutes).
    Poll /api/v1/data/status to see when new data is available.
    """
    if _job_status["running"]:
        return {"error": "A download job is already running."}

    def _run_all():
        _job_status["running"] = True
        try:
            result = _fetcher.fetch_and_save(mode="all")
            _job_status["last_result"] = result
            _job_status["last_run"]    = datetime.utcnow().isoformat()
            logger.info(f"fetch/all complete: {result}")
        except Exception as e:
            logger.error(f"fetch/all failed: {e}", exc_info=True)
        finally:
            _job_status["running"] = False

    background_tasks.add_task(_run_all)
    return {
        "status":  "started",
        "message": "Full historical download started in background (2018–present). "
                   "Poll GET /api/v1/data/status to track progress.",
        "started_at": datetime.utcnow().isoformat(),
    }


@router.get("/job-status")
async def job_status():
    """Check whether a background download is currently running."""
    return {**_job_status, "checked_at": datetime.utcnow().isoformat()}


@router.get("/visualize")
async def visualize_data():
    """
    Compute real chart-ready statistics from loaded DataFrames.
    Returns data for:
      - District distribution volumes (bar chart)
      - Commodity breakdown (pie chart)
      - Monthly transaction trend (line chart)
      - Shop network health (active vs inactive)
      - Card type distribution
      - Top 10 districts by beneficiaries
    """
    try:
        shops, bene, txn = _data_svc.get_all_data()

        # ── 1. Shop network health ────────────────────────────────────────────
        total_shops   = len(shops)
        active_shops  = int(shops["is_active"].sum()) if "is_active" in shops.columns else total_shops
        inactive_shops = total_shops - active_shops
        shop_health = {
            "total":    total_shops,
            "active":   active_shops,
            "inactive": inactive_shops,
            "active_pct": round(active_shops / max(total_shops, 1) * 100, 1),
        }

        # ── 2. District-level distribution volumes ────────────────────────────
        district_volumes = []
        if "district" in txn.columns and "quantity_kg" in txn.columns:
            dv = (
                txn.groupby("district")["quantity_kg"]
                .sum()
                .reset_index()
                .rename(columns={"quantity_kg": "total_kg"})
                .sort_values("total_kg", ascending=False)
                .head(15)
            )
            dv["total_kg"] = dv["total_kg"].round(1)
            district_volumes = dv.to_dict("records")

        # ── 3. Commodity breakdown ────────────────────────────────────────────
        commodity_breakdown = []
        if "commodity" in txn.columns and "quantity_kg" in txn.columns:
            cb = (
                txn.groupby("commodity")["quantity_kg"]
                .sum()
                .reset_index()
                .rename(columns={"quantity_kg": "total_kg"})
                .sort_values("total_kg", ascending=False)
            )
            cb["total_kg"] = cb["total_kg"].round(1)
            commodity_breakdown = cb.to_dict("records")

        # ── 4. Monthly transaction trend ──────────────────────────────────────
        monthly_trend = []
        if "transaction_date" in txn.columns:
            txn_copy = txn.copy()
            txn_copy["transaction_date"] = pd.to_datetime(
                txn_copy["transaction_date"], errors="coerce"
            )
            txn_copy = txn_copy.dropna(subset=["transaction_date"])
            txn_copy["ym"] = txn_copy["transaction_date"].dt.to_period("M").astype(str)
            mt = (
                txn_copy.groupby("ym")
                .agg(transactions=("transaction_id" if "transaction_id" in txn_copy.columns else "fps_shop_id", "count"),
                     total_kg=("quantity_kg", "sum"))
                .reset_index()
                .sort_values("ym")
                .tail(24)
            )
            mt["total_kg"] = mt["total_kg"].round(1)
            monthly_trend = mt.to_dict("records")

        # ── 5. Card type distribution ─────────────────────────────────────────
        card_type_dist = []
        if "card_type" in bene.columns:
            ct = (
                bene["card_type"].value_counts().reset_index()
                .rename(columns={"index": "card_type", "card_type": "count",
                                  "count": "count"})
            )
            # pandas ≥ 2.0 value_counts() already named correctly
            if "card_type" not in ct.columns:
                ct.columns = ["card_type", "count"]
            card_type_dist = ct.to_dict("records")

        # ── 6. Top districts by beneficiaries ────────────────────────────────
        top_districts_bene = []
        if "district" in bene.columns:
            td = (
                bene.groupby("district")
                .size()
                .reset_index(name="beneficiary_count")
                .sort_values("beneficiary_count", ascending=False)
                .head(10)
            )
            top_districts_bene = td.to_dict("records")

        # ── 7. District-level active shops ────────────────────────────────────
        district_shops = []
        if "district" in shops.columns:
            ds = shops.groupby("district").agg(
                total=("shop_id", "count"),
                active=("is_active", "sum") if "is_active" in shops.columns else ("shop_id", "count"),
            ).reset_index()
            ds["active"] = ds["active"].astype(int)
            ds = ds.sort_values("total", ascending=False).head(15)
            district_shops = ds.to_dict("records")

        # ── 8. Summary KPIs ───────────────────────────────────────────────────
        total_volume_kg = round(float(txn["quantity_kg"].sum()), 1) if "quantity_kg" in txn.columns else 0
        total_bene      = len(bene)
        total_txn       = len(txn)

        return {
            "generated_at":        datetime.utcnow().isoformat(),
            "kpis": {
                "total_shops":        total_shops,
                "active_shops":       active_shops,
                "total_beneficiaries": total_bene,
                "total_transactions": total_txn,
                "total_volume_kg":    total_volume_kg,
            },
            "shop_health":          shop_health,
            "district_volumes":     district_volumes,
            "commodity_breakdown":  commodity_breakdown,
            "monthly_trend":        monthly_trend,
            "card_type_distribution": card_type_dist,
            "top_districts_beneficiaries": top_districts_bene,
            "district_shops":       district_shops,
        }
    except Exception as e:
        logger.error(f"visualize failed: {e}", exc_info=True)
        return {"error": str(e), "generated_at": datetime.utcnow().isoformat()}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _earliest(files: list) -> Optional[str]:
    valid = [f for f in files if f.get("year") and f.get("month")]
    if not valid:
        return None
    f = min(valid, key=lambda x: (x["year"], x["month"]))
    return f"{f['year']}-{f['month']:02d}"

def _latest(files: list) -> Optional[str]:
    valid = [f for f in files if f.get("year") and f.get("month")]
    if not valid:
        return None
    f = max(valid, key=lambda x: (x["year"], x["month"]))
    return f"{f['year']}-{f['month']:02d}"
