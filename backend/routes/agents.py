"""API routes for orchestrator and agent operations."""
import asyncio
import logging
import threading
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional, List, Dict, Any
from datetime import datetime

_logger = logging.getLogger(__name__)

from agents.orchestrator_agent import OrchestratorAgent, WorkflowTrigger
from agents.reporting_agent import ReportingAgent
from agents.fraud_detection_agent import FraudDetectionAgent
from agents.demand_forecast_agent import DemandForecastAgent
from agents.scoring_agent import ScoringAgent
from app.constants import CommodityType
from database.schemas import (
    AgentQueryRequest, AgentQueryResponse, OrchestratorStatus,
    ChatRequest, ChatResponse, ChatSessionInfo,
)
from services.data_ingestion import DataIngestionService

router = APIRouter(prefix="/api/v1/agents", tags=["Agents"])

# Module-level singletons (in production use dependency injection)
_orchestrator: Optional[OrchestratorAgent] = None
_reporting_agent: Optional[ReportingAgent] = None
_fraud_agent: Optional[FraudDetectionAgent] = None
_forecast_agent: Optional[DemandForecastAgent] = None
_scoring_agent: Optional[ScoringAgent] = None
_data_service: DataIngestionService = DataIngestionService()

# Cached pipeline results — updated each time the pipeline runs so /chat can
# retrieve context without re-running the full pipeline every request.
_last_pipeline_results: Dict[str, Any] = {
    "fraud_results": {},
    "forecast_results": {},
    "geo_results": {},
}


def get_orchestrator() -> OrchestratorAgent:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = OrchestratorAgent()
    return _orchestrator


def get_reporting_agent() -> ReportingAgent:
    global _reporting_agent
    if _reporting_agent is None:
        _reporting_agent = ReportingAgent()
    return _reporting_agent


def get_fraud_agent() -> FraudDetectionAgent:
    global _fraud_agent
    if _fraud_agent is None:
        _fraud_agent = FraudDetectionAgent()
    return _fraud_agent


def get_forecast_agent() -> DemandForecastAgent:
    global _forecast_agent
    if _forecast_agent is None:
        _forecast_agent = DemandForecastAgent()
    return _forecast_agent


def get_scoring_agent() -> ScoringAgent:
    global _scoring_agent
    if _scoring_agent is None:
        _scoring_agent = ScoringAgent()
    return _scoring_agent


@router.get("/status", response_model=OrchestratorStatus)
async def get_orchestrator_status():
    """Get current orchestrator and agent health status."""
    orch = get_orchestrator()
    return orch.get_status()


@router.post("/run/monthly")
async def run_monthly_pipeline(background_tasks: BackgroundTasks):
    """Trigger full monthly batch pipeline (async)."""
    orch = get_orchestrator()
    shops_df, bene_df, txn_df = _data_service.get_all_data()

    async def _run():
        global _last_pipeline_results
        result = await orch.run(
            trigger=WorkflowTrigger.MONTHLY_BATCH,
            shops_df=shops_df,
            beneficiaries_df=bene_df,
            transactions_df=txn_df,
        )
        _last_pipeline_results = {
            "fraud_results": result.get("fraud_results", {}),
            "forecast_results": result.get("forecast_results", {}),
            "geo_results": result.get("geo_results", {}),
        }

    background_tasks.add_task(_run)
    return {"message": "Monthly pipeline started", "trigger": "monthly", "started_at": datetime.utcnow().isoformat()}


@router.post("/run/full")
async def run_full_pipeline():
    """Run full pipeline synchronously and return results."""
    global _last_pipeline_results
    orch = get_orchestrator()
    shops_df, bene_df, txn_df = _data_service.get_all_data()

    result = await orch.run(
        trigger=WorkflowTrigger.FULL_PIPELINE,
        shops_df=shops_df,
        beneficiaries_df=bene_df,
        transactions_df=txn_df,
    )
    _last_pipeline_results = {
        "fraud_results": result.get("fraud_results", {}),
        "forecast_results": result.get("forecast_results", {}),
        "geo_results": result.get("geo_results", {}),
    }
    return result


@router.post("/query", response_model=AgentQueryResponse)
async def natural_language_query(request: AgentQueryRequest):
    """Answer a natural language query using the RAG Reporting Agent + Claude."""
    orch = get_orchestrator()
    shops_df, bene_df, txn_df = _data_service.get_all_data()

    result = await orch.run(
        trigger=WorkflowTrigger.NL_QUERY,
        shops_df=shops_df,
        beneficiaries_df=bene_df,
        transactions_df=txn_df,
        nl_query=request.query,
        session_id=request.session_id,
    )

    nl_response = result.get("reporting_results", {}).get("nl_query_response") or {}
    return {
        "query": request.query,
        "answer": nl_response.get("answer", "No response generated"),
        "agent_used": nl_response.get("agent_used", "reporting_rag"),
        "data": nl_response.get("rag_store_stats"),
        "generated_at": nl_response.get("generated_at") or datetime.utcnow().isoformat(),
    }


# ── RAG Multi-Turn Chat ────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Multi-turn RAG chatbot for PDS officials.

    Uses TF-IDF retrieval over indexed agent outputs + Claude for answer generation.
    Pass the same session_id across requests to maintain conversation history.

    On first call (cold RAG store), immediately indexes raw shop/beneficiary data
    (<1s) so the chatbot is usable instantly, then triggers the full fraud+forecast
    pipeline in the background. Subsequent calls use the fully-indexed store.
    """
    global _last_pipeline_results
    reporting = get_reporting_agent()
    shops_df, bene_df, txn_df = _data_service.get_all_data()

    # Warm up RAG store if cold — index raw data instantly, run pipeline in bg
    if not reporting.rag_store.is_fitted:
        # Index shops + beneficiaries immediately (<1s) so chat is usable now
        reporting.index_agent_outputs(
            fraud_results=_last_pipeline_results.get("fraud_results", {}),
            forecast_results=_last_pipeline_results.get("forecast_results", {}),
            geo_results=_last_pipeline_results.get("geo_results", {}),
            shops_df=shops_df,
            beneficiaries_df=bene_df,
        )
        # Kick off full pipeline in a background THREAD (not a coroutine) so it
        # runs in its own event loop and never blocks the main uvicorn event loop.
        def _bg_pipeline_thread():
            global _last_pipeline_results

            async def _pipeline():
                try:
                    orch = get_orchestrator()
                    result = await orch.run(
                        trigger=WorkflowTrigger.FULL_PIPELINE,
                        shops_df=shops_df,
                        beneficiaries_df=bene_df,
                        transactions_df=txn_df,
                    )
                    _last_pipeline_results.update({
                        "fraud_results": result.get("fraud_results", {}),
                        "forecast_results": result.get("forecast_results", {}),
                        "geo_results": result.get("geo_results", {}),
                    })
                    # Re-index the RAG store with the enriched pipeline results
                    _reporting = get_reporting_agent()
                    _shops_df, _bene_df, _ = _data_service.get_all_data()
                    _reporting.index_agent_outputs(
                        fraud_results=_last_pipeline_results.get("fraud_results", {}),
                        forecast_results=_last_pipeline_results.get("forecast_results", {}),
                        geo_results=_last_pipeline_results.get("geo_results", {}),
                        shops_df=_shops_df,
                        beneficiaries_df=_bene_df,
                    )
                    _logger.info("Background pipeline complete — RAG store re-indexed with full pipeline results")
                except Exception as exc:
                    _logger.warning(f"Background pipeline failed: {exc}")

            asyncio.run(_pipeline())

        threading.Thread(target=_bg_pipeline_thread, daemon=True).start()

    # Use the cached (most recent) pipeline results for RAG context
    nl_response = await reporting.answer_nl_query(
        query=request.query,
        shops_df=shops_df,
        beneficiaries_df=bene_df,
        fraud_results=_last_pipeline_results.get("fraud_results", {}),
        geo_results=_last_pipeline_results.get("geo_results", {}),
        forecast_results=_last_pipeline_results.get("forecast_results", {}),
        session_id=request.session_id,
    )

    return {
        "query": request.query,
        "answer": nl_response.get("answer", "No answer generated"),
        "session_id": nl_response.get("session_id", request.session_id),
        "conversation_turn": nl_response.get("conversation_turn", 0),
        "rag_sources": nl_response.get("rag_sources", []),
        "agent_used": nl_response.get("agent_used", "reporting_rag"),
        "generated_at": datetime.utcnow(),
    }


@router.get("/chat/sessions", response_model=List[ChatSessionInfo])
async def list_chat_sessions():
    """List all active chat sessions and their turn counts."""
    reporting = get_reporting_agent()
    return reporting.list_sessions()


@router.delete("/chat/sessions/{session_id}")
async def clear_chat_session(session_id: str):
    """Clear conversation history for a specific session."""
    reporting = get_reporting_agent()
    reporting.clear_session(session_id)
    return {"message": f"Session '{session_id}' cleared", "session_id": session_id}


@router.get("/chat/rag/stats")
async def rag_store_stats():
    """Return current RAG store statistics (document count, sources, vocab size)."""
    reporting = get_reporting_agent()
    return {
        "rag_store": reporting.rag_store.stats(),
        "rag_store_fitted": reporting.rag_store.is_fitted,
        "last_indexed_at": reporting._last_indexed_at,
        "active_sessions": len(reporting.list_sessions()),
    }


# ── Scoring Endpoints ─────────────────────────────────────────────────────────

@router.get("/scores/shops")
async def get_shop_scores(
    district: Optional[str] = None,
    band: Optional[str] = None,
    top_n: Optional[int] = None,
    bottom_n: Optional[int] = None,
):
    """
    Shop performance scores (0-100 composite).

    Query params:
      district : Filter by district name (case-insensitive)
      band     : Filter by performance band (Exemplary/Good/Average/Below Average/Critical)
      top_n    : Return only top N shops by score
      bottom_n : Return only bottom N shops by score
    """
    scoring = get_scoring_agent()
    shops_df, bene_df, txn_df = _data_service.get_all_data()

    fraud_alerts = _last_pipeline_results.get("fraud_results", {}).get("alerts", [])
    result = await scoring.run(
        transactions_df=txn_df,
        shops_df=shops_df,
        beneficiaries_df=bene_df,
        fraud_alerts=fraud_alerts,
    )

    scores = result["shop_scores"]

    # Apply filters
    if district:
        scores = [s for s in scores if str(s.get("district", "")).lower() == district.lower()]
    if band:
        scores = [s for s in scores if s.get("performance_band", "").lower() == band.lower()]
    if top_n:
        scores = sorted(scores, key=lambda x: x.get("total_score", 0), reverse=True)[:top_n]
    if bottom_n:
        scores = sorted(scores, key=lambda x: x.get("total_score", 0))[:bottom_n]

    return {
        "shop_scores":    scores,
        "total_shops":    len(scores),
        "summary":        result["shop_summary"],
        "generated_at":   result["generated_at"],
    }


@router.get("/scores/shops/{shop_id}")
async def get_shop_score_detail(shop_id: str):
    """Detailed performance score for a single FPS shop."""
    scoring = get_scoring_agent()
    shops_df, bene_df, txn_df = _data_service.get_all_data()

    fraud_alerts = _last_pipeline_results.get("fraud_results", {}).get("alerts", [])
    result = await scoring.run(
        transactions_df=txn_df,
        shops_df=shops_df,
        beneficiaries_df=bene_df,
        fraud_alerts=fraud_alerts,
    )

    matching = [
        s for s in result["shop_scores"]
        if str(s.get("fps_shop_id", "")) == str(shop_id)
    ]
    if not matching:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Shop '{shop_id}' not found")

    return {
        "shop":         matching[0],
        "generated_at": result["generated_at"],
    }


@router.get("/scores/districts")
async def get_district_scores(risk_level: Optional[str] = None):
    """
    District health index scores (0-100).

    Query params:
      risk_level : Filter by risk level (Critical/High/Moderate/Low)
    """
    scoring = get_scoring_agent()
    shops_df, bene_df, txn_df = _data_service.get_all_data()

    fraud_alerts = _last_pipeline_results.get("fraud_results", {}).get("alerts", [])
    result = await scoring.run(
        transactions_df=txn_df,
        shops_df=shops_df,
        beneficiaries_df=bene_df,
        fraud_alerts=fraud_alerts,
    )

    districts = result["district_scores"]
    if risk_level:
        districts = [d for d in districts if d.get("risk_level", "").lower() == risk_level.lower()]

    return {
        "district_scores":  districts,
        "total_districts":  len(districts),
        "summary":          result["district_summary"],
        "generated_at":     result["generated_at"],
    }
