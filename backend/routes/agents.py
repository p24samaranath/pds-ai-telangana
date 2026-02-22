"""API routes for orchestrator and agent operations."""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional, List, Dict, Any
from datetime import datetime

from agents.orchestrator_agent import OrchestratorAgent, WorkflowTrigger
from agents.reporting_agent import ReportingAgent
from database.schemas import (
    AgentQueryRequest, AgentQueryResponse, OrchestratorStatus,
    ChatRequest, ChatResponse, ChatSessionInfo,
)
from services.data_ingestion import DataIngestionService

router = APIRouter(prefix="/api/v1/agents", tags=["Agents"])

# Module-level singletons (in production use dependency injection)
_orchestrator: Optional[OrchestratorAgent] = None
_reporting_agent: Optional[ReportingAgent] = None
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
async def chat(request: ChatRequest):
    """
    Multi-turn RAG chatbot for PDS officials.

    Uses TF-IDF retrieval over indexed agent outputs + Claude for answer generation.
    Pass the same session_id across requests to maintain conversation history.

    On first call (cold RAG store), triggers a full pipeline run to index fresh data.
    Subsequent calls reuse the cached index — no pipeline re-run needed.
    """
    global _last_pipeline_results
    reporting = get_reporting_agent()
    shops_df, bene_df, txn_df = _data_service.get_all_data()

    # Warm up RAG store if cold (first request after startup)
    if not reporting.rag_store.is_fitted:
        orch = get_orchestrator()
        pipeline_result = await orch.run(
            trigger=WorkflowTrigger.FULL_PIPELINE,
            shops_df=shops_df,
            beneficiaries_df=bene_df,
            transactions_df=txn_df,
        )
        _last_pipeline_results = {
            "fraud_results": pipeline_result.get("fraud_results", {}),
            "forecast_results": pipeline_result.get("forecast_results", {}),
            "geo_results": pipeline_result.get("geo_results", {}),
        }
        # reporting_agent.run() inside orch already indexes — no double-index needed

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
