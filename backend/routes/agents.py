"""API routes for orchestrator and agent operations."""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional
from datetime import datetime

from agents.orchestrator_agent import OrchestratorAgent, WorkflowTrigger
from database.schemas import AgentQueryRequest, AgentQueryResponse, OrchestratorStatus
from services.data_ingestion import DataIngestionService

router = APIRouter(prefix="/api/v1/agents", tags=["Agents"])

# Module-level singletons (in production use dependency injection)
_orchestrator: Optional[OrchestratorAgent] = None
_data_service: DataIngestionService = DataIngestionService()


def get_orchestrator() -> OrchestratorAgent:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = OrchestratorAgent()
    return _orchestrator


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
        await orch.run(
            trigger=WorkflowTrigger.MONTHLY_BATCH,
            shops_df=shops_df,
            beneficiaries_df=bene_df,
            transactions_df=txn_df,
        )

    background_tasks.add_task(_run)
    return {"message": "Monthly pipeline started", "trigger": "monthly", "started_at": datetime.utcnow().isoformat()}


@router.post("/run/full")
async def run_full_pipeline():
    """Run full pipeline synchronously and return results (may be slow)."""
    orch = get_orchestrator()
    shops_df, bene_df, txn_df = _data_service.get_all_data()

    result = await orch.run(
        trigger=WorkflowTrigger.FULL_PIPELINE,
        shops_df=shops_df,
        beneficiaries_df=bene_df,
        transactions_df=txn_df,
    )
    return result


@router.post("/query", response_model=AgentQueryResponse)
async def natural_language_query(request: AgentQueryRequest):
    """Answer a natural language query using the Reporting Agent + Claude."""
    orch = get_orchestrator()
    shops_df, bene_df, txn_df = _data_service.get_all_data()

    result = await orch.run(
        trigger=WorkflowTrigger.NL_QUERY,
        shops_df=shops_df,
        beneficiaries_df=bene_df,
        transactions_df=txn_df,
        nl_query=request.query,
    )

    nl_response = result.get("reporting_results", {}).get("nl_query_response", {})
    return {
        "query": request.query,
        "answer": nl_response.get("answer", "No response generated"),
        "agent_used": nl_response.get("agent_used", "reporting"),
        "data": nl_response.get("data"),
        "generated_at": datetime.utcnow(),
    }
