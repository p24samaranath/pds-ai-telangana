"""
Orchestrator Agent — Master Coordinator
LangGraph-style stateful multi-agent orchestration for PDS optimization.
Coordinates Demand Forecast, Fraud Detection, Geospatial, and Reporting agents.
"""
import asyncio
import pandas as pd
from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime
from enum import Enum
import logging

from agents.demand_forecast_agent import DemandForecastAgent
from agents.fraud_detection_agent import FraudDetectionAgent
from agents.geospatial_agent import GeospatialAgent
from agents.reporting_agent import ReportingAgent
from app.constants import CommodityType
from app.config import settings

logger = logging.getLogger(__name__)


# ── Shared State Schema ───────────────────────────────────────────────────────

class OrchestratorState(TypedDict, total=False):
    """Shared memory passed between agents in the graph."""
    shops_df: Any
    beneficiaries_df: Any
    transactions_df: Any
    fraud_results: Dict
    forecast_results: Dict
    geo_results: Dict
    reporting_results: Dict
    nl_query: Optional[str]
    trigger: str           # "monthly", "realtime", "query", "geo_change"
    errors: List[str]
    started_at: str
    completed_at: Optional[str]


class WorkflowTrigger(str, Enum):
    MONTHLY_BATCH = "monthly"
    REALTIME_FRAUD = "realtime"
    NL_QUERY = "query"
    GEO_CHANGE = "geo_change"
    FULL_PIPELINE = "full"


# ── Orchestrator Agent ────────────────────────────────────────────────────────

class OrchestratorAgent:
    """
    Master coordinator for the PDS multi-agent system.

    Responsibilities:
    - Ingest data triggers and decide agent routing
    - Maintain shared state across agents
    - Escalate critical fraud alerts immediately
    - Generate executive summaries via Reporting Agent
    - Log every decision for audit trail
    """

    def __init__(self):
        self.demand_agent = DemandForecastAgent()
        self.fraud_agent = FraudDetectionAgent()
        self.geo_agent = GeospatialAgent()
        self.reporting_agent = ReportingAgent()

        # Shared memory (in production: Pinecone / Weaviate vector store)
        self._memory: Dict[str, Any] = {
            "last_fraud_check": None,
            "last_forecast_run": None,
            "last_geo_analysis": None,
            "pending_alerts": [],
            "historical_decisions": [],
        }

    # ── Memory ────────────────────────────────────────────────────────────────

    def _store_decision(self, agent: str, action: str, outcome: str):
        self._memory["historical_decisions"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "agent": agent,
            "action": action,
            "outcome": outcome,
        })
        # Keep last 100 decisions in memory
        self._memory["historical_decisions"] = self._memory["historical_decisions"][-100:]

    # ── Routing Logic ─────────────────────────────────────────────────────────

    def _route_agents(self, trigger: WorkflowTrigger) -> List[str]:
        """Decide which agents to invoke based on the trigger type."""
        routing = {
            WorkflowTrigger.MONTHLY_BATCH: [
                "fraud_detection", "demand_forecast", "geospatial", "reporting"
            ],
            WorkflowTrigger.REALTIME_FRAUD: ["fraud_detection"],
            WorkflowTrigger.NL_QUERY: ["reporting"],
            WorkflowTrigger.GEO_CHANGE: ["geospatial", "reporting"],
            WorkflowTrigger.FULL_PIPELINE: [
                "fraud_detection", "demand_forecast", "geospatial", "reporting"
            ],
        }
        return routing.get(trigger, ["reporting"])

    # ── Critical Alert Escalation ─────────────────────────────────────────────

    async def _handle_critical_alerts(self, fraud_results: Dict) -> List[Dict]:
        """
        Immediately escalate Critical fraud alerts.
        In production: auto-block cards via ePoS API + SMS to field officers.
        """
        critical = fraud_results.get("critical_alerts", [])
        escalated = []

        for alert in critical:
            # Check memory: has this dealer been flagged before?
            dealer_id = alert.get("fps_shop_id")
            prev_flags = [
                d for d in self._memory["historical_decisions"]
                if d.get("agent") == "fraud_detection"
                and str(dealer_id) in d.get("outcome", "")
            ]

            context = "repeat offender" if prev_flags else "first occurrence"
            action_taken = (
                f"Auto-escalated Critical alert {alert['alert_id']} "
                f"for shop {dealer_id} ({context}). "
                f"Recommended: {alert.get('recommended_action', 'Field verification')}"
            )

            self._store_decision(
                "orchestrator", "escalate_critical_alert", action_taken
            )
            self._memory["pending_alerts"].append(alert)

            escalated.append({
                **alert,
                "escalated_at": datetime.utcnow().isoformat(),
                "context": context,
                "orchestrator_action": action_taken,
            })
            logger.warning(f"CRITICAL ALERT ESCALATED: {alert['alert_id']} — {context}")

        return escalated

    # ── Workflow: Monthly Batch ───────────────────────────────────────────────

    async def _run_monthly_batch(self, state: OrchestratorState) -> OrchestratorState:
        """
        Workflow 1: Monthly stock planning
        1. Fraud detection on current month
        2. Demand forecasting for next 3 months
        3. Geospatial accessibility check
        4. Reporting + executive summary
        """
        shops_df = state["shops_df"]
        beneficiaries_df = state["beneficiaries_df"]
        transactions_df = state["transactions_df"]

        logger.info("Orchestrator: Starting monthly batch pipeline")

        # Step 1 — Fraud Detection
        fraud_results = await self.fraud_agent.run(transactions_df)
        state["fraud_results"] = fraud_results
        self._memory["last_fraud_check"] = datetime.utcnow().isoformat()

        # Step 2 — Demand Forecasting (parallel per commodity)
        shops_list = shops_df.to_dict("records") if not shops_df.empty else []
        forecast_results = await self.demand_agent.run(
            shops=shops_list,
            transactions_df=transactions_df,
            commodities=[CommodityType.RICE, CommodityType.WHEAT],
            months_ahead=3,
        )
        state["forecast_results"] = forecast_results
        self._memory["last_forecast_run"] = datetime.utcnow().isoformat()

        # Step 3 — Geospatial
        geo_results = await self.geo_agent.run(
            shops_df=shops_df,
            beneficiaries_df=beneficiaries_df,
            transactions_df=transactions_df,
        )
        state["geo_results"] = geo_results
        self._memory["last_geo_analysis"] = datetime.utcnow().isoformat()

        # Step 4 — Reporting
        reporting_results = await self.reporting_agent.run(
            shops_df=shops_df,
            beneficiaries_df=beneficiaries_df,
            transactions_df=transactions_df,
            fraud_results=fraud_results,
            forecast_results=forecast_results,
            geo_results=geo_results,
            nl_query=state.get("nl_query"),
        )
        state["reporting_results"] = reporting_results

        self._store_decision("orchestrator", "monthly_batch", "completed_successfully")
        return state

    # ── Workflow: Real-Time Fraud ─────────────────────────────────────────────

    async def _run_realtime_fraud(self, state: OrchestratorState) -> OrchestratorState:
        """Workflow 2: Stream-mode fraud check for incoming transactions."""
        fraud_results = await self.fraud_agent.run(state["transactions_df"])
        state["fraud_results"] = fraud_results

        # Immediately escalate Critical alerts
        if fraud_results.get("critical_alerts"):
            await self._handle_critical_alerts(fraud_results)

        return state

    # ── Workflow: NL Query ────────────────────────────────────────────────────

    async def _run_nl_query(self, state: OrchestratorState) -> OrchestratorState:
        """Workflow 3: Answer a natural language query from an official."""
        reporting_results = await self.reporting_agent.run(
            shops_df=state.get("shops_df", pd.DataFrame()),
            beneficiaries_df=state.get("beneficiaries_df", pd.DataFrame()),
            transactions_df=state.get("transactions_df", pd.DataFrame()),
            fraud_results=state.get("fraud_results", {}),
            forecast_results=state.get("forecast_results", {}),
            geo_results=state.get("geo_results", {}),
            nl_query=state.get("nl_query"),
        )
        state["reporting_results"] = reporting_results
        return state

    # ── Main Entry Point ──────────────────────────────────────────────────────

    async def run(
        self,
        trigger: WorkflowTrigger,
        shops_df: pd.DataFrame,
        beneficiaries_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        nl_query: Optional[str] = None,
        commodities: Optional[List[CommodityType]] = None,
    ) -> Dict[str, Any]:
        """
        Primary orchestrator entry point. Accepts a trigger type and all data,
        routes to appropriate sub-agents, and returns consolidated results.
        """
        started_at = datetime.utcnow()
        logger.info(f"Orchestrator started — trigger: {trigger.value}")

        state: OrchestratorState = {
            "shops_df": shops_df,
            "beneficiaries_df": beneficiaries_df,
            "transactions_df": transactions_df,
            "nl_query": nl_query,
            "trigger": trigger.value,
            "errors": [],
            "started_at": started_at.isoformat(),
            "completed_at": None,
            "fraud_results": {},
            "forecast_results": {},
            "geo_results": {},
            "reporting_results": {},
        }

        try:
            if trigger in (WorkflowTrigger.MONTHLY_BATCH, WorkflowTrigger.FULL_PIPELINE):
                state = await self._run_monthly_batch(state)
            elif trigger == WorkflowTrigger.REALTIME_FRAUD:
                state = await self._run_realtime_fraud(state)
            elif trigger == WorkflowTrigger.NL_QUERY:
                state = await self._run_nl_query(state)
            elif trigger == WorkflowTrigger.GEO_CHANGE:
                geo_results = await self.geo_agent.run(
                    shops_df=shops_df,
                    beneficiaries_df=beneficiaries_df,
                    transactions_df=transactions_df,
                    force_rerun=True,
                )
                state["geo_results"] = geo_results
        except Exception as e:
            logger.error(f"Orchestrator error: {e}", exc_info=True)
            state["errors"].append(str(e))

        completed_at = datetime.utcnow()
        duration_s = (completed_at - started_at).total_seconds()
        state["completed_at"] = completed_at.isoformat()

        logger.info(f"Orchestrator completed in {duration_s:.1f}s — trigger: {trigger.value}")

        return {
            "trigger": trigger.value,
            "started_at": state["started_at"],
            "completed_at": state["completed_at"],
            "duration_seconds": round(duration_s, 2),
            "fraud_results": state.get("fraud_results", {}),
            "forecast_results": state.get("forecast_results", {}),
            "geo_results": state.get("geo_results", {}),
            "reporting_results": state.get("reporting_results", {}),
            "errors": state.get("errors", []),
            "pending_alerts_count": len(self._memory["pending_alerts"]),
            "memory_snapshot": {
                "last_fraud_check": self._memory["last_fraud_check"],
                "last_forecast_run": self._memory["last_forecast_run"],
                "last_geo_analysis": self._memory["last_geo_analysis"],
            },
        }

    def get_status(self) -> Dict[str, Any]:
        """Return current orchestrator health / status."""
        return {
            "status": "running",
            "active_agents": ["demand_forecast", "fraud_detection", "geospatial", "reporting"],
            "last_fraud_check": self._memory["last_fraud_check"],
            "last_forecast_run": self._memory["last_forecast_run"],
            "last_geo_analysis": self._memory["last_geo_analysis"],
            "pending_alerts": len(self._memory["pending_alerts"]),
            "decisions_logged": len(self._memory["historical_decisions"]),
            "system_health": "healthy",
        }
