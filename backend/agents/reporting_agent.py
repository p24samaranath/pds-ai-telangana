"""
Reporting & Alerting Agent
Synthesises outputs from all agents into dashboards, reports, and NL query responses.

Phase 3 upgrade: RAG-enabled NL chatbot
- Indexes all agent outputs into an in-memory TF-IDF vector store (RAGStore)
- Retrieves the top-k most relevant chunks before every LLM call
- Maintains per-session conversation history for multi-turn dialogue

LLM Fallback Chain (in priority order):
  1. Anthropic Claude  — primary; used when ANTHROPIC_API_KEY is set
  2. Google Gemini     — fallback; used when GEMINI_API_KEY is set and Claude fails/unavailable
  3. Rule-based text   — final safety net; always works, no API key needed
"""
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# ── LLM client imports (each guarded so missing packages never crash startup) ──

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from google import genai as google_genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from app.config import settings
from services.rag_store import RAGStore

logger = logging.getLogger(__name__)

# Number of RAG chunks to include in each LLM prompt
RAG_TOP_K = 6
# Maximum conversation turns kept per session (to stay within context limits)
MAX_HISTORY_TURNS = 10


class ReportingAgent:
    """
    Synthesises multi-agent outputs into human-readable reports and
    answers natural language queries from PDS officials.

    RAG chatbot architecture:
      1. index() — agent outputs → RAGStore (TF-IDF vectors)
      2. answer_nl_query() — query → retrieve top-k chunks → LLM with context
      3. Conversation history — per session_id, up to MAX_HISTORY_TURNS turns

    LLM fallback chain:
      Anthropic Claude  →  Google Gemini  →  Rule-based text
    """

    def __init__(self):
        # ── Anthropic client ────────────────────────────────────────────────
        self.anthropic_client: Optional[Any] = None
        if ANTHROPIC_AVAILABLE and settings.ANTHROPIC_API_KEY:
            try:
                self.anthropic_client = Anthropic(api_key=settings.ANTHROPIC_API_KEY)
                logger.info("ReportingAgent: Anthropic Claude client initialised")
            except Exception as e:
                logger.warning(f"ReportingAgent: Anthropic init failed — {e}")

        # ── Google Gemini client (google-genai SDK) ─────────────────────────
        self.gemini_client: Optional[Any] = None
        if GEMINI_AVAILABLE and settings.GEMINI_API_KEY:
            try:
                self.gemini_client = google_genai.Client(api_key=settings.GEMINI_API_KEY)
                logger.info(
                    f"ReportingAgent: Google Gemini client initialised "
                    f"(model={settings.GEMINI_MODEL})"
                )
            except Exception as e:
                logger.warning(f"ReportingAgent: Gemini init failed — {e}")

        # Log active LLM
        if self.anthropic_client:
            logger.info("ReportingAgent: active LLM = Anthropic Claude (primary)")
        elif self.gemini_client:
            logger.info("ReportingAgent: active LLM = Google Gemini (fallback)")
        else:
            logger.warning(
                "ReportingAgent: no LLM API key configured — "
                "using rule-based fallback only. "
                "Set ANTHROPIC_API_KEY or GEMINI_API_KEY in .env"
            )

        self.rag_store = RAGStore()
        # conversation_history: {session_id: [{"role": ..., "content": ...}, ...]}
        self._conversation_history: Dict[str, List[Dict[str, str]]] = {}
        self._last_indexed_at: Optional[str] = None

    # ── LLM Call Helper ────────────────────────────────────────────────────────

    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        max_tokens: int = 800,
    ) -> tuple[str, str]:
        """
        Try LLMs in priority order: Anthropic → Gemini → fallback.

        Returns (answer_text, llm_used) where llm_used is one of:
          "claude", "gemini", "fallback"
        """
        # 1. Anthropic Claude
        if self.anthropic_client:
            try:
                response = self.anthropic_client.messages.create(
                    model=settings.CLAUDE_MODEL,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=messages,
                )
                return response.content[0].text, "claude"
            except Exception as e:
                logger.warning(
                    f"Anthropic Claude call failed ({e}); trying Gemini fallback…"
                )

        # 2. Google Gemini (google-genai SDK)
        if self.gemini_client:
            try:
                from google.genai import types as genai_types

                # Build a flat contents list for the new SDK.
                # Convert role "assistant" → "model" as Gemini requires.
                # Prepend the system instruction into the first user turn.
                contents = []
                for i, msg in enumerate(messages):
                    role = "model" if msg["role"] == "assistant" else "user"
                    text = msg["content"]
                    # Inject system prompt before the first user message
                    if i == 0 and msg["role"] == "user":
                        text = f"{system_prompt}\n\n{text}"
                    contents.append(
                        genai_types.Content(
                            role=role,
                            parts=[genai_types.Part(text=text)],
                        )
                    )

                response = self.gemini_client.models.generate_content(
                    model=settings.GEMINI_MODEL,
                    contents=contents,
                    config=genai_types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                    ),
                )
                return response.text, "gemini"
            except Exception as e:
                logger.warning(
                    f"Google Gemini call failed ({e}); falling back to rule-based response"
                )

        # 3. Rule-based fallback (always succeeds)
        return None, "fallback"

    # ── RAG Indexing ───────────────────────────────────────────────────────────

    def index_agent_outputs(
        self,
        fraud_results: Dict,
        forecast_results: Dict,
        geo_results: Dict,
        shops_df: Optional[pd.DataFrame] = None,
        beneficiaries_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Index all agent outputs into the RAG store.
        Call this after each pipeline run so the chatbot has up-to-date context.
        """
        n_docs = self.rag_store.index(
            fraud_results=fraud_results,
            forecast_results=forecast_results,
            geo_results=geo_results,
            shops_df=shops_df,
            beneficiaries_df=beneficiaries_df,
        )
        self._last_indexed_at = datetime.utcnow().isoformat()
        logger.info(f"ReportingAgent: RAG store indexed {n_docs} documents")
        return {
            "indexed_documents": n_docs,
            "indexed_at": self._last_indexed_at,
            "store_stats": self.rag_store.stats(),
        }

    # ── Dashboard Metrics ──────────────────────────────────────────────────────

    def build_dashboard_metrics(
        self,
        shops_df: pd.DataFrame,
        beneficiaries_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        fraud_results: Dict,
        forecast_results: Dict,
        geo_results: Dict,
    ) -> Dict[str, Any]:
        """Aggregate all agent outputs into a single dashboard payload."""
        total_shops = len(shops_df)
        active_shops = int(shops_df["is_active"].sum()) if "is_active" in shops_df.columns else total_shops
        total_bene = len(beneficiaries_df)

        now = datetime.utcnow()
        txn_month = transactions_df.copy()
        if "transaction_date" in txn_month.columns:
            txn_month["transaction_date"] = pd.to_datetime(txn_month["transaction_date"])
            txn_month = txn_month[
                (txn_month["transaction_date"].dt.year == now.year)
                & (txn_month["transaction_date"].dt.month == now.month)
            ]

        fraud_summary = fraud_results.get("summary", {})
        total_alerts = fraud_summary.get("total_alerts", 0)
        critical_alerts = fraud_summary.get("critical", 0)
        fraud_rings = fraud_summary.get("fraud_rings_detected", 0)

        forecasts = forecast_results.get("forecasts", [])
        avg_accuracy = 0.90  # Placeholder until actual MAPE is computed

        pct_within_3km = geo_results.get("pct_beneficiaries_within_threshold_km", 0)

        monthly_dist = {}
        if "commodity" in transactions_df.columns and "quantity_kg" in transactions_df.columns:
            monthly_dist = (
                txn_month.groupby("commodity")["quantity_kg"]
                .sum()
                .round(2)
                .to_dict()
            )

        fraud_alerts = fraud_results.get("alerts", [])
        district_fraud: Dict[str, int] = {}
        for alert in fraud_alerts:
            shop_id = alert.get("fps_shop_id")
            if shop_id and not shops_df.empty:
                match = shops_df[shops_df["shop_id"].astype(str) == str(shop_id)]
                if not match.empty:
                    district = match.iloc[0].get("district", "Unknown")
                    district_fraud[district] = district_fraud.get(district, 0) + 1

        top_fraud_districts = [
            {"district": d, "alert_count": c}
            for d, c in sorted(district_fraud.items(), key=lambda x: -x[1])[:5]
        ]

        return {
            "total_fps_shops": total_shops,
            "active_fps_shops": active_shops,
            "total_beneficiaries": total_bene,
            "transactions_this_month": len(txn_month),
            "fraud_alerts_open": total_alerts,
            "fraud_alerts_critical": critical_alerts,
            "fraud_rings_detected": fraud_rings,
            "avg_forecast_accuracy": avg_accuracy,
            "beneficiaries_within_3km_pct": pct_within_3km,
            "districts_covered": beneficiaries_df["district"].nunique()
                if "district" in beneficiaries_df.columns else 0,
            "top_fraud_districts": top_fraud_districts,
            "monthly_distribution_kg": monthly_dist,
            "last_updated": now.isoformat(),
        }

    # ── Executive Summary ──────────────────────────────────────────────────────

    async def generate_executive_summary(
        self,
        fraud_results: Dict,
        forecast_results: Dict,
        geo_results: Dict,
        district: Optional[str] = None,
    ) -> str:
        """
        Generate a human-readable executive summary.

        Tries Claude → Gemini → rule-based fallback.
        """
        context = f"""
PDS System Status Report — {datetime.utcnow().strftime('%B %Y')}
District focus: {district or 'All Telangana'}

FRAUD DETECTION:
- Total alerts: {fraud_results.get('summary', {}).get('total_alerts', 0)}
- Critical: {fraud_results.get('summary', {}).get('critical', 0)}
- High: {fraud_results.get('summary', {}).get('high', 0)}
- Fraud rings detected: {fraud_results.get('summary', {}).get('fraud_rings_detected', 0)}
- Transactions analysed: {fraud_results.get('summary', {}).get('transactions_analysed', 0)}

DEMAND FORECAST:
- Total forecasts generated: {forecast_results.get('total_forecasts', 0)}
- Risk flags (over/understock): {forecast_results.get('total_risk_flags', 0)}

GEOSPATIAL:
- Accessibility score: {geo_results.get('overall_accessibility_score', 0):.1%}
- Underserved beneficiaries: {geo_results.get('underserved_count', 0)}
- New FPS recommendations: {len(geo_results.get('new_location_recommendations', []))}
- Underperforming shops: {len(geo_results.get('underperforming_shops', []))}
"""
        system_prompt = (
            "You are a senior analyst for India's Public Distribution System. "
            "Write a concise 3-paragraph executive summary for district officials "
            "based on this data. Highlight the most urgent actions needed."
        )
        messages = [{"role": "user", "content": context}]

        answer, llm_used = self._call_llm(messages, system_prompt, max_tokens=600)

        if llm_used == "fallback" or answer is None:
            logger.info("generate_executive_summary: using rule-based fallback")
            return self._fallback_summary(fraud_results, forecast_results, geo_results)

        logger.info(f"generate_executive_summary: answered via {llm_used}")
        return answer

    def _fallback_summary(self, fraud: Dict, forecast: Dict, geo: Dict) -> str:
        crit = fraud.get("summary", {}).get("critical", 0)
        rings = fraud.get("summary", {}).get("fraud_rings_detected", 0)
        risk = forecast.get("total_risk_flags", 0)
        underserved = geo.get("underserved_count", 0)
        return (
            f"PDS System Report — {datetime.utcnow().strftime('%B %Y')}\n\n"
            f"Fraud Detection: {fraud.get('summary', {}).get('total_alerts', 0)} alerts raised, "
            f"including {crit} critical cases and {rings} coordinated fraud rings requiring immediate attention.\n\n"
            f"Demand Forecasting: {forecast.get('total_forecasts', 0)} forecasts generated "
            f"with {risk} stock risk flags identified.\n\n"
            f"Geospatial: {underserved} beneficiaries are currently underserved (>5 km from FPS). "
            f"Accessibility score: {geo.get('overall_accessibility_score', 0):.1%}."
        )

    # ── RAG-Powered NL Query ───────────────────────────────────────────────────

    async def answer_nl_query(
        self,
        query: str,
        shops_df: pd.DataFrame,
        beneficiaries_df: pd.DataFrame,
        fraud_results: Dict,
        geo_results: Dict,
        forecast_results: Dict,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        """
        Answer a natural language question from a PDS official using RAG + LLM.

        Flow:
          1. Retrieve top-k relevant chunks from RAGStore
          2. Build LLM prompt with retrieved context + conversation history
          3. Call LLM (Claude → Gemini → rule-based fallback)
          4. Append turn to session history
        """
        # --- Step 1: Auto-index if the store is empty (first call) ---
        if not self.rag_store.is_fitted:
            self.index_agent_outputs(
                fraud_results=fraud_results,
                forecast_results=forecast_results,
                geo_results=geo_results,
                shops_df=shops_df,
                beneficiaries_df=beneficiaries_df,
            )

        # --- Step 2: Retrieve relevant context ---
        retrieved_context = self.rag_store.retrieve_as_text(query, k=RAG_TOP_K)
        rag_chunks = self.rag_store.retrieve(query, k=RAG_TOP_K)
        rag_sources = list({doc.source for doc, _ in rag_chunks})

        # --- Step 3: Build conversation history for this session ---
        history = self._get_history(session_id)

        # Build the user turn: retrieved context + question
        user_content = (
            f"RETRIEVED CONTEXT (from PDS knowledge base — use this to answer):\n"
            f"{retrieved_context}\n\n"
            f"OFFICIAL QUERY: {query}"
        )

        messages = history + [{"role": "user", "content": user_content}]

        system_prompt = (
            "You are an AI assistant for India's Public Distribution System (PDS) in Telangana. "
            "You have access to real-time data on FPS shops, beneficiaries, fraud alerts, "
            "demand forecasts, geospatial coverage, and coordinated fraud ring analysis. "
            "Each turn you will receive RETRIEVED CONTEXT — the most relevant facts from the "
            "live knowledge base. Use this context to ground your answers in real data. "
            "Be concise, specific, and cite numbers when available. "
            "If recommending actions, be precise. "
            "If the context doesn't contain enough information, say so rather than guessing."
        )

        # --- Step 4: Call LLM with fallback chain ---
        answer, llm_used = self._call_llm(messages, system_prompt, max_tokens=800)

        if llm_used == "fallback" or answer is None:
            logger.info(f"answer_nl_query [{session_id}]: using rule-based fallback")
            answer = (
                f"Query received: '{query}'.\n\n"
                f"No LLM API key is configured (set ANTHROPIC_API_KEY or GEMINI_API_KEY). "
                f"Here is the raw retrieved context that would have been used to answer:\n\n"
                f"{retrieved_context}"
            )
            llm_used = "fallback"
        else:
            logger.info(f"answer_nl_query [{session_id}]: answered via {llm_used}")

        # --- Step 5: Persist turn to history (store raw query, not context) ---
        self._append_history(session_id, query, answer)

        return self._build_response(
            query, answer, session_id, rag_sources, retrieved_context, llm_used
        )

    def _build_response(
        self,
        query: str,
        answer: str,
        session_id: str,
        rag_sources: List[str],
        retrieved_context: str,
        llm_used: str = "claude",
    ) -> Dict[str, Any]:
        return {
            "query": query,
            "answer": answer,
            "agent_used": f"reporting_rag/{llm_used}",
            "session_id": session_id,
            "conversation_turn": len(self._get_history(session_id)) // 2,
            "rag_sources": rag_sources,
            "retrieved_context": retrieved_context,
            "rag_store_stats": self.rag_store.stats(),
            "generated_at": datetime.utcnow().isoformat(),
        }

    # ── Conversation History Management ───────────────────────────────────────

    def _get_history(self, session_id: str) -> List[Dict[str, str]]:
        return self._conversation_history.get(session_id, [])

    def _append_history(self, session_id: str, query: str, answer: str) -> None:
        """
        Append a user+assistant turn to the session history.
        Stores only the raw query (not the retrieved context) as the user message
        so future turns have a clean conversation log.
        """
        history = self._conversation_history.setdefault(session_id, [])
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})
        # Trim to MAX_HISTORY_TURNS (each turn = 2 messages)
        if len(history) > MAX_HISTORY_TURNS * 2:
            self._conversation_history[session_id] = history[-(MAX_HISTORY_TURNS * 2):]

    def clear_session(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        self._conversation_history.pop(session_id, None)
        logger.info(f"ReportingAgent: cleared session '{session_id}'")

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions with their turn counts."""
        return [
            {
                "session_id": sid,
                "turns": len(hist) // 2,
                "messages": len(hist),
            }
            for sid, hist in self._conversation_history.items()
        ]

    # ── Full Reporting Run ─────────────────────────────────────────────────────

    async def run(
        self,
        shops_df: pd.DataFrame,
        beneficiaries_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        fraud_results: Dict,
        forecast_results: Dict,
        geo_results: Dict,
        nl_query: Optional[str] = None,
        session_id: str = "default",
    ) -> Dict[str, Any]:
        """Compile all reporting outputs and (re)index the RAG store."""
        # Always re-index after a fresh pipeline run
        index_info = self.index_agent_outputs(
            fraud_results=fraud_results,
            forecast_results=forecast_results,
            geo_results=geo_results,
            shops_df=shops_df,
            beneficiaries_df=beneficiaries_df,
        )

        dashboard = self.build_dashboard_metrics(
            shops_df, beneficiaries_df, transactions_df,
            fraud_results, forecast_results, geo_results,
        )

        summary = await self.generate_executive_summary(
            fraud_results, forecast_results, geo_results
        )

        nl_response = None
        if nl_query:
            nl_response = await self.answer_nl_query(
                nl_query, shops_df, beneficiaries_df,
                fraud_results, geo_results, forecast_results,
                session_id=session_id,
            )

        return {
            "agent": "reporting",
            "dashboard_metrics": dashboard,
            "executive_summary": summary,
            "nl_query_response": nl_response,
            "rag_index": index_info,
            "generated_at": datetime.utcnow().isoformat(),
        }
