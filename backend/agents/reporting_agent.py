"""
Reporting & Alerting Agent
Synthesises outputs from all agents into dashboards, reports, and NL query responses.
"""
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

try:
    from anthropic import Anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from app.config import settings

logger = logging.getLogger(__name__)


class ReportingAgent:
    """
    Synthesises multi-agent outputs into human-readable reports and
    answers natural language queries from PDS officials.
    """

    def __init__(self):
        self.client = Anthropic(api_key=settings.ANTHROPIC_API_KEY) if ANTHROPIC_AVAILABLE else None
        self._context_store: Dict[str, Any] = {}   # In-memory context (replace with vector DB)

    # ── Dashboard Metrics ─────────────────────────────────────────────────────

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

        # Transactions this month
        now = datetime.utcnow()
        txn_month = transactions_df.copy()
        if "transaction_date" in txn_month.columns:
            txn_month["transaction_date"] = pd.to_datetime(txn_month["transaction_date"])
            txn_month = txn_month[
                (txn_month["transaction_date"].dt.year == now.year)
                & (txn_month["transaction_date"].dt.month == now.month)
            ]

        # Fraud summary
        fraud_summary = fraud_results.get("summary", {})
        total_alerts = fraud_summary.get("total_alerts", 0)
        critical_alerts = fraud_summary.get("critical", 0)

        # Forecast accuracy
        forecasts = forecast_results.get("forecasts", [])
        avg_accuracy = 0.90  # Placeholder until actual MAPE is computed

        # Geo accessibility
        pct_within_3km = geo_results.get("pct_beneficiaries_within_threshold_km", 0)

        # Monthly distribution by commodity
        monthly_dist = {}
        if "commodity" in transactions_df.columns and "quantity_kg" in transactions_df.columns:
            monthly_dist = (
                txn_month.groupby("commodity")["quantity_kg"]
                .sum()
                .round(2)
                .to_dict()
            )

        # Top fraud districts
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
            "avg_forecast_accuracy": avg_accuracy,
            "beneficiaries_within_3km_pct": pct_within_3km,
            "districts_covered": beneficiaries_df["district"].nunique()
                if "district" in beneficiaries_df.columns else 0,
            "top_fraud_districts": top_fraud_districts,
            "monthly_distribution_kg": monthly_dist,
            "last_updated": now.isoformat(),
        }

    # ── Executive Summary ─────────────────────────────────────────────────────

    async def generate_executive_summary(
        self,
        fraud_results: Dict,
        forecast_results: Dict,
        geo_results: Dict,
        district: Optional[str] = None,
    ) -> str:
        """Generate a human-readable executive summary via Claude."""
        context = f"""
PDS System Status Report — {datetime.utcnow().strftime('%B %Y')}
District focus: {district or 'All Telangana'}

FRAUD DETECTION:
- Total alerts: {fraud_results.get('summary', {}).get('total_alerts', 0)}
- Critical: {fraud_results.get('summary', {}).get('critical', 0)}
- High: {fraud_results.get('summary', {}).get('high', 0)}
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

        if not ANTHROPIC_AVAILABLE or not self.client:
            return self._fallback_summary(fraud_results, forecast_results, geo_results)

        try:
            response = self.client.messages.create(
                model=settings.CLAUDE_MODEL,
                max_tokens=600,
                messages=[{
                    "role": "user",
                    "content": (
                        f"You are a senior analyst for India's Public Distribution System. "
                        f"Write a concise 3-paragraph executive summary for district officials "
                        f"based on this data. Highlight the most urgent actions needed.\n\n{context}"
                    ),
                }],
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            return self._fallback_summary(fraud_results, forecast_results, geo_results)

    def _fallback_summary(self, fraud: Dict, forecast: Dict, geo: Dict) -> str:
        crit = fraud.get("summary", {}).get("critical", 0)
        risk = forecast.get("total_risk_flags", 0)
        underserved = geo.get("underserved_count", 0)
        return (
            f"PDS System Report — {datetime.utcnow().strftime('%B %Y')}\n\n"
            f"Fraud Detection: {fraud.get('summary', {}).get('total_alerts', 0)} alerts raised, "
            f"including {crit} critical cases requiring immediate attention.\n\n"
            f"Demand Forecasting: {forecast.get('total_forecasts', 0)} forecasts generated "
            f"with {risk} stock risk flags identified.\n\n"
            f"Geospatial: {underserved} beneficiaries are currently underserved (>5 km from FPS). "
            f"Accessibility score: {geo.get('overall_accessibility_score', 0):.1%}."
        )

    # ── Natural Language Query ────────────────────────────────────────────────

    async def answer_nl_query(
        self,
        query: str,
        shops_df: pd.DataFrame,
        beneficiaries_df: pd.DataFrame,
        fraud_results: Dict,
        geo_results: Dict,
        forecast_results: Dict,
    ) -> Dict[str, Any]:
        """
        Answer a natural language question from a PDS official using agent outputs.
        """
        # Build structured context
        context_data = {
            "fraud_summary": fraud_results.get("summary", {}),
            "total_alerts": len(fraud_results.get("alerts", [])),
            "underserved_zones": len(geo_results.get("underserved_zones", [])),
            "new_shop_recommendations": geo_results.get("new_location_recommendations", [])[:3],
            "risk_flags": forecast_results.get("risk_flags", [])[:5],
        }

        if not ANTHROPIC_AVAILABLE or not self.client:
            return {
                "query": query,
                "answer": f"Query received: '{query}'. Claude API unavailable — please configure ANTHROPIC_API_KEY.",
                "agent_used": "reporting_fallback",
                "data": context_data,
                "generated_at": datetime.utcnow().isoformat(),
            }

        system_prompt = (
            "You are an AI assistant for India's PDS (Public Distribution System). "
            "You have access to real-time data on FPS shops, beneficiaries, fraud alerts, "
            "demand forecasts, and geospatial coverage analysis. "
            "Answer questions concisely and accurately. "
            "If recommending actions, be specific. "
            "Always cite data when making claims."
        )

        user_message = (
            f"Official query: {query}\n\n"
            f"Current system data:\n{context_data}\n\n"
            "Please provide a clear, actionable answer."
        )

        try:
            response = self.client.messages.create(
                model=settings.CLAUDE_MODEL,
                max_tokens=800,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            answer = response.content[0].text
        except Exception as e:
            logger.error(f"NL query Claude error: {e}")
            answer = f"Unable to process query at this time. Error: {str(e)}"

        return {
            "query": query,
            "answer": answer,
            "agent_used": "reporting",
            "data": context_data,
            "generated_at": datetime.utcnow().isoformat(),
        }

    # ── Full Reporting Run ────────────────────────────────────────────────────

    async def run(
        self,
        shops_df: pd.DataFrame,
        beneficiaries_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        fraud_results: Dict,
        forecast_results: Dict,
        geo_results: Dict,
        nl_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compile all reporting outputs."""
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
            )

        return {
            "agent": "reporting",
            "dashboard_metrics": dashboard,
            "executive_summary": summary,
            "nl_query_response": nl_response,
            "generated_at": datetime.utcnow().isoformat(),
        }
