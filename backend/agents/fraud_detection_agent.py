"""
Fraud Detection Agent
Continuously monitors PDS transactions for anomalous and fraudulent patterns.
Combines rule engine + Isolation Forest + DBSCAN.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid
import logging

from ml_models.fraud_detection.rule_engine import RuleBasedFraudDetector
from ml_models.fraud_detection.isolation_forest import IsolationForestDetector, DBSCANFraudDetector
from app.constants import FraudSeverity, AlertStatus
from app.config import settings

logger = logging.getLogger(__name__)


class FraudDetectionAgent:
    """
    Autonomous fraud detection agent.
    - Rule engine for deterministic patterns
    - Isolation Forest for statistical point anomalies
    - DBSCAN for coordinated fraud clusters
    - Deduplicates alerts before returning
    - Escalates Critical alerts to Orchestrator
    """

    def __init__(self):
        self.rule_engine = RuleBasedFraudDetector()
        self.if_detector = IsolationForestDetector(
            contamination=0.05,
            n_estimators=200,
        )
        self.dbscan_detector = DBSCANFraudDetector(eps=0.3, min_samples=5)
        self.trained = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, historical_df: pd.DataFrame) -> Dict:
        """Train ML models on historical transaction data."""
        result = self.if_detector.train(historical_df)
        self.trained = True
        logger.info(f"Fraud Detection Agent trained: {result}")
        return result

    # ── Alert Deduplication ───────────────────────────────────────────────────

    def _deduplicate_alerts(self, alerts: List[Dict]) -> List[Dict]:
        """Remove duplicate alerts for the same transaction/card."""
        seen = set()
        deduped = []
        for alert in alerts:
            # Dedup key: highest-confidence alert per card or transaction
            key_parts = []
            if "card_id" in alert:
                key_parts.append(f"card:{alert['card_id']}")
            if "fps_shop_id" in alert:
                key_parts.append(f"shop:{alert['fps_shop_id']}")
            if "pattern" in alert:
                key_parts.append(f"pattern:{alert['pattern']}")
            key = "|".join(key_parts)

            if key not in seen:
                seen.add(key)
                deduped.append(alert)
        return deduped

    # ── Severity Mapping ──────────────────────────────────────────────────────

    def _determine_severity(self, score: float) -> FraudSeverity:
        cfg = settings
        if score >= cfg.FRAUD_SEVERITY_CRITICAL:
            return FraudSeverity.CRITICAL
        elif score >= cfg.FRAUD_SEVERITY_HIGH:
            return FraudSeverity.HIGH
        elif score >= cfg.FRAUD_SEVERITY_MEDIUM:
            return FraudSeverity.MEDIUM
        return FraudSeverity.LOW

    # ── Enrich Alert ─────────────────────────────────────────────────────────

    @staticmethod
    def _enum_val(v):
        """Return the .value of an enum, or the string itself."""
        return v.value if hasattr(v, "value") else str(v) if v is not None else None

    def _enrich_alert(self, raw_alert: Dict) -> Dict:
        """Standardise and enrich a raw alert dict into the canonical format."""
        score = raw_alert.get("anomaly_score", 0.5)
        severity = raw_alert.get("severity")
        severity = self._enum_val(severity) if severity is not None else self._determine_severity(score).value

        pattern_raw = raw_alert.get("pattern", raw_alert.get("fraud_pattern", "unknown"))
        fraud_pattern = self._enum_val(pattern_raw)

        # Serialise any enum values in transaction_ids list
        txn_ids = raw_alert.get("transaction_ids", [])
        if not isinstance(txn_ids, list):
            txn_ids = []

        return {
            "alert_id": str(uuid.uuid4()),
            "beneficiary_card_id": raw_alert.get("card_id"),
            "fps_shop_id": str(raw_alert["fps_shop_id"]) if raw_alert.get("fps_shop_id") else None,
            "transaction_ids": txn_ids,
            "severity": severity,
            "fraud_pattern": fraud_pattern,
            "anomaly_score": round(float(score), 3),
            "description": raw_alert.get("explanation", "Anomaly detected"),
            "explanation": raw_alert.get("explanation", ""),
            "recommended_action": raw_alert.get("recommended_action", "Review manually"),
            "status": AlertStatus.OPEN.value,
            "model": raw_alert.get("model", "ensemble"),
            "detected_at": datetime.utcnow().isoformat(),
        }

    # ── Main Detection Run ────────────────────────────────────────────────────

    async def run(
        self,
        transactions_df: pd.DataFrame,
        historical_df: Optional[pd.DataFrame] = None,
        threshold: float = None,
    ) -> Dict[str, Any]:
        """
        Run full fraud detection pipeline on a batch of transactions.

        Args:
            transactions_df: Current period transactions to analyse
            historical_df: Historical data for model training (uses current if None)
            threshold: Anomaly score cutoff (defaults to settings value)
        """
        if threshold is None:
            threshold = settings.FRAUD_ALERT_THRESHOLD

        train_df = historical_df if historical_df is not None else transactions_df

        # Train if needed
        if not self.trained and len(train_df) >= 20:
            self.train(train_df)

        all_raw_alerts: List[Dict] = []

        # 1. Rule engine (deterministic)
        rule_alerts, rule_count = self.rule_engine.run_all_rules(transactions_df)
        all_raw_alerts.extend(rule_alerts)
        logger.info(f"Rule engine: {rule_count} alerts")

        # 2. Isolation Forest (statistical point anomalies)
        if self.trained and len(transactions_df) > 0:
            _, if_alerts = self.if_detector.detect(transactions_df, threshold=threshold)
            all_raw_alerts.extend(if_alerts)
            logger.info(f"Isolation Forest: {len(if_alerts)} alerts")

        # 3. DBSCAN (cluster-level coordination fraud)
        if len(transactions_df) >= 10:
            _, cluster_alerts = self.dbscan_detector.detect_clusters(transactions_df)
            all_raw_alerts.extend(cluster_alerts)
            logger.info(f"DBSCAN clusters: {len(cluster_alerts)} alerts")

        # Enrich + deduplicate
        enriched = [self._enrich_alert(a) for a in all_raw_alerts]
        enriched = self._deduplicate_alerts(enriched)

        # Partition by severity
        critical = [a for a in enriched if a["severity"] == FraudSeverity.CRITICAL.value]
        high = [a for a in enriched if a["severity"] == FraudSeverity.HIGH.value]
        medium = [a for a in enriched if a["severity"] == FraudSeverity.MEDIUM.value]
        low = [a for a in enriched if a["severity"] == FraudSeverity.LOW.value]

        logger.info(
            f"Fraud Detection Agent complete: {len(enriched)} alerts "
            f"(Critical: {len(critical)}, High: {len(high)}, "
            f"Medium: {len(medium)}, Low: {len(low)})"
        )

        return {
            "agent": "fraud_detection",
            "alerts": enriched,
            "summary": {
                "total_alerts": len(enriched),
                "critical": len(critical),
                "high": len(high),
                "medium": len(medium),
                "low": len(low),
                "transactions_analysed": len(transactions_df),
            },
            "critical_alerts": critical,        # Sent directly to Orchestrator
            "generated_at": datetime.utcnow().isoformat(),
        }

    # ── Real-time single transaction scoring ─────────────────────────────────

    async def score_transaction(self, transaction: Dict) -> Dict:
        """
        Score a single incoming ePoS transaction in real-time (<500 ms target).
        Returns immediately with rule-based check + IF score.
        """
        df = pd.DataFrame([transaction])

        # Fast rule checks
        rule_hit = None
        if transaction.get("hour_of_day", 12) < 8 or transaction.get("hour_of_day", 12) >= 20:
            rule_hit = {
                "pattern": "temporal_anomaly",
                "severity": "Medium",
                "explanation": "Transaction outside operating hours (08:00–20:00)",
                "anomaly_score": 0.65,
            }

        # IF score
        if_score = 0.0
        if self.trained:
            try:
                scores = self.if_detector.score_transactions(df)
                if_score = float(scores.iloc[0])
            except Exception:
                pass

        final_score = max(if_score, rule_hit["anomaly_score"] if rule_hit else 0.0)
        severity = self._determine_severity(final_score).value

        return {
            "transaction_id": transaction.get("transaction_id"),
            "anomaly_score": round(final_score, 3),
            "severity": severity,
            "is_flagged": final_score >= settings.FRAUD_ALERT_THRESHOLD,
            "rule_triggered": rule_hit is not None,
            "explanation": rule_hit["explanation"] if rule_hit else f"IF anomaly score: {if_score:.2f}",
            "scored_at": datetime.utcnow().isoformat(),
        }
