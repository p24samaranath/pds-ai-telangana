"""
Fraud Detection Agent
======================
Continuously monitors PDS transactions for anomalous and fraudulent patterns.

Pipeline (in order)
-------------------
1. Aggregate long-format transactions → shop-level features
2. Statistical district-baseline detector (Z-score)
3. Rule-based deterministic rules (real data columns)
4. Isolation Forest (shop-level point anomalies)
5. DBSCAN (district-level cluster detection)
6. Graph fraud ring detector (card-sharing networks — graceful no-op on shop-level data)
"""
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ml_models.fraud_detection.statistical_detector import StatisticalFraudDetector
from ml_models.fraud_detection.rule_engine import RuleBasedFraudDetector
from ml_models.fraud_detection.isolation_forest import IsolationForestDetector, DBSCANFraudDetector
from ml_models.fraud_detection.graph_fraud_detector import GraphFraudRingDetector
from app.constants import FraudSeverity, AlertStatus
from app.config import settings

logger = logging.getLogger(__name__)


class FraudDetectionAgent:
    """
    Autonomous fraud detection agent for Telangana PDS.

    Works at the **shop level** (one aggregate row per shop per period) so that
    all detectors see real data columns (rice_per_card, collection_rate, etc.)
    instead of synthesised per-transaction features.

    Target: alert rate < 8% of shops (≈ 1,400 of 17,434).
    """

    def __init__(self):
        self.statistical_detector = StatisticalFraudDetector()
        self.rule_engine = RuleBasedFraudDetector()
        self.if_detector = IsolationForestDetector(
            contamination=None,   # auto-estimate from data
            n_estimators=200,
        )
        self.dbscan_detector = DBSCANFraudDetector(eps=0.5, min_samples=5)
        self.graph_ring_detector = GraphFraudRingDetector(
            min_ring_size=3,
            min_edge_weight=1,
        )
        self.trained = False

    # ── Shop-Level Feature Computation ────────────────────────────────────────

    def _compute_shop_features(
        self,
        transactions_df: pd.DataFrame,
        beneficiaries_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Aggregate long-format transactions into one row per shop with fraud
        detection features.

        Input columns (normalised by DataIngestionService):
            fps_shop_id, district, commodity, quantity_kg, total_cards,
            transaction_date, [total_transactions]

        Output features per shop:
            rice_total_kg, wheat_total_kg, sugar_total_kg, kerosene_total_kg,
            rice_per_card, wheat_per_card, sugar_per_card, kerosene_per_card,
            collection_rate, over_entitlement_ratio, commodity_count,
            district_rice_z, [cards_nfsa_aay, cards_nfsa_phh]
        """
        df = transactions_df.copy()

        # ── 1. Aggregate commodity quantities per shop ─────────────────────────
        def _sum_commodity(commodity_name):
            sub = df[df["commodity"] == commodity_name]
            return sub.groupby("fps_shop_id")["quantity_kg"].sum()

        rice_q  = _sum_commodity("rice").rename("rice_total_kg")
        wheat_q = _sum_commodity("wheat").rename("wheat_total_kg")
        sugar_q = _sum_commodity("sugar").rename("sugar_total_kg")
        kero_q  = _sum_commodity("kerosene").rename("kerosene_total_kg")

        # ── 2. Static fields (take first row per shop) ─────────────────────────
        static_cols = ["district", "total_cards"]
        if "total_transactions" in df.columns:
            static_cols.append("total_transactions")

        base = (
            df.groupby("fps_shop_id")[static_cols]
            .first()
        )

        # ── 3. Build shop-level DataFrame ──────────────────────────────────────
        shop_df = (
            base
            .join(rice_q,  how="left")
            .join(wheat_q, how="left")
            .join(sugar_q, how="left")
            .join(kero_q,  how="left")
        )
        for col in ["rice_total_kg", "wheat_total_kg", "sugar_total_kg", "kerosene_total_kg"]:
            shop_df[col] = shop_df[col].fillna(0.0)

        # ── 4. Commodity count (how many commodities distributed) ──────────────
        comm_count = (
            df[df["quantity_kg"] > 0]
            .groupby("fps_shop_id")["commodity"]
            .nunique()
            .rename("commodity_count")
        )
        shop_df = shop_df.join(comm_count, how="left")
        shop_df["commodity_count"] = shop_df["commodity_count"].fillna(0)

        shop_df = shop_df.reset_index()   # fps_shop_id becomes column

        # ── 5. Per-card features ───────────────────────────────────────────────
        cards = shop_df["total_cards"].clip(lower=1)
        shop_df["rice_per_card"]      = shop_df["rice_total_kg"]      / cards
        shop_df["wheat_per_card"]     = shop_df["wheat_total_kg"]     / cards
        shop_df["sugar_per_card"]     = shop_df["sugar_total_kg"]     / cards
        shop_df["kerosene_per_card"]  = shop_df["kerosene_total_kg"]  / cards
        shop_df["total_kg"]           = shop_df[["rice_total_kg", "wheat_total_kg", "sugar_total_kg"]].sum(axis=1)
        shop_df["over_entitlement_ratio"] = shop_df["rice_per_card"] / 35.0   # AAY ceiling

        # ── 6. Collection rate ─────────────────────────────────────────────────
        if "total_transactions" in shop_df.columns and shop_df["total_transactions"].notna().any():
            shop_df["collection_rate"] = (
                shop_df["total_transactions"] / cards
            ).clip(upper=2.0)
        else:
            # Proxy: use rice_per_card relative to PHH entitlement (5 kg/card minimum)
            # collection_rate > 1 → shop distributed above average; < 0.05 → ghost
            shop_df["collection_rate"] = (
                (shop_df["rice_per_card"] / 5.0).clip(0, 2.0)
            )

        # ── 7. District robust Z-score for rice_per_card ───────────────────────
        def _district_z(group):
            vals = group["rice_per_card"].fillna(0.0)
            median = vals.median()
            iqr = vals.quantile(0.75) - vals.quantile(0.25)
            group["district_rice_z"] = 0.6745 * (vals - median) / (iqr + 1e-8)
            return group

        # Save district before groupby (pandas drops the groupby column)
        district_col = shop_df[["fps_shop_id", "district"]].copy()
        shop_df = shop_df.groupby("district", group_keys=False).apply(_district_z)
        if "district" not in shop_df.columns:
            shop_df = shop_df.merge(district_col, on="fps_shop_id", how="left")

        # ── 8. Optional: merge AAY/PHH card counts from beneficiaries ──────────
        if beneficiaries_df is not None and "fps_shop_id" in beneficiaries_df.columns:
            for card_col in ["cards_nfsa_aay", "cards_nfsa_phh"]:
                if card_col in beneficiaries_df.columns:
                    counts = (
                        beneficiaries_df.groupby("fps_shop_id")[card_col]
                        .first()
                        .rename(card_col)
                    )
                    shop_df = shop_df.set_index("fps_shop_id").join(counts, how="left").reset_index()

        logger.info(
            f"Shop features computed: {len(shop_df)} shops, "
            f"columns={[c for c in shop_df.columns if c not in ['fps_shop_id', 'district']]}"
        )
        return shop_df

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, shop_df: pd.DataFrame) -> Dict:
        """Train Isolation Forest on shop-level features."""
        result = self.if_detector.train(shop_df)
        self.trained = True
        logger.info(f"Fraud Detection Agent trained: {result}")
        return result

    # ── Alert Deduplication ───────────────────────────────────────────────────

    def _deduplicate_alerts(self, alerts: List[Dict]) -> List[Dict]:
        """Remove duplicate alerts for the same shop+pattern combination."""
        seen = set()
        deduped = []
        for alert in alerts:
            if alert.get("model") == "graph_fraud_ring_detector":
                key = f"graph_ring:{alert.get('ring_id', id(alert))}"
            else:
                shop_id = alert.get("fps_shop_id", "")
                pattern = alert.get("fraud_pattern", alert.get("pattern", "unknown"))
                if hasattr(pattern, "value"):
                    pattern = pattern.value
                key = f"shop:{shop_id}|pattern:{pattern}"

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
        score    = raw_alert.get("anomaly_score", 0.5)
        severity = raw_alert.get("severity")
        severity = (
            self._enum_val(severity)
            if severity is not None
            else self._determine_severity(score).value
        )

        pattern_raw  = raw_alert.get("pattern", raw_alert.get("fraud_pattern", "unknown"))
        fraud_pattern = self._enum_val(pattern_raw)

        txn_ids = raw_alert.get("transaction_ids", [])
        if not isinstance(txn_ids, list):
            txn_ids = []

        enriched: Dict = {
            "alert_id":            str(uuid.uuid4()),
            "beneficiary_card_id": raw_alert.get("card_id"),
            "fps_shop_id":         str(raw_alert["fps_shop_id"]) if raw_alert.get("fps_shop_id") else None,
            "district":            raw_alert.get("district", ""),
            "transaction_ids":     txn_ids,
            "severity":            severity,
            "fraud_pattern":       fraud_pattern,
            "anomaly_score":       round(float(score), 3),
            "description":         raw_alert.get("explanation", "Anomaly detected"),
            "explanation":         raw_alert.get("explanation", ""),
            "recommended_action":  raw_alert.get("recommended_action", "Review manually"),
            "status":              AlertStatus.OPEN.value,
            "model":               raw_alert.get("model", "ensemble"),
            "detected_at":         datetime.utcnow().isoformat(),
        }

        # Preserve ring-specific fields for graph ring alerts
        if raw_alert.get("model") == "graph_fraud_ring_detector":
            enriched["ring_id"]       = raw_alert.get("ring_id")
            enriched["ring_metadata"] = raw_alert.get("ring_metadata", {})

        # Preserve district z-score metadata from statistical detector
        for extra in ("district_rice_z", "district_collection_z"):
            if extra in raw_alert:
                enriched[extra] = raw_alert[extra]

        return enriched

    # ── Main Detection Run ────────────────────────────────────────────────────

    async def run(
        self,
        transactions_df: pd.DataFrame,
        historical_df: Optional[pd.DataFrame] = None,
        threshold: float = None,
        beneficiaries_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Run full fraud detection pipeline on a batch of PDS transactions.

        Args:
            transactions_df : Long-format normalised transactions
            historical_df   : Historical data for IF training (uses current if None)
            threshold       : Anomaly score cutoff (defaults to settings value)
            beneficiaries_df: Optional beneficiary data for AAY/PHH card counts
        """
        if threshold is None:
            threshold = settings.FRAUD_ALERT_THRESHOLD

        # ── Step 0: Aggregate to shop level ─────────────────────────────────
        shop_df = self._compute_shop_features(transactions_df, beneficiaries_df)

        # Use historical shop features for training if provided, else current
        train_df = shop_df if historical_df is None else self._compute_shop_features(historical_df)

        if not self.trained and len(train_df) >= 20:
            self.train(train_df)

        all_raw_alerts: List[Dict] = []

        # ── Step 1: Statistical district-baseline (Z-score) ─────────────────
        _, stat_alerts = self.statistical_detector.detect(shop_df)
        all_raw_alerts.extend(stat_alerts)
        logger.info(f"Statistical detector: {len(stat_alerts)} alerts")

        # ── Step 2: Rule engine (deterministic, real data columns) ───────────
        rule_alerts, rule_count = self.rule_engine.run_all_rules(shop_df)
        all_raw_alerts.extend(rule_alerts)
        logger.info(f"Rule engine: {rule_count} alerts")

        # ── Step 3: Isolation Forest (shop-level point anomalies) ────────────
        if self.trained and len(shop_df) > 0:
            _, if_alerts = self.if_detector.detect(shop_df, threshold=threshold)
            all_raw_alerts.extend(if_alerts)
            logger.info(f"Isolation Forest: {len(if_alerts)} anomalous shops")

        # ── Step 4: DBSCAN (district-level cluster detection) ────────────────
        if len(shop_df) >= 10:
            _, cluster_alerts = self.dbscan_detector.detect_clusters(shop_df)
            all_raw_alerts.extend(cluster_alerts)
            logger.info(f"DBSCAN clusters: {len(cluster_alerts)} alerts")

        # ── Step 5: Graph ring detector (card-sharing; graceful no-op on shop-level data)
        if len(transactions_df) >= 5:
            try:
                _, ring_alerts = self.graph_ring_detector.detect_rings(transactions_df)
                all_raw_alerts.extend(ring_alerts)
                logger.info(f"Graph ring detector: {len(ring_alerts)} ring alerts")
            except Exception as exc:
                logger.warning(f"Graph ring detector skipped: {exc}")

        # ── Enrich + deduplicate ─────────────────────────────────────────────
        enriched = [self._enrich_alert(a) for a in all_raw_alerts]
        enriched = self._deduplicate_alerts(enriched)

        # ── Partition by severity ────────────────────────────────────────────
        critical = [a for a in enriched if a["severity"] == FraudSeverity.CRITICAL.value]
        high     = [a for a in enriched if a["severity"] == FraudSeverity.HIGH.value]
        medium   = [a for a in enriched if a["severity"] == FraudSeverity.MEDIUM.value]
        low      = [a for a in enriched if a["severity"] == FraudSeverity.LOW.value]
        ring_alerts_enriched = [a for a in enriched if a.get("model") == "graph_fraud_ring_detector"]

        alert_rate = len(enriched) / max(len(shop_df), 1)
        logger.info(
            f"Fraud Detection complete: {len(enriched)} alerts / {len(shop_df)} shops "
            f"= {alert_rate:.1%} alert rate "
            f"(Critical: {len(critical)}, High: {len(high)}, "
            f"Medium: {len(medium)}, Low: {len(low)})"
        )

        return {
            "agent": "fraud_detection",
            "alerts": enriched,
            "summary": {
                "total_alerts":        len(enriched),
                "total_shops_analysed": len(shop_df),
                "alert_rate":          round(alert_rate, 3),
                "critical":            len(critical),
                "high":                len(high),
                "medium":              len(medium),
                "low":                 len(low),
                "transactions_analysed": len(transactions_df),
                "fraud_rings_detected": len(ring_alerts_enriched),
            },
            "critical_alerts":  critical,
            "fraud_rings":      self.graph_ring_detector.ring_report,
            "graph_summary":    self.graph_ring_detector.get_graph_summary(),
            "generated_at":     datetime.utcnow().isoformat(),
        }

    # ── Real-time single shop scoring ─────────────────────────────────────────

    async def score_transaction(self, transaction: Dict) -> Dict:
        """
        Score an incoming ePoS shop report in real-time.
        Expects shop-level aggregate fields (rice_per_card, collection_rate, etc.)
        or a raw long-format row that will be treated as a single-shop batch.
        """
        df = pd.DataFrame([transaction])

        # Quick rule check: over-entitlement
        rule_hit = None
        rpc = transaction.get("rice_per_card", 0)
        if rpc and float(rpc) > 35.0:
            rule_hit = {
                "pattern": "entitlement_breach",
                "severity": "Critical",
                "explanation": f"Rice per card ({rpc:.1f} kg) exceeds AAY ceiling of 35 kg",
                "anomaly_score": min(1.0, 0.85 + (float(rpc) - 35) / 35),
            }
        elif float(transaction.get("collection_rate", 0.5)) > 1.05:
            cr = float(transaction["collection_rate"])
            rule_hit = {
                "pattern": "phantom_transactions",
                "severity": "Critical",
                "explanation": f"Collection rate {cr:.2f} > 1.05 (phantom transactions)",
                "anomaly_score": min(1.0, 0.90 + (cr - 1.05) * 0.5),
            }

        # IF score (on whatever features are available in the row)
        if_score = 0.0
        if self.trained:
            try:
                scores = self.if_detector.score_transactions(df)
                if_score = float(scores.iloc[0])
            except Exception:
                pass

        final_score = max(if_score, rule_hit["anomaly_score"] if rule_hit else 0.0)
        severity    = self._determine_severity(final_score).value

        return {
            "fps_shop_id":  transaction.get("fps_shop_id"),
            "anomaly_score": round(final_score, 3),
            "severity":      severity,
            "is_flagged":    final_score >= settings.FRAUD_ALERT_THRESHOLD,
            "rule_triggered": rule_hit is not None,
            "explanation":   rule_hit["explanation"] if rule_hit else f"IF score: {if_score:.2f}",
            "scored_at":     datetime.utcnow().isoformat(),
        }
