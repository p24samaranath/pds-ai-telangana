"""
Statistical District-Baseline Fraud Detector
=============================================
Uses a modified Z-score (robust to outliers) to compare each shop against its
district peer group.  Operates on shop-level aggregate features derived from
real Telangana PDS wide-format data.

Algorithm
---------
1. For each feature, compute district-level *median* and *IQR* (robust stats).
2. Modified Z-score:  z = 0.6745 × (x − median) / (IQR + ε)
3. Composite anomaly score = weighted sum of per-metric |z|, normalised to [0,1].
4. Hard rules (physics-based) immediately produce Critical/High alerts for
   impossible values, independently of district statistics.
"""
import logging

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

from app.constants import FraudSeverity

logger = logging.getLogger(__name__)

# ── Sigma thresholds ───────────────────────────────────────────────────────────
THRESHOLD_CRITICAL = 3.0   # composite sigma → Critical
THRESHOLD_HIGH     = 2.0   # → High
THRESHOLD_MEDIUM   = 1.5   # → Medium

# ── Physics-based hard-rule thresholds ────────────────────────────────────────
AAY_RICE_CEILING_KG    = 35.0   # Maximum rice per card (AAY entitlement)
COLLECTION_RATE_MAX    = 1.05   # collection_rate > 1.05 = impossible (phantom txns)
GHOST_SHOP_RATE        = 0.05   # collection_rate < 5% = near-zero distribution
GHOST_SHOP_MIN_CARDS   = 100    # only flag ghost-shop rule for large beneficiary bases

# ── Feature weights for composite score (must sum to 1.0) ─────────────────────
FEATURE_WEIGHTS = {
    "rice_per_card":      0.35,
    "collection_rate":    0.25,
    "over_entitlement":   0.20,
    "kerosene_per_card":  0.10,
    "wheat_per_card":     0.10,
}


class StatisticalFraudDetector:
    """
    District-baseline Z-score anomaly detector for PDS shop-level data.

    For each shop feature the detector:
      1. Computes district-level median and IQR (robust to outliers).
      2. Computes a modified Z-score per shop per feature.
      3. Weights and sums |z| values into a composite anomaly score.

    Hard rules (physics-based) bypass statistical thresholds for impossible
    values — e.g. collection_rate > 1.05 is always Critical.
    """

    def detect(
        self, shop_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Detect fraudulent shops from shop-level aggregated features.

        Parameters
        ----------
        shop_df : DataFrame
            One row per shop with columns:
            fps_shop_id, district, rice_per_card, wheat_per_card,
            collection_rate, over_entitlement_ratio, kerosene_per_card,
            commodity_count, total_cards

        Returns
        -------
        (scored_df, alerts)
        """
        if shop_df.empty:
            return shop_df, []

        df = shop_df.copy()

        # Compute district-level robust statistics for each feature
        df = self._compute_z_scores(df)

        # Compute per-shop composite anomaly score
        df = self._compute_anomaly_scores(df)

        alerts: List[Dict] = []

        # 1. Hard rules (impossible / physics-based violations)
        hard_alerts = self._apply_hard_rules(df)
        alerts.extend(hard_alerts)
        hard_rule_shops = {a["fps_shop_id"] for a in hard_alerts}

        # 2. Statistical anomalies (z-score based)
        for _, row in df.iterrows():
            shop_id = str(row["fps_shop_id"])
            if shop_id in hard_rule_shops:
                continue  # already flagged by a hard rule

            score = float(row.get("anomaly_score", 0.0))
            if score < 0.20:
                continue

            severity = self._score_to_severity(score)
            if severity == FraudSeverity.LOW.value:
                continue  # skip low — only surface actionable alerts

            alerts.append({
                "fps_shop_id": shop_id,
                "district": str(row.get("district", "")),
                "anomaly_score": round(score, 3),
                "severity": severity,
                "fraud_pattern": self._infer_pattern(row),
                "explanation": self._build_explanation(row, score),
                "recommended_action": self._recommended_action(severity),
                "model": "statistical_z_score",
                "district_rice_z": round(float(row.get("rice_z", 0.0)), 2),
                "district_collection_z": round(float(row.get("coll_z", 0.0)), 2),
            })

        logger.info(
            f"Statistical detector: {len(alerts)} alerts from {len(df)} shops "
            f"(hard: {len(hard_alerts)}, statistical: {len(alerts) - len(hard_alerts)})"
        )
        return df, alerts

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _compute_z_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute modified Z-scores per district for each fraud-signal feature."""
        feature_z_map = {
            "rice_per_card":   "rice_z",
            "collection_rate": "coll_z",
            "kerosene_per_card": "kero_z",
            "wheat_per_card":  "wheat_z",
        }

        for feat, z_col in feature_z_map.items():
            if feat not in df.columns:
                continue

            def _z_for_group(group, _feat=feat, _z=z_col):
                vals = group[_feat].fillna(0.0)
                median = vals.median()
                iqr = vals.quantile(0.75) - vals.quantile(0.25)
                group[_z] = 0.6745 * (vals - median) / (iqr + 1e-8)
                return group

            df = df.groupby("district", group_keys=False).apply(_z_for_group)

        return df

    def _compute_anomaly_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute weighted composite anomaly score per shop, normalised to [0,1]."""
        z_col_weights = [
            ("rice_z",  FEATURE_WEIGHTS["rice_per_card"]),
            ("coll_z",  FEATURE_WEIGHTS["collection_rate"]),
            ("kero_z",  FEATURE_WEIGHTS["kerosene_per_card"]),
            ("wheat_z", FEATURE_WEIGHTS["wheat_per_card"]),
        ]

        composite = pd.Series(0.0, index=df.index)
        total_weight = 0.0

        for z_col, weight in z_col_weights:
            if z_col in df.columns:
                composite += weight * df[z_col].abs().clip(upper=6.0)
                total_weight += weight

        # Over-entitlement signal (not purely z-score based)
        if "over_entitlement_ratio" in df.columns:
            oe = (df["over_entitlement_ratio"] - 1.0).clip(lower=0) * 3.0
            composite += FEATURE_WEIGHTS["over_entitlement"] * oe
            total_weight += FEATURE_WEIGHTS["over_entitlement"]

        # Normalise to [0, 1]: max possible = 6σ × total_weight
        max_possible = 6.0 * max(total_weight, 1e-8)
        df["anomaly_score"] = (composite / max_possible).clip(0.0, 1.0)
        return df

    def _apply_hard_rules(self, df: pd.DataFrame) -> List[Dict]:
        """Apply physics-based hard rules that immediately flag Critical/High."""
        alerts: List[Dict] = []

        for _, row in df.iterrows():
            shop_id = str(row["fps_shop_id"])
            district = str(row.get("district", ""))
            cards = max(float(row.get("total_cards", 1)), 1)

            cr = float(row.get("collection_rate", 0.5)) if pd.notna(row.get("collection_rate")) else None
            rpc = float(row.get("rice_per_card", 0.0)) if pd.notna(row.get("rice_per_card")) else None

            # 1. Phantom transactions: collection_rate > 1.05 (impossible)
            if cr is not None and cr > COLLECTION_RATE_MAX:
                alerts.append({
                    "fps_shop_id": shop_id,
                    "district": district,
                    "anomaly_score": min(1.0, 0.88 + (cr - COLLECTION_RATE_MAX) * 0.5),
                    "severity": FraudSeverity.CRITICAL.value,
                    "fraud_pattern": "phantom_transactions",
                    "explanation": (
                        f"Shop {shop_id} recorded {cr:.2f}× more transactions than "
                        f"registered cards — physically impossible. Classic phantom "
                        f"beneficiary data entry (collection_rate = {cr:.2f})."
                    ),
                    "recommended_action": "Suspend shop immediately; forensic audit of ePoS device",
                    "model": "statistical_z_score",
                })
                continue  # skip other hard rules for same shop

            # 2. Ghost distribution: near-zero collection for large shop
            if cr is not None and cr < GHOST_SHOP_RATE and cards >= GHOST_SHOP_MIN_CARDS:
                alerts.append({
                    "fps_shop_id": shop_id,
                    "district": district,
                    "anomaly_score": 0.82,
                    "severity": FraudSeverity.HIGH.value,
                    "fraud_pattern": "ghost_distribution",
                    "explanation": (
                        f"Shop {shop_id} distributed to only {cr:.1%} of its "
                        f"{int(cards)} registered cards. Near-zero distribution "
                        f"with a large base suggests siphoning or a non-operational shop."
                    ),
                    "recommended_action": "Field verification within 48 h; check physical operations",
                    "model": "statistical_z_score",
                })
                continue

            # 3. Entitlement breach: rice_per_card > AAY ceiling (35 kg)
            if rpc is not None and rpc > AAY_RICE_CEILING_KG:
                pct_over = (rpc - AAY_RICE_CEILING_KG) / AAY_RICE_CEILING_KG
                alerts.append({
                    "fps_shop_id": shop_id,
                    "district": district,
                    "anomaly_score": min(1.0, 0.85 + pct_over * 0.10),
                    "severity": FraudSeverity.CRITICAL.value,
                    "fraud_pattern": "entitlement_breach",
                    "explanation": (
                        f"Shop {shop_id} distributed {rpc:.1f} kg rice/card, "
                        f"exceeding the AAY ceiling of {AAY_RICE_CEILING_KG} kg "
                        f"({pct_over:.0%} over limit). Indicates inflated records "
                        f"or diversion of grain supplies."
                    ),
                    "recommended_action": "Freeze stock records; cross-verify with warehouse dispatch",
                    "model": "statistical_z_score",
                })

        return alerts

    # ── Utility ────────────────────────────────────────────────────────────────

    @staticmethod
    def _score_to_severity(score: float) -> str:
        if score >= 0.55:
            return FraudSeverity.CRITICAL.value
        elif score >= 0.38:
            return FraudSeverity.HIGH.value
        elif score >= 0.22:
            return FraudSeverity.MEDIUM.value
        return FraudSeverity.LOW.value

    @staticmethod
    def _infer_pattern(row: pd.Series) -> str:
        oe  = float(row.get("over_entitlement_ratio", 0))
        cr  = float(row.get("collection_rate", 1))
        rz  = abs(float(row.get("rice_z", 0)))
        cz  = abs(float(row.get("coll_z", 0)))

        if oe > 1.2:
            return "entitlement_breach"
        if cr < 0.30:
            return "ghost_distribution"
        if rz > cz:
            return "dealer_skimming"
        return "low_collection"

    @staticmethod
    def _build_explanation(row: pd.Series, score: float) -> str:
        parts = []
        rpc  = float(row.get("rice_per_card", 0))
        cr   = float(row.get("collection_rate", 1))
        rice_z = abs(float(row.get("rice_z", 0)))
        coll_z = abs(float(row.get("coll_z", 0)))

        if rice_z > 2.0:
            parts.append(
                f"rice distribution ({rpc:.1f} kg/card) deviates "
                f"{rice_z:.1f}σ from district median"
            )
        if coll_z > 2.0:
            parts.append(
                f"collection rate ({cr:.0%}) is {coll_z:.1f}σ from district norm"
            )
        if not parts:
            parts.append(f"composite statistical anomaly (score: {score:.2f})")

        return (
            f"Shop {row.get('fps_shop_id', '?')} flagged: " + "; ".join(parts) + "."
        )

    @staticmethod
    def _recommended_action(severity: str) -> str:
        return {
            FraudSeverity.CRITICAL.value: "Suspend distribution; escalate to district collector",
            FraudSeverity.HIGH.value:     "Field verification within 48 hours",
            FraudSeverity.MEDIUM.value:   "Review records; spot-check next distribution cycle",
            FraudSeverity.LOW.value:      "Monitor for 2 more months before action",
        }.get(severity, "Review manually")
