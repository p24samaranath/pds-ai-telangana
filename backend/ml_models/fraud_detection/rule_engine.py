"""
Rule-Based Fraud Detection Engine
==================================
Deterministic rules operating on **shop-level aggregate features** derived from
real Telangana PDS wide-format data.

All previous rules based on per-transaction columns (biometric_verified,
hour_of_day, card_id across shops) have been replaced with rules that work
on real monthly aggregates: rice_per_card, collection_rate, wheat_per_card,
kerosene_per_card, total_cards, etc.

Each rule returns a list of alert dicts in the same schema expected by
FraudDetectionAgent._enrich_alert().
"""
import logging
from typing import Dict, List, Tuple

import pandas as pd

from app.constants import FraudSeverity

logger = logging.getLogger(__name__)


class RuleBasedFraudDetector:
    """
    Implements deterministic fraud rules for the PDS system.
    Works on shop-level aggregate features (one row per shop per period).
    """

    DEFAULT_CONFIG: Dict = {
        "aay_rice_ceiling_kg":         35.0,   # AAY maximum rice per card
        "phantom_txn_threshold":        1.05,   # collection_rate > 1.05 → impossible
        "ghost_shop_rate":              0.05,   # collection_rate < 5% → near-zero dist.
        "ghost_shop_min_cards":         100,    # minimum cards to flag ghost-shop rule
        "low_collection_threshold":     0.40,   # < 40% collection rate → flag
        "over_collection_percentile":   95,     # rice_per_card > district p95 → flag
        "kerosene_spike_multiplier":    2.0,    # > 2× district p90 → flag
        "non_wheat_district_threshold": 0.95,   # if >95% shops have 0 wheat → non-wheat
    }

    def __init__(self, rules_config: Dict = None):
        self.config = {**self.DEFAULT_CONFIG, **(rules_config or {})}

    # ── Rule 1: Phantom Transactions ──────────────────────────────────────────

    def detect_phantom_transactions(self, df: pd.DataFrame) -> List[Dict]:
        """Flag shops whose collection_rate exceeds 1.05 (more txns than cards)."""
        if "collection_rate" not in df.columns:
            return []

        thresh = self.config["phantom_txn_threshold"]
        alerts = []
        for _, row in df[df["collection_rate"] > thresh].iterrows():
            cr = float(row["collection_rate"])
            alerts.append({
                "fps_shop_id": str(row["fps_shop_id"]),
                "district":    str(row.get("district", "")),
                "pattern":     "phantom_transactions",
                "severity":    FraudSeverity.CRITICAL,
                "anomaly_score": min(1.0, 0.90 + (cr - thresh) * 0.50),
                "explanation": (
                    f"Shop {row['fps_shop_id']} recorded {cr:.2f}× more transactions "
                    f"than registered cards — physically impossible. Classic phantom "
                    f"beneficiary fraud."
                ),
                "recommended_action": "Immediate suspension; forensic audit of ePoS device",
            })
        return alerts

    # ── Rule 2: Entitlement Breach ────────────────────────────────────────────

    def detect_entitlement_breach(self, df: pd.DataFrame) -> List[Dict]:
        """Flag shops distributing rice above the AAY ceiling (35 kg/card)."""
        if "rice_per_card" not in df.columns:
            return []

        ceiling = self.config["aay_rice_ceiling_kg"]
        alerts = []
        for _, row in df[df["rice_per_card"] > ceiling].iterrows():
            rpc = float(row["rice_per_card"])
            pct_over = (rpc - ceiling) / ceiling
            alerts.append({
                "fps_shop_id": str(row["fps_shop_id"]),
                "district":    str(row.get("district", "")),
                "pattern":     "entitlement_breach",
                "severity":    FraudSeverity.CRITICAL,
                "anomaly_score": min(1.0, 0.85 + pct_over * 0.15),
                "explanation": (
                    f"Shop {row['fps_shop_id']} distributed {rpc:.1f} kg rice/card, "
                    f"exceeding the AAY entitlement ceiling of {ceiling:.0f} kg "
                    f"({pct_over:.0%} over limit). Indicates inflated records or "
                    f"grain diversion."
                ),
                "recommended_action": "Cross-verify with state warehouse dispatch records",
            })
        return alerts

    # ── Rule 3: Ghost Distribution ────────────────────────────────────────────

    def detect_ghost_shops(self, df: pd.DataFrame) -> List[Dict]:
        """Flag large shops with near-zero collection rates."""
        if "collection_rate" not in df.columns or "total_cards" not in df.columns:
            return []

        ghost_rate = self.config["ghost_shop_rate"]
        min_cards  = self.config["ghost_shop_min_cards"]
        mask = (df["collection_rate"] < ghost_rate) & (df["total_cards"] >= min_cards)
        alerts = []
        for _, row in df[mask].iterrows():
            cr    = float(row["collection_rate"])
            cards = int(row.get("total_cards", 0))
            alerts.append({
                "fps_shop_id": str(row["fps_shop_id"]),
                "district":    str(row.get("district", "")),
                "pattern":     "ghost_distribution",
                "severity":    FraudSeverity.HIGH,
                "anomaly_score": 0.80,
                "explanation": (
                    f"Shop {row['fps_shop_id']} served only {cr:.1%} of {cards} "
                    f"registered cards. Near-zero distribution for a large shop "
                    f"suggests it is non-operational while claiming stock."
                ),
                "recommended_action": "Field verification within 48 h; check physical operations",
            })
        return alerts

    # ── Rule 4: Over-Collection (district p95) ────────────────────────────────

    def detect_over_collection(self, df: pd.DataFrame) -> List[Dict]:
        """Flag shops with rice_per_card above district 95th percentile."""
        if "rice_per_card" not in df.columns or "district" not in df.columns:
            return []

        ceiling = self.config["aay_rice_ceiling_kg"]
        pct = self.config["over_collection_percentile"] / 100
        alerts = []

        for district, group in df.groupby("district"):
            p95 = group["rice_per_card"].quantile(pct)
            # Flag shops above p95 but below AAY ceiling (ceiling is handled by rule 2)
            over = group[
                (group["rice_per_card"] > p95) &
                (group["rice_per_card"] <= ceiling)
            ]
            for _, row in over.iterrows():
                rpc = float(row["rice_per_card"])
                alerts.append({
                    "fps_shop_id": str(row["fps_shop_id"]),
                    "district":    str(district),
                    "pattern":     "above_district_p95",
                    "severity":    FraudSeverity.HIGH,
                    "anomaly_score": 0.72,
                    "explanation": (
                        f"Shop {row['fps_shop_id']} distributed {rpc:.1f} kg rice/card, "
                        f"exceeding the {district} district 95th percentile ({p95:.1f} kg). "
                        f"Suggests over-claiming relative to district norms."
                    ),
                    "recommended_action": "Cross-check beneficiary list; demand field audit",
                })
        return alerts

    # ── Rule 5: Low Collection Rate ───────────────────────────────────────────

    def detect_low_collection(self, df: pd.DataFrame) -> List[Dict]:
        """Flag shops with collection rates below 40% (but not ghost-shop level)."""
        if "collection_rate" not in df.columns:
            return []

        thresh      = self.config["low_collection_threshold"]
        ghost_rate  = self.config["ghost_shop_rate"]

        # Minimum cards filter: ignore tiny shops with large statistical variance
        min_cards = 50
        cards_col = "total_cards" if "total_cards" in df.columns else None
        mask = (
            (df["collection_rate"] >= ghost_rate) &
            (df["collection_rate"] < thresh)
        )
        if cards_col:
            mask = mask & (df[cards_col] >= min_cards)

        alerts = []
        for _, row in df[mask].iterrows():
            cr = float(row["collection_rate"])
            alerts.append({
                "fps_shop_id": str(row["fps_shop_id"]),
                "district":    str(row.get("district", "")),
                "pattern":     "low_collection",
                "severity":    FraudSeverity.MEDIUM,
                "anomaly_score": round(0.55 + (thresh - cr) * 0.30, 3),
                "explanation": (
                    f"Shop {row['fps_shop_id']} collection rate is {cr:.0%} "
                    f"(target: >{thresh:.0%}). Persistent low engagement may indicate "
                    f"beneficiary exclusion, accessibility barriers, or an inactive shop."
                ),
                "recommended_action": "Beneficiary awareness campaign; verify accessibility",
            })
        return alerts

    # ── Rule 6: Off-District Commodity ────────────────────────────────────────

    def detect_commodity_anomaly(self, df: pd.DataFrame) -> List[Dict]:
        """Flag shops distributing wheat in districts where wheat is not allocated."""
        if "wheat_per_card" not in df.columns or "district" not in df.columns:
            return []

        threshold = self.config["non_wheat_district_threshold"]
        # Identify districts where the overwhelming majority of shops have 0 wheat
        district_zero_rate = df.groupby("district")["wheat_per_card"].apply(
            lambda x: (x == 0).mean()
        )
        non_wheat_districts = district_zero_rate[district_zero_rate > threshold].index

        alerts = []
        flagged = df[
            df["district"].isin(non_wheat_districts) &
            (df["wheat_per_card"] > 0)
        ]
        for _, row in flagged.iterrows():
            wpc = float(row["wheat_per_card"])
            alerts.append({
                "fps_shop_id": str(row["fps_shop_id"]),
                "district":    str(row.get("district", "")),
                "pattern":     "off_district_commodity",
                "severity":    FraudSeverity.MEDIUM,
                "anomaly_score": 0.60,
                "explanation": (
                    f"Shop {row['fps_shop_id']} distributed {wpc:.2f} kg wheat/card in "
                    f"{row.get('district', 'a')} district where wheat is not allocated "
                    f"(>95% of shops report zero wheat). Possible record manipulation."
                ),
                "recommended_action": "Verify commodity allocation list; check ePoS entries",
            })
        return alerts

    # ── Rule 7: Kerosene Spike ────────────────────────────────────────────────

    def detect_kerosene_spike(self, df: pd.DataFrame) -> List[Dict]:
        """Flag shops distributing kerosene at > 2× district 90th percentile."""
        if "kerosene_per_card" not in df.columns or "district" not in df.columns:
            return []

        multiplier = self.config["kerosene_spike_multiplier"]
        alerts = []

        for district, group in df.groupby("district"):
            p90 = group["kerosene_per_card"].quantile(0.90)
            if p90 < 0.1:   # district barely distributes kerosene — skip
                continue
            threshold = p90 * multiplier
            for _, row in group[group["kerosene_per_card"] > threshold].iterrows():
                kpc = float(row["kerosene_per_card"])
                ratio = kpc / p90
                alerts.append({
                    "fps_shop_id": str(row["fps_shop_id"]),
                    "district":    str(district),
                    "pattern":     "kerosene_spike",
                    "severity":    FraudSeverity.MEDIUM,
                    "anomaly_score": min(0.75, 0.55 + (ratio - multiplier) * 0.10),
                    "explanation": (
                        f"Shop {row['fps_shop_id']} distributed {kpc:.2f} L kerosene/card, "
                        f"{ratio:.1f}× the {district} district 90th percentile ({p90:.2f} L). "
                        f"Unusually high kerosene claims relative to district peers."
                    ),
                    "recommended_action": "Audit kerosene stock movement; verify beneficiary receipts",
                })
        return alerts

    # ── Rule 8: No Distribution ───────────────────────────────────────────────

    def detect_no_distribution(self, df: pd.DataFrame) -> List[Dict]:
        """Flag active shops with zero rice distribution for the entire period."""
        if "rice_total_kg" not in df.columns:
            return []

        alerts = []
        zero_shops = df[df["rice_total_kg"] == 0]
        for _, row in zero_shops.iterrows():
            cards = int(row.get("total_cards", 0))
            alerts.append({
                "fps_shop_id": str(row["fps_shop_id"]),
                "district":    str(row.get("district", "")),
                "pattern":     "no_distribution",
                "severity":    FraudSeverity.LOW,
                "anomaly_score": 0.35,
                "explanation": (
                    f"Shop {row['fps_shop_id']} recorded zero rice distribution this "
                    f"period for {cards} registered cards. Could indicate shop closure, "
                    f"data entry failure, or deliberate omission."
                ),
                "recommended_action": "Verify shop operational status; follow up with mandal officer",
            })
        return alerts

    # ── Orchestrate all rules ─────────────────────────────────────────────────

    def run_all_rules(self, df: pd.DataFrame) -> Tuple[List[Dict], int]:
        """Run all rules on shop-level aggregate data and return combined alerts."""
        all_alerts: List[Dict] = []
        all_alerts.extend(self.detect_phantom_transactions(df))
        all_alerts.extend(self.detect_entitlement_breach(df))
        all_alerts.extend(self.detect_ghost_shops(df))
        all_alerts.extend(self.detect_over_collection(df))
        all_alerts.extend(self.detect_low_collection(df))
        all_alerts.extend(self.detect_commodity_anomaly(df))
        all_alerts.extend(self.detect_kerosene_spike(df))
        all_alerts.extend(self.detect_no_distribution(df))

        logger.info(
            f"Rule engine: {len(all_alerts)} alerts from {len(df)} shops"
        )
        return all_alerts, len(all_alerts)
