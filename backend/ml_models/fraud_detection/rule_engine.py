"""
Rule-Based Fraud Detection Engine
Deterministic rules for known PDS fraud patterns.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import logging
from app.constants import FraudPattern, FraudSeverity

logger = logging.getLogger(__name__)


class RuleBasedFraudDetector:
    """
    Implements deterministic fraud rules for the PDS system.
    Each rule returns a list of flagged transaction IDs with explanations.
    """

    def __init__(self, rules_config: Dict = None):
        self.config = rules_config or {
            "duplicate_window_hours": 24,
            "bulk_transaction_threshold_pct": 0.40,   # >40% txns in last hour → flag
            "min_biometric_rate": 0.80,                # <80% biometric → flag dealer
            "operating_hours_start": 8,
            "operating_hours_end": 20,
            "max_daily_cards_per_dealer": 200,
        }

    # ── Rule 1: Duplicate Transactions ────────────────────────────────────────

    def detect_duplicate_transactions(self, df: pd.DataFrame) -> List[Dict]:
        """Flag same card used at ≥2 different shops in same month."""
        alerts = []
        df["month_key"] = pd.to_datetime(df["transaction_date"]).dt.to_period("M")

        grouped = df.groupby(["card_id", "month_key"])["fps_shop_id"].nunique()
        duplicates = grouped[grouped > 1].reset_index()

        for _, row in duplicates.iterrows():
            txn_mask = (
                (df["card_id"] == row["card_id"])
                & (df["month_key"] == row["month_key"])
            )
            flagged_txns = df[txn_mask]["transaction_id"].tolist()
            alerts.append({
                "pattern": FraudPattern.DUPLICATE_TRANSACTION,
                "severity": FraudSeverity.HIGH,
                "card_id": row["card_id"],
                "transaction_ids": flagged_txns,
                "anomaly_score": 0.85,
                "explanation": (
                    f"Card {row['card_id']} used at {row['fps_shop_id']} different FPS shops "
                    f"in {row['month_key']}. This indicates either a duplicate card or "
                    f"a dealer-level record manipulation."
                ),
                "recommended_action": "Block card; trigger field verification",
            })
        return alerts

    # ── Rule 2: Outside Operating Hours ───────────────────────────────────────

    def detect_after_hours_transactions(self, df: pd.DataFrame) -> List[Dict]:
        """Flag transactions logged outside official FPS operating hours."""
        alerts = []
        df["hour"] = pd.to_datetime(df["transaction_date"]).dt.hour
        start = self.config["operating_hours_start"]
        end = self.config["operating_hours_end"]

        after_hours = df[(df["hour"] < start) | (df["hour"] >= end)]
        for fps_id, group in after_hours.groupby("fps_shop_id"):
            if len(group) >= 5:
                alerts.append({
                    "pattern": FraudPattern.TEMPORAL_ANOMALY,
                    "severity": FraudSeverity.MEDIUM,
                    "fps_shop_id": fps_id,
                    "transaction_ids": group["transaction_id"].tolist(),
                    "anomaly_score": 0.65,
                    "explanation": (
                        f"Shop {fps_id} logged {len(group)} transactions outside operating hours "
                        f"({start}:00–{end}:00). This may indicate back-dated entries or "
                        f"unauthorized system access."
                    ),
                    "recommended_action": "Suspend dealer; audit ePoS device logs",
                })
        return alerts

    # ── Rule 3: Month-End Bulk Transactions ───────────────────────────────────

    def detect_month_end_bulk(self, df: pd.DataFrame) -> List[Dict]:
        """Flag shops where >40% of monthly transactions occur in the last day."""
        alerts = []
        df["date"] = pd.to_datetime(df["transaction_date"]).dt.date
        df["month_key"] = pd.to_datetime(df["transaction_date"]).dt.to_period("M")

        for (fps_id, month), group in df.groupby(["fps_shop_id", "month_key"]):
            month_end = group["date"].max()
            last_day = group[group["date"] == month_end]
            ratio = len(last_day) / max(len(group), 1)

            if ratio > self.config["bulk_transaction_threshold_pct"]:
                alerts.append({
                    "pattern": FraudPattern.TEMPORAL_ANOMALY,
                    "severity": FraudSeverity.HIGH,
                    "fps_shop_id": fps_id,
                    "transaction_ids": last_day["transaction_id"].tolist(),
                    "anomaly_score": 0.75,
                    "explanation": (
                        f"Shop {fps_id} processed {ratio:.0%} of its {month} transactions "
                        f"on the final day ({month_end}). Typical month-end bulk data entry "
                        f"strongly suggests ghost beneficiary transactions."
                    ),
                    "recommended_action": "Trigger field verification; cross-check biometric logs",
                })
        return alerts

    # ── Rule 4: Low Biometric Rate ────────────────────────────────────────────

    def detect_low_biometric_compliance(self, df: pd.DataFrame) -> List[Dict]:
        """Flag dealers where biometric verification rate < threshold."""
        alerts = []
        if "biometric_verified" not in df.columns:
            return alerts

        dealer_bio = df.groupby("fps_shop_id").agg(
            total=("transaction_id", "count"),
            verified=("biometric_verified", "sum"),
        )
        dealer_bio["bio_rate"] = dealer_bio["verified"] / dealer_bio["total"].clip(lower=1)

        low_bio = dealer_bio[
            (dealer_bio["bio_rate"] < self.config["min_biometric_rate"])
            & (dealer_bio["total"] >= 10)
        ]

        for fps_id, row in low_bio.iterrows():
            alerts.append({
                "pattern": FraudPattern.DEALER_SKIMMING,
                "severity": FraudSeverity.CRITICAL if row["bio_rate"] < 0.5 else FraudSeverity.HIGH,
                "fps_shop_id": fps_id,
                "transaction_ids": [],
                "anomaly_score": 1.0 - row["bio_rate"],
                "explanation": (
                    f"Shop {fps_id} has only {row['bio_rate']:.0%} biometric verification "
                    f"across {int(row['total'])} transactions. This strongly indicates "
                    f"dealer-level skimming — rations logged but not actually distributed."
                ),
                "recommended_action": "Suspend dealer license; initiate audit",
            })
        return alerts

    # ── Rule 5: Card Used at Too Many Shops ───────────────────────────────────

    def detect_card_multi_shop(self, df: pd.DataFrame,
                               threshold: int = 2) -> List[Dict]:
        """Flag cards that transact at more than `threshold` shops."""
        alerts = []
        card_shops = df.groupby("card_id")["fps_shop_id"].nunique()
        suspicious = card_shops[card_shops > threshold]

        for card_id, shop_count in suspicious.items():
            txn_ids = df[df["card_id"] == card_id]["transaction_id"].tolist()
            alerts.append({
                "pattern": FraudPattern.GHOST_BENEFICIARY,
                "severity": FraudSeverity.CRITICAL,
                "card_id": card_id,
                "transaction_ids": txn_ids,
                "anomaly_score": min(1.0, (shop_count - threshold) * 0.25 + 0.70),
                "explanation": (
                    f"Card {card_id} has been used at {shop_count} different FPS shops. "
                    f"Legitimate beneficiaries are assigned to a single shop. This suggests "
                    f"a ghost beneficiary card being used across multiple locations."
                ),
                "recommended_action": "Block card immediately; notify district officer",
            })
        return alerts

    # ── Orchestrate all rules ─────────────────────────────────────────────────

    def run_all_rules(self, df: pd.DataFrame) -> Tuple[List[Dict], int]:
        """Run all rules and return combined alerts + total count."""
        all_alerts = []
        all_alerts.extend(self.detect_duplicate_transactions(df))
        all_alerts.extend(self.detect_after_hours_transactions(df))
        all_alerts.extend(self.detect_month_end_bulk(df))
        all_alerts.extend(self.detect_low_biometric_compliance(df))
        all_alerts.extend(self.detect_card_multi_shop(df))

        logger.info(
            f"Rule engine produced {len(all_alerts)} alerts from {len(df)} transactions"
        )
        return all_alerts, len(all_alerts)
