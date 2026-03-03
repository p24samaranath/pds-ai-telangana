"""
District Health Scorer
=======================
Composite 0–100 health index for each Telangana district, aggregating:

  1. Average shop performance score  (35%) — from ShopPerformanceScorer
  2. Fraud alert burden              (30%) — % of shops with High/Critical alerts
  3. Supply adequacy                 (20%) — district-level supply gap %
  4. Coverage equity                 (15%) — std deviation of collection rates
                                            (lower std = more equitable)

Dashboard outputs
-----------------
* district_health_index   : 0–100 (higher = healthier)
* risk_level              : Critical / High / Moderate / Low
* top_issue               : Primary concern driving the score down
* action_priority         : Suggested next step for district officials
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WEIGHTS = {
    "avg_shop_score":    0.35,
    "fraud_burden_inv":  0.30,   # inverted: lower fraud% → higher score
    "supply_adequacy":   0.20,
    "coverage_equity":   0.15,
}


class DistrictHealthScorer:
    """
    Aggregate shop-level metrics to a district health dashboard.
    """

    def score_districts(
        self,
        scored_shop_df: pd.DataFrame,
        fraud_alerts: Optional[List[Dict]] = None,
    ) -> pd.DataFrame:
        """
        Compute health scores for all districts.

        Args:
            scored_shop_df : Shop-level DataFrame (output of ShopPerformanceScorer)
            fraud_alerts   : Enriched alert list from FraudDetectionAgent

        Returns:
            DataFrame with one row per district:
            district, n_shops, avg_shop_score, fraud_alert_rate,
            supply_gap_pct, coverage_equity_score,
            district_health_index, risk_level, top_issue, action_priority
        """
        if "district" not in scored_shop_df.columns:
            logger.warning("No 'district' column — returning empty district scores")
            return pd.DataFrame()

        df = scored_shop_df.copy()

        # Build fraud burden lookup per district
        fraud_burden = self._compute_fraud_burden(df, fraud_alerts)

        # Aggregate shop metrics per district
        agg: Dict = {
            "fps_shop_id":    "count",
            "total_score":    "mean",
            "collection_rate": ["mean", "std"],
        }
        for col in ["rice_gap_pct", "rice_per_card", "supply_status"]:
            if col in df.columns:
                agg[col] = "mean" if col != "supply_status" else "count"

        dist_df = df.groupby("district").agg(
            n_shops=("fps_shop_id", "count"),
            avg_shop_score=("total_score", "mean"),
            avg_collection_rate=("collection_rate", "mean"),
            std_collection_rate=("collection_rate", "std"),
        ).reset_index()

        # Supply gap (from entitlement model results, if available)
        if "rice_gap_pct" in df.columns:
            gap_agg = df.groupby("district")["rice_gap_pct"].mean().rename("avg_rice_gap_pct")
            dist_df = dist_df.join(gap_agg, on="district")
        else:
            dist_df["avg_rice_gap_pct"] = 0.0

        # Fraud burden
        dist_df = dist_df.join(fraud_burden, on="district")
        dist_df["fraud_alert_rate"] = dist_df["fraud_alert_rate"].fillna(0.0)

        # ── Sub-scores ────────────────────────────────────────────────────────
        dist_df["sub_avg_score"]       = dist_df["avg_shop_score"].clip(0, 100)
        dist_df["sub_fraud_inv"]       = (100 * (1 - dist_df["fraud_alert_rate"].clip(0, 1))).clip(0, 100)
        dist_df["sub_supply_adequacy"] = self._supply_adequacy_score(dist_df["avg_rice_gap_pct"])
        dist_df["sub_equity"]          = self._equity_score(dist_df["std_collection_rate"])

        # ── Composite health index ────────────────────────────────────────────
        dist_df["district_health_index"] = (
            WEIGHTS["avg_shop_score"]   * dist_df["sub_avg_score"]
            + WEIGHTS["fraud_burden_inv"] * dist_df["sub_fraud_inv"]
            + WEIGHTS["supply_adequacy"]  * dist_df["sub_supply_adequacy"]
            + WEIGHTS["coverage_equity"]  * dist_df["sub_equity"]
        ).round(1).clip(0, 100)

        # ── Risk level ────────────────────────────────────────────────────────
        dist_df["risk_level"] = pd.cut(
            dist_df["district_health_index"],
            bins=[-1, 30, 50, 70, 100],
            labels=["Critical", "High", "Moderate", "Low"],
        ).astype(str)

        # ── Top issue and action priority ─────────────────────────────────────
        dist_df["top_issue"]        = dist_df.apply(self._top_issue, axis=1)
        dist_df["action_priority"]  = dist_df.apply(self._action_priority, axis=1)

        # ── Sort by health index ascending (worst first) ──────────────────────
        dist_df = dist_df.sort_values("district_health_index").reset_index(drop=True)
        dist_df["district_rank"] = range(1, len(dist_df) + 1)

        logger.info(
            f"District health scored: {len(dist_df)} districts — "
            f"Critical: {(dist_df['risk_level']=='Critical').sum()}, "
            f"High: {(dist_df['risk_level']=='High').sum()}"
        )
        return dist_df

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_fraud_burden(
        shop_df: pd.DataFrame, alerts: Optional[List[Dict]]
    ) -> pd.Series:
        """Fraction of shops in each district with High/Critical fraud alerts."""
        if not alerts or "district" not in shop_df.columns:
            return pd.Series(dtype=float, name="fraud_alert_rate")

        shop_count = shop_df.groupby("district")["fps_shop_id"].count()
        flagged_shops = set(
            str(a.get("fps_shop_id", ""))
            for a in alerts
            if a.get("severity") in ("Critical", "High")
        )
        shop_df_c = shop_df.copy()
        shop_df_c["_flagged"] = shop_df_c["fps_shop_id"].astype(str).isin(flagged_shops)
        flagged_count = shop_df_c.groupby("district")["_flagged"].sum()

        burden = (flagged_count / shop_count.clip(lower=1)).rename("fraud_alert_rate")
        return burden

    @staticmethod
    def _supply_adequacy_score(gap_pct: pd.Series) -> pd.Series:
        """
        Convert rice_gap_pct to a 0–100 score.
        gap_pct > 0 → under-supplied (negative score)
        gap_pct < 0 → over-supplied (also negative — fraud risk)
        gap_pct ≈ 0 → perfect
        """
        score = 100 - gap_pct.abs().clip(0, 100)
        return score.clip(0, 100)

    @staticmethod
    def _equity_score(std_cr: pd.Series) -> pd.Series:
        """
        Low standard deviation of collection rates = equitable distribution.
        std = 0 → score 100; std = 0.5 → score 0.
        """
        score = 100 * (1 - std_cr.fillna(0.3).clip(0, 0.5) / 0.5)
        return score.clip(0, 100)

    @staticmethod
    def _top_issue(row: pd.Series) -> str:
        issues = {
            "Fraud burden":       100 - row.get("sub_fraud_inv", 100),
            "Shop performance":   100 - row.get("sub_avg_score", 100),
            "Supply gap":         100 - row.get("sub_supply_adequacy", 100),
            "Coverage inequity":  100 - row.get("sub_equity", 100),
        }
        return max(issues, key=issues.get)

    @staticmethod
    def _action_priority(row: pd.Series) -> str:
        idx   = row.get("district_health_index", 70)
        issue = row.get("top_issue", "")

        if idx < 30:
            return f"URGENT: Immediate district collector intervention required — {issue}"
        elif idx < 50:
            return f"HIGH: District-level audit within 7 days — focus on {issue}"
        elif idx < 70:
            return f"MEDIUM: Monthly review — address {issue} systematically"
        return "LOW: Continue monitoring; share best practices with lower-scoring districts"

    # ── Convenience ───────────────────────────────────────────────────────────

    def critical_districts(self, dist_df: pd.DataFrame) -> pd.DataFrame:
        """Return districts in Critical or High risk."""
        return dist_df[dist_df["risk_level"].isin(["Critical", "High"])]

    def summary(self, dist_df: pd.DataFrame) -> Dict:
        """Dashboard summary statistics."""
        return {
            "total_districts":  len(dist_df),
            "mean_health_index": round(float(dist_df["district_health_index"].mean()), 1),
            "risk_counts": {
                level: int((dist_df["risk_level"] == level).sum())
                for level in ["Critical", "High", "Moderate", "Low"]
            },
            "most_at_risk": (
                dist_df.nsmallest(3, "district_health_index")["district"].tolist()
                if "district" in dist_df.columns else []
            ),
            "best_performers": (
                dist_df.nlargest(3, "district_health_index")["district"].tolist()
                if "district" in dist_df.columns else []
            ),
        }
