"""
Shop Performance Scorer
========================
Composite 0–100 score for each FPS (Fair Price Shop) based on:

  1. Distribution efficiency   (30%) — how much of entitlement was distributed
  2. Fraud risk               (25%) — inverse of anomaly score from fraud models
  3. Reach / coverage         (20%) — collection rate (% of cards served)
  4. Commodity diversity      (15%) — number of distinct commodities distributed
  5. Consistency              (10%) — low variance across months (if multi-month)

Scoring philosophy
------------------
* 90–100 : Exemplary — used as best-practice benchmark
* 70–89  : Good — performing above district median
* 50–69  : Average — meets minimum standards
* 30–49  : Below average — intervention recommended
* 0–29   : Critical — immediate corrective action required

All sub-scores are normalised to [0, 100] before weighting.
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Component weights (must sum to 1.0) ───────────────────────────────────────
WEIGHTS = {
    "distribution_efficiency": 0.30,
    "fraud_risk_score":        0.25,
    "coverage_rate":           0.20,
    "commodity_diversity":     0.15,
    "consistency":             0.10,
}

# ── Reference values ──────────────────────────────────────────────────────────
AAY_RICE_ENTITLEMENT_KG  = 35.0
MAX_COMMODITY_COUNT       = 6       # rice, wheat, sugar, kerosene, red gram, salt
EXCELLENT_COLLECTION_RATE = 0.90   # 90% coverage = perfect score


class ShopPerformanceScorer:
    """
    Compute a 0–100 composite performance score for each FPS shop.
    Works on the same shop-level aggregate DataFrame produced by
    FraudDetectionAgent._compute_shop_features().
    """

    def score_shops(
        self,
        shop_df: pd.DataFrame,
        fraud_alerts: Optional[List[Dict]] = None,
    ) -> pd.DataFrame:
        """
        Compute performance scores for all shops.

        Args:
            shop_df      : One row per shop with fraud/distribution features
            fraud_alerts : Optional list of enriched fraud alert dicts
                           (used to set fraud_risk sub-score)

        Returns:
            shop_df with added columns:
                score_distribution, score_fraud_risk, score_coverage,
                score_diversity, score_consistency,
                total_score, performance_band, rank, district_rank
        """
        df = shop_df.copy()

        # Build anomaly score lookup from fraud_alerts (overrides field if present)
        anomaly_lookup: Dict[str, float] = {}
        if fraud_alerts:
            for alert in fraud_alerts:
                sid = str(alert.get("fps_shop_id", ""))
                score = float(alert.get("anomaly_score", 0))
                anomaly_lookup[sid] = max(anomaly_lookup.get(sid, 0), score)

        # ── Sub-scores ─────────────────────────────────────────────────────────
        df["score_distribution"] = self._score_distribution_efficiency(df)
        df["score_fraud_risk"]   = self._score_fraud_risk(df, anomaly_lookup)
        df["score_coverage"]     = self._score_coverage(df)
        df["score_diversity"]    = self._score_diversity(df)
        df["score_consistency"]  = self._score_consistency(df)

        # ── Composite score ────────────────────────────────────────────────────
        df["total_score"] = (
            WEIGHTS["distribution_efficiency"] * df["score_distribution"]
            + WEIGHTS["fraud_risk_score"]       * df["score_fraud_risk"]
            + WEIGHTS["coverage_rate"]          * df["score_coverage"]
            + WEIGHTS["commodity_diversity"]    * df["score_diversity"]
            + WEIGHTS["consistency"]            * df["score_consistency"]
        ).round(1).clip(0, 100)

        # ── Performance band ───────────────────────────────────────────────────
        df["performance_band"] = pd.cut(
            df["total_score"],
            bins=[-1, 30, 50, 70, 90, 100],
            labels=["Critical", "Below Average", "Average", "Good", "Exemplary"],
        ).astype(str)

        # ── Global rank (1 = best) ─────────────────────────────────────────────
        df["rank"] = df["total_score"].rank(ascending=False, method="min").astype(int)

        # ── District rank ──────────────────────────────────────────────────────
        if "district" in df.columns:
            df["district_rank"] = (
                df.groupby("district")["total_score"]
                .rank(ascending=False, method="min")
                .astype(int)
            )
        else:
            df["district_rank"] = df["rank"]

        logger.info(
            f"Performance scores computed for {len(df)} shops. "
            f"Bands: Exemplary={( df['performance_band']=='Exemplary').sum()}, "
            f"Good={( df['performance_band']=='Good').sum()}, "
            f"Critical={( df['performance_band']=='Critical').sum()}"
        )
        return df

    # ── Sub-score methods (each returns Series in [0, 100]) ──────────────────

    def _score_distribution_efficiency(self, df: pd.DataFrame) -> pd.Series:
        """
        How much rice did the shop distribute relative to AAY entitlement ceiling?
        Score 100 if rice_per_card ≈ 30–35 kg (good distribution, not fraudulent).
        Score penalises both under-distribution AND over-distribution.
        """
        if "rice_per_card" not in df.columns:
            return pd.Series(50.0, index=df.index)

        rpc = df["rice_per_card"].fillna(0).clip(lower=0)
        target = 25.0   # typical realistic target (mix of AAY + PHH)

        # Scores peak at target, drop for under/over
        scores = 100 - (rpc - target).abs() * 3.0
        return scores.clip(0, 100)

    def _score_fraud_risk(
        self, df: pd.DataFrame, anomaly_lookup: Dict[str, float]
    ) -> pd.Series:
        """
        Inverse of anomaly score: 100 = clean, 0 = maximum fraud risk.
        Uses pre-computed anomaly scores from fraud detectors.
        """
        if "anomaly_score" in df.columns and not df["anomaly_score"].isna().all():
            anomaly = df["anomaly_score"].fillna(0).clip(0, 1)
        elif anomaly_lookup and "fps_shop_id" in df.columns:
            anomaly = df["fps_shop_id"].astype(str).map(anomaly_lookup).fillna(0)
        else:
            # No fraud data: assume clean
            return pd.Series(75.0, index=df.index)

        return (100 * (1 - anomaly)).clip(0, 100)

    def _score_coverage(self, df: pd.DataFrame) -> pd.Series:
        """
        Collection rate (fraction of cards served).
        Score 100 at 90%+ collection. Linear decay to 0 at 0%.
        """
        if "collection_rate" not in df.columns:
            return pd.Series(60.0, index=df.index)

        cr = df["collection_rate"].fillna(0).clip(0, 1)
        return (100 * cr / EXCELLENT_COLLECTION_RATE).clip(0, 100)

    def _score_diversity(self, df: pd.DataFrame) -> pd.Series:
        """
        Number of distinct commodities distributed.
        Score 100 if all 6 commodities are distributed.
        """
        if "commodity_count" not in df.columns:
            return pd.Series(50.0, index=df.index)

        cc = df["commodity_count"].fillna(0).clip(0, MAX_COMMODITY_COUNT)
        return (100 * cc / MAX_COMMODITY_COUNT).round(1)

    def _score_consistency(self, df: pd.DataFrame) -> pd.Series:
        """
        Low variance = consistent service.
        With single-month data this defaults to a neutral 70.
        With multi-month data uses coefficient of variation on rice_per_card.
        """
        # If we only have one-month snapshot, we can't compute variance
        if "rice_per_card_cv" in df.columns:
            cv = df["rice_per_card_cv"].fillna(0.3)
            return (100 * (1 - cv.clip(0, 1))).clip(0, 100)
        return pd.Series(70.0, index=df.index)

    # ── Convenience: top/bottom shops ─────────────────────────────────────────

    def top_performers(self, scored_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Return the top-n shops by total_score."""
        return scored_df.nlargest(n, "total_score")

    def bottom_performers(self, scored_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Return the bottom-n shops by total_score."""
        return scored_df.nsmallest(n, "total_score")

    def score_summary(self, scored_df: pd.DataFrame) -> Dict:
        """Aggregate statistics across all scored shops."""
        return {
            "total_shops":   len(scored_df),
            "mean_score":    round(float(scored_df["total_score"].mean()), 1),
            "median_score":  round(float(scored_df["total_score"].median()), 1),
            "std_score":     round(float(scored_df["total_score"].std()), 1),
            "band_counts": {
                band: int((scored_df["performance_band"] == band).sum())
                for band in ["Exemplary", "Good", "Average", "Below Average", "Critical"]
            },
            "weights_used": WEIGHTS,
        }
