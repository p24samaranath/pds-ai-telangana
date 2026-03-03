"""
Isolation Forest & DBSCAN Anomaly Detection for PDS Fraud
==========================================================
Operates on **shop-level aggregate features** derived from real Telangana PDS
wide-format data (one row per shop per period).

Key changes from previous version
----------------------------------
* Feature set replaced: biometric_verified / hour_of_day / day_of_month (all
  simulated) → rice_per_card / collection_rate / over_entitlement_ratio /
  kerosene_per_card / district_rice_z (all from real data).
* Contamination is estimated dynamically with Tukey's IQR rule instead of
  being hardcoded at 5 %.
* DBSCAN updated to operate on shop-level features (time-based clustering
  removed since we have monthly aggregates, not per-transaction timestamps).
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """
    Shop-level point anomaly detection using Isolation Forest.
    Detects individual suspicious shops without reference to district neighbours.
    """

    FEATURE_COLUMNS = [
        "rice_per_card",            # primary fraud signal
        "collection_rate",          # transactions / registered cards
        "over_entitlement_ratio",   # rice_per_card / 35 (AAY ceiling)
        "wheat_per_card",           # secondary commodity signal
        "kerosene_per_card",        # tertiary commodity signal
        "commodity_count",          # how many commodities distributed
        "district_rice_z",          # z-score vs district mean (pre-computed)
        "sugar_per_card",           # sugar distribution signal
    ]

    def __init__(
        self,
        contamination: Optional[float] = None,
        n_estimators: int = 200,
        random_state: int = 42,
    ):
        # contamination=None → auto-estimate from data (recommended)
        self._contamination_override = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model: Optional[IsolationForest] = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self._fitted_features: List[str] = []

    # ── Feature Engineering ────────────────────────────────────────────────────

    def _build_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Select available feature columns and return (array, column_names)."""
        available = [c for c in self.FEATURE_COLUMNS if c in df.columns]
        if not available:
            raise ValueError(
                "No valid feature columns found in shop DataFrame. "
                f"Expected one of: {self.FEATURE_COLUMNS}"
            )
        feature_df = df[available].fillna(0.0).replace([np.inf, -np.inf], 0.0)
        return feature_df.values, available

    # ── Contamination Estimation ───────────────────────────────────────────────

    def _estimate_contamination(self, rice_per_card: pd.Series) -> float:
        """
        Estimate anomaly rate using Tukey's IQR rule on the primary signal.
        More reliable than hardcoding 5%.
        """
        q1 = rice_per_card.quantile(0.25)
        q3 = rice_per_card.quantile(0.75)
        iqr = q3 - q1
        n_outliers = (
            (rice_per_card < q1 - 3 * iqr) |
            (rice_per_card > q3 + 3 * iqr)
        ).sum()
        estimated = n_outliers / max(len(rice_per_card), 1)
        # Clip to a sensible range: 1%–15%
        return float(np.clip(estimated, 0.01, 0.15))

    # ── Training ───────────────────────────────────────────────────────────────

    def train(self, shop_df: pd.DataFrame) -> Dict:
        """Train Isolation Forest on shop-level features."""
        if len(shop_df) < 20:
            return {"status": "insufficient_data", "n_samples": len(shop_df)}

        features, available = self._build_features(shop_df)
        scaled = self.scaler.fit_transform(features)

        # Estimate contamination if not explicitly overridden
        contamination = self._contamination_override
        if contamination is None:
            if "rice_per_card" in shop_df.columns:
                contamination = self._estimate_contamination(
                    shop_df["rice_per_card"].fillna(0)
                )
            else:
                contamination = 0.05

        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(scaled)
        self.is_trained = True
        self._fitted_features = available

        logger.info(
            f"Isolation Forest trained: {len(shop_df)} shops, "
            f"contamination={contamination:.3f}, features={available}"
        )
        return {
            "status": "success",
            "n_samples": len(shop_df),
            "contamination": contamination,
            "features": available,
        }

    # ── Scoring ────────────────────────────────────────────────────────────────

    def score_transactions(self, shop_df: pd.DataFrame) -> pd.Series:
        """Return anomaly scores in [0, 1] — higher means more anomalous."""
        features, _ = self._build_features(shop_df)

        if not self.is_trained:
            logger.warning("Isolation Forest not trained — fitting on inference data (not ideal)")
            scaled = self.scaler.fit_transform(features)
            self.model = IsolationForest(
                contamination=0.05,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1,
            )
            self.model.fit(scaled)
            self.is_trained = True
        else:
            scaled = self.scaler.transform(features)

        # decision_function returns negative scores; flip and normalise to [0, 1]
        raw = self.model.decision_function(scaled)
        score_range = raw.max() - raw.min()
        scores = 1 - (raw - raw.min()) / (score_range + 1e-8)
        return pd.Series(scores, index=shop_df.index)

    # ── Detection ──────────────────────────────────────────────────────────────

    def detect(
        self,
        shop_df: pd.DataFrame,
        threshold: float = 0.6,
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """Detect anomalous shops and return (scored_df, alerts)."""
        scores = self.score_transactions(shop_df)
        result_df = shop_df.copy()
        result_df["anomaly_score"] = scores
        result_df["is_anomaly"] = scores >= threshold

        alerts = []
        for _, row in result_df[result_df["is_anomaly"]].iterrows():
            score = float(row["anomaly_score"])
            alerts.append({
                "fps_shop_id": str(row.get("fps_shop_id", "")),
                "district":    str(row.get("district", "")),
                "anomaly_score": round(score, 3),
                "severity":    _score_to_severity(score),
                "fraud_pattern": "statistical_anomaly",
                "explanation":  _generate_if_explanation(row, score),
                "model": "isolation_forest",
            })

        logger.info(f"Isolation Forest: {len(alerts)} anomalous shops")
        return result_df, alerts


class DBSCANFraudDetector:
    """
    Density-based clustering to detect systemic (district-level) fraud clusters.
    With shop-level monthly aggregates, clusters shops by anomaly profile rather
    than by transaction timestamp.
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler()

    def detect_clusters(
        self, shop_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """Identify clusters of similarly anomalous shops."""
        df = shop_df.copy()

        # Choose available clustering features
        feature_cols = [
            c for c in ["rice_per_card", "collection_rate", "kerosene_per_card"]
            if c in df.columns
        ]
        if not feature_cols or len(df) < self.min_samples:
            return df, []

        features = self.scaler.fit_transform(df[feature_cols].fillna(0.0))
        labels = DBSCAN(
            eps=self.eps, min_samples=self.min_samples, n_jobs=-1
        ).fit_predict(features)
        df["cluster_label"] = labels

        cluster_alerts = []
        for cluster_id in set(labels):
            if cluster_id == -1:   # noise points — not a cluster
                continue
            cluster = df[df["cluster_label"] == cluster_id]
            if len(cluster) < self.min_samples:
                continue

            districts = (
                cluster["district"].nunique() if "district" in cluster.columns else 1
            )
            avg_rpc = (
                float(cluster["rice_per_card"].mean())
                if "rice_per_card" in cluster.columns else 0.0
            )
            shop_ids = (
                cluster["fps_shop_id"].tolist()
                if "fps_shop_id" in cluster.columns else []
            )

            cluster_alerts.append({
                "cluster_id":      int(cluster_id),
                "shops_involved":  len(cluster),
                "districts_involved": districts,
                "fps_shop_ids":    shop_ids,
                "avg_rice_per_card": round(avg_rpc, 2),
                "anomaly_score":   0.70,
                "severity":        "High" if districts > 1 else "Medium",
                "explanation": (
                    f"DBSCAN identified a cluster of {len(cluster)} shops across "
                    f"{districts} district(s) with similar anomalous distribution "
                    f"patterns. Average rice/card: {avg_rpc:.1f} kg."
                ),
                "model": "dbscan",
            })

        logger.info(f"DBSCAN: {len(cluster_alerts)} fraud clusters")
        return df, cluster_alerts


# ── Helpers ────────────────────────────────────────────────────────────────────

def _score_to_severity(score: float) -> str:
    if score >= 0.85:
        return "Critical"
    elif score >= 0.70:
        return "High"
    elif score >= 0.50:
        return "Medium"
    return "Low"


def _generate_if_explanation(row: pd.Series, score: float) -> str:
    reasons = []
    rpc = float(row.get("rice_per_card", 0))
    cr  = float(row.get("collection_rate", 1))
    oe  = float(row.get("over_entitlement_ratio", 0))

    if rpc > 35.0:
        reasons.append(f"rice/card ({rpc:.1f} kg) exceeds AAY ceiling of 35 kg")
    elif rpc > 20.0:
        reasons.append(f"high rice distribution ({rpc:.1f} kg/card)")
    if cr < 0.3:
        reasons.append(f"low collection rate ({cr:.0%})")
    if oe > 1.2:
        reasons.append(f"over-entitlement ratio of {oe:.2f}×")
    if not reasons:
        reasons.append("statistical outlier across multiple features")

    return (
        f"Isolation Forest flagged shop {row.get('fps_shop_id', '?')} "
        f"(score: {score:.2f}): " + "; ".join(reasons) + "."
    )
