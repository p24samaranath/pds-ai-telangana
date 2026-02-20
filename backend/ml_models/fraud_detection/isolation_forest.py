"""
Isolation Forest & DBSCAN Anomaly Detection for PDS Fraud
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import logging
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """
    Point anomaly detection using Isolation Forest.
    Detects individual suspicious transactions without reference to neighbors.
    """

    FEATURE_COLUMNS = [
        "quantity_kg",
        "hour_of_day",
        "day_of_month",
        "cards_transacted_same_day",
        "shop_daily_volume",
        "deviation_from_avg",
        "biometric_flag",
    ]

    def __init__(self, contamination: float = 0.05, n_estimators: int = 200,
                 random_state: int = 42):
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.is_trained = False

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
        df["hour_of_day"] = df["transaction_date"].dt.hour
        df["day_of_month"] = df["transaction_date"].dt.day

        # Cards transacted at same shop on same day
        daily_counts = (
            df.groupby(["fps_shop_id", df["transaction_date"].dt.date])
            ["card_id"].transform("count")
        )
        df["cards_transacted_same_day"] = daily_counts

        # Daily volume per shop
        daily_volume = (
            df.groupby(["fps_shop_id", df["transaction_date"].dt.date])
            ["quantity_kg"].transform("sum")
        )
        df["shop_daily_volume"] = daily_volume

        # Deviation from shop monthly average
        shop_avg = df.groupby("fps_shop_id")["quantity_kg"].transform("mean")
        df["deviation_from_avg"] = (df["quantity_kg"] - shop_avg).abs()

        # Biometric absence flag
        df["biometric_flag"] = (~df.get("biometric_verified", pd.Series([True] * len(df)))).astype(int)

        # Fill any remaining NaN
        feature_df = df[self.FEATURE_COLUMNS].fillna(0)
        return feature_df.values

    def train(self, transactions_df: pd.DataFrame) -> Dict:
        if len(transactions_df) < 20:
            return {"status": "insufficient_data"}

        features = self._build_features(transactions_df)
        scaled = self.scaler.fit_transform(features)
        self.model.fit(scaled)
        self.is_trained = True
        logger.info(f"Isolation Forest trained on {len(transactions_df)} transactions")
        return {"status": "success", "n_samples": len(transactions_df)}

    def score_transactions(self, transactions_df: pd.DataFrame) -> pd.Series:
        """Return anomaly scores (higher = more anomalous, 0–1 range)."""
        features = self._build_features(transactions_df)
        if not self.is_trained:
            logger.warning("Model not trained — fitting on inference data (not ideal)")
            scaled = self.scaler.fit_transform(features)
            self.model.fit(scaled)
            self.is_trained = True
        else:
            scaled = self.scaler.transform(features)

        # decision_function returns negative scores; flip and normalise to 0-1
        raw_scores = self.model.decision_function(scaled)
        score_range = raw_scores.max() - raw_scores.min()
        anomaly_scores = 1 - (raw_scores - raw_scores.min()) / (score_range + 1e-8)
        return pd.Series(anomaly_scores, index=transactions_df.index)

    def detect(self, transactions_df: pd.DataFrame,
               threshold: float = 0.6) -> Tuple[pd.DataFrame, List[Dict]]:
        scores = self.score_transactions(transactions_df)
        result_df = transactions_df.copy()
        result_df["anomaly_score"] = scores
        result_df["is_anomaly"] = scores >= threshold

        anomalous = result_df[result_df["is_anomaly"]]
        alerts = []
        for _, row in anomalous.iterrows():
            score = row["anomaly_score"]
            alerts.append({
                "transaction_id": row.get("transaction_id"),
                "fps_shop_id": row.get("fps_shop_id"),
                "card_id": row.get("card_id"),
                "anomaly_score": round(float(score), 3),
                "severity": _score_to_severity(score),
                "explanation": _generate_if_explanation(row, score),
                "model": "isolation_forest",
            })

        logger.info(f"Isolation Forest flagged {len(alerts)} anomalies")
        return result_df, alerts


class DBSCANFraudDetector:
    """
    Density-based spatial clustering to detect coordinated fraud clusters.
    Identifies groups of suspicious transactions in time-location space.
    """

    def __init__(self, eps: float = 0.3, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler()

    def detect_clusters(self, transactions_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict]]:
        df = transactions_df.copy()
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])

        # Features: normalised time + shop-id + quantity
        df["time_numeric"] = (
            df["transaction_date"] - df["transaction_date"].min()
        ).dt.total_seconds() / 86400  # Days

        feature_cols = ["time_numeric", "quantity_kg"]
        if "fps_shop_id" in df.columns:
            df["shop_numeric"] = pd.factorize(df["fps_shop_id"])[0]
            feature_cols.append("shop_numeric")

        features = self.scaler.fit_transform(df[feature_cols].fillna(0))
        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1).fit_predict(features)
        df["cluster_label"] = labels

        # label == -1 → noise (isolated points), other labels → clusters
        cluster_alerts = []
        for cluster_id in set(labels):
            if cluster_id == -1:
                continue
            cluster = df[df["cluster_label"] == cluster_id]
            if len(cluster) >= self.min_samples:
                shops = cluster["fps_shop_id"].nunique() if "fps_shop_id" in cluster.columns else 1
                cluster_alerts.append({
                    "cluster_id": int(cluster_id),
                    "transaction_count": len(cluster),
                    "shops_involved": shops,
                    "transaction_ids": cluster.get("transaction_id", pd.Series([])).tolist(),
                    "total_quantity_kg": float(cluster["quantity_kg"].sum()),
                    "time_span_hours": float(
                        (cluster["transaction_date"].max() - cluster["transaction_date"].min())
                        .total_seconds() / 3600
                    ),
                    "anomaly_score": 0.75,
                    "severity": "High" if shops > 1 else "Medium",
                    "explanation": (
                        f"DBSCAN detected a dense cluster of {len(cluster)} transactions "
                        f"across {shops} shop(s) within a short time window. This pattern "
                        f"suggests coordinated fraud or ghost beneficiary bulk processing."
                    ),
                    "model": "dbscan",
                })

        logger.info(f"DBSCAN identified {len(cluster_alerts)} suspicious clusters")
        return df, cluster_alerts


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    if row.get("hour_of_day", 12) < 8 or row.get("hour_of_day", 12) >= 20:
        reasons.append("transaction outside operating hours")
    if row.get("deviation_from_avg", 0) > 50:
        reasons.append(f"quantity deviates significantly from shop average")
    if row.get("cards_transacted_same_day", 0) > 100:
        reasons.append(f"unusually high card volume at this shop today")
    if row.get("biometric_flag", 0) == 1:
        reasons.append("biometric verification absent")

    if not reasons:
        reasons.append("statistical outlier across multiple features")

    return (
        f"Isolation Forest flagged this transaction (score: {score:.2f}) because: "
        + "; ".join(reasons) + "."
    )
