"""
Scoring Agent
==============
Orchestrates shop performance scoring and district health indexing.

Wraps ShopPerformanceScorer and DistrictHealthScorer to produce a
self-contained scoring report from shop-level features and fraud alerts.
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from agents.fraud_detection_agent import FraudDetectionAgent
from ml_models.scoring.shop_performance_scorer import ShopPerformanceScorer
from ml_models.scoring.district_health_scorer import DistrictHealthScorer

logger = logging.getLogger(__name__)


class ScoringAgent:
    """
    Autonomous scoring agent for PDS shops and districts.

    Pipeline:
      1. Compute shop-level features via FraudDetectionAgent._compute_shop_features()
      2. Run shop performance scorer (0–100 composite score)
      3. Run district health scorer (0–100 health index)
      4. Return structured report
    """

    def __init__(self):
        self._fraud_agent    = FraudDetectionAgent()
        self.shop_scorer     = ShopPerformanceScorer()
        self.district_scorer = DistrictHealthScorer()

    async def run(
        self,
        transactions_df: pd.DataFrame,
        shops_df: Optional[pd.DataFrame] = None,
        beneficiaries_df: Optional[pd.DataFrame] = None,
        fraud_alerts: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """
        Run full scoring pipeline.

        Args:
            transactions_df : Long-format normalised transactions
            shops_df        : Optional shop metadata (for shop_name, GPS)
            beneficiaries_df: Optional beneficiary data (for supply gap)
            fraud_alerts    : Pre-computed fraud alerts (avoids re-running fraud pipeline)

        Returns:
            Dict with:
              shop_scores          : List of shop score dicts
              district_scores      : List of district health dicts
              shop_summary         : Aggregate score statistics
              district_summary     : Aggregate district health statistics
              generated_at         : ISO timestamp
        """
        # ── Step 1: Compute shop-level aggregate features ─────────────────────
        shop_features_df = self._fraud_agent._compute_shop_features(
            transactions_df, beneficiaries_df
        )

        # ── Step 2: Merge shop metadata if available ──────────────────────────
        if shops_df is not None and not shops_df.empty:
            id_col = "fps_shop_id" if "fps_shop_id" in shops_df.columns else "shop_id"
            meta_cols = [id_col] + [
                c for c in ["shop_name", "latitude", "longitude", "mandal", "is_active"]
                if c in shops_df.columns
            ]
            meta = shops_df[meta_cols].rename(columns={id_col: "fps_shop_id"})
            meta["fps_shop_id"] = meta["fps_shop_id"].astype(str)
            shop_features_df = shop_features_df.merge(meta, on="fps_shop_id", how="left")

        # ── Step 3: Shop performance scoring ──────────────────────────────────
        scored_df = self.shop_scorer.score_shops(
            shop_df=shop_features_df,
            fraud_alerts=fraud_alerts,
        )
        shop_summary = self.shop_scorer.score_summary(scored_df)

        # ── Step 4: District health scoring ───────────────────────────────────
        dist_df = self.district_scorer.score_districts(
            scored_shop_df=scored_df,
            fraud_alerts=fraud_alerts,
        )
        dist_summary = self.district_scorer.summary(dist_df) if not dist_df.empty else {}

        # ── Step 5: Serialise ─────────────────────────────────────────────────
        shop_records = _safe_to_records(scored_df)
        dist_records = _safe_to_records(dist_df) if not dist_df.empty else []

        logger.info(
            f"ScoringAgent complete: {len(shop_records)} shops scored, "
            f"{len(dist_records)} districts."
        )

        return {
            "agent":            "scoring",
            "shop_scores":      shop_records,
            "district_scores":  dist_records,
            "shop_summary":     shop_summary,
            "district_summary": dist_summary,
            "generated_at":     datetime.utcnow().isoformat(),
        }


def _safe_to_records(df: pd.DataFrame) -> List[Dict]:
    """Convert DataFrame to JSON-serialisable list of dicts."""
    import numpy as np
    records = []
    for row in df.to_dict("records"):
        clean = {}
        for k, v in row.items():
            if isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = None if np.isnan(v) else float(v)
            elif isinstance(v, (np.bool_,)):
                clean[k] = bool(v)
            elif isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                clean[k] = None
            else:
                clean[k] = v
        records.append(clean)
    return records
