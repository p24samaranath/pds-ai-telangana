"""
Geospatial Optimizer Agent
Analyses FPS shop network coverage, identifies underserved zones,
and recommends new shop placements.
"""
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
import logging

from ml_models.optimization.geospatial_optimizer import GeospatialOptimizer
from app.config import settings

logger = logging.getLogger(__name__)


class GeospatialAgent:
    """
    Autonomous geospatial analysis agent.
    - Computes per-beneficiary distance to nearest active FPS
    - Identifies underserved zones (>5 km from FPS)
    - Recommends new FPS locations via K-Means centroid analysis
    - Computes Voronoi service zones
    - Flags underperforming shops for consolidation
    - Reanalyses automatically when shop network changes
    """

    def __init__(self):
        self.optimizer = GeospatialOptimizer(
            max_acceptable_distance_km=settings.MAX_ACCEPTABLE_DISTANCE_KM
        )
        self._last_shop_count: int = 0

    def _network_changed(self, shops_df: pd.DataFrame) -> bool:
        current_count = len(shops_df[shops_df["is_active"] == True])
        changed = current_count != self._last_shop_count
        self._last_shop_count = current_count
        return changed

    async def run(
        self,
        shops_df: pd.DataFrame,
        beneficiaries_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        n_new_shop_recommendations: int = 5,
        force_rerun: bool = False,
    ) -> Dict[str, Any]:
        """
        Full geospatial analysis pipeline.

        Args:
            shops_df: FPS shop master data with lat/lon
            beneficiaries_df: Beneficiary locations
            transactions_df: Transaction history for utilisation analysis
            n_new_shop_recommendations: How many new sites to propose
            force_rerun: Bypass change-detection cache
        """
        network_changed = force_rerun or self._network_changed(shops_df)

        # 1. Underserved zones
        underserved_df, accessibility_score = self.optimizer.find_underserved_zones(
            beneficiaries_df, shops_df
        )

        underserved_zones = []
        for _, row in underserved_df.head(100).iterrows():
            underserved_zones.append({
                "district": row.get("district", "Unknown"),
                "village": row.get("village"),
                "latitude": row.get("latitude"),
                "longitude": row.get("longitude"),
                "nearest_fps_distance_km": round(float(row.get("nearest_fps_distance_km", 0)), 2),
                "affected_beneficiaries": int(row.get("members_count", 1)),
                "priority_score": round(
                    float(row.get("members_count", 1)) /
                    max(float(row.get("nearest_fps_distance_km", 1)), 0.1), 2
                ),
            })

        # 2. New FPS location recommendations
        new_locations = []
        if len(underserved_df) > 0:
            new_locations = self.optimizer.recommend_new_fps_locations(
                beneficiaries_df, shops_df, n_new_shops=n_new_shop_recommendations
            )

        # 3. Voronoi zones
        voronoi = self.optimizer.compute_voronoi_zones(shops_df)

        # 4. Underperforming shops
        underperforming = []
        if not transactions_df.empty:
            underperforming = self.optimizer.flag_underperforming_shops(
                shops_df, transactions_df, low_utilization_threshold=0.30
            )

        # 5. District accessibility scores
        district_scores = {}
        if "district" in beneficiaries_df.columns:
            district_scores = self.optimizer.district_accessibility_scores(
                beneficiaries_df, shops_df
            )

        # Summary statistics
        total_beneficiaries = len(beneficiaries_df)
        underserved_count = len(underserved_df)
        pct_within_3km = round(
            (total_beneficiaries - underserved_count) / max(total_beneficiaries, 1) * 100, 1
        )

        logger.info(
            f"Geospatial Agent: {underserved_count} underserved beneficiaries, "
            f"{len(new_locations)} location recommendations, "
            f"accessibility score: {accessibility_score:.2f}"
        )

        return {
            "agent": "geospatial",
            "network_reanalysed": network_changed,
            "overall_accessibility_score": round(accessibility_score, 3),
            "pct_beneficiaries_within_threshold_km": pct_within_3km,
            "underserved_count": underserved_count,
            "total_beneficiaries": total_beneficiaries,
            "underserved_zones": underserved_zones,
            "new_location_recommendations": new_locations,
            "voronoi_zones": voronoi,
            "underperforming_shops": underperforming,
            "district_accessibility_scores": district_scores,
            "generated_at": datetime.utcnow().isoformat(),
        }
