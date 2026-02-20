"""
Geospatial Optimizer: K-Means clustering, Voronoi tessellation,
underserved zone detection, and new FPS location recommendations.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

# Approximate km per degree at Telangana's latitude (~17°N)
KM_PER_DEGREE_LAT = 111.0
KM_PER_DEGREE_LON = 105.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in km between two lat/lon points."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


class GeospatialOptimizer:
    """
    Analyzes FPS shop network coverage and recommends optimizations.
    """

    def __init__(self, max_acceptable_distance_km: float = 5.0):
        self.max_acceptable_km = max_acceptable_distance_km

    # ── Distance Matrix ───────────────────────────────────────────────────────

    def compute_distance_matrix(
        self, beneficiary_locs: np.ndarray, shop_locs: np.ndarray
    ) -> np.ndarray:
        """
        Returns (n_beneficiaries x n_shops) distance matrix in km.
        Uses Euclidean approximation scaled to km for speed; switch to haversine
        for production accuracy.
        """
        # Scale degrees to km
        b_scaled = beneficiary_locs * np.array([KM_PER_DEGREE_LAT, KM_PER_DEGREE_LON])
        s_scaled = shop_locs * np.array([KM_PER_DEGREE_LAT, KM_PER_DEGREE_LON])
        return cdist(b_scaled, s_scaled)

    # ── Nearest Shop Distance ─────────────────────────────────────────────────

    def nearest_shop_distances(
        self, beneficiaries_df: pd.DataFrame, shops_df: pd.DataFrame
    ) -> pd.Series:
        """Return per-beneficiary distance (km) to the nearest active FPS shop.
        Beneficiaries missing lat/lon receive NaN (treated as unknown distance).
        """
        active_shops = shops_df[shops_df["is_active"] == True].dropna(subset=["latitude", "longitude"])
        if active_shops.empty:
            return pd.Series([np.inf] * len(beneficiaries_df), index=beneficiaries_df.index)

        # Drop rows with null coordinates from beneficiaries
        valid_bene = beneficiaries_df.dropna(subset=["latitude", "longitude"])
        if valid_bene.empty:
            return pd.Series([np.nan] * len(beneficiaries_df), index=beneficiaries_df.index)

        b_locs = valid_bene[["latitude", "longitude"]].values.astype(float)
        s_locs = active_shops[["latitude", "longitude"]].values.astype(float)

        dist_matrix = self.compute_distance_matrix(b_locs, s_locs)
        min_distances = dist_matrix.min(axis=1)

        # Align back to original index
        result = pd.Series(np.nan, index=beneficiaries_df.index)
        result.loc[valid_bene.index] = min_distances
        return result

    # ── Underserved Zones ─────────────────────────────────────────────────────

    def find_underserved_zones(
        self, beneficiaries_df: pd.DataFrame, shops_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, float]:
        """
        Identify beneficiaries > max_acceptable_km from nearest active FPS.
        Returns flagged beneficiaries DataFrame + overall accessibility score (0-1).
        """
        distances = self.nearest_shop_distances(beneficiaries_df, shops_df)
        beneficiaries_df = beneficiaries_df.copy()
        beneficiaries_df["nearest_fps_distance_km"] = distances
        # Only count as underserved if distance is known AND exceeds threshold
        beneficiaries_df["is_underserved"] = distances.fillna(0) > self.max_acceptable_km

        underserved = beneficiaries_df[beneficiaries_df["is_underserved"]]
        # Only score against beneficiaries with known distances
        known = distances.notna().sum()
        total = max(known, 1)
        accessibility_score = 1.0 - len(underserved) / total

        logger.info(
            f"{len(underserved)}/{total} beneficiaries underserved "
            f"(accessibility score: {accessibility_score:.2f})"
        )
        return underserved, accessibility_score

    # ── New FPS Location Recommendations (K-Means) ────────────────────────────

    def recommend_new_fps_locations(
        self,
        beneficiaries_df: pd.DataFrame,
        shops_df: pd.DataFrame,
        n_new_shops: int = 5,
    ) -> List[Dict]:
        """
        Cluster underserved beneficiaries and recommend centroid as new FPS location.
        """
        underserved, _ = self.find_underserved_zones(beneficiaries_df, shops_df)
        if underserved.empty:
            return []

        valid = underserved.dropna(subset=["latitude", "longitude"])
        n_clusters = min(n_new_shops, len(valid))
        if n_clusters == 0:
            return []

        coords = valid[["latitude", "longitude"]].values
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(coords)

        recommendations = []
        for rank, centroid in enumerate(kmeans.cluster_centers_, start=1):
            lat, lon = centroid

            # Beneficiaries that would be served by this new shop
            cluster_mask = kmeans.labels_ == (rank - 1)
            affected = valid[cluster_mask]
            projected_coverage = int(affected["members_count"].sum()) if "members_count" in affected.columns else len(affected)

            # Current vs projected average distance
            current_avg = float(affected["nearest_fps_distance_km"].mean())
            projected_distances = np.array([
                haversine_km(lat, lon, r["latitude"], r["longitude"])
                for _, r in affected.iterrows()
            ])
            projected_avg = float(projected_distances.mean())

            recommendations.append({
                "rank": rank,
                "recommended_lat": round(lat, 6),
                "recommended_lon": round(lon, 6),
                "district": affected["district"].mode()[0] if "district" in affected.columns else "Unknown",
                "projected_coverage": projected_coverage,
                "current_avg_distance_km": round(current_avg, 2),
                "projected_avg_distance_km": round(projected_avg, 2),
                "distance_reduction_km": round(current_avg - projected_avg, 2),
                "priority_score": round(projected_coverage / max(projected_avg, 0.1), 2),
                "justification": (
                    f"Centroid of {len(affected)} underserved beneficiaries. "
                    f"A shop here would reduce average travel from "
                    f"{current_avg:.1f} km to {projected_avg:.1f} km "
                    f"for ~{projected_coverage} people."
                ),
            })

        recommendations.sort(key=lambda x: x["priority_score"], reverse=True)
        return recommendations

    # ── Voronoi Service Zone Analysis ─────────────────────────────────────────

    def compute_voronoi_zones(self, shops_df: pd.DataFrame) -> Dict:
        """
        Compute Voronoi tessellation for active FPS shops.
        Returns zone areas and shop-to-zone assignment.
        """
        active = shops_df[shops_df["is_active"] == True].dropna(subset=["latitude", "longitude"])
        if len(active) < 4:
            return {"error": "Need at least 4 active shops for Voronoi", "zones": []}

        points = active[["latitude", "longitude"]].values
        vor = Voronoi(points)

        zones = []
        for i, (shop_idx, region_idx) in enumerate(zip(range(len(active)), vor.point_region)):
            region = vor.regions[region_idx]
            if -1 in region or len(region) == 0:
                area_km2 = None
            else:
                vertices = vor.vertices[region]
                # Shoelace formula for polygon area (rough km² approximation)
                x = vertices[:, 1] * KM_PER_DEGREE_LON
                y = vertices[:, 0] * KM_PER_DEGREE_LAT
                area_km2 = round(0.5 * abs(
                    np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
                ), 2)

            shop_row = active.iloc[i]
            zones.append({
                "shop_id": shop_row.get("shop_id", i),
                "shop_name": shop_row.get("shop_name", f"Shop {i}"),
                "latitude": float(shop_row["latitude"]),
                "longitude": float(shop_row["longitude"]),
                "voronoi_area_km2": area_km2,
            })

        return {"zones": zones, "total_shops": len(active)}

    # ── Underperforming FPS Detection ─────────────────────────────────────────

    def flag_underperforming_shops(
        self,
        shops_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        low_utilization_threshold: float = 0.30,
    ) -> List[Dict]:
        """
        Flag FPS shops with low utilization that have a nearby alternative shop.
        """
        utilization = (
            transactions_df.groupby("fps_shop_id")["card_id"]
            .nunique()
            .reset_index(name="cards_served")
        )
        merged = shops_df.merge(utilization, left_on="shop_id", right_on="fps_shop_id", how="left")
        merged["cards_served"] = merged["cards_served"].fillna(0)
        merged["utilization_rate"] = merged["cards_served"] / merged["total_cards"].clip(lower=1)

        flagged = []
        low_util = merged[merged["utilization_rate"] < low_utilization_threshold]

        for _, shop in low_util.iterrows():
            # Check if there's another active shop within 3 km
            active_others = merged[
                (merged["shop_id"] != shop["shop_id"]) & (merged["is_active"])
            ].dropna(subset=["latitude", "longitude"])

            if active_others.empty or pd.isna(shop.get("latitude")):
                continue

            distances = active_others.apply(
                lambda r: haversine_km(shop["latitude"], shop["longitude"],
                                       r["latitude"], r["longitude"]),
                axis=1,
            )
            min_dist = float(distances.min())

            if min_dist <= 3.0:
                flagged.append({
                    "shop_id": shop["shop_id"],
                    "shop_name": shop.get("shop_name"),
                    "district": shop.get("district"),
                    "utilization_rate": round(float(shop["utilization_rate"]), 2),
                    "cards_served": int(shop["cards_served"]),
                    "total_cards": int(shop.get("total_cards", 0)),
                    "nearest_alternative_km": round(min_dist, 2),
                    "recommendation": "Consider consolidation with nearby shop",
                })

        return flagged

    # ── District Accessibility Score ──────────────────────────────────────────

    def district_accessibility_scores(
        self, beneficiaries_df: pd.DataFrame, shops_df: pd.DataFrame
    ) -> Dict[str, float]:
        """Return 0–1 accessibility score per district."""
        distances = self.nearest_shop_distances(beneficiaries_df, shops_df)
        beneficiaries_df = beneficiaries_df.copy()
        beneficiaries_df["distance"] = distances

        scores = {}
        for district, group in beneficiaries_df.groupby("district"):
            known = group["distance"].notna()
            if known.sum() == 0:
                scores[district] = 1.0   # no data → assume ok
                continue
            served = (group.loc[known, "distance"] <= self.max_acceptable_km).sum()
            total = max(known.sum(), 1)
            scores[district] = round(float(served) / total, 3)
        return scores
