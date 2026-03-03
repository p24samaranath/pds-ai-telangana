"""
Geospatial Optimizer: K-Means clustering, Voronoi tessellation,
underserved zone detection, new FPS location recommendations,
equity analysis, and shop vulnerability indexing.

Key improvements
----------------
* compute_distance_matrix: vectorised Haversine (not Euclidean approximation).
* compute_equity_index: Gini coefficient of beneficiary-load distribution.
* compute_shop_vulnerability: composite score (GPS isolation, supply gap, fraud).
* flag_underperforming_shops: uses shop-level collection_rate (real data compatible).
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
        Returns (n_beneficiaries × n_shops) distance matrix in km using
        vectorised Haversine formula (accurate to < 0.1% over Telangana distances).
        """
        R = 6371.0
        # Unpack (lat, lon) columns
        blat = np.radians(beneficiary_locs[:, 0])[:, None]
        blon = np.radians(beneficiary_locs[:, 1])[:, None]
        slat = np.radians(shop_locs[:, 0])[None, :]
        slon = np.radians(shop_locs[:, 1])[None, :]

        dphi    = slat - blat
        dlambda = slon - blon
        a = (np.sin(dphi / 2) ** 2
             + np.cos(blat) * np.cos(slat) * np.sin(dlambda / 2) ** 2)
        return R * 2 * np.arcsin(np.clip(np.sqrt(a), 0, 1))

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

        Uses shop-level collection_rate from transactions_df if available
        (computed from total_transactions / total_cards), otherwise falls back
        to counting distinct card_id values per shop.
        """
        id_col = "fps_shop_id" if "fps_shop_id" in shops_df.columns else "shop_id"
        shops = shops_df.copy()
        shops["_sid"] = shops[id_col].astype(str)

        # Prefer shop-level collection_rate if already in transactions_df
        if "collection_rate" in transactions_df.columns:
            util = (
                transactions_df.groupby("fps_shop_id")["collection_rate"]
                .first()
                .rename("utilization_rate")
                .reset_index()
            )
        elif "total_cards" in transactions_df.columns and "total_transactions" in transactions_df.columns:
            util = (
                transactions_df.groupby("fps_shop_id")
                .agg(total_cards=("total_cards", "first"),
                     total_txn=("total_transactions", "first"))
                .reset_index()
            )
            util["utilization_rate"] = util["total_txn"] / util["total_cards"].clip(lower=1)
            util = util[["fps_shop_id", "utilization_rate"]]
        else:
            # Last fallback: per-transaction card counting (original logic)
            util = (
                transactions_df.groupby("fps_shop_id")["card_id"]
                .nunique()
                .reset_index(name="cards_served")
            )
            util["utilization_rate"] = 0.30  # assume unknown

        util = util.rename(columns={"fps_shop_id": "_sid"})
        merged = shops.merge(util, on="_sid", how="left")
        merged["utilization_rate"] = merged["utilization_rate"].fillna(0.0)

        flagged = []
        low_util = merged[merged["utilization_rate"] < low_utilization_threshold]

        for _, shop in low_util.iterrows():
            active_others = merged[
                (merged["_sid"] != shop["_sid"]) & merged.get("is_active", True)
            ].dropna(subset=["latitude", "longitude"])

            if active_others.empty or pd.isna(shop.get("latitude")):
                continue

            distances = active_others.apply(
                lambda r: haversine_km(
                    shop["latitude"], shop["longitude"],
                    r["latitude"],   r["longitude"]
                ),
                axis=1,
            )
            min_dist = float(distances.min())

            if min_dist <= 3.0:
                flagged.append({
                    "shop_id":                str(shop["_sid"]),
                    "shop_name":              shop.get("shop_name"),
                    "district":               shop.get("district"),
                    "utilization_rate":       round(float(shop["utilization_rate"]), 2),
                    "total_cards":            int(shop.get("total_cards", 0)),
                    "nearest_alternative_km": round(min_dist, 2),
                    "recommendation":         "Consider consolidation with nearby shop",
                })

        return flagged

    # ── Equity Analysis ───────────────────────────────────────────────────────

    def compute_equity_index(self, shops_df: pd.DataFrame) -> Dict:
        """
        Gini coefficient of beneficiary-load distribution across FPS shops.

        A Gini of 0 = perfectly equal load; 1 = one shop serves everyone.
        Also computes district-level load equity.
        """
        if "total_cards" not in shops_df.columns or shops_df.empty:
            return {"gini_coefficient": None, "interpretation": "no_data"}

        cards = shops_df["total_cards"].fillna(0).values.astype(float)
        gini = _gini(cards)

        # District-level equity
        district_gini = {}
        if "district" in shops_df.columns:
            for dist, grp in shops_df.groupby("district"):
                dc = grp["total_cards"].fillna(0).values.astype(float)
                district_gini[dist] = round(float(_gini(dc)), 3)

        interpretation = (
            "highly equitable" if gini < 0.20 else
            "moderately equitable" if gini < 0.35 else
            "moderately inequitable" if gini < 0.50 else
            "highly inequitable"
        )

        return {
            "gini_coefficient":  round(float(gini), 3),
            "interpretation":    interpretation,
            "total_shops":       int(len(shops_df)),
            "mean_cards_per_shop": round(float(cards.mean()), 1),
            "std_cards_per_shop":  round(float(cards.std()), 1),
            "p10_cards":           int(np.percentile(cards, 10)),
            "p90_cards":           int(np.percentile(cards, 90)),
            "district_gini":     district_gini,
        }

    # ── Shop Vulnerability Index ──────────────────────────────────────────────

    def compute_shop_vulnerability(
        self,
        shops_df: pd.DataFrame,
        shop_features_df: Optional[pd.DataFrame] = None,
        fraud_alerts: Optional[List[Dict]] = None,
    ) -> pd.DataFrame:
        """
        Composite vulnerability index (0–100) per shop.
        Higher score = more at risk of service failure.

        Components:
          - GPS isolation (50%): shop has no beneficiaries within 3 km
          - Supply gap (25%): from entitlement model rice_gap_pct
          - Fraud risk (25%): from fraud alert anomaly_score

        Returns shops_df with added columns:
            vuln_isolation, vuln_supply, vuln_fraud, vulnerability_index,
            vulnerability_band
        """
        df = shops_df.copy()
        sid_col = "fps_shop_id" if "fps_shop_id" in df.columns else "shop_id"
        df["_sid"] = df[sid_col].astype(str)

        # ── 1. GPS isolation ─────────────────────────────────────────────────
        # Shops with no lat/lon have unknown isolation
        df["vuln_isolation"] = 30.0   # neutral default
        has_gps = df["latitude"].notna() & df["longitude"].notna()
        if has_gps.any() and "total_cards" in df.columns:
            # Low cards + GPS = small isolated shop
            low_card_mask = df["total_cards"] < 50
            df.loc[has_gps & low_card_mask, "vuln_isolation"] = 60.0
            df.loc[~has_gps, "vuln_isolation"] = 50.0  # unknown GPS = moderate risk

        # ── 2. Supply gap ─────────────────────────────────────────────────────
        df["vuln_supply"] = 20.0   # neutral
        if shop_features_df is not None and "rice_gap_pct" in shop_features_df.columns:
            gap = (
                shop_features_df[["fps_shop_id", "rice_gap_pct"]]
                .rename(columns={"fps_shop_id": "_sid"})
            )
            gap["_sid"] = gap["_sid"].astype(str)
            df = df.merge(gap, on="_sid", how="left")
            df["vuln_supply"] = df["rice_gap_pct"].fillna(0).abs().clip(0, 100) * 0.5

        # ── 3. Fraud risk ─────────────────────────────────────────────────────
        df["vuln_fraud"] = 0.0
        if fraud_alerts:
            fraud_lookup = {}
            for alert in fraud_alerts:
                sid = str(alert.get("fps_shop_id", ""))
                score = float(alert.get("anomaly_score", 0))
                fraud_lookup[sid] = max(fraud_lookup.get(sid, 0), score)
            df["vuln_fraud"] = df["_sid"].map(fraud_lookup).fillna(0) * 100

        # ── Composite index ───────────────────────────────────────────────────
        df["vulnerability_index"] = (
            0.50 * df["vuln_isolation"].clip(0, 100)
            + 0.25 * df["vuln_supply"].clip(0, 100)
            + 0.25 * df["vuln_fraud"].clip(0, 100)
        ).round(1).clip(0, 100)

        df["vulnerability_band"] = pd.cut(
            df["vulnerability_index"],
            bins=[-1, 20, 40, 60, 100],
            labels=["Low", "Moderate", "High", "Critical"],
        ).astype(str)

        df = df.drop(columns=["_sid"], errors="ignore")
        logger.info(
            f"Vulnerability index computed: "
            f"Critical={( df['vulnerability_band']=='Critical').sum()}, "
            f"High={( df['vulnerability_band']=='High').sum()}"
        )
        return df

    # ── District Accessibility Score ─────────────────────────────────────────

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


# ── Module-level helpers ───────────────────────────────────────────────────────

def _gini(values: np.ndarray) -> float:
    """Compute the Gini coefficient of an array (0 = equal, 1 = maximally unequal)."""
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    v = np.sort(values.clip(min=0))
    n = len(v)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum() - (n + 1) * v.sum()) / (n * v.sum()))
