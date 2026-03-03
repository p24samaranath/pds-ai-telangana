"""
Physics-Based Entitlement Demand Model
========================================
Computes expected PDS commodity demand from beneficiary card type and
government-mandated entitlement rules — no historical data required.

Telangana monthly entitlements (Government Order MS No.23, 2020):
  AAY (Antyodaya Anna Yojana — poorest of poor):
    35 kg rice, 1 kg sugar, 3 L kerosene  [per card / per month]

  PHH (Priority Household):
    5 kg rice per person, 0.5 kg sugar per person, 3 L kerosene [per card]
    (family size determines rice/sugar; kerosene is flat per card)

Use-cases
---------
1. Expected demand forecast (deterministic — entitlements don't change monthly)
2. Supply gap analysis: actual_distributed vs expected_entitlement
3. District-level under/over-supply dashboard
"""
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Government entitlement rates ───────────────────────────────────────────────
AAY_PER_CARD: Dict[str, float] = {
    "rice":      35.0,   # kg per card per month
    "sugar":      1.0,   # kg per card per month
    "kerosene":   3.0,   # litres per card per month
}

PHH_PER_PERSON: Dict[str, float] = {
    "rice":   5.0,   # kg per person per month
    "sugar":  0.5,   # kg per person per month
}

PHH_PER_CARD: Dict[str, float] = {
    "kerosene": 3.0,   # litres — fixed per card, NOT per person
}

# Telangana average household size (NSSO 2019-20)
TELANGANA_AVG_FAMILY_SIZE: float = 3.4

# Expected beneficiary absence rate (not everyone collects every month)
TYPICAL_COLLECTION_RATE: float = 0.88   # 88% of beneficiaries collect in a typical month


class EntitlementDemandModel:
    """
    Deterministic demand estimator based on government ration entitlements.
    Works from card-type counts (AAY/PHH) rather than transaction history.
    """

    def compute_expected_demand(
        self,
        shop_id: str,
        aay_cards: int,
        phh_cards: int,
        avg_family_size: float = TELANGANA_AVG_FAMILY_SIZE,
        collection_rate: float = TYPICAL_COLLECTION_RATE,
    ) -> Dict:
        """
        Compute monthly expected demand for one FPS shop.

        Args:
            shop_id          : FPS shop identifier
            aay_cards        : Number of AAY ration cards
            phh_cards        : Number of PHH ration cards
            avg_family_size  : Average family members per PHH card
            collection_rate  : Fraction of beneficiaries expected to collect

        Returns:
            Dict with expected_rice_kg, expected_sugar_kg, expected_kerosene_litres
        """
        # ── AAY demand (fixed per card) ────────────────────────────────────────
        rice_aay    = aay_cards * AAY_PER_CARD["rice"]
        sugar_aay   = aay_cards * AAY_PER_CARD["sugar"]
        kero_aay    = aay_cards * AAY_PER_CARD["kerosene"]

        # ── PHH demand (per person) ────────────────────────────────────────────
        phh_persons = phh_cards * avg_family_size
        rice_phh    = phh_persons * PHH_PER_PERSON["rice"]
        sugar_phh   = phh_persons * PHH_PER_PERSON["sugar"]
        kero_phh    = phh_cards * PHH_PER_CARD["kerosene"]

        # ── Totals (full entitlement, pre-collection-rate adjustment) ─────────
        total_rice  = rice_aay + rice_phh
        total_sugar = sugar_aay + sugar_phh
        total_kero  = kero_aay + kero_phh

        # ── Adjusted demand (accounting for typical absence rate) ─────────────
        adj_rice  = total_rice  * collection_rate
        adj_sugar = total_sugar * collection_rate
        # Kerosene is usually fully distributed (government must meet demand)

        return {
            "shop_id":                    shop_id,
            "aay_cards":                  int(aay_cards),
            "phh_cards":                  int(phh_cards),
            "total_cards":                int(aay_cards + phh_cards),
            "expected_rice_kg":           round(total_rice, 1),
            "expected_rice_kg_adjusted":  round(adj_rice, 1),
            "expected_sugar_kg":          round(total_sugar, 1),
            "expected_kerosene_litres":   round(total_kero, 1),
            "model":                      "entitlement_physics",
            "breakdown": {
                "aay_rice_kg":  round(rice_aay, 1),
                "phh_rice_kg":  round(rice_phh, 1),
                "aay_sugar_kg": round(sugar_aay, 1),
                "phh_sugar_kg": round(sugar_phh, 1),
            },
        }

    def compute_supply_gap(
        self,
        shop_df: pd.DataFrame,
        beneficiaries_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute supply gap (expected − actual) for each shop.

        Input:
            shop_df         : Shop-level aggregate features (from FraudDetectionAgent)
            beneficiaries_df: Raw beneficiary/card-status data with AAY/PHH counts

        Output (shop_df with added columns):
            expected_rice_kg, rice_gap_kg, rice_gap_pct, supply_status
        """
        df = shop_df.copy()

        # ── Step 1: Attach AAY/PHH card counts ────────────────────────────────
        has_card_types = (
            "cards_nfsa_aay" in df.columns and
            df["cards_nfsa_aay"].notna().any()
        )

        if not has_card_types and beneficiaries_df is not None:
            bene = beneficiaries_df.copy()
            id_col = "fps_shop_id" if "fps_shop_id" in bene.columns else "shop_id"
            if id_col in bene.columns:
                bene = bene.rename(columns={id_col: "fps_shop_id"})
                for card_col in ["cards_nfsa_aay", "cards_nfsa_phh"]:
                    if card_col in bene.columns:
                        cnts = bene.groupby("fps_shop_id")[card_col].first().rename(card_col)
                        df = (
                            df.set_index("fps_shop_id")
                            .join(cnts, how="left")
                            .reset_index()
                        )
                        df[card_col] = df[card_col].fillna(0).astype(int)

        # ── Step 2: Fallback — assume 10% AAY / 90% PHH mix ─────────────────
        if "cards_nfsa_aay" not in df.columns:
            tc = df.get("total_cards", pd.Series(0, index=df.index))
            df["cards_nfsa_aay"] = (tc * 0.10).round().astype(int)
            df["cards_nfsa_phh"] = (tc * 0.90).round().astype(int)

        # ── Step 3: Compute expected demand per shop ──────────────────────────
        expected_rows = []
        for _, row in df.iterrows():
            exp = self.compute_expected_demand(
                shop_id=str(row.get("fps_shop_id", "")),
                aay_cards=int(row.get("cards_nfsa_aay", 0)),
                phh_cards=int(row.get("cards_nfsa_phh", 0)),
            )
            expected_rows.append(exp)

        exp_df = pd.DataFrame(expected_rows)
        exp_df = exp_df.rename(columns={"shop_id": "fps_shop_id"})

        merge_cols = ["fps_shop_id", "expected_rice_kg", "expected_rice_kg_adjusted",
                      "expected_sugar_kg", "expected_kerosene_litres"]
        df = df.merge(
            exp_df[merge_cols],
            on="fps_shop_id", how="left",
        )

        # ── Step 4: Compute supply gaps ────────────────────────────────────────
        if "rice_total_kg" in df.columns and "expected_rice_kg" in df.columns:
            df["rice_gap_kg"]  = (df["expected_rice_kg"] - df["rice_total_kg"]).round(1)
            denom              = df["expected_rice_kg"].clip(lower=1)
            df["rice_gap_pct"] = (df["rice_gap_kg"] / denom * 100).round(1)

            # Supply status classification
            df["supply_status"] = "normal"
            df.loc[df["rice_gap_pct"] > 80, "supply_status"] = "critically_under_supplied"
            df.loc[(df["rice_gap_pct"] > 20) & (df["rice_gap_pct"] <= 80),
                   "supply_status"] = "under_supplied"
            df.loc[df["rice_gap_pct"] < -10, "supply_status"] = "over_supplied"   # fraud risk

        logger.info(
            f"Supply gap computed for {len(df)} shops — "
            f"under: {(df.get('supply_status', pd.Series()) == 'under_supplied').sum()}, "
            f"over: {(df.get('supply_status', pd.Series()) == 'over_supplied').sum()}"
        )
        return df

    def district_supply_summary(self, shop_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate supply gap metrics to district level."""
        if "district" not in shop_df.columns:
            return pd.DataFrame()

        agg: Dict[str, str] = {"fps_shop_id": "count"}
        for col in [
            "rice_total_kg", "expected_rice_kg", "rice_gap_kg",
            "kerosene_total_kg", "expected_kerosene_litres",
            "total_cards", "cards_nfsa_aay", "cards_nfsa_phh",
        ]:
            if col in shop_df.columns:
                agg[col] = "sum"

        dist_df = (
            shop_df.groupby("district")
            .agg(agg)
            .rename(columns={"fps_shop_id": "n_shops"})
            .reset_index()
        )

        if "rice_gap_kg" in dist_df.columns and "expected_rice_kg" in dist_df.columns:
            denom = dist_df["expected_rice_kg"].clip(lower=1)
            dist_df["district_supply_gap_pct"] = (
                dist_df["rice_gap_kg"] / denom * 100
            ).round(1)

        return dist_df

    def forecast_entitlement(
        self,
        aay_cards: int,
        phh_cards: int,
        months_ahead: int = 3,
        avg_family_size: float = TELANGANA_AVG_FAMILY_SIZE,
    ) -> List[Dict]:
        """
        Generate a deterministic multi-month entitlement forecast.

        Since entitlements are fixed by law, monthly demand is constant
        (the small variance comes from collection rate fluctuations).
        """
        base = self.compute_expected_demand(
            shop_id="",
            aay_cards=aay_cards,
            phh_cards=phh_cards,
            avg_family_size=avg_family_size,
        )
        rice_base = base["expected_rice_kg"]

        results = []
        for m in range(months_ahead):
            results.append({
                "month_offset":         m + 1,
                "predicted_quantity_kg": rice_base,
                "confidence_lower":     round(rice_base * 0.85, 1),   # 15% absence
                "confidence_upper":     round(rice_base * 1.02, 1),   # 2% over-draw
                "model":                "entitlement_physics",
                "entitlement_breakdown": {
                    "aay_rice_kg": aay_cards * AAY_PER_CARD["rice"],
                    "phh_rice_kg": phh_cards * avg_family_size * PHH_PER_PERSON["rice"],
                },
            })
        return results
