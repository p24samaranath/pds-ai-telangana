"""
Synthetic multi-period district-level SCM data generator.

Ground truth sourced directly from Open Data Telangana CSVs
(beneficiaries.csv June 2025, fps_shops.csv June 2025,
 transactions.csv May 2025) — so every figure is calibrated
to the real Telangana PDS universe.

Key real-data anchors
─────────────────────
  Total beneficiary cards : 9,183,183
  Total FPS shops         : 17,434  (all active)
  Monthly rice (May 2025) : 170.22 M kg  (~21.39 kg/card)
  Monthly wheat           : 3.57 M kg (6 districts only)
  Monthly sugar           : 76.5 T   (scattered)
  Districts               : 33
"""

import numpy as np
from typing import Dict, List, Any

# ── Real district registry  ─────────────────────────────────────────────────
# All fields derived from beneficiaries.csv + fps_shops.csv + transactions.csv
# GPS = real district headquarters (corrected from known-bad shop-average entries)
# Distances to Hyderabad computed via haversine(17.385, 78.487, lat, lon) in km
# Fraud seed calibrated from fraud-detection agent output (higher = more flagged)

TELANGANA_DISTRICTS: List[Dict[str, Any]] = [
    # "district" matches the CSV's district column exactly
    {"id": "D01", "name": "Adilabad",                  "beneficiaries": 192_757,  "shops": 356,  "lat": 19.664, "lon": 78.532, "dist_km": 283, "rice_kg": 3_887_966, "wheat_kg":        0, "sugar_kg":   185, "fraud_seed": 0.22},
    {"id": "D02", "name": "Bhadrdri Kothagudem",       "beneficiaries": 297_189,  "shops": 443,  "lat": 17.549, "lon": 80.616, "dist_km": 224, "rice_kg": 4_803_811, "wheat_kg":        0, "sugar_kg": 5_596, "fraud_seed": 0.20},
    {"id": "D03", "name": "Hanumakonda",               "beneficiaries": 231_516,  "shops": 414,  "lat": 17.977, "lon": 79.598, "dist_km": 145, "rice_kg": 3_993_309, "wheat_kg":        0, "sugar_kg":   216, "fraud_seed": 0.12},
    {"id": "D04", "name": "Hyderabad",                 "beneficiaries": 647_282,  "shops": 700,  "lat": 17.385, "lon": 78.487, "dist_km":   0, "rice_kg":14_443_412, "wheat_kg":2_245_346, "sugar_kg":22_422, "fraud_seed": 0.07},
    {"id": "D05", "name": "Jagityal",                  "beneficiaries": 318_732,  "shops": 592,  "lat": 18.793, "lon": 78.741, "dist_km": 179, "rice_kg": 5_519_789, "wheat_kg":        0, "sugar_kg":    22, "fraud_seed": 0.15},
    {"id": "D06", "name": "Janagaon",                  "beneficiaries": 163_283,  "shops": 335,  "lat": 17.727, "lon": 79.152, "dist_km": 114, "rice_kg": 2_601_429, "wheat_kg":        0, "sugar_kg": 2_246, "fraud_seed": 0.10},
    {"id": "D07", "name": "Jayashankar Bhupalpalli",   "beneficiaries": 125_589,  "shops": 277,  "lat": 18.432, "lon": 80.006, "dist_km": 198, "rice_kg": 2_060_033, "wheat_kg":        0, "sugar_kg":     0, "fraud_seed": 0.18},
    {"id": "D08", "name": "Jogulamba Gadwal",          "beneficiaries": 164_357,  "shops": 335,  "lat": 16.226, "lon": 77.799, "dist_km": 182, "rice_kg": 3_266_397, "wheat_kg":    6_456, "sugar_kg": 2_024, "fraud_seed": 0.17},
    {"id": "D09", "name": "Kamareddy",                 "beneficiaries": 256_732,  "shops": 592,  "lat": 18.322, "lon": 78.336, "dist_km": 131, "rice_kg": 5_188_866, "wheat_kg":        0, "sugar_kg": 2_438, "fraud_seed": 0.12},
    {"id": "D10", "name": "Karimnagar",                "beneficiaries": 290_402,  "shops": 566,  "lat": 18.438, "lon": 79.132, "dist_km": 162, "rice_kg": 5_326_297, "wheat_kg":        0, "sugar_kg":     1, "fraud_seed": 0.16},
    {"id": "D11", "name": "Khammam",                   "beneficiaries": 415_905,  "shops": 748,  "lat": 17.247, "lon": 80.150, "dist_km": 195, "rice_kg": 6_752_910, "wheat_kg":        0, "sugar_kg": 9_476, "fraud_seed": 0.14},
    {"id": "D12", "name": "Kumarambheem Asifabad",     "beneficiaries": 141_904,  "shops": 314,  "lat": 19.364, "lon": 79.286, "dist_km": 273, "rice_kg": 2_804_329, "wheat_kg":        0, "sugar_kg":     0, "fraud_seed": 0.25},
    {"id": "D13", "name": "Mahabubabad",               "beneficiaries": 243_204,  "shops": 558,  "lat": 17.601, "lon": 80.002, "dist_km": 168, "rice_kg": 3_663_948, "wheat_kg":        0, "sugar_kg":   900, "fraud_seed": 0.13},
    {"id": "D14", "name": "Mahbubnagar",               "beneficiaries": 245_463,  "shops": 506,  "lat": 16.733, "lon": 77.983, "dist_km": 110, "rice_kg": 4_242_163, "wheat_kg":    3_852, "sugar_kg":     7, "fraud_seed": 0.19},
    {"id": "D15", "name": "Manchiryala",               "beneficiaries": 223_844,  "shops": 423,  "lat": 18.873, "lon": 79.439, "dist_km": 202, "rice_kg": 3_862_099, "wheat_kg":        0, "sugar_kg":   199, "fraud_seed": 0.15},
    {"id": "D16", "name": "Medak",                     "beneficiaries": 216_716,  "shops": 520,  "lat": 18.045, "lon": 78.262, "dist_km":  90, "rice_kg": 3_881_136, "wheat_kg":        0, "sugar_kg":     0, "fraud_seed": 0.11},
    {"id": "D17", "name": "Medchal",                   "beneficiaries": 537_810,  "shops": 618,  "lat": 17.618, "lon": 78.562, "dist_km":  29, "rice_kg":12_310_868, "wheat_kg":  839_556, "sugar_kg": 7_941, "fraud_seed": 0.08},
    {"id": "D18", "name": "Mulugu",                    "beneficiaries":  94_628,  "shops": 222,  "lat": 18.196, "lon": 80.100, "dist_km": 207, "rice_kg": 1_535_514, "wheat_kg":        0, "sugar_kg":     0, "fraud_seed": 0.21},
    {"id": "D19", "name": "Nagarkarnool",              "beneficiaries": 243_722,  "shops": 552,  "lat": 16.477, "lon": 78.322, "dist_km": 126, "rice_kg": 4_015_431, "wheat_kg":        0, "sugar_kg":   433, "fraud_seed": 0.16},
    {"id": "D20", "name": "Nalgonda",                  "beneficiaries": 484_210,  "shops": 997,  "lat": 17.047, "lon": 79.267, "dist_km":  93, "rice_kg": 7_666_553, "wheat_kg":        0, "sugar_kg": 6_380, "fraud_seed": 0.13},
    {"id": "D21", "name": "Narayanpet",                "beneficiaries": 145_684,  "shops": 301,  "lat": 16.745, "lon": 77.491, "dist_km": 163, "rice_kg": 2_768_061, "wheat_kg":        0, "sugar_kg":   682, "fraud_seed": 0.15},
    {"id": "D22", "name": "Nirmal",                    "beneficiaries": 219_972,  "shops": 412,  "lat": 19.096, "lon": 78.338, "dist_km": 225, "rice_kg": 4_002_726, "wheat_kg":        0, "sugar_kg":     0, "fraud_seed": 0.17},
    {"id": "D23", "name": "Nizamabad",                 "beneficiaries": 403_510,  "shops": 759,  "lat": 18.672, "lon": 78.094, "dist_km": 163, "rice_kg": 7_815_400, "wheat_kg":        0, "sugar_kg":     0, "fraud_seed": 0.14},
    {"id": "D24", "name": "Peddapalli",                "beneficiaries": 223_553,  "shops": 410,  "lat": 18.618, "lon": 79.376, "dist_km": 184, "rice_kg": 3_704_959, "wheat_kg":        0, "sugar_kg":     1, "fraud_seed": 0.13},
    {"id": "D25", "name": "Rajanna Siricilla",         "beneficiaries": 177_851,  "shops": 345,  "lat": 18.386, "lon": 78.837, "dist_km": 155, "rice_kg": 3_032_772, "wheat_kg":        0, "sugar_kg": 6_138, "fraud_seed": 0.13},
    {"id": "D26", "name": "Ranga Reddy",               "beneficiaries": 572_792,  "shops": 936,  "lat": 17.246, "lon": 78.372, "dist_km":  18, "rice_kg":12_953_556, "wheat_kg":  467_188, "sugar_kg": 5_404, "fraud_seed": 0.09},
    {"id": "D27", "name": "Sangareddy",                "beneficiaries": 381_017,  "shops": 845,  "lat": 17.619, "lon": 78.089, "dist_km":  55, "rice_kg": 7_841_638, "wheat_kg":        0, "sugar_kg":     0, "fraud_seed": 0.10},
    {"id": "D28", "name": "Siddipet",                  "beneficiaries": 298_985,  "shops": 686,  "lat": 18.103, "lon": 78.847, "dist_km": 110, "rice_kg": 5_505_663, "wheat_kg":    7_598, "sugar_kg": 2_461, "fraud_seed": 0.10},
    {"id": "D29", "name": "Suryapet",                  "beneficiaries": 326_057,  "shops": 735,  "lat": 17.141, "lon": 79.623, "dist_km": 130, "rice_kg": 5_153_375, "wheat_kg":        0, "sugar_kg": 1_296, "fraud_seed": 0.12},
    {"id": "D30", "name": "Vikarabad",                 "beneficiaries": 251_097,  "shops": 588,  "lat": 17.338, "lon": 77.908, "dist_km":  73, "rice_kg": 4_870_765, "wheat_kg":        0, "sugar_kg":     0, "fraud_seed": 0.10},
    {"id": "D31", "name": "Wanaparthy",                "beneficiaries": 161_316,  "shops": 325,  "lat": 16.367, "lon": 78.065, "dist_km": 142, "rice_kg": 2_608_385, "wheat_kg":        0, "sugar_kg":    10, "fraud_seed": 0.15},
    {"id": "D32", "name": "Warangal",                  "beneficiaries": 267_141,  "shops": 509,  "lat": 17.977, "lon": 79.598, "dist_km": 145, "rice_kg": 4_329_349, "wheat_kg":        0, "sugar_kg":     0, "fraud_seed": 0.11},
    {"id": "D33", "name": "Yadadri Bhuvanagiri",       "beneficiaries": 218_963,  "shops": 515,  "lat": 17.509, "lon": 78.882, "dist_km":  56, "rice_kg": 3_807_576, "wheat_kg":        0, "sugar_kg":     1, "fraud_seed": 0.09},
]

# ── Validation totals (assert against real data) ───────────────────────────
_TOTAL_CARDS_REAL  = 9_183_183
_TOTAL_SHOPS_REAL  = 17_434
_TOTAL_RICE_REAL   = 170_220_485   # May 2025 actuals (kg)
_TOTAL_WHEAT_REAL  =   3_569_996
_TOTAL_SUGAR_REAL  =      76_479

# Quick sanity
_cards  = sum(d["beneficiaries"] for d in TELANGANA_DISTRICTS)
_shops  = sum(d["shops"]         for d in TELANGANA_DISTRICTS)
_rice   = sum(d["rice_kg"]       for d in TELANGANA_DISTRICTS)
_wheat  = sum(d["wheat_kg"]      for d in TELANGANA_DISTRICTS)
assert _cards == _TOTAL_CARDS_REAL,  f"Card mismatch: {_cards} vs {_TOTAL_CARDS_REAL}"
assert _shops == _TOTAL_SHOPS_REAL,  f"Shop mismatch: {_shops} vs {_TOTAL_SHOPS_REAL}"
assert _rice  == _TOTAL_RICE_REAL,   f"Rice mismatch: {_rice}  vs {_TOTAL_RICE_REAL}"

# ── Monthly seasonal multipliers (index 0 = January) ──────────────────────
# Based on Telangana festive calendar & harvest cycles:
#   Sankranti (Jan) +10%, Ugadi (Mar/Apr) +12%, Bonalu (Jul) +5%
#   Dussehra (Oct) +8%,  Diwali (Nov) +10%
#   Summer lean (Apr-Jun) -5%: fewer people collect when migrating
_SEASONAL = np.array([
    1.10,  # Jan  Sankranti
    1.02,  # Feb
    1.12,  # Mar  Ugadi
    0.97,  # Apr  migration starts
    0.95,  # May  (real transaction baseline month)
    0.94,  # Jun
    1.05,  # Jul  Bonalu
    1.00,  # Aug
    1.03,  # Sep
    1.08,  # Oct  Dussehra
    1.10,  # Nov  Diwali
    1.05,  # Dec
])


class DistrictDataGenerator:
    """
    Generate multi-period SCM simulation data for all 33 Telangana districts.

    All base figures are anchored to the real Open Data Telangana CSVs:
      - beneficiaries.csv  (June 2025)
      - fps_shops.csv      (June 2025, 17,434 active shops)
      - transactions.csv   (May 2025 actuals for rice/wheat/sugar kg)

    Parameters
    ----------
    n_periods : int
        Simulation horizon in months (default 24).
    seed : int
        Random seed.
    start_month_offset : int
        Calendar month of period-0 (0=Jan, 4=May). Defaults to 4 (May 2025).
    commodity : str
        Primary commodity: 'rice' (default), 'wheat', 'sugar'.
    """

    def __init__(
        self,
        n_periods: int = 24,
        seed: int = 42,
        start_month_offset: int = 4,
        commodity: str = "rice",
    ):
        self.n_periods   = n_periods
        self.rng         = np.random.default_rng(seed)
        self.start_month = start_month_offset
        self.commodity   = commodity
        self.N           = len(TELANGANA_DISTRICTS)
        self.districts   = TELANGANA_DISTRICTS

    # ── Public interface ───────────────────────────────────────────────────

    def generate(self) -> Dict[str, Any]:
        """
        Return simulation inputs fully calibrated to real data.

        Keys
        ----
        district_meta       list[dict]  — 33 district records
        transport_cost      ndarray [N] — ₹/kg (linear distance model)
        initial_inventory   ndarray [N] — kg (1.8 × May 2025 monthly)
        initial_fraud_prob  ndarray [N] — calibrated from fraud-detection agent
        demand_mean         ndarray [N, T]  — monthly demand forecast (kg)
        demand_std          ndarray [N, T]  — demand std (15% CV)
        supply_schedule     ndarray [T]     — state-level supply available (kg)
        commodity_mix       dict        — {rice, wheat, sugar} annual fractions
        """
        demand_mean, demand_std = self._build_demand_profiles()
        return {
            "district_meta":      self.districts,
            "transport_cost":     self._build_transport_costs(),
            "initial_inventory":  self._build_initial_inventory(demand_mean),
            "initial_fraud_prob": np.array([d["fraud_seed"] for d in self.districts]),
            "demand_mean":        demand_mean,
            "demand_std":         demand_std,
            "supply_schedule":    self._build_supply_schedule(demand_mean),
            "commodity_mix":      {"rice": 0.965, "wheat": 0.020, "sugar": 0.001, "other": 0.014},
        }

    def district_names(self) -> List[str]:
        return [d["name"] for d in self.districts]

    def real_monthly_totals(self) -> Dict[str, float]:
        """Return real May 2025 monthly distribution totals (kg)."""
        return {
            "rice_kg":   _TOTAL_RICE_REAL,
            "wheat_kg":  _TOTAL_WHEAT_REAL,
            "sugar_kg":  _TOTAL_SUGAR_REAL,
            "total_cards": _TOTAL_CARDS_REAL,
            "total_shops": _TOTAL_SHOPS_REAL,
        }

    # ── Private helpers ────────────────────────────────────────────────────

    def _base_demand_kg(self, district: Dict) -> float:
        """
        Use real May 2025 transaction data as the base monthly demand.
        Apply a 10% uplift for non-collection buffer (some beneficiaries
        collect late or in the following month).
        """
        col = f"{self.commodity}_kg"
        base = district.get(col, 0) or district.get("rice_kg", 0)
        return base * 1.10   # 10% uplift for late collection & waste

    def _build_demand_profiles(self):
        """
        D_{i,t} ~ LogNormal(μ_{i,t}, σ_i)
        μ_{i,t} = base_i × seasonal_{(start+t) % 12} × (1.02)^(t // 12)
        """
        N, T = self.N, self.n_periods
        demand_mean = np.zeros((N, T))
        demand_std  = np.zeros((N, T))

        for i, d in enumerate(self.districts):
            base = self._base_demand_kg(d)
            for t in range(T):
                month     = (self.start_month + t) % 12
                growth    = (1.02) ** (t // 12)          # 2% YoY
                seasonal  = _SEASONAL[month]
                # ±3% idiosyncratic noise (permanent factor per district per period)
                noise     = self.rng.normal(1.0, 0.015)
                mu        = max(base * seasonal * growth * noise, 1.0)
                demand_mean[i, t] = mu
                demand_std[i, t]  = mu * 0.15            # CV = 15%

        return demand_mean, demand_std

    def _build_transport_costs(self) -> np.ndarray:
        """
        Transport cost ₹/kg using linear distance model:
          cost_i = ₹0.40/kg (base, Hyderabad) + ₹0.011/km × dist_km

        Calibrated so that:
          Hyderabad (0 km)   → ₹0.40/kg
          Kumarambheem (273 km) → ₹3.40/kg
        """
        return np.array([
            0.40 + 0.011 * d["dist_km"]
            for d in self.districts
        ])

    def _build_initial_inventory(self, demand_mean: np.ndarray) -> np.ndarray:
        """
        Start with 1.8 months of buffer stock per district
        (consistent with TSCSC warehousing norms).
        Add ±15% noise.
        """
        base  = demand_mean[:, 0]
        noise = self.rng.uniform(0.85, 1.15, size=self.N)
        return base * 1.8 * noise

    def _build_supply_schedule(self, demand_mean: np.ndarray) -> np.ndarray:
        """
        State-level supply available each period.
        Procurement fraction varies 87–95% of total demand
        (models seasonal FCI release patterns).
        """
        total_demand = demand_mean.sum(axis=0)           # [T]
        # Procurement typically tighter in Feb–Apr (inter-season gap)
        base_fraction = np.array([
            0.91, 0.88, 0.87, 0.89, 0.91, 0.93,         # P0–P5
            0.94, 0.95, 0.93, 0.91, 0.90, 0.91,         # P6–P11
            0.91, 0.88, 0.87, 0.89, 0.91, 0.93,         # P12–P17 (repeat)
            0.94, 0.95, 0.93, 0.91, 0.90, 0.91,         # P18–P23
        ] + [0.91] * max(0, self.n_periods - 24))        # extra periods
        frac = base_fraction[:self.n_periods]
        # Add ±1% random noise
        frac = frac + self.rng.uniform(-0.01, 0.01, size=self.n_periods)
        return total_demand * np.clip(frac, 0.82, 0.97)
