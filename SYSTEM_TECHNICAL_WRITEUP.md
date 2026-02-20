# PDS AI Optimization System — Technical Writeup
### Telangana Fair Price Shop Fraud Detection, Demand Forecasting & Geospatial Optimization

> **Version:** 1.0.0 | **Data Snapshot:** May–June 2025 | **Scope:** 33 Districts, 17,434 FPS Shops, 9.18 M Ration Cards

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Raw Data: Fields, Sources & Meaning](#2-raw-data-fields-sources--meaning)
3. [Data Fetching Script](#3-data-fetching-script)
4. [Data Processing Pipeline](#4-data-processing-pipeline)
5. [Data Anomalies Observed](#5-data-anomalies-observed)
6. [Data Timing & Staleness Analysis](#6-data-timing--staleness-analysis)
7. [ML Model 1 — Rule-Based Fraud Detector](#7-ml-model-1--rule-based-fraud-detector)
8. [ML Model 2 — Isolation Forest Point Anomaly Detector](#8-ml-model-2--isolation-forest-point-anomaly-detector)
9. [ML Model 3 — DBSCAN Coordinated Fraud Cluster Detector](#9-ml-model-3--dbscan-coordinated-fraud-cluster-detector)
10. [ML Model 4 — LSTM Demand Forecaster](#10-ml-model-4--lstm-demand-forecaster)
11. [ML Model 5 — Prophet Demand Forecaster](#11-ml-model-5--prophet-demand-forecaster)
12. [ML Model 6 — Ensemble Forecaster](#12-ml-model-6--ensemble-forecaster)
13. [ML Model 7 — Geospatial K-Means Optimizer](#13-ml-model-7--geospatial-k-means-optimizer)
14. [Multi-Agent Orchestration](#14-multi-agent-orchestration)
15. [Improvements Based on Available Data](#15-improvements-based-on-available-data)

---

## 1. Architecture Overview

The system follows a **LangGraph-inspired stateful multi-agent pattern**. Five specialized agents are coordinated by a master Orchestrator that routes work based on a `WorkflowTrigger` enum and maintains shared memory for audit, repeat-offender detection, and escalation.

```
┌──────────────────────────────────────────────────────────┐
│              OrchestratorAgent (Master Router)            │
│  • Shared state: shops_df, bene_df, txn_df              │
│  • Memory: pending_alerts, last_run timestamps           │
│  • Triggers: MONTHLY_BATCH | REALTIME_FRAUD | NL_QUERY   │
│              GEO_CHANGE | FULL_PIPELINE                   │
└─────────────┬───────────────────────────────────────────┘
              │
    ┌─────────┼──────────┬───────────────┐
    ▼         ▼          ▼               ▼
FraudAgent  ForecastAgent  GeoAgent  ReportingAgent
Rule + IF   LSTM+Prophet   KMeans+    Claude API +
+ DBSCAN    Ensemble       Voronoi    Dashboard KPIs
```

**Trigger routing:**

| Trigger | Agents activated | Use case |
|---|---|---|
| `MONTHLY_BATCH` | All four | End-of-month planning cycle |
| `REALTIME_FRAUD` | Fraud only | Live ePoS transaction check |
| `NL_QUERY` | Reporting only | Official asks a question in plain English |
| `GEO_CHANGE` | Geo + Reporting | Shop opened/closed |
| `FULL_PIPELINE` | All four | Full dashboard refresh |

---

## 2. Raw Data: Fields, Sources & Meaning

Data is sourced from the **Open Data Telangana** portal via three REST datasets. All are monthly snapshots at the FPS-shop level — there is no individual beneficiary transaction row in the public data.

### 2.1 FPS Shops (`fps_shops.csv`) — 17,434 rows

**Source dataset UUID:** `6b5a1abe-6bf8-4c28-a400-2e42640de641`
**API filename pattern:** `shop-status-details_M_YYYY.csv`
**Snapshot month:** June 2025

| Column (raw API) | Renamed to | Type | Meaning |
|---|---|---|---|
| `distCode` | `district_code` | int | Telangana district numeric code (e.g., 532 = Adilabad) |
| `distName` | `district` | str | District name (33 unique values) |
| `officeCode` | `office_code` | str | Mandal/circle office code |
| `officeName` | `office_name` | str | Mandal/circle office name |
| `shopNo` | `shop_id` | str | Unique FPS shop identifier (e.g., `1901001`) |
| `address` | `address` | str | Physical address of the FPS shop |
| `longitude` | `longitude` | float | WGS84 decimal longitude |
| `latitude` | `latitude` | float | WGS84 decimal latitude |
| `fpsStatus` | `is_active_str` → `is_active` | bool | "Active" → True, else False |
| `fpsType` | `shop_type` | str | `"Normal Shop"` or `"kerosene Shop"` |
| `dateTime` | `last_updated` | datetime | When this record was last updated in the state system |
| — | `source_month` | int | Month of the API snapshot used |
| — | `source_year` | int | Year of the API snapshot used |
| — | `downloaded_at` | ISO str | UTC timestamp when we fetched this row |

**Key facts:**
- 17,262 Normal Shops + 172 Kerosene-only shops
- 17,368 of 17,434 shops have valid GPS coordinates (66 shops have lat=0 or lon=0)
- All 17,434 shops are currently marked `is_active = True` in this snapshot
- Latitude range: 16.5°–19.9°N | Longitude range: 77.4°–81.3°E (covers Telangana state)

---

### 2.2 Transactions (`transactions.csv`) — 17,254 rows

**Source dataset UUID:** `4a7337ce-08bd-4f74-8e00-bd6eb83eb4ff`
**API filename pattern:** `shop-wise-trans-details_M_YYYY.csv`
**Snapshot month:** May 2025
**Grain:** One row per FPS shop per month

| Column (raw API) | Renamed to | Type | Meaning |
|---|---|---|---|
| `distCode` | `district_code` | int | District numeric code |
| `distName` | `district` | str | District name |
| `officeCode` | `office_code` | str | Mandal code |
| `officeName` | `office_name` | str | Mandal name |
| `shopNo` | `shop_id` | str | FPS shop ID |
| `month` | `month` | int | Reporting month (1–12) |
| `year` | `year` | int | Reporting year |
| `noOfRcs` | `total_cards` | int | Number of ration cards registered at this shop |
| `noOfTrans` | `total_transactions` | int | Total transactions completed this month |
| `riceAfsc` | `rice_afsc_kg` | float | Rice lifted under AFSC scheme (kg) |
| `riceFsc` | `rice_fsc_kg` | float | Rice lifted under FSC scheme (kg) |
| `riceAap` | `rice_aap_kg` | float | Rice lifted under AAP scheme (kg) |
| `wheat` | `wheat_kg` | float | Wheat lifted (kg) |
| `sugar` | `sugar_kg` | float | Sugar lifted (kg) |
| `rgdal` | `red_gram_dal_kg` | float | Red gram dal lifted (kg) |
| `kerosene` | `kerosene_litres` | float | Kerosene distributed (litres) |
| `totalAmount` | `total_amount` | float | Total value in Indian Rupees |
| `salt` | `salt_kg` | float | Salt distributed (kg) |
| `otherShopTransCnt` | `other_shop_transactions` | int | Transactions served from other FPS shops |
| — | `transaction_date` | date | Derived: `YYYY-MM-01` from year+month |

**Commodity totals for May 2025 (all 33 districts):**

| Commodity | Total | Notes |
|---|---|---|
| Rice FSC | **151.14 M kg** | Dominant scheme — covers ~95% of rice |
| Rice AFSC | 19.05 M kg | Supplementary allocation |
| Rice AAP | 0.03 M kg | Minimal uptake |
| Wheat | 3.57 M kg | Only Hyderabad, Ranga Reddy, Medchal corridors |
| Sugar | 76.5 K kg | Selective distribution |
| Salt | 115 kg | Negligible (19 shops only) |
| Red Gram Dal | 0 kg | Not distributed May 2025 |
| Kerosene | 0 kg | Not distributed May 2025 |

**Top 5 districts by total rice (May 2025):**

| District | Rice (M kg) | Shops | Beneficiary Cards |
|---|---|---|---|
| Hyderabad | 14.44 | 700 | 628,679 |
| Ranga Reddy | 12.95 | 936 | 580,566 |
| Medchal | 12.31 | 618 | 563,702 |
| Sangareddy | 7.84 | 845 | 345,004 |
| Nizamabad | 7.82 | 759 | 367,585 |

---

### 2.3 Beneficiaries (`beneficiaries.csv`) — 17,275 rows

**Source dataset UUID:** `84e47ac8-2c24-43a4-a045-2a7bec7212e5`
**API filename pattern:** `fpshop-card-status_M_YYYY.csv`
**Snapshot month:** June 2025
**Grain:** One row per FPS shop per month — aggregated card counts, not individual beneficiaries

| Column (raw API) | Renamed to | Type | Meaning |
|---|---|---|---|
| `distCode` | `district_code` | int | District code |
| `distName` | `district` | str | District name |
| `officeCode` | `office_code` | str | Mandal code |
| `officeName` | `office_name` | str | Mandal name |
| `shopNo` | `shop_id` | str | FPS shop ID |
| `month` | `month` | int | Reporting month |
| `year` | `year` | int | Reporting year |
| `rcNfsaAay` | `cards_nfsa_aay` | int | NFSA Antyodaya Anna Yojana (AAY) ration cards |
| `unitsNfsaAay` | `units_nfsa_aay` | int | Family members under NFSA AAY |
| `rcNfsaPhh` | `cards_nfsa_phh` | int | NFSA Priority Household (PHH) cards |
| `unitsNfsaPhh` | `units_nfsa_phh` | int | Members under NFSA PHH |
| `totalRcNfsa` | `total_cards_nfsa` | int | Total NFSA cards (AAY + PHH) |
| `totalUnitsNfsa` | `total_units_nfsa` | int | Total NFSA members |
| `rcStateAay` | `cards_state_aay` | int | State-scheme AAY cards |
| `unitsStateAay` | `units_state_aay` | int | Members under State AAY |
| `rcStatePhh` | `cards_state_phh` | int | State-scheme PHH cards |
| `unitsStatePhh` | `units_state_phh` | int | Members under State PHH |
| `rcStateAap` | `cards_state_aap` | int | State AAP (Annapurna) cards |
| `unitsStateAap` | `units_state_aap` | int | Members under State AAP |
| `totalRcState` | `total_cards_state` | int | Total state-scheme cards |
| `totalUnitsState` | `total_units_state` | int | Total state-scheme members |
| `totalRcs` | `total_cards` | int | Grand total cards (NFSA + State) |
| `totalUnits` | `total_units` | int | Grand total members |
| `cardTypeId` | `card_type_id` | str | Internal card type ID (mostly null) |
| `units` | `units` | int | Units (additional, mostly 0) |
| `gasCylinders` | `gas_cylinders` | int | LPG cylinders (all 0 in this snapshot) |
| `mroAsoApprDate` | `approval_date` | str | MRO/ASO approval date (all null in this snapshot) |
| `cardPoolType` | `card_pool_type` | str | Card pool category (all null) |

**Telangana-wide beneficiary totals (June 2025):**

| Category | Cards | Members |
|---|---|---|
| NFSA AAY (poorest) | 565,025 | ~1.86 M |
| NFSA PHH (priority) | 4,895,114 | ~15.9 M |
| State AAY | ~33,000 | ~110 K |
| State PHH | 3,716,589 | ~9.7 M |
| State AAP | ~57,000 | ~185 K |
| **Grand Total** | **9,183,183** | **30,129,181** |

---

## 3. Data Fetching Script

### File: `backend/services/telangana_fetcher.py`

The `TelanganaFetcher` class handles all interaction with the Open Data Telangana REST API.

#### 3.1 API Structure

```
GET https://data.telangana.gov.in/api/1/metastore/schemas/dataset/items/{UUID}?show-reference-ids
```

Returns a JSON object with a `"distribution"` array. Each element has a `"data"` sub-object containing `"title"` and `"downloadURL"`. The download URL points directly to a CSV file.

**Dataset UUIDs:**
```python
DATASET_IDS = {
    "transactions":  "4a7337ce-08bd-4f74-8e00-bd6eb83eb4ff",
    "card_status":   "84e47ac8-2c24-43a4-a045-2a7bec7212e5",
    "geo_locations": "6b5a1abe-6bf8-4c28-a400-2e42640de641",
}
```

#### 3.2 Month/Year Extraction from Filenames

CSV filenames follow the pattern `shop-wise-trans-details_5_2025.csv`. The `_extract_month_year()` function uses three regex patterns in priority order:

```python
# Pattern 1 (URL): "_M_YYYY.csv" or "_MM_YYYY.csv"
re.search(r"_(\d{1,2})_(\d{4})\.csv", url)

# Pattern 2 (Title): "to DD-MM-YYYY"
re.search(r"to\s+\d{1,2}-(\d{1,2})-(\d{4})", title)

# Pattern 3 (Title fallback): "MM-YYYY"
re.search(r"(\d{1,2})-(\d{4})", title)
```

#### 3.3 Download Methods

| Method | What it does | When to use |
|---|---|---|
| `discover()` | Fetches the manifest of all available files from all 3 datasets | Always called first |
| `download_latest()` | Sorts by (year, month) desc, downloads only the newest file per dataset | Daily/weekly refresh |
| `download_range(from_y, from_m, to_y, to_m)` | Filters manifest by date range, downloads matching files | Backfill periods |
| `download_all()` | Downloads every file in the manifest (2018–present) | Initial historical load |
| `fetch_and_save(mode)` | Convenience wrapper calling one of the above + `build_master_files()` | Used by API routes |

#### 3.4 Caching Logic

Before downloading a file, the fetcher checks if `data/raw/{dataset}_{year}_{month:02d}.csv` already exists:

```python
if dest.exists():
    return pd.read_csv(dest)   # serve from disk cache
```

This prevents redundant downloads on repeated calls.

#### 3.5 Column Renaming

After download, `_normalise(df, dataset_name)` applies the rename map for the dataset:

```python
TRANSACTIONS_RENAME = {
    "distCode" → "district_code",
    "distName" → "district",
    "shopNo"   → "shop_id",
    "noOfRcs"  → "total_cards",
    "noOfTrans"→ "total_transactions",
    "riceAfsc" → "rice_afsc_kg",
    "riceFsc"  → "rice_fsc_kg",
    "riceAap"  → "rice_aap_kg",
    "wheat"    → "wheat_kg",
    "sugar"    → "sugar_kg",
    "rgdal"    → "red_gram_dal_kg",
    "kerosene" → "kerosene_litres",
    "totalAmount" → "total_amount",
    "salt"     → "salt_kg",
    "otherShopTransCnt" → "other_shop_transactions",
    ...
}
```

#### 3.6 Master File Assembly (`build_master_files()`)

After downloading, three master files are assembled:

| Master file | Source | Transform |
|---|---|---|
| `fps_shops.csv` | `geo_locations` | `fpsStatus == "Active"` → `is_active` bool |
| `transactions.csv` | `transactions` | Adds `transaction_date = YYYY-MM-01` |
| `beneficiaries.csv` | `card_status` | Saved as-is; normalised downstream |

#### 3.7 Rate Limiting & Error Handling

- **0.5 second delay** between HTTP requests (`RATE_DELAY = 0.5`)
- **30 second timeout** per request (`TIMEOUT = 30`)
- **`User-Agent` header** set to `PDS-AI-Optimizer/1.0 (research)` for transparency
- Failed downloads log an error and return `None` (gracefully skipped)
- `on_bad_lines="skip"` prevents parse failures on malformed CSV rows

---

## 4. Data Processing Pipeline

### File: `backend/services/data_ingestion.py`

#### 4.1 Priority Loading Chain

```
load_shops() / load_transactions() / load_beneficiaries()
    │
    ├─ 1. Master CSV exists? (fps_shops.csv / transactions.csv / beneficiaries.csv)
    │      → YES: read directly
    │
    ├─ 2. Monthly CSVs exist? (geo_locations_*.csv / transactions_*.csv / card_status_*.csv)
    │      → YES: glob + concat all matching files
    │
    └─ 3. No files found → generate synthetic demo data (seeded random)
```

#### 4.2 `_normalise_shops(df)` — FPS Shop Normalisation

**Input:** Raw geo CSV with `is_active_str` or legacy `fpsStatus` column
**Output:** DataFrame with canonical columns used by all agents

| Step | Action |
|---|---|
| `is_active` | Convert string "Active" → True, else False |
| `shop_id` | Cast to string; remap from `shopNo` if needed |
| `shop_name` | Synthesise as `"{district} FPS {shop_id}"` if absent |
| `total_cards` | Default 0 (real geo file has no card count; joined from card_status separately) |

#### 4.3 `_normalise_transactions(df)` — Wide-to-Long Transformation

The real Telangana CSV has **one row per shop per month** with commodity values in separate columns. The ML models expect **one row per commodity** (long format). This function melts the wide format:

```
Input (wide):
shop_id | month | rice_afsc_kg | rice_fsc_kg | wheat_kg | sugar_kg
1901001 | 5     | 1330.0       | 17604.0     | 0.0      | 0.0

Output (long):
shop_id | month | commodity | quantity_kg | transaction_date
1901001 | 5     | rice      | 1330.0      | 2025-05-01
1901001 | 5     | rice      | 17604.0     | 2025-05-01
```

**Commodity mapping applied:**
```python
commodity_map = {
    "rice_afsc_kg"    → "rice",
    "rice_fsc_kg"     → "rice",
    "rice_aap_kg"     → "rice",
    "wheat_kg"        → "wheat",
    "sugar_kg"        → "sugar",
    "red_gram_dal_kg" → "red_gram_dal",
    "kerosene_litres" → "kerosene",
    "salt_kg"         → "salt",
}
```

Only rows where the commodity column `> 0` are kept. The function also generates:
- `transaction_id`: `"{fps_shop_id}_{commodity}_{transaction_date}"`
- `card_id`: `"{fps_shop_id}_card"` (proxy, as individual card data is not public)
- `status`: `"completed"`
- `biometric_verified`: `True` (assumed; real ePoS logs not in public dataset)

#### 4.4 `_normalise_beneficiaries(df)` — Card Status Transformation

| Step | Action |
|---|---|
| `fps_shop_id` | From `shop_id` |
| `card_id` | Synthesised as `"{fps_shop_id}_{index}"` |
| `card_type` | "AAY" if `cards_nfsa_aay > 0`, else "PHH" |
| `members_count` | `total_units / total_cards`, clipped 1–10 |
| `is_active` | Default True |
| `latitude / longitude` | `None` (card_status has no coordinates) |

#### 4.5 Synthetic Data Fallback

When no CSVs exist (e.g., first run before fetch), the system generates:
- **150 synthetic shops** across 10 major districts; 95% active; lat/lon ± 0.5° from district centres
- **5,000 synthetic beneficiaries**: 2/3 PHH, 1/3 AAY; 1–6 members; 97% active
- **12 months of synthetic transactions**: ~1,500 beneficiaries × 2 commodities per month with **5% injected fraud** patterns (after-hours, bulk × 3, high volume)

The synthetic fallback uses `random.seed(42)` and `np.random.seed(42)` for reproducibility.

---

## 5. Data Anomalies Observed

### 5.1 Single Month of Transaction History

**Observation:** The `transactions.csv` master file contains data from **May 2025 only** (1 month). All `transaction_date` values are `2025-05-01`.

**Impact:** LSTM and Prophet models require ≥ 3 months of history per shop to fit meaningful time-series patterns. With one month, the system falls back to the **growth-adjusted heuristic** (2% monthly growth assumed), which returns reasonable but purely extrapolated forecasts.

**Root cause:** The fetcher was run in `"latest"` mode, downloading only the most recent snapshot. Historical data from 2018–2024 is available on the portal but not yet downloaded.

---

### 5.2 Missing Wheat Distribution in 27 of 33 Districts

**Observation:** Wheat is distributed (`wheat_kg > 0`) in only 1,331 of 17,254 rows — exclusively in Hyderabad, Ranga Reddy, Medchal, Siddipet, and surrounding urban districts.

**Interpretation:** Wheat is a supplementary commodity under the NFSA framework. It is allocated only to certain APL/PHH card categories in select districts; in predominantly rural districts, rice is the sole cereal.

**Impact on models:** Forecasting wheat demand for non-distributing districts will always return near-zero, which is correct behavior. The growth-heuristic fallback correctly returns 0 for shops with no wheat history.

---

### 5.3 No Kerosene or Red Gram Dal in May 2025

**Observation:** `kerosene_litres` and `red_gram_dal_kg` are 0 across all 17,254 rows.

**Interpretation:** Kerosene and red gram dal distribution is seasonal/episodic and may not have been distributed this month. The public portal only reflects actual lifts, not allocations.

**Impact:** These commodities exist in the schema but produce empty forecasts for this period. Models will handle gracefully with the heuristic fallback.

---

### 5.4 Rice Subsidy Scheme Split (AFSC vs FSC)

**Observation:** Rice is split across three columns: `rice_afsc_kg`, `rice_fsc_kg`, `rice_aap_kg`. FSC dominates (151 M kg vs 19 M kg AFSC vs 0.03 M kg AAP).

**Interpretation:**
- **FSC (Food Security Card):** Main NFSA PHH entitlement (5 kg/person)
- **AFSC (Additional FSC):** Supplementary state scheme
- **AAP (Antyodaya Anna Yojana Plus):** State variant for the ultra-poor

**Impact:** The normalisation pipeline merges all three into a single `"rice"` commodity row (three rows per shop). This inflates the total rice quantity per shop by a factor of 3 in the long-format DataFrame. Forecasts for "rice" represent total rice across all sub-schemes combined.

---

### 5.5 `other_shop_transactions` Nearly Universal

**Observation:** 17,159 of 17,254 rows (99.5%) have `other_shop_transactions > 0`.

**Interpretation:** This column counts beneficiaries from other shops who were served at this shop (portability transactions). In Telangana's One Nation One Ration Card (ONORC) system, beneficiaries can lift rations at any FPS. Nearly every shop services some portable transactions.

**Impact:** The system doesn't currently distinguish portability transactions from regular ones. This could create false positives in fraud detection (multi-shop card usage rules) if individual card-level data were available.

---

### 5.6 Salt Distribution Nearly Non-Existent

**Observation:** Only 19 shops distributed salt (max 25 kg). This is extremely sparse.

**Interpretation:** Salt is typically not a PDS commodity in Telangana; these 19 rows may be data entry artifacts or pilot distributions.

**Impact:** Salt forecasting returns near-zero results; this is correct behavior.

---

### 5.7 66 Shops with Zero/Null Coordinates

**Observation:** 66 of 17,434 shops have `latitude = 0.0` or `longitude = 0.0` (effectively missing coordinates despite not being null).

**Impact:** These shops are excluded from distance matrix computation in the Geospatial Optimizer. Their beneficiaries cannot be assessed for accessibility. A filter on `lat > 5.0 and lon > 60.0` (bounds check for India) would cleanly exclude them.

---

### 5.8 Beneficiary Data Has No GPS Coordinates

**Observation:** The `beneficiaries.csv` (card_status snapshot) contains no `latitude` or `longitude` columns. The Geospatial Optimizer assigns `None` for all beneficiary coordinates.

**Impact:** Distance-to-shop calculations cannot be performed at the individual beneficiary level. The optimizer's underserved zone detection returns 0 underserved beneficiaries (all NaN distances are treated as "within range"). New shop recommendations are therefore not generated from real data.

**Root cause:** The Open Data Telangana portal's card_status dataset is shop-level aggregated data, not individual beneficiary records. Individual GPS data would require integration with state civil registration or census data.

---

### 5.9 Beneficiary Data One Month Ahead of Transactions

**Observation:**
- `transactions.csv` → May 2025 (month 5)
- `beneficiaries.csv` → June 2025 (month 6)
- `fps_shops.csv` → June 2025 (month 6; last_updated July 2025)

The beneficiary card count (June) is used to assess shop load but transactions are from May. A beneficiary registered in June did not transact in May.

**Impact:** Per-shop `total_cards` figures reflect June registrations while transaction volumes reflect May activity. Utilization rate calculations (`cards_served / total_cards`) may be slightly off.

---

### 5.10 253 Shops Have No Transaction Record

**Observation:** 17,434 shops exist in the geo master but only 17,254 shops appear in the transaction data — a gap of 180 shops (after deduplication). After matching, 253 shops have no transaction data at all.

**Interpretation:** These may be newly activated shops (opened after May 2025 cutoff), shops that distributed nothing in May (rare but possible), or data quality gaps in the portal.

**Impact:** Demand forecasting for these shops returns `"no_data"` error. They appear in the geo map but have no forecast or fraud data.

---

## 6. Data Timing & Staleness Analysis

### 6.1 Download Timestamp vs Data Period

| File | Data period | Downloaded at |
|---|---|---|
| `transactions.csv` | May 2025 | 2026-02-19 01:24 UTC |
| `beneficiaries.csv` | June 2025 | 2026-02-19 01:24 UTC |
| `fps_shops.csv` | June 2025 (last_updated: July 2025) | 2026-02-19 01:24 UTC |

The data is **approximately 8–9 months stale** relative to the download date. This is expected: the Open Data Telangana portal publishes with a 2–3 month lag, and the fetcher was run on 19 February 2026 using the "latest available" files.

### 6.2 Effective Data Lag

```
Real-world events (May 2025)
    → State PDS database (same day)
        → Portal publication (August–September 2025, ~3 month lag)
            → Our download (February 2026)
                → Displayed in dashboard (~Feb 2026)

Total end-to-end lag: ~9 months
```

### 6.3 Implications for Fraud Detection

The rule-based fraud detector flags temporal anomalies (after-hours transactions) by parsing `transaction_date`. With all dates set to `2025-05-01` (first of month), individual transaction hour information is lost. The hour-based rules rely on `hour_of_day` derived from `transaction_date.dt.hour`, which will always return `0` for all transactions.

**Result:** After-hours rules fire on all transactions (hour 0 < 8 AM threshold), producing a large number of medium-severity temporal alerts (~17K). This is a **data artifact**, not real fraud.

### 6.4 Implications for Demand Forecasting

LSTM and Prophet are designed for monthly time-series data. With a single month, there is no series to fit. The heuristic fallback produces reasonable extrapolations, but confidence intervals are wide (±15%) and the model label clearly shows `"growth_adjusted_heuristic"` so users can interpret accordingly.

### 6.5 Recommended Refresh Schedule

| Action | Frequency | Command |
|---|---|---|
| Fetch latest data | Monthly (1st of each month) | `POST /api/v1/data/fetch/latest` |
| Full historical backfill | Once | `POST /api/v1/data/fetch/all` |
| ML model retrain | After ≥3 months of data available | Automatic (MAPE threshold) |

---

## 7. ML Model 1 — Rule-Based Fraud Detector

### File: `backend/ml_models/fraud_detection/rule_engine.py`

Rules are **deterministic and interpretable** — each corresponds to a known PDS fraud pattern documented in government audit reports. They run first in the pipeline and produce the majority of alerts.

### 7.1 Input Fields Required

| Field | Type | Source |
|---|---|---|
| `transaction_date` | datetime | Normalised transactions |
| `fps_shop_id` | str | Normalised transactions |
| `card_id` | str | Normalised transactions |
| `transaction_id` | str | Normalised transactions |
| `quantity_kg` | float | Normalised transactions |
| `biometric_verified` | bool | Normalised transactions (synthetic: True) |

### 7.2 Rule 1 — Duplicate Transactions

**What it detects:** Same ration card used at two or more different FPS shops in the same calendar month.

**Logic:**
```
group by (card_id, month)
→ count distinct fps_shop_id
→ flag if count > 1
```

**Why it's fraud:** A legitimate beneficiary is registered at exactly one FPS shop. Multi-shop usage indicates either a cloned/duplicated card or a dealer-level record manipulation where the card was logged at a different shop.

**Severity:** HIGH | **Score:** 0.85
**Action:** Block card; trigger field verification

### 7.3 Rule 2 — After-Hours Transactions

**What it detects:** FPS shops with ≥5 transactions logged outside the official operating window (08:00–20:00).

**Logic:**
```
df["hour"] = transaction_date.dt.hour
after_hours = df[(hour < 8) | (hour >= 20)]
for each fps_shop_id with ≥5 after-hours rows → alert
```

**Why it's fraud:** FPS shops are legally required to operate 08:00–20:00. Transactions logged at other times indicate backdated entries (fraud) or unauthorized ePoS device access.

**Severity:** MEDIUM | **Score:** 0.65
**Action:** Suspend dealer; audit ePoS device logs

> **⚠ Current data caveat:** All transactions in the current dataset have `transaction_date = 2025-05-01` (midnight UTC), so `hour = 0` for every row. This rule triggers on **all transactions**, producing ~17K medium alerts that are data artifacts, not real fraud.

### 7.4 Rule 3 — Month-End Bulk Transactions

**What it detects:** Shops where more than 40% of the month's transactions occur on the final calendar day of the month.

**Logic:**
```
for each (fps_shop_id, month):
    month_end = max(transaction_date.date)
    last_day_count = count(date == month_end)
    ratio = last_day_count / total_count
    if ratio > 0.40 → alert
```

**Why it's fraud:** Ghost beneficiary processing — entering a large batch of fictitious transactions at month-end to account for commodity that was stolen or sold on the black market. Normal distribution should be spread across the month.

**Severity:** HIGH | **Score:** 0.75
**Action:** Trigger field verification; cross-check biometric logs

### 7.5 Rule 4 — Low Biometric Compliance

**What it detects:** Dealers where biometric verification (thumbprint scan) is absent for >20% of transactions, among shops with ≥10 transactions.

**Logic:**
```
dealer_bio = group by fps_shop_id
           → bio_rate = sum(biometric_verified) / count(transaction_id)
flag if bio_rate < 0.80 AND total_transactions >= 10
```

**Why it's fraud:** India's ePoS mandate requires biometric authentication for every PDS distribution. Low biometric rates indicate the dealer is processing transactions without beneficiary presence — a strong signal of ration skimming.

**Severity:** CRITICAL if bio_rate < 0.50, HIGH otherwise
**Score:** `1.0 - bio_rate` (e.g., 60% bio rate → score 0.40 → HIGH)
**Action:** Suspend dealer license; initiate audit

### 7.6 Rule 5 — Card Used at Too Many Shops

**What it detects:** Ration cards used at more than 2 distinct FPS shops across the dataset.

**Logic:**
```
card_shops = group by card_id → count distinct fps_shop_id
flag if count > 2
score = min(1.0, (shop_count - 2) * 0.25 + 0.70)
```

**Why it's fraud:** Under NFSA rules, beneficiaries are assigned to a specific shop. While the ONORC (One Nation One Ration Card) portability scheme allows inter-state transactions, appearing at 3+ shops in the same state is anomalous and suggests a ghost beneficiary card being circulated by multiple dealers.

**Severity:** CRITICAL | **Action:** Block card; notify district officer

---

## 8. ML Model 2 — Isolation Forest Point Anomaly Detector

### File: `backend/ml_models/fraud_detection/isolation_forest.py`

### 8.1 What It Is

**Isolation Forest** is an unsupervised anomaly detection algorithm. It works by randomly partitioning the feature space with decision trees. Anomalous points are isolated with fewer splits (shorter average path length). The algorithm is particularly effective for high-dimensional tabular data with rare fraud events.

**Parameters:**
- `contamination = 0.05` (assumes 5% of data is anomalous)
- `n_estimators = 200` (200 isolation trees for stability)
- `random_state = 42` (reproducible results)
- `n_jobs = -1` (uses all CPU cores)

### 8.2 Input Fields Used

| Feature | Derived from | Purpose |
|---|---|---|
| `quantity_kg` | Raw transaction | Base quantity signal |
| `hour_of_day` | `transaction_date.dt.hour` | Time anomaly |
| `day_of_month` | `transaction_date.dt.day` | End-of-month patterns |
| `cards_transacted_same_day` | Group `fps_shop_id + date → count card_id` | Concurrent load |
| `shop_daily_volume` | Group `fps_shop_id + date → sum quantity_kg` | Daily throughput |
| `deviation_from_avg` | `|quantity - shop_monthly_mean|` | Per-shop outlier |
| `biometric_flag` | `NOT biometric_verified` (int 0/1) | Missing auth signal |

### 8.3 How It Works

```
Step 1: Build features for all transactions
Step 2: StandardScaler normalisation (mean=0, std=1 per feature)
Step 3: Fit 200 isolation trees on scaled features
Step 4: decision_function() → negative scores for anomalies
Step 5: Normalize to [0, 1]:
        score = 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
        (high score = more anomalous)
Step 6: Flag transactions where score >= threshold (default 0.60)
```

### 8.4 Severity Thresholds

| Score range | Severity |
|---|---|
| ≥ 0.85 | CRITICAL |
| ≥ 0.70 | HIGH |
| ≥ 0.50 | MEDIUM |
| < 0.50 | LOW |

### 8.5 When It Catches What Rules Miss

- Transactions with *just-within-hours* timing but unusually large quantity + low biometric
- Shops with gradual creeping fraud (individually not extreme but collectively anomalous)
- Combinations of features that are each slightly off but together highly suspicious

### 8.6 Explanation Generation

For each flagged transaction, human-readable explanations are generated:
```
"Isolation Forest flagged this transaction (score: 0.73) because:
 transaction outside operating hours; quantity deviates significantly from shop average."
```

---

## 9. ML Model 3 — DBSCAN Coordinated Fraud Cluster Detector

### File: `backend/ml_models/fraud_detection/isolation_forest.py`

### 9.1 What It Is

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) groups transactions that are densely packed together in time-quantity-shop space. Unlike Isolation Forest which finds individual anomalies, DBSCAN finds **coordinated groups** — evidence of organised fraud rings.

**Parameters:**
- `eps = 0.3` (neighborhood radius in normalized space)
- `min_samples = 5` (minimum cluster size to flag)

### 9.2 Input Fields Used

| Feature | Derived from | Purpose |
|---|---|---|
| `time_numeric` | `(transaction_date - min_date).total_seconds / 86400` | Days since start |
| `quantity_kg` | Raw | Volume signal |
| `shop_numeric` | `factorize(fps_shop_id)` (int) | Shop identity |

All features are StandardScaler-normalized before DBSCAN runs.

### 9.3 How It Works

```
Step 1: Normalize 3 features (time, quantity, shop_id)
Step 2: DBSCAN.fit_predict → labels (-1 = noise, 0+ = cluster id)
Step 3: For each cluster (label != -1):
        - count transactions in cluster
        - count distinct shops
        - sum total_quantity_kg
        - compute time span (max_date - min_date)
Step 4: Alert if cluster size >= min_samples (5)
```

### 9.4 Alert Output

```python
{
    "cluster_id": int,
    "transaction_count": int,
    "shops_involved": int,          # distinct shops in cluster
    "transaction_ids": [list],
    "total_quantity_kg": float,
    "time_span_hours": float,
    "anomaly_score": 0.75,          # fixed for cluster alerts
    "severity": "High" if shops > 1 else "Medium",
    "explanation": "DBSCAN detected a dense cluster of N transactions across M shops...",
    "model": "dbscan",
}
```

### 9.5 What It Catches

- Multiple shops processing an unusually high number of transactions in a narrow time window (flash mob fraud)
- Coordinated bulk entry across shops by a single operator
- Ghost beneficiary rings where the same phantom transaction pattern appears at multiple shops

---

## 10. ML Model 4 — LSTM Demand Forecaster

### File: `backend/ml_models/demand_forecast/lstm_model.py`

### 10.1 What It Is

A **Long Short-Term Memory (LSTM)** recurrent neural network for monthly time-series demand forecasting. LSTMs capture long-range temporal dependencies and non-linear patterns that simple regression cannot model.

**PyTorch Architecture:**
```
Input (batch, seq_len=6, features=8)
    → LSTM layer 1 (hidden=64, dropout=0.2)
    → LSTM layer 2 (hidden=64)
    → Linear(64 → 1)
    → ReLU
Output: predicted_quantity_kg (scalar)
```

### 10.2 Input Fields Required

| Feature | Computed from | Role |
|---|---|---|
| `quantity_lifted` | Monthly sum of `quantity_kg` per shop/commodity | Target variable and feature |
| `active_cards` | Count distinct `card_id` per month | Demand proxy |
| `month_sin` | `sin(2π × month / 12)` | Seasonal encoding — captures cyclical month pattern |
| `month_cos` | `cos(2π × month / 12)` | Orthogonal seasonal component |
| `lag_1` | `quantity_lifted` shifted by 1 month | Autoregressive signal |
| `lag_2` | `quantity_lifted` shifted by 2 months | Longer autoregressive signal |
| `rolling_avg_3` | 3-month rolling mean | Trend smoothing |
| `rolling_std_3` | 3-month rolling std dev | Volatility signal |

**Sequence length:** 6 months lookback. The model sees the last 6 months and predicts the next month.

### 10.3 Training Process

```
Step 1: Engineer 8 features from monthly aggregated data
Step 2: Drop rows with NaN (created by lags, rolling windows)
Step 3: Normalize all features: (x - mean) / (std + 1e-8)
Step 4: Create sliding windows: X[i] = data[i:i+6], y[i] = data[i+6, qty_col]
Step 5: Convert to PyTorch tensors
Step 6: Adam optimizer, MSELoss, N epochs (default 30–50)
Step 7: Save model weights + scaler params to disk
```

**Minimum data requirement:** `SEQUENCE_LENGTH + 2 = 8` months of monthly data.

### 10.4 Prediction Process

```
Step 1: Take last 6 months as initial sequence
Step 2: For each future month m:
    a. Feed sequence through LSTM
    b. Denormalize: pred = norm_pred * std[0] + mean[0]
    c. Clip negative values to 0
    d. Confidence interval: [pred × 0.90, pred × 1.10] (±10% heuristic)
    e. Append predicted step to sequence (autoregressive rollout)
Step 3: Return list of {month_offset, predicted_quantity_kg, CI_lower, CI_upper}
```

### 10.5 Fallback — Exponential Smoothing

When PyTorch is unavailable or training fails:
```python
alpha = 0.3
smoothed = last_value
for v in last_6_values:
    smoothed = alpha * v + (1 - alpha) * smoothed
# Project forward with constant smoothed value
# CI: ±15% of predicted
```

### 10.6 Auto-Retraining Logic

The `DemandForecastAgent` tracks per-shop MAPE (Mean Absolute Percentage Error). When actual data arrives:
```python
mape = mean(|actual - predicted| / actual)
if mape > MODEL_RETRAIN_THRESHOLD_MAPE:  # default 15%
    → set needs_retrain = True for next run
```

---

## 11. ML Model 5 — Prophet Demand Forecaster

### File: `backend/ml_models/demand_forecast/prophet_model.py`

### 11.1 What It Is

**Facebook Prophet** is a decomposable time-series model that fits additive or multiplicative components: trend + seasonality + holidays + residuals. It is robust to missing data and handles multiple seasonality periods gracefully — ideal for PDS distribution with annual harvest season cycles and major Indian holidays.

### 11.2 Configuration

```python
Prophet(
    seasonality_mode = "multiplicative",  # distribution scales with seasons
    changepoint_prior_scale = 0.05,       # conservative trend changes
    interval_width = 0.90,                # 90% confidence interval
    weekly_seasonality = False,           # monthly data: no weekly cycle
    daily_seasonality  = False,           # monthly data: no daily cycle
    yearly_seasonality = True,            # harvest/crop seasons
)
# Custom monthly seasonality:
model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
```

### 11.3 Indian Holidays Modelled

11 major holidays with a ±1-day effect window:

| Holiday | Date (2024) | Effect window |
|---|---|---|
| Sankranti / Pongal | 2024-01-15 | ±1 day |
| Republic Day | 2024-01-26 | ±1 day |
| Holi | 2024-03-25 | ±1 day |
| Ram Navami | 2024-04-17 | ±1 day |
| Eid ul-Fitr | 2024-04-10 | ±1 day |
| Eid ul-Adha | 2024-06-17 | ±1 day |
| Independence Day | 2024-08-15 | ±1 day |
| Ganesh Chaturthi | 2024-09-07 | ±1 day |
| Dussehra | 2024-10-12 | ±1 day |
| Diwali | 2024-11-01 | ±1 day |
| Christmas | 2024-12-25 | ±1 day |

### 11.4 Input Fields Required

Prophet expects two columns in training data:
- `ds`: datetime (Prophet convention) — set to `YYYY-MM-01`
- `y`: float — the monthly quantity to forecast

These are derived from the monthly aggregated DataFrame.

### 11.5 Prediction Output

```python
{
    "month_offset": m,                   # 1, 2, 3, ...
    "predicted_quantity_kg": float,      # yhat
    "confidence_lower": float,           # yhat_lower
    "confidence_upper": float,           # yhat_upper
    "model": "prophet",
}
```

### 11.6 Fallback — Linear Growth Heuristic

When Prophet is unavailable:
```python
base = 500.0 kg  # default base
predicted = base * (1 + 0.01 * month_offset)  # 1% monthly growth
# CI: ±15%
```

---

## 12. ML Model 6 — Ensemble Forecaster

### File: `backend/ml_models/demand_forecast/prophet_model.py`

### 12.1 What It Is

The **EnsembleForecaster** combines LSTM and Prophet predictions using a weighted average. Ensemble methods reduce variance: if one model is wrong, the other may compensate.

### 12.2 Default Weights

```python
lstm_weight   = 0.5
prophet_weight = 0.5  # equal weighting by default
```

### 12.3 Combination Logic

```python
blended_quantity = lstm_weight * lstm_pred + prophet_weight * prophet_pred
blended_lower    = min(lstm_lower, prophet_lower)   # wider CI → more conservative
blended_upper    = max(lstm_upper, prophet_upper)
```

### 12.4 Bayesian Weight Update

When MAPE data is available for both models, weights are updated dynamically:

```python
# Lower MAPE → higher weight (better model gets more say)
total_error = lstm_mape + prophet_mape
lstm_weight   = prophet_mape / total_error   # inverted
prophet_weight = lstm_mape / total_error
```

### 12.5 Risk Flag Detection

After blending, the ensemble checks for stock risk:

```python
if predicted_quantity < blended_lower * 0.7:
    risk_flag = "understock_risk"   # severely below lower bound
elif predicted_quantity > blended_upper * 1.3:
    risk_flag = "overstock_risk"    # severely above upper bound
```

---

## 13. ML Model 7 — Geospatial K-Means Optimizer

### File: `backend/ml_models/optimization/geospatial_optimizer.py`

### 13.1 Distance Computation

**Fast distance matrix (used for bulk operations):**
```python
KM_PER_DEGREE_LAT = 111.0   # ~constant worldwide
KM_PER_DEGREE_LON = 105.0   # at Telangana's ~17°N latitude

b_scaled = beneficiary_locs * [111.0, 105.0]   # convert to km
s_scaled = shop_locs        * [111.0, 105.0]
dist_matrix = cdist(b_scaled, s_scaled)          # euclidean in km
```

**Accurate Haversine (used for per-recommendation metrics):**
```python
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0  # Earth radius km
    φ1, φ2 = radians(lat1), radians(lat2)
    a = sin²(Δφ/2) + cos(φ1)·cos(φ2)·sin²(Δλ/2)
    return R · 2 · arcsin(√a)
```

### 13.2 Underserved Zone Detection

```python
max_acceptable_km = 5.0   # configurable

for each beneficiary:
    distance = nearest active shop (with valid coords)
    is_underserved = distance > 5.0 km

accessibility_score = 1.0 - (underserved_count / total_with_known_distance)
```

### 13.3 K-Means New Shop Recommendations

```python
# 1. Identify underserved beneficiaries
underserved = find_underserved_zones()

# 2. Cluster their locations
kmeans = KMeans(n_clusters=n_new_shops, random_state=42, n_init=10)
kmeans.fit(underserved[["latitude", "longitude"]])

# 3. For each centroid:
for rank, centroid in enumerate(kmeans.cluster_centers_):
    # - Count beneficiaries that would be served (cluster members)
    # - Compute current avg distance (before new shop)
    # - Compute projected avg distance (haversine to centroid)
    # - Priority score = coverage / max(projected_distance, 0.1)
    # - Rank by priority score descending
```

### 13.4 Voronoi Service Zones

Tessellates the map into service zones for each active shop:
```
scipy.spatial.Voronoi(shop_coordinates)
→ polygon per shop
→ area via shoelace formula (approximate km²)
→ shops with open regions (edge effects) → area = None
```

### 13.5 Underperforming Shop Detection

```python
utilization_rate = cards_served_this_month / total_cards_registered
flag if rate < 0.30 AND nearest_alternative_shop < 3.0 km
→ recommend consolidation
```

### 13.6 Input Fields Required

| Model | Required fields |
|---|---|
| Distance matrix | `latitude`, `longitude` (shops and beneficiaries) |
| K-Means recommendations | Beneficiary `latitude`, `longitude`, `members_count`, `district` |
| Voronoi | Shop `latitude`, `longitude`, `is_active` |
| Underperforming | Shop `total_cards`; Transaction `fps_shop_id`, `card_id` |
| Accessibility scores | Beneficiary `latitude`, `longitude`, `district` |

---

## 14. Multi-Agent Orchestration

### 14.1 OrchestratorAgent State

The orchestrator maintains shared state as a TypedDict:

```python
OrchestratorState = {
    "shops_df":         pd.DataFrame,      # FPS shop master
    "beneficiaries_df": pd.DataFrame,      # Card status snapshot
    "transactions_df":  pd.DataFrame,      # Monthly transactions (long format)
    "fraud_results":    Dict,              # FraudDetectionAgent output
    "forecast_results": Dict,              # DemandForecastAgent output
    "geo_results":      Dict,              # GeospatialAgent output
    "reporting_results":Dict,              # ReportingAgent output
    "nl_query":         Optional[str],     # Natural language query if any
    "trigger":          str,               # "monthly" | "realtime" | "query" | ...
    "errors":           List[str],         # Agent-level errors
    "started_at":       str,               # ISO timestamp
    "completed_at":     Optional[str],
}
```

### 14.2 Shared Memory (Persistent Across Runs)

```python
self._memory = {
    "last_fraud_check":       datetime,
    "last_forecast_run":      datetime,
    "last_geo_analysis":      datetime,
    "pending_alerts":         [Critical alerts awaiting field action],
    "historical_decisions":   [Last 100 orchestrator actions with timestamps],
}
```

### 14.3 Critical Alert Escalation

When fraud detection returns critical alerts, the orchestrator runs `_handle_critical_alerts()`:

1. For each critical alert, check `_memory["historical_decisions"]` for same `fps_shop_id`
2. If shop has been flagged before → tag as **"repeat offender"** → escalated severity
3. Log decision with timestamp, agent, action, outcome
4. In production: trigger SMS to field officer, auto-block card in ePoS system

### 14.4 Dashboard Shop Cap

To keep dashboard response time under 30 seconds, the orchestrator runs LSTM forecasting on a **stratified sample of 30 shops** (1 per district approximately), while fraud detection runs on all 40K transaction rows.

---

## 15. Improvements Based on Available Data

### 15.1 Immediate Improvements (Data Already Available)

**1. Historical data backfill (highest priority)**
The Open Data Telangana portal has transaction data from 2018 onward. Running `fetch_and_save(mode="all")` would provide 7+ years of monthly data — enough for robust LSTM and Prophet model training. Without this, forecasts are heuristic-only.

**2. Zero-coordinate shop filter**
Add a bounds check when loading shops: `lat > 5.0 AND lon > 60.0`. This removes 66 shops with `lat=0.0` / `lon=0.0` from geospatial analysis.

**3. Rice sub-scheme differentiation**
Currently AFSC, FSC, and AAP rice are merged into a single `"rice"` commodity row. Separating them would allow the system to forecast each scheme independently — useful for budget planning since AFSC/FSC have different subsidy rates.

**4. Hour-accurate transaction timestamps**
The current data only has month-level timestamps (`2025-05-01`). If the state ePoS system provides daily/hourly transaction data (which it collects internally), loading that data would make the after-hours fraud rule (`Rule 2`) and time-based Isolation Forest features meaningful.

**5. Wheat distribution district filter**
Pre-filter wheat forecasts to only run for the ~8 districts that actually distribute wheat. This removes meaningless zero-forecast noise for the other 25 districts.

---

### 15.2 Medium-Term Improvements (Require Additional Data Sources)

**6. Individual beneficiary GPS coordinates**
Geospatial underserved zone analysis is currently non-functional because beneficiaries have no coordinates. Linking with:
- Census village-level lat/lon data (public)
- State Socio-Economic Caste Census (SECC) — addresses per beneficiary
- Aadhaar-linked address geocoding

This would make K-Means new shop recommendations fully operational.

**7. Real biometric compliance rates**
The current data sets `biometric_verified = True` for all transactions (since the public data doesn't include per-transaction biometric status). Integrating with ePoS device logs would make Rule 4 (low biometric compliance) and the Isolation Forest `biometric_flag` feature meaningful.

**8. Portability transaction separation**
The `other_shop_transactions` column counts ONORC portable transactions but doesn't break them down by card. Adding a separate portability dataset would allow Rule 5 (multi-shop card usage) to correctly distinguish legitimate portability from ghost cards.

**9. Card activation/cancellation history**
The current card_status snapshot shows the state at one moment. A time-series of card activations/cancellations per shop would allow detection of sudden bulk card additions (a known fraud vector for inflating ration entitlements).

---

### 15.3 Advanced Improvements (Model-Level)

**10. MC Dropout for LSTM confidence intervals**
Current LSTM confidence intervals use a fixed ±10% heuristic. Monte Carlo Dropout (running inference N times with dropout enabled) would produce statistically meaningful uncertainty estimates.

**11. SHAP values for fraud explainability**
Add `shap.TreeExplainer` for Isolation Forest predictions. This would produce feature-level contributions for each flagged transaction (e.g., "flagged primarily because shop_daily_volume was 4.2σ above mean").

**12. Commodity allocation validation rule**
Combine beneficiary data (cards × card_type) with COMMODITY_ALLOCATIONS constants to compute the maximum legitimate allocation per shop per month:
```
max_rice_kg = sum(AAY_cards × 35 kg + PHH_cards × 5 kg × avg_family_size)
if actual_rice_lifted > max_rice_kg × 1.05 → fraud alert
```
This is a high-precision rule using data already available.

**13. Seasonal Prophet tuning per commodity**
Different commodities have different seasonality. Rice follows harvest cycles (July–August lift), wheat follows Rabi crop, kerosene follows winter. Training commodity-specific Prophet models (rather than one generic model) would improve forecast accuracy significantly.

**14. District-level ensemble weighting**
Urban districts (Hyderabad, Medchal) have stable month-to-month demand patterns → give LSTM higher weight. Rural districts have higher seasonal variability → give Prophet higher weight. This can be implemented as a district lookup table.

**15. Graph-based coordinated fraud detection**
Build a bipartite graph of (cards × shops) and apply community detection (Louvain algorithm) to find clusters of cards that all transact at the same subset of shops on the same days. This is more powerful than DBSCAN for identifying organised fraud rings.

---

*Document generated: February 2026 | Codebase: `/Users/samaranathreddymukka/Downloads/SalarySe/Scrape/`*
