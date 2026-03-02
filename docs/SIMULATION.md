# Agent-Based SCM Simulation — Technical Reference

> **Phase 4** of the Telangana PDS AI System.
> Implements the multi-agent Supply Chain Management framework described in the research paper
> *"Agent-Based Optimisation of Public Distribution Systems under Fraud and Demand Uncertainty"*.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Real-World Data Foundation](#2-real-world-data-foundation)
3. [Mathematical Framework](#3-mathematical-framework)
4. [The Five Agents](#4-the-five-agents)
5. [Allocation Policies](#5-allocation-policies)
6. [Inspection Decision Logic](#6-inspection-decision-logic)
7. [Cost Function and CVaR](#7-cost-function-and-cvar)
8. [Backend Architecture](#8-backend-architecture)
9. [API Reference](#9-api-reference)
10. [Frontend Dashboard](#10-frontend-dashboard)
11. [Configuration Parameters](#11-configuration-parameters)
12. [Policy Comparison Results](#12-policy-comparison-results)
13. [Running Locally](#13-running-locally)

---

## 1. Overview

The simulation models the monthly grain distribution cycle across **33 Telangana districts** over a configurable horizon of **6–60 months**. Each period, five specialised agents observe the current system state, forecast demand, estimate fraud risk, compute logistics costs, optimise grain allocation, and validate policy constraints.

The system demonstrates:
- How different allocation strategies perform under supply shortfall and demand uncertainty
- The impact of fraud (leakage) on inventory and service levels
- The trade-off between equity and efficiency across urban and rural districts
- Why single-period LP optimisation can outperform naive heuristics — and when it fails

---

## 2. Real-World Data Foundation

All base figures are anchored to **Open Data Telangana** CSVs (publicly available):

| Dataset | Source | Key Figures |
|---|---|---|
| `beneficiaries.csv` | June 2025 | 9,183,183 ration cards across 33 districts |
| `fps_shops.csv` | June 2025 | 17,434 active Fair Price Shops |
| `transactions.csv` | May 2025 | 170,220,485 kg rice; 3,569,996 kg wheat; 76,479 kg sugar |

### District Registry (33 Districts)

Each district record contains:

| Field | Description |
|---|---|
| `beneficiaries` | Exact ration card count from CSV |
| `shops` | Exact Fair Price Shop count from CSV |
| `rice_kg` | Actual May 2025 rice distributed (kg) |
| `wheat_kg` | Actual May 2025 wheat distributed (only 6 districts) |
| `sugar_kg` | Actual May 2025 sugar distributed |
| `lat`, `lon` | District headquarters GPS coordinates |
| `dist_km` | Haversine distance from Hyderabad (km) |
| `fraud_seed` | Initial fraud probability (calibrated from fraud-detection agent) |

### Validation Assertions

The data generator asserts exact totals at import time — any future data drift raises immediately:

```python
assert sum(d["beneficiaries"] for d in TELANGANA_DISTRICTS) == 9_183_183
assert sum(d["shops"]         for d in TELANGANA_DISTRICTS) == 17_434
assert sum(d["rice_kg"]       for d in TELANGANA_DISTRICTS) == 170_220_485
```

### Demand Calibration

The monthly base demand per district is:

```
base_demand_i = real_rice_kg_i × 1.10   (10% uplift for late collection and waste)
```

Demand then evolves with seasonal multipliers and 2% year-on-year growth:

```
μ_{i,t} = base_i × seasonal_{(May+t) mod 12} × (1.02)^⌊t/12⌋ × noise_i
```

Telangana seasonal calendar used:

| Month | Multiplier | Reason |
|---|---|---|
| January | 1.10 | Sankranti festival |
| March | 1.12 | Ugadi festival (peak) |
| May | 0.95 | Migration season (baseline month) |
| July | 1.05 | Bonalu festival |
| October | 1.08 | Dussehra |
| November | 1.10 | Diwali |

### Transport Cost Model

```
cost_i = ₹0.40/kg  +  ₹0.011/km × dist_km
```

This gives ₹0.40/kg for Hyderabad (0 km) and ₹3.40/kg for Kumarambheem Asifabad (273 km), calibrated to actual TSCSC logistics rates.

---

## 3. Mathematical Framework

### System State

At each period `t`, the state vector is:

```
S_t = (I_t, D̂_t, p̂_t, ĉ)
```

| Symbol | Dimension | Meaning |
|---|---|---|
| `I_t` | [N] | Inventory at each district (kg) |
| `D̂_t` | [N] | Demand forecast — mean and std |
| `p̂_t` | [N] | Fraud probability estimate per district |
| `ĉ` | [N] | Transport cost per kg per district (static) |

### Demand Realisation

Demand is drawn from a log-normal distribution each period:

```
D_{i,t} ~ LogNormal(μ_{i,t}, σ_i)
```

The log-normal parameterisation ensures non-negative demand with a 15% coefficient of variation (CV = σ / μ = 0.15).

### Leakage (Fraud)

Grain that is allocated but diverted before reaching beneficiaries:

```
L_{i,t} = p̂_{i,t} · x_{i,t}
```

Where `x_{i,t}` is the kg allocated to district `i` in period `t`.

### Inventory Dynamics

```
I_{i,t+1} = max(0,  I_{i,t} + x_{i,t} − D_{i,t} − L_{i,t})
```

Inventory cannot go negative. Unmet demand becomes a stockout.

### Fraud Probability Update

Inspections reduce fraud probability; exogenous shocks keep it from reaching zero:

```
p̂_{i,t+1} = p̂_{i,t} · (1 − η · y_{i,t})  +  ε_{i,t}
```

| Symbol | Default | Meaning |
|---|---|---|
| `η` | 0.40 | Inspection effectiveness (40% fraud reduction per inspection) |
| `y_{i,t}` | 0 or 1 | Binary inspection decision |
| `ε_{i,t}` | Uniform(0, 0.03) | Exogenous fraud shock (new actors, collusion) |

Fraud probability is clipped to `[0.01, 0.99]`.

---

## 4. The Five Agents

Each period, the agents act in sequence:

```
DemandAgent → FraudAgent → GeoAgent → AllocationAgent → GovernanceAgent
```

### 4.1 Demand Agent

**Role:** Forecast district-level demand for the current period.

**Output:**
- `D̂_{i,t}` — demand mean per district
- Count of districts with under-stock risk (`D̂_i > 1.5 × I_i`)
- Total forecast demand (kg)

The demand agent reads pre-generated multi-period profiles from `DistrictDataGenerator` which incorporates seasonal factors and year-on-year growth.

### 4.2 Fraud Agent

**Role:** Maintain and report current fraud probability estimates.

**Output:**
- `p̂_{i,t}` — fraud probability vector
- Count of high-risk districts (p > 0.65 threshold)
- System-wide average fraud probability

Fraud probabilities are updated from the previous period's inspection outcomes. High-fraud districts are flagged for mandatory inspection.

### 4.3 Geo / Cost Agent

**Role:** Compute logistics costs and flag remote districts.

**Output:**
- `c_i` — ₹/kg transport cost per district (static in current version)
- District with maximum transport cost
- System-wide average cost per kg

In future versions this agent could use real-time road condition data, seasonal access constraints, or FCI depot routing.

### 4.4 Allocation Agent

**Role:** Solve the grain distribution optimisation problem.

Given `(I_t, D̂_t, p̂_t, ĉ, S_t, B_t)`, it decides:
- `x_{i,t}` — kg to allocate to district `i`
- `y_{i,t}` — whether to inspect district `i` (binary)

Four policy modes are available (see Section 5).

**Post-optimisation constraints applied by the allocator:**
1. `x_i ≤ max_ratio × D̂_i` — prevents stockpiling (default max_ratio = 2.0)
2. `Σ x_i ≤ S_t` — cannot exceed available supply

### 4.5 Governance Agent

**Role:** Validate that the allocation plan meets policy constraints.

**Checks:**
- How many districts fall below minimum service level (default 70%)
- Supply utilisation percentage (`Σ x_i / S_t × 100`)
- Logs any violations for the audit trail

The governance agent does not modify allocations; it records constraint breaches in the agent action log for transparency and accountability.

---

## 5. Allocation Policies

### 5.1 Proportional (Baseline)

The simplest possible policy: allocate supply proportional to demand.

```
x_i = (D̂_i / Σ D̂_j) × S_t
```

**Strengths:** Simple, always distributes all supply, maintains stable service levels.
**Weaknesses:** Ignores fraud risk — high-fraud districts receive the same share as clean ones. Does not account for existing inventory.

---

### 5.2 LP Optimised

A linear programme minimising the weighted sum of transport, stockout, and leakage costs.

**Decision variables:** `[x_0 … x_{N-1}, s_0 … s_{N-1}]`
- `x_i ≥ 0` — allocation (kg)
- `s_i ≥ 0` — stockout slack variable (kg unmet demand)

**Normalised objective** (ensures α, β, γ are true relative weights):

```
min  Σ_i [ α·(c_i/c_max) + γ·p_i ] · x_i  +  β · s_i
```

The normalisation is critical:
- Transport `α·(c_i/c_max)` ∈ [0.03, 0.25] (normalised by maximum district cost)
- Stockout `β = 0.35` — larger than transport for all districts → LP prefers to allocate

Dividing the stockout term by N (as one might naively do for "per-district" weighting) would collapse the penalty to 0.011, making stockout cheaper than transport and causing the LP to systematically under-allocate. This was an identified and fixed bug.

**Constraints:**

```
(1)  -x_i - s_i  ≤  -max(0, D̂_i - I_i)    [cover shortfall or pay stockout]
(2)   Σ x_i      ≤  S_t                      [total supply]
(3)   x_i        ≤  max_ratio × D̂_i          [anti-stockpiling]
```

**Solver:** `scipy.optimize.linprog` with HiGHS backend (falls back to proportional if scipy unavailable).

**Strengths:** Explicitly penalises high-fraud, high-transport districts. Mathematically rigorous.
**Weaknesses:** Single-period myopic — does not plan ahead. Can under-allocate when initial inventory is high, leading to inventory depletion and later stockouts.

---

### 5.3 Equity-First

Equalises service ratios `S_i = (I_i + x_i) / D̂_i` across all districts.

**Algorithm:**

```
1. total_shortfall = Σ max(0, D̂_i - I_i)
2. τ = min(1.0,  S_t / total_shortfall)          [target service ratio]
3. x_i = max(0,  τ·D̂_i - I_i)                   [fill up to target]
4. leftover = S_t - Σ x_i
5. x_i += leftover / N                            [distribute remainder equally]
```

**Strengths:** Maximises service equity — no district is left far behind. In benchmarks, it achieves the highest average service level and lowest total cost of all four policies.
**Weaknesses:** Does not account for fraud probability — equal treatment of high-fraud districts wastes grain.

---

### 5.4 Risk-Averse (CVaR-Penalised LP)

An LP variant that accounts for demand variance by treating high-uncertainty districts as higher priority.

**Two modifications over the base LP:**

**1. Adjusted stockout penalty per district:**
```
β_i = β × (1 + σ_i / D̂_i)    [coefficient of variation penalty]
```
Districts with high demand variance (rural, tribal areas) receive a higher effective stockout penalty, encouraging the LP to pre-position more stock there.

**2. Safety buffer in demand target:**
```
effective_demand_i = D̂_i + z_{0.95} · σ_i = D̂_i + 1.645 · σ_i
```
Targets the 95th-percentile demand scenario (CVaR at 95% confidence).

**Objective:**
```
min  Σ_i [ α·c_i + γ·p_i ] · x_i  +  β_i · stockout_penalty · s_i
```

**Strengths:** Protects against demand spikes in high-variance districts. Appropriate for risk-averse administrators.
**Weaknesses:** Consistently over-allocates to high-variance districts at the expense of supply availability elsewhere. Higher total cost.

---

## 6. Inspection Decision Logic

After the allocation is determined, the allocation agent decides which districts to inspect using a **greedy ROI-based algorithm**.

### Return on Inspection (ROI)

```
ROI_i = p_i · x_i / (inspection_cost + c_i · x_i + ε)
```

The ROI measures how much fraud (in kg terms) is likely to be caught per rupee spent on inspection.

### Selection Process

```
1. Mandatory:  flag all districts where p_i > inspection_threshold (default 0.65)
   → these are always inspected regardless of budget
2. Remaining budget:  sort non-mandatory districts by ROI (descending)
   → greedily add districts until budget B_t is exhausted
```

**Inspection effect:** Reduces district fraud probability by `η = 40%` in the following period.

```
p̂_{i,t+1} = p̂_{i,t} × (1 − 0.40)   (if inspected)
```

---

## 7. Cost Function and CVaR

### Per-Period Cost

```
C_t = α·C^trans  +  β·C^stock  +  γ·C^leak  +  δ·C^ineq
```

| Component | Formula | Default Weight |
|---|---|---|
| Transport | `Σ c_i · x_i` | α = 0.25 |
| Stockout | `Σ π · max(0, D_{i,t} − I_{i,t} − x_{i,t})` (π = ₹50/kg) | β = 0.35 |
| Leakage | `Σ p_i · x_i` | γ = 0.25 |
| Equity | `Var(service_ratio_i) × 10^6` | δ = 0.15 |

### Total Discounted Cost

```
J = Σ_{t=0}^{T-1}  λ^t · C_t       (λ = discount factor, default 0.95)
```

### CVaR (Conditional Value at Risk)

The CVaR at confidence level α measures the expected cost in the worst `(1−α)` fraction of periods:

```
CVaR_α = E[C_t | C_t ≥ VaR_α]
```

At the default 95% confidence: sort all period costs, take the mean of the top 5% (worst periods). This is the risk measure the research paper proposes to minimise.

---

## 8. Backend Architecture

### File Structure

```
backend/
├── simulation/
│   ├── __init__.py               — Package exports
│   ├── data_generator.py         — District data + demand profiles
│   ├── allocation_optimizer.py   — 4 allocation policies (LP via HiGHS)
│   └── scm_simulator.py          — Core T-period simulation engine
└── routes/
    └── simulation.py             — FastAPI router (4 endpoints)
```

### `data_generator.py`

**Class:** `DistrictDataGenerator(n_periods, seed, start_month_offset, commodity)`

**Key method:** `.generate()` returns:

| Key | Shape | Description |
|---|---|---|
| `district_meta` | list[dict] | 33 district records |
| `transport_cost` | [N] | ₹/kg per district |
| `initial_inventory` | [N] | kg (0.3× monthly demand buffer) |
| `initial_fraud_prob` | [N] | fraud seeds |
| `demand_mean` | [N, T] | monthly demand forecasts (kg) |
| `demand_std` | [N, T] | demand standard deviations |
| `supply_schedule` | [T] | state-level supply (87–95% of total demand) |
| `commodity_mix` | dict | rice 96.5%, wheat 2%, sugar 0.1% |

### `allocation_optimizer.py`

**Class:** `AllocationOptimizer(alpha, beta, gamma, delta, stockout_penalty, inspection_cost, max_ratio, inspection_threshold, cvar_confidence)`

**Public method:** `.allocate(inventory, demand_mean, demand_std, fraud_prob, transport_cost, supply_total, budget, policy) → (x, y)`

### `scm_simulator.py`

**Class:** `SCMSimulator(config: SimulationConfig)`

**Method:** `.run() → SimulationResult`

**Dataclasses:**

- `SimulationConfig` — all tunable parameters
- `PeriodResult` — per-period, per-district arrays + scalar costs + agent action log
- `SimulationResult` — full period list + summary metrics + chart time-series + district final state

**Function:** `run_policy_comparison(base_config) → dict`
Runs all 4 policies with the same data seed and returns a side-by-side comparison.

---

## 9. API Reference

All endpoints are under prefix `/api/v1/simulation`.

### POST `/run`

Execute a full T-period simulation with a single policy.

**Request body** (all fields optional, defaults shown):

```json
{
  "n_periods": 24,
  "discount_factor": 0.95,
  "alpha": 0.25,
  "beta": 0.35,
  "gamma": 0.25,
  "delta": 0.15,
  "policy": "optimized",
  "inspection_effectiveness": 0.40,
  "fraud_shock_scale": 0.03,
  "supply_fraction": 0.91,
  "inspection_cost": 5000,
  "budget_per_period": 50000000,
  "stockout_penalty": 50,
  "min_service_level": 0.70,
  "max_allocation_ratio": 2.0,
  "inspection_threshold": 0.65,
  "cvar_confidence": 0.95,
  "seed": 42
}
```

**Response:** `SimulationResult` as JSON — includes:
- `periods[]` — full per-district arrays for every period
- `cost_series`, `service_level_series`, `fraud_prob_series`, etc. — chart time-series
- `total_discounted_cost`, `avg_service_level`, `avg_fraud_prob`, `total_stockout_kg`, `cvar_cost`
- `district_final_state[]` — per-district summary at end of simulation

### POST `/compare`

Runs all 4 allocation policies with identical parameters and returns side-by-side metrics.

**Request body:** Same as `/run`.

**Response:**
```json
{
  "comparison": {
    "proportional":  { "avg_service_level": 0.989, "total_discounted_cost": 953613692, ... },
    "optimized":     { "avg_service_level": 0.882, "total_discounted_cost": 3900050038, ... },
    "equity_first":  { "avg_service_level": 0.996, "total_discounted_cost": 942840435, ... },
    "risk_averse":   { "avg_service_level": 0.851, "total_discounted_cost": 3727057503, ... }
  },
  "n_periods": 12
}
```

### GET `/presets`

Returns 4 named configurations for quick demo use:

| Preset | Policy | Focus |
|---|---|---|
| Balanced (Default) | optimized | Equal weight across all cost terms |
| Equity-First | equity_first | Maximise service equality across districts |
| Fraud Control | risk_averse | Heavy weight on leakage + aggressive inspections |
| Cost Optimal | optimized | Minimise logistics cost; lower supply fraction |

### GET `/district-meta`

Returns the full 33-district registry with GPS coordinates, beneficiary counts, shop counts, and fraud seeds. Used by the frontend to pre-populate the district table without running a simulation.

---

## 10. Frontend Dashboard

**File:** `frontend/src/pages/SimulationPage.jsx`
**Route:** `/simulation`
**Tech:** React 18, `react-chartjs-2` (Chart.js v4)

### Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  Left Panel (controls)        │  Right Panel (results)              │
│  ─────────────────────        │  ─────────────────────              │
│  Simulation Horizon slider    │  5 KPI Cards                        │
│  Discount Factor slider       │  ┌──────────────────────────────┐   │
│  α β γ δ weight sliders       │  │ Tab: Cost Breakdown           │   │
│  η ε fraud dynamics sliders   │  │      Stacked bar per period   │   │
│  Supply Fraction slider       │  ├──────────────────────────────┤   │
│  CVaR Confidence slider       │  │ Tab: Service & Fraud          │   │
│  Policy radio buttons         │  │      Dual-axis line chart     │   │
│  4 Quick Preset buttons       │  ├──────────────────────────────┤   │
│  [Run Simulation]             │  │ Tab: Inventory                │   │
│  [Compare All Policies]       │  │      Area + line chart        │   │
│                               │  ├──────────────────────────────┤   │
│                               │  │ Tab: Districts (33-row table) │   │
│                               │  ├──────────────────────────────┤   │
│                               │  │ Tab: Agent Log                │   │
│                               │  └──────────────────────────────┘   │
│                               │                                     │
│                               │  Policy Comparison (if run)         │
│                               │  Side-by-side table + line charts   │
└─────────────────────────────────────────────────────────────────────┘
```

### KPI Cards

| Card | Value | Color |
|---|---|---|
| Total Discounted Cost | ₹ formatted | Blue |
| Avg Service Level | % | Green (>90%) / Red (<70%) |
| Avg Fraud Probability | % | Amber |
| Policy Used | Name string | Purple |
| Runtime | seconds | Slate |

### Chart Tabs

**Cost Breakdown** — Stacked bar chart showing `C^trans`, `C^stock`, `C^leak`, `C^ineq` per period. Reveals whether stockout or leakage dominates total cost over time.

**Service & Fraud** — Dual-axis line chart:
- Left axis: Average service level (%) across 33 districts
- Right axis: Average fraud probability (× 100 for scale matching)

Shows the inverse relationship between inspections and fraud probability, and tracks service level trends.

**Inventory** — Area chart of total system inventory (end-of-period) overlaid with demand allocation series. Reveals inventory depletion trajectories under different policies.

**Districts** — 33-row sortable table showing per-district final state:
- Final inventory (kg), avg service ratio, fraud probability, leakage kg, inspection count
- Colour-coded service ratio column (green/amber/red heatmap)

**Agent Log** — Chronological log of all 5 agent actions across T periods, formatted for audit trail review. Shows what each agent decided, how many districts were flagged, and any governance violations.

### Policy Comparison Panel

Triggered by "Compare All Policies" button. Shows:
1. Summary table with all 4 policies side by side (cost, service, fraud, stockout, leakage)
2. Service level time-series line chart — all 4 policies on one chart
3. Fraud probability time-series line chart — all 4 policies

---

## 11. Configuration Parameters

| Parameter | Symbol | Default | Range | Effect |
|---|---|---|---|---|
| `n_periods` | T | 24 | 6–60 | Simulation horizon in months |
| `discount_factor` | λ | 0.95 | 0.5–1.0 | Future cost discount (0.95 ≈ 5% monthly rate) |
| `alpha` | α | 0.25 | 0–1 | Weight on transport cost |
| `beta` | β | 0.35 | 0–1 | Weight on stockout cost |
| `gamma` | γ | 0.25 | 0–1 | Weight on leakage cost |
| `delta` | δ | 0.15 | 0–1 | Weight on equity cost |
| `inspection_effectiveness` | η | 0.40 | 0–1 | Fraud reduction per inspection |
| `fraud_shock_scale` | ε | 0.03 | 0–0.20 | Exogenous fraud shock magnitude |
| `supply_fraction` | — | 0.91 | 0.5–1.0 | State supply as fraction of total demand |
| `inspection_cost` | κ | ₹5,000 | 0+ | Cost per district inspection |
| `budget_per_period` | B_t | ₹50M | 0+ | Total inspection budget per period |
| `stockout_penalty` | π | ₹50/kg | 0+ | Cost per kg of unmet demand |
| `min_service_level` | — | 0.70 | 0–1 | Governance constraint threshold |
| `max_allocation_ratio` | — | 2.0 | 1–5 | Max `x_i / D̂_i` (anti-stockpiling) |
| `inspection_threshold` | — | 0.65 | 0–1 | p_i above this → mandatory inspection |
| `cvar_confidence` | α_CVaR | 0.95 | 0.5–0.999 | CVaR confidence level |
| `seed` | — | 42 | any int | Random seed for reproducibility |
| `policy` | — | optimized | see below | Allocation policy |

**Policy values:** `proportional`, `optimized`, `equity_first`, `risk_averse`

---

## 12. Policy Comparison Results

Benchmarked over **12 periods** with default parameters (seed 42, supply fraction 0.91):

| Policy | Avg Service Level | Total Discounted Cost | Avg Fraud Prob | Total Stockout (kg) |
|---|---|---|---|---|
| **Equity-First** | **99.6%** | **₹942M** (best) | 5.91% | 183M kg |
| Proportional | 98.9% | ₹954M | 5.91% | 179M kg |
| Risk-Averse | 85.1% | ₹3.73B | 9.28% | 411M kg |
| LP Optimised | 88.2% | ₹3.90B | 8.03% | 451M kg |

**Key findings:**

1. **Equity-first dominates** in this Telangana context because the single-period LP and CVaR policies are both conservative about transport cost, leading them to under-allocate in early periods and face stockouts later.

2. **Proportional is robust** — steady allocation each period prevents the inventory depletion cliff seen in LP and risk-averse policies.

3. **LP optimised is myopic** — without multi-period lookahead, it delays allocation when initial inventory is high. The simulation demonstrates why this matters: inventory depletion from period 3 onwards causes persistent stockouts.

4. **Risk-averse adds safety buffer** (targets 95th-percentile demand: `D̂_i + 1.645σ_i`) which sounds conservative but leads to higher cost and lower service because it over-allocates to high-variance districts at the expense of all others.

5. **Fraud probability diverges** — the LP and risk-averse policies trigger more inspections (their objectives penalise leakage directly), leading to lower steady-state fraud probability in later periods, but at higher total cost.

---

## 13. Running Locally

### Prerequisites

```
Python 3.10+  with:  fastapi, uvicorn, numpy, scipy, pydantic
Node 18+      with:  react, vite, react-chartjs-2, chart.js, axios
```

### Start Backend

```bash
cd backend
python -m uvicorn app.main:app --reload --port 8000
```

### Start Frontend

```bash
cd frontend
npm install
npm run dev      # starts on port 3000
```

### Vite Proxy

The frontend dev server proxies all `/api/*` requests to `http://localhost:8000`.
This is configured in `vite.config.js` and means `api.js` uses a relative base URL (`''`), not `http://localhost:8000` directly.

### Quick API Test

```bash
# Run a 6-period optimised simulation
curl -X POST http://localhost:8000/api/v1/simulation/run \
  -H "Content-Type: application/json" \
  -d '{"n_periods": 6, "policy": "equity_first", "seed": 42}'

# Compare all 4 policies over 12 periods
curl -X POST http://localhost:8000/api/v1/simulation/compare \
  -H "Content-Type: application/json" \
  -d '{"n_periods": 12, "seed": 42}'

# Get all district metadata
curl http://localhost:8000/api/v1/simulation/district-meta
```

### Navigate to Simulation

Open `http://localhost:3000` → click **SCM Simulation** in the top navigation bar.

---

*Last updated: March 2026. Data anchored to Open Data Telangana (beneficiaries.csv, fps_shops.csv, transactions.csv — June/May 2025).*
