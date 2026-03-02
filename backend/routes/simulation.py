"""
Simulation API Routes
POST /api/v1/simulation/run        — run a single-policy simulation
GET  /api/v1/simulation/presets    — return preset configurations
POST /api/v1/simulation/compare    — run all 4 policies and compare
GET  /api/v1/simulation/district-meta — static district registry
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

router = APIRouter(prefix="/api/v1/simulation", tags=["simulation"])
logger = logging.getLogger(__name__)


# ── Request / Response schemas ─────────────────────────────────────────────

class SimulationRequest(BaseModel):
    n_periods:            int   = Field(24,   ge=6,  le=60,   description="Simulation horizon (months)")
    discount_factor:      float = Field(0.95, ge=0.5, le=1.0, description="λ — future cost discount")
    alpha:                float = Field(0.25, ge=0.0, le=1.0, description="Transport cost weight")
    beta:                 float = Field(0.35, ge=0.0, le=1.0, description="Stockout cost weight")
    gamma:                float = Field(0.25, ge=0.0, le=1.0, description="Leakage cost weight")
    delta:                float = Field(0.15, ge=0.0, le=1.0, description="Equity cost weight")
    inspection_effectiveness: float = Field(0.40, ge=0.0, le=1.0, description="η — fraud reduction per inspection")
    fraud_shock_scale:    float = Field(0.03, ge=0.0, le=0.20, description="ε scale for exogenous fraud shocks")
    supply_fraction:      float = Field(0.91, ge=0.5, le=1.0,  description="Supply as fraction of total demand")
    inspection_cost:      float = Field(5_000.0, ge=0, description="κ — cost per inspection (₹)")
    budget_per_period:    float = Field(50_000_000.0, ge=0, description="B_t — total budget per period (₹)")
    stockout_penalty:     float = Field(50.0,  ge=0, description="π — penalty per kg of unmet demand (₹)")
    min_service_level:    float = Field(0.70,  ge=0.0, le=1.0)
    max_allocation_ratio: float = Field(2.0,   ge=1.0, le=5.0)
    inspection_threshold: float = Field(0.65,  ge=0.0, le=1.0)
    cvar_confidence:      float = Field(0.95,  ge=0.5, le=0.999)
    policy: str = Field("optimized", description="proportional | optimized | equity_first | risk_averse")
    seed:   int = Field(42, description="Random seed for reproducibility")


# ── Helper ─────────────────────────────────────────────────────────────────

def _result_to_dict(result) -> Dict[str, Any]:
    """Convert SimulationResult dataclass to JSON-serialisable dict."""
    from dataclasses import asdict
    d = asdict(result)
    # Convert PeriodResult list
    periods_out = []
    for pr in result.periods:
        from dataclasses import asdict as _asdict
        p = _asdict(pr)
        periods_out.append(p)
    d["periods"] = periods_out
    return d


# ── Routes ─────────────────────────────────────────────────────────────────

@router.post("/run")
async def run_simulation(req: SimulationRequest):
    """
    Execute a full T-period Agent-Based SCM simulation.

    Returns complete period-by-period results plus summary statistics,
    time-series data for charts, and per-district final state.
    """
    try:
        from simulation.scm_simulator import SCMSimulator, SimulationConfig
        cfg = SimulationConfig(
            n_periods=req.n_periods,
            discount_factor=req.discount_factor,
            alpha=req.alpha,
            beta=req.beta,
            gamma=req.gamma,
            delta=req.delta,
            inspection_effectiveness=req.inspection_effectiveness,
            fraud_shock_scale=req.fraud_shock_scale,
            supply_fraction=req.supply_fraction,
            inspection_cost=req.inspection_cost,
            budget_per_period=req.budget_per_period,
            stockout_penalty=req.stockout_penalty,
            min_service_level=req.min_service_level,
            max_allocation_ratio=req.max_allocation_ratio,
            inspection_threshold=req.inspection_threshold,
            cvar_confidence=req.cvar_confidence,
            policy=req.policy,
            seed=req.seed,
        )
        sim = SCMSimulator(cfg)
        result = sim.run()
        return _result_to_dict(result)
    except Exception as exc:
        logger.exception("Simulation run failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/presets")
async def get_presets():
    """Return 4 preset configurations for quick demo."""
    return {
        "presets": [
            {
                "name":        "Balanced (Default)",
                "description": "Equal weight across all cost objectives",
                "policy":      "optimized",
                "alpha": 0.25, "beta": 0.35, "gamma": 0.25, "delta": 0.15,
                "n_periods": 24, "discount_factor": 0.95,
                "supply_fraction": 0.91,
            },
            {
                "name":        "Equity-First",
                "description": "Maximise service equality across districts",
                "policy":      "equity_first",
                "alpha": 0.10, "beta": 0.30, "gamma": 0.15, "delta": 0.45,
                "n_periods": 24, "discount_factor": 0.95,
                "supply_fraction": 0.91,
            },
            {
                "name":        "Fraud Control",
                "description": "Heavy weight on leakage reduction and inspections",
                "policy":      "risk_averse",
                "alpha": 0.15, "beta": 0.25, "gamma": 0.50, "delta": 0.10,
                "n_periods": 24, "discount_factor": 0.95,
                "supply_fraction": 0.91,
                "inspection_threshold": 0.40,
                "inspection_effectiveness": 0.60,
            },
            {
                "name":        "Cost Optimal",
                "description": "Minimise total logistics & procurement cost",
                "policy":      "optimized",
                "alpha": 0.50, "beta": 0.30, "gamma": 0.15, "delta": 0.05,
                "n_periods": 24, "discount_factor": 0.97,
                "supply_fraction": 0.88,
            },
        ]
    }


@router.post("/compare")
async def compare_policies(req: SimulationRequest):
    """
    Run all 4 allocation policies with identical parameters and return
    a side-by-side comparison of summary statistics and time-series.
    """
    try:
        from simulation.scm_simulator import run_policy_comparison, SimulationConfig
        cfg = SimulationConfig(
            n_periods=req.n_periods,
            discount_factor=req.discount_factor,
            alpha=req.alpha,
            beta=req.beta,
            gamma=req.gamma,
            delta=req.delta,
            inspection_effectiveness=req.inspection_effectiveness,
            fraud_shock_scale=req.fraud_shock_scale,
            supply_fraction=req.supply_fraction,
            inspection_cost=req.inspection_cost,
            budget_per_period=req.budget_per_period,
            stockout_penalty=req.stockout_penalty,
            min_service_level=req.min_service_level,
            max_allocation_ratio=req.max_allocation_ratio,
            inspection_threshold=req.inspection_threshold,
            cvar_confidence=req.cvar_confidence,
            policy="optimized",  # overridden per policy
            seed=req.seed,
        )
        comparison = run_policy_comparison(cfg)
        return {"comparison": comparison, "n_periods": req.n_periods}
    except Exception as exc:
        logger.exception("Policy comparison failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/district-meta")
async def get_district_meta():
    """Return static district registry with geo coordinates."""
    from simulation.data_generator import TELANGANA_DISTRICTS
    return {"districts": TELANGANA_DISTRICTS, "count": len(TELANGANA_DISTRICTS)}
