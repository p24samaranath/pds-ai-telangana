"""
SCM Simulation Engine — Agent-Based Multi-Period Optimization

Implements the mathematical framework from the research paper:

  State:     S_t = (I_t, D̂_t, p̂_t, ĉ)
  Demand:    D_{i,t} ~ LogNormal(μ_{i,t}, σ_i)
  Leakage:   L_{i,t} = p̂_{i,t} · x_{i,t}
  Inventory: I_{i,t+1} = max(0, I_{i,t} + x_{i,t} − D_{i,t} − L_{i,t})
  Fraud upd: p̂_{i,t+1} = p̂_{i,t}·(1 − η·y_{i,t}) + ε_{i,t}
  Cost:      C_t = α·C^trans + β·C^stock + γ·C^leak + δ·C^ineq

CVaR of total discounted cost is computed at the end.
"""

import numpy as np
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

from .data_generator import DistrictDataGenerator
from .allocation_optimizer import AllocationOptimizer

logger = logging.getLogger(__name__)


# ── Config & Data Classes ──────────────────────────────────────────────────

@dataclass
class SimulationConfig:
    n_periods:            int   = 24
    discount_factor:      float = 0.95
    # Cost weights
    alpha:                float = 0.25   # transport
    beta:                 float = 0.35   # stockout
    gamma:                float = 0.25   # leakage
    delta:                float = 0.15   # equity
    # Fraud dynamics
    inspection_effectiveness: float = 0.40   # η
    fraud_shock_scale:    float = 0.03        # ε ~ Uniform(0, shock_scale)
    # Supply / budget
    supply_fraction:      float = 0.91        # fraction of total demand
    inspection_cost:      float = 5_000.0     # κ (₹)
    budget_per_period:    float = 50_000_000  # B_t (₹)
    # Cost parameters
    stockout_penalty:     float = 50.0        # π_i (₹/kg)
    transport_cost_base:  float = 2.0         # base cost multiplier
    # Policy
    min_service_level:    float = 0.70
    max_allocation_ratio: float = 2.0
    inspection_threshold: float = 0.65
    cvar_confidence:      float = 0.95
    # Policy name
    policy:               str   = "optimized"
    # Demand noise
    demand_cv:            float = 0.15
    seed:                 int   = 42


@dataclass
class PeriodResult:
    period:           int
    # Per-district arrays (stored as lists for JSON serialisation)
    inventory_start:  List[float] = field(default_factory=list)
    inventory_end:    List[float] = field(default_factory=list)
    demand_mean:      List[float] = field(default_factory=list)
    demand_realized:  List[float] = field(default_factory=list)
    fraud_prob_start: List[float] = field(default_factory=list)
    fraud_prob_end:   List[float] = field(default_factory=list)
    service_ratio:    List[float] = field(default_factory=list)
    allocation:       List[float] = field(default_factory=list)
    inspections:      List[int]   = field(default_factory=list)
    leakage:          List[float] = field(default_factory=list)
    # Scalar costs
    cost_transport:   float = 0.0
    cost_stockout:    float = 0.0
    cost_leakage:     float = 0.0
    cost_equity:      float = 0.0
    cost_total:       float = 0.0
    # Period summary
    n_stockouts:      int   = 0
    n_inspections:    int   = 0
    total_demand_kg:  float = 0.0
    total_supply_kg:  float = 0.0
    supply_util_pct:  float = 0.0
    avg_service_ratio: float = 0.0
    avg_fraud_prob:   float = 0.0
    # Agent actions
    agent_actions:    Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationResult:
    config:           dict
    policy:           str
    n_districts:      int
    district_names:   List[str]
    n_periods:        int
    periods:          List[PeriodResult]
    # Summary
    total_discounted_cost: float = 0.0
    avg_service_level:     float = 0.0
    avg_fraud_prob:        float = 0.0
    total_leakage_kg:      float = 0.0
    total_stockout_kg:     float = 0.0
    cvar_cost:             float = 0.0
    runtime_seconds:       float = 0.0
    # Time-series (for charts)
    cost_series:            List[float] = field(default_factory=list)
    cost_transport_series:  List[float] = field(default_factory=list)
    cost_stockout_series:   List[float] = field(default_factory=list)
    cost_leakage_series:    List[float] = field(default_factory=list)
    cost_equity_series:     List[float] = field(default_factory=list)
    service_level_series:   List[float] = field(default_factory=list)
    fraud_prob_series:      List[float] = field(default_factory=list)
    inventory_total_series: List[float] = field(default_factory=list)
    allocation_total_series: List[float] = field(default_factory=list)
    # Per-district final state
    district_final_state:   List[Dict] = field(default_factory=list)


# ── Simulator ─────────────────────────────────────────────────────────────

class SCMSimulator:
    """
    Multi-period agent-based SCM simulator.

    Usage
    -----
    sim = SCMSimulator(config)
    result = sim.run()
    """

    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.rng = np.random.default_rng(config.seed)

        # Build data
        gen = DistrictDataGenerator(
            n_periods=config.n_periods,
            seed=config.seed,
        )
        self._data = gen.generate()
        self.N = len(self._data["district_meta"])
        self.district_names = [d["name"] for d in self._data["district_meta"]]

        # Allocation optimizer
        self.optimizer = AllocationOptimizer(
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
            delta=config.delta,
            stockout_penalty=config.stockout_penalty,
            inspection_cost=config.inspection_cost,
            max_ratio=config.max_allocation_ratio,
            inspection_threshold=config.inspection_threshold,
            cvar_confidence=config.cvar_confidence,
        )

    def run(self) -> SimulationResult:
        """Execute the full T-period simulation and return structured results."""
        t0 = time.time()
        cfg = self.cfg
        N, T = self.N, cfg.n_periods

        # ── Initialise state ──────────────────────────────────────────────
        I = self._data["initial_inventory"].copy()       # [N]
        p = self._data["initial_fraud_prob"].copy()      # [N]
        c = self._data["transport_cost"].copy()          # [N]
        demand_mean_all = self._data["demand_mean"]      # [N, T]
        demand_std_all  = self._data["demand_std"]       # [N, T]
        supply_sched    = self._data["supply_schedule"]  # [T]

        periods: List[PeriodResult] = []
        period_costs: List[float] = []

        for t in range(T):
            D_hat = demand_mean_all[:, t]
            D_std = demand_std_all[:, t]
            S_t = supply_sched[t] * cfg.supply_fraction / cfg.supply_fraction
            # (supply_fraction already baked in data_generator; use as-is)
            S_t = supply_sched[t]
            B_t = cfg.budget_per_period

            # ── Agent Actions ─────────────────────────────────────────────
            agent_actions = {}

            # 1. Demand Agent: observe D̂_t
            agent_actions["demand_agent"] = {
                "action": "forecast_demand",
                "districts_with_understock_risk": int((D_hat > I * 1.5).sum()),
                "total_forecast_demand_kg": float(D_hat.sum()),
            }

            # 2. Fraud Agent: update p̂_t
            agent_actions["fraud_agent"] = {
                "action": "estimate_fraud_probability",
                "high_risk_districts": int((p > cfg.inspection_threshold).sum()),
                "avg_fraud_prob": float(p.mean()),
            }

            # 3. Geo Agent: transport cost (static, but logged)
            agent_actions["geo_agent"] = {
                "action": "compute_transport_costs",
                "avg_cost_per_kg": float(c.mean()),
                "max_cost_district": self.district_names[int(c.argmax())],
            }

            # 4. Allocation Agent: solve optimisation
            x, y = self.optimizer.allocate(
                inventory=I,
                demand_mean=D_hat,
                demand_std=D_std,
                fraud_prob=p,
                transport_cost=c,
                supply_total=S_t,
                budget=B_t,
                policy=cfg.policy,
            )
            agent_actions["allocation_agent"] = {
                "action": f"allocate_{cfg.policy}",
                "total_allocated_kg": float(x.sum()),
                "districts_receiving_allocation": int((x > 0).sum()),
                "n_inspections_recommended": int(y.sum()),
            }

            # 5. Governance Agent: validate constraints
            violations = []
            min_coverage = np.where(D_hat > 0, (I + x) / D_hat, 1.0)
            under_min = (min_coverage < cfg.min_service_level).sum()
            if under_min > 0:
                violations.append(f"{under_min} districts below min service level {cfg.min_service_level}")
            agent_actions["governance_agent"] = {
                "action": "validate_constraints",
                "policy_violations": violations,
                "supply_utilisation_pct": float(100 * x.sum() / max(S_t, 1)),
            }

            # ── Simulate Period ───────────────────────────────────────────
            # Realise demand from LogNormal
            sigma = np.log(1 + (D_std / np.maximum(D_hat, 1)) ** 2) ** 0.5
            mu_ln = np.log(np.maximum(D_hat, 1)) - 0.5 * sigma ** 2
            D_real = self.rng.lognormal(mu_ln, sigma)

            # Leakage
            L = p * x                                            # [N]

            # Effective supply after leakage
            effective_supply = np.maximum(I + x - L, 0.0)

            # Stockout: demand not met
            stockout = np.maximum(D_real - effective_supply, 0.0)

            # Service ratio (clamped to [0, 1])
            service_ratio = np.clip(
                np.where(D_real > 0, (effective_supply) / D_real, 1.0), 0.0, 1.0
            )

            # Inventory update
            I_new = np.maximum(effective_supply - D_real, 0.0)

            # Fraud probability update
            p_new = p * (1.0 - cfg.inspection_effectiveness * y) + \
                    self.rng.uniform(0, cfg.fraud_shock_scale, size=N)
            p_new = np.clip(p_new, 0.01, 0.99)

            # ── Cost Computation ──────────────────────────────────────────
            C_trans  = float((c * x).sum())
            C_stock  = float((cfg.stockout_penalty * stockout).sum())
            C_leak   = float((p * x).sum())
            # Equity: variance of service ratios (scaled for magnitude)
            C_ineq   = float(np.var(service_ratio) * 1e6)
            C_total  = (cfg.alpha * C_trans +
                        cfg.beta  * C_stock +
                        cfg.gamma * C_leak  +
                        cfg.delta * C_ineq)

            period_costs.append(C_total)

            # ── Record ────────────────────────────────────────────────────
            pr = PeriodResult(
                period=t,
                inventory_start=I.tolist(),
                inventory_end=I_new.tolist(),
                demand_mean=D_hat.tolist(),
                demand_realized=D_real.tolist(),
                fraud_prob_start=p.tolist(),
                fraud_prob_end=p_new.tolist(),
                service_ratio=service_ratio.tolist(),
                allocation=x.tolist(),
                inspections=y.tolist(),
                leakage=L.tolist(),
                cost_transport=C_trans,
                cost_stockout=C_stock,
                cost_leakage=C_leak,
                cost_equity=C_ineq,
                cost_total=C_total,
                n_stockouts=int((stockout > 0).sum()),
                n_inspections=int(y.sum()),
                total_demand_kg=float(D_real.sum()),
                total_supply_kg=float(x.sum()),
                supply_util_pct=float(100 * x.sum() / max(S_t, 1)),
                avg_service_ratio=float(service_ratio.mean()),
                avg_fraud_prob=float(p.mean()),
                agent_actions=agent_actions,
            )
            periods.append(pr)

            # Advance state
            I = I_new
            p = p_new

        # ── Summary ───────────────────────────────────────────────────────
        lam = cfg.discount_factor
        discounted_cost = sum(
            (lam ** t) * period_costs[t] for t in range(T)
        )
        avg_service = np.mean([pr.avg_service_ratio for pr in periods])
        avg_fraud   = np.mean([pr.avg_fraud_prob    for pr in periods])
        total_leak  = sum(sum(pr.leakage) for pr in periods)
        total_stock = sum(
            max(0, pr.total_demand_kg - pr.total_supply_kg)
            for pr in periods
        )

        # CVaR at given confidence level
        sorted_costs = np.sort(period_costs)
        cvar_idx = int(np.ceil(cfg.cvar_confidence * T))
        cvar = float(np.mean(sorted_costs[cvar_idx:]) if cvar_idx < T else sorted_costs[-1])

        # Final district state
        district_final = []
        final = periods[-1] if periods else None
        for i, d in enumerate(self._data["district_meta"]):
            district_final.append({
                "id": d["id"],
                "name": d["name"],
                "shops": d["shops"],
                "beneficiaries": d["beneficiaries"],
                "final_inventory_kg": round(final.inventory_end[i], 1) if final else 0,
                "final_fraud_prob": round(final.fraud_prob_end[i], 4) if final else 0,
                "avg_service_ratio": round(
                    np.mean([pr.service_ratio[i] for pr in periods]), 4
                ) if periods else 0,
                "total_leakage_kg": round(
                    sum(pr.leakage[i] for pr in periods), 1
                ) if periods else 0,
                "inspected_periods": int(
                    sum(pr.inspections[i] for pr in periods)
                ) if periods else 0,
                "lat": d["lat"],
                "lon": d["lon"],
                "dist_km": d["dist_km"],
            })

        result = SimulationResult(
            config=asdict(cfg),
            policy=cfg.policy,
            n_districts=N,
            district_names=self.district_names,
            n_periods=T,
            periods=periods,
            total_discounted_cost=round(discounted_cost, 2),
            avg_service_level=round(float(avg_service), 4),
            avg_fraud_prob=round(float(avg_fraud), 4),
            total_leakage_kg=round(float(total_leak), 1),
            total_stockout_kg=round(float(total_stock), 1),
            cvar_cost=round(cvar, 2),
            runtime_seconds=round(time.time() - t0, 3),
            cost_series=[round(pc, 2) for pc in period_costs],
            cost_transport_series=[round(pr.cost_transport, 2) for pr in periods],
            cost_stockout_series=[round(pr.cost_stockout,  2) for pr in periods],
            cost_leakage_series=[round(pr.cost_leakage,   2) for pr in periods],
            cost_equity_series=[round(pr.cost_equity,     2) for pr in periods],
            service_level_series=[round(pr.avg_service_ratio, 4) for pr in periods],
            fraud_prob_series=[round(pr.avg_fraud_prob, 4) for pr in periods],
            inventory_total_series=[
                round(sum(pr.inventory_end), 1) for pr in periods
            ],
            allocation_total_series=[
                round(pr.total_supply_kg, 1) for pr in periods
            ],
            district_final_state=district_final,
        )
        return result


def run_policy_comparison(
    base_config: Optional[SimulationConfig] = None,
) -> Dict[str, Any]:
    """
    Run all 4 policies with identical data seeds and return a comparison dict.
    """
    if base_config is None:
        base_config = SimulationConfig()

    policies = ["proportional", "optimized", "equity_first", "risk_averse"]
    comparison = {}

    for policy in policies:
        import copy
        cfg = copy.copy(base_config)
        cfg.policy = policy
        sim = SCMSimulator(cfg)
        res = sim.run()
        comparison[policy] = {
            "policy": policy,
            "total_discounted_cost": res.total_discounted_cost,
            "avg_service_level":     res.avg_service_level,
            "avg_fraud_prob":        res.avg_fraud_prob,
            "total_leakage_kg":      res.total_leakage_kg,
            "total_stockout_kg":     res.total_stockout_kg,
            "cvar_cost":             res.cvar_cost,
            "cost_series":           res.cost_series,
            "service_level_series":  res.service_level_series,
            "fraud_prob_series":     res.fraud_prob_series,
        }
    return comparison
