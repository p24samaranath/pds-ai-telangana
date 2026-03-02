"""
Allocation Optimizer — implements 4 policies from the research paper.

Given system state (I, D_hat, p, c, S_t, B_t) → decides:
  x_{i,t} : kg allocated to district i in period t
  y_{i,t} : binary inspection indicator for district i

Policies
--------
1. proportional   — x_i ∝ D̂_i  (naïve baseline)
2. optimized      — LP minimising α·C^trans + β·C^stock + γ·C^leak
3. equity_first   — equalize service ratios S_i = (I_i + x_i) / D̂_i
4. risk_averse    — CVaR-penalized LP (higher β for high-variance districts)
"""

import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class AllocationOptimizer:
    """
    Parameters
    ----------
    alpha : float   transport cost weight
    beta  : float   stockout cost weight
    gamma : float   leakage cost weight
    delta : float   equity weight
    stockout_penalty : float  ₹/kg penalty for unmet demand
    inspection_cost  : float  ₹ per inspection
    max_ratio        : float  max x_i / D̂_i (prevent stockpiling)
    inspection_threshold : float  inspect if p_i > threshold
    """

    def __init__(
        self,
        alpha: float = 0.25,
        beta: float  = 0.35,
        gamma: float = 0.25,
        delta: float = 0.15,
        stockout_penalty: float = 50.0,
        inspection_cost: float  = 5_000.0,
        max_ratio: float        = 2.0,
        inspection_threshold: float = 0.65,
        cvar_confidence: float  = 0.95,
    ):
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta
        self.stockout_penalty = stockout_penalty
        self.inspection_cost  = inspection_cost
        self.max_ratio        = max_ratio
        self.inspection_threshold = inspection_threshold
        self.cvar_confidence  = cvar_confidence

    # ── Public API ─────────────────────────────────────────────────────────

    def allocate(
        self,
        inventory:    np.ndarray,   # I_{i,t}  [N]
        demand_mean:  np.ndarray,   # D̂_mean   [N]
        demand_std:   np.ndarray,   # D̂_std    [N]
        fraud_prob:   np.ndarray,   # p_{i,t}  [N]
        transport_cost: np.ndarray, # c_i      [N]
        supply_total: float,        # S_t
        budget:       float,        # B_t
        policy:       str = "optimized",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        x : np.ndarray [N]  — kg allocated per district
        y : np.ndarray [N]  — 1=inspect, 0=skip (int)
        """
        N = len(inventory)
        # Shortfall = how much we need to bring inventory to demand level
        shortfall = np.maximum(demand_mean - inventory, 0.0)

        if policy == "proportional":
            x = self._policy_proportional(demand_mean, supply_total)
        elif policy == "optimized":
            x = self._policy_lp(
                inventory, demand_mean, fraud_prob,
                transport_cost, supply_total,
            )
        elif policy == "equity_first":
            x = self._policy_equity(inventory, demand_mean, supply_total)
        elif policy == "risk_averse":
            x = self._policy_risk_averse(
                inventory, demand_mean, demand_std,
                fraud_prob, transport_cost, supply_total,
            )
        else:
            x = self._policy_proportional(demand_mean, supply_total)

        # Clip to max ratio and supply
        max_alloc = self.max_ratio * np.maximum(demand_mean, 1.0)
        x = np.clip(x, 0.0, max_alloc)
        # Renormalise to supply
        total = x.sum()
        if total > supply_total and total > 0:
            x = x * supply_total / total

        # Inspection decisions
        y = self._decide_inspections(fraud_prob, x, transport_cost, budget)
        return x, y

    # ── Policies ───────────────────────────────────────────────────────────

    def _policy_proportional(
        self,
        demand_mean: np.ndarray,
        supply_total: float,
    ) -> np.ndarray:
        """Allocate proportional to demand forecast."""
        total_demand = demand_mean.sum()
        if total_demand <= 0:
            return np.ones(len(demand_mean)) * supply_total / len(demand_mean)
        return (demand_mean / total_demand) * supply_total

    def _policy_lp(
        self,
        inventory:      np.ndarray,
        demand_mean:    np.ndarray,
        fraud_prob:     np.ndarray,
        transport_cost: np.ndarray,
        supply_total:   float,
    ) -> np.ndarray:
        """
        LP optimisation — minimise α·C^trans + β·C^stock + γ·C^leak

        Variables: [x_0…x_{N-1}, s_0…s_{N-1}]
          x_i  ≥ 0   : allocation
          s_i  ≥ 0   : stockout slack (s_i ≥ D̂_i − I_i − x_i)

        Objective (NORMALISED): (α·c̃_i + γ·p_i)·x_i + β·s̃_i
          where c̃_i = c_i / c_max  ∈ [0,1]
                s̃_i = s_i / D_total  (fraction unmet)
          Normalisation ensures α,β,γ are true relative weights, not
          dominated by raw ₹ magnitudes.

        Subject to:
          (1) −x_i − s_i ≤ −max(0, D̂_i − I_i)   [stockout cover]
          (2) Σ x_i ≤ S_t                          [supply]
        """
        try:
            from scipy.optimize import linprog
        except ImportError:
            logger.warning("scipy not available — falling back to proportional policy")
            return self._policy_proportional(demand_mean, supply_total)

        N = len(demand_mean)
        c_max   = max(transport_cost.max(), 1e-9)
        D_total = max(demand_mean.sum(), 1e-9)

        # Normalised objective vector (length 2N)
        # Transport: alpha * (c_i / c_max)  in [alpha*c_min/c_max, alpha] = [0.03, 0.25]
        # Stockout:  beta * 1.0             = 0.35
        # This ensures stockout penalty > transport for all districts (LP prefers to allocate).
        # Do NOT divide by N — that would make stockout 33x cheaper than transport.
        c_obj = np.concatenate([
            self.alpha * (transport_cost / c_max) + self.gamma * fraud_prob,  # x part
            self.beta  * np.ones(N),   # s part: penalise each stockout equally
        ])

        # Inequality constraints Ax ≤ b
        rows_A, rows_b = [], []

        # (1) per-district stockout: -x_i - s_i ≤ -(D̂_i - I_i)+
        for i in range(N):
            row = np.zeros(2 * N)
            row[i] = -1.0     # -x_i
            row[N + i] = -1.0  # -s_i
            rows_A.append(row)
            rows_b.append(-max(0.0, demand_mean[i] - inventory[i]))

        # (2) supply: Σ x_i ≤ S_t
        supply_row = np.zeros(2 * N)
        supply_row[:N] = 1.0
        rows_A.append(supply_row)
        rows_b.append(supply_total)

        A_ub = np.array(rows_A)
        b_ub = np.array(rows_b)

        # Bounds
        max_alloc = self.max_ratio * np.maximum(demand_mean, 1.0)
        bounds = [(0, float(mx)) for mx in max_alloc] + [(0, None)] * N

        result = linprog(
            c_obj, A_ub=A_ub, b_ub=b_ub,
            bounds=bounds, method="highs",
        )

        if result.success:
            return result.x[:N]
        else:
            logger.warning(f"LP failed ({result.message}) — falling back to proportional")
            return self._policy_proportional(demand_mean, supply_total)

    def _policy_equity(
        self,
        inventory:    np.ndarray,
        demand_mean:  np.ndarray,
        supply_total: float,
    ) -> np.ndarray:
        """
        Equity-First: equalise service ratios S_i = (I_i + x_i) / D̂_i.

        Algorithm:
          1. Target service ratio τ = min(1, S_t / Σ max(0, D̂_i - I_i))
          2. Each district gets x_i = max(0, τ·D̂_i - I_i)
          3. Renormalise to supply total
        """
        N = len(demand_mean)
        shortfall = np.maximum(demand_mean - inventory, 0.0)
        total_shortfall = shortfall.sum()

        if total_shortfall <= 0:
            # Inventory already covers everything; give a small base allocation
            return np.ones(N) * supply_total / N

        tau = min(1.0, supply_total / total_shortfall)
        x = np.maximum(0.0, tau * demand_mean - inventory)

        # Distribute leftover equally
        used = x.sum()
        leftover = max(0.0, supply_total - used)
        x += leftover / N
        return x

    def _policy_risk_averse(
        self,
        inventory:      np.ndarray,
        demand_mean:    np.ndarray,
        demand_std:     np.ndarray,
        fraud_prob:     np.ndarray,
        transport_cost: np.ndarray,
        supply_total:   float,
    ) -> np.ndarray:
        """
        CVaR-penalized LP:
          - Adjust stockout penalty β_i upward for high-variance districts
            β_i = β · (1 + σ_i / D̂_i)  [coefficient of variation penalty]
          - Add a safety buffer: target D̂_i + z_α·σ_i
            (z_{0.95} = 1.645)
        """
        try:
            from scipy.optimize import linprog
        except ImportError:
            return self._policy_proportional(demand_mean, supply_total)

        N = len(demand_mean)
        z_alpha = 1.645  # 95th percentile
        cv = np.where(demand_mean > 0, demand_std / demand_mean, 0.0)
        beta_adj = self.beta * (1.0 + cv)            # higher for risky districts
        effective_demand = demand_mean + z_alpha * demand_std  # upside buffer

        c_obj = np.concatenate([
            self.alpha * transport_cost + self.gamma * fraud_prob,
            beta_adj * self.stockout_penalty * np.ones(N),
        ])

        rows_A, rows_b = [], []
        for i in range(N):
            row = np.zeros(2 * N)
            row[i] = -1.0
            row[N + i] = -1.0
            rows_A.append(row)
            rows_b.append(-max(0.0, effective_demand[i] - inventory[i]))

        supply_row = np.zeros(2 * N)
        supply_row[:N] = 1.0
        rows_A.append(supply_row)
        rows_b.append(supply_total)

        A_ub = np.array(rows_A)
        b_ub = np.array(rows_b)

        max_alloc = self.max_ratio * np.maximum(effective_demand, 1.0)
        bounds = [(0, float(mx)) for mx in max_alloc] + [(0, None)] * N

        result = linprog(
            c_obj, A_ub=A_ub, b_ub=b_ub,
            bounds=bounds, method="highs",
        )

        if result.success:
            return result.x[:N]
        else:
            return self._policy_proportional(demand_mean, supply_total)

    # ── Inspection decisions ───────────────────────────────────────────────

    def _decide_inspections(
        self,
        fraud_prob:      np.ndarray,
        allocation:      np.ndarray,
        transport_cost:  np.ndarray,
        budget:          float,
    ) -> np.ndarray:
        """
        Greedy inspection selection:
          ROI_i = p_i · x_i / (inspection_cost + c_i·x_i)
        Select top-k districts by ROI until inspection budget exhausted.
        Always inspect if p_i > inspection_threshold (regardless of budget).
        """
        N = len(fraud_prob)
        y = np.zeros(N, dtype=int)

        # Mandatory inspections
        mandatory = fraud_prob > self.inspection_threshold
        y[mandatory] = 1
        budget_used = mandatory.sum() * self.inspection_cost

        # ROI-based selection for remaining budget
        roi = np.where(
            allocation > 0,
            fraud_prob * allocation / (self.inspection_cost + transport_cost * allocation + 1e-9),
            0.0,
        )
        roi[mandatory] = -np.inf  # already selected

        order = np.argsort(-roi)
        for i in order:
            if roi[i] <= 0:
                break
            if budget_used + self.inspection_cost > budget:
                break
            y[i] = 1
            budget_used += self.inspection_cost

        return y
