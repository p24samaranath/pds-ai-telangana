"""
Agent-Based SCM Simulation Package

Implements the mathematical framework from:
  "Agent-Based Supply Chain Management" research paper

State: S_t = (I_t, D̂_t, p̂_t, ĉ)
  I_t  — inventory vector [N districts]
  D̂_t  — demand belief distribution [N x 2] (mean, std)
  p̂_t  — fraud probability vector [N]
  ĉ    — geographic transport cost vector [N]

Agents:
  DemandAgent     — forecasts D̂_t from history
  FraudAgent      — estimates p̂_t, recommends inspections y_{i,t}
  GeoAgent        — computes ĉ, accessibility
  AllocationAgent — solves multi-objective optimisation for x_{i,t}
  GovernanceAgent — enforces policy constraints

Cost: C_t = α·C^trans + β·C^stock + γ·C^leak + δ·C^ineq
"""
from .data_generator import DistrictDataGenerator
from .scm_simulator import SCMSimulator, SimulationConfig, SimulationResult
from .allocation_optimizer import AllocationOptimizer

__all__ = [
    "DistrictDataGenerator",
    "SCMSimulator",
    "SimulationConfig",
    "SimulationResult",
    "AllocationOptimizer",
]
