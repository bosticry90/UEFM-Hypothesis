# toe/lp_orchestration.py
from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, List
from scipy.optimize import linprog

__all__ = ["schedule_grid_lp"]

def _normalize_method(method: Optional[str]) -> str:
    """
    Map legacy/alias method names to SciPy-supported strings.
    SciPy >=1.6 supports: 'highs', 'highs-ds', 'highs-ipm'.
    """
    if method is None:
        return "highs"
    m = method.lower()
    if m in {"highs", "highs-ds", "highs-ipm"}:
        return m
    # tolerate common aliases
    if m in {"highs-simplex", "simplex", "revised simplex"}:
        return "highs-ds"
    if m in {"interior-point", "ipm"}:
        return "highs-ipm"
    return "highs"

def _smoothed_inputs(
    c: np.ndarray,
    A_ub: np.ndarray,
    b_ub: np.ndarray,
    bounds: List[Tuple[float, float]],
    seed: Optional[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[float, float]]]:
    """
    Light perturbation + column scaling (smoothed-analysis-friendly).
    """
    rng = np.random.default_rng(seed)
    eps = 1e-9

    c_ = c + eps * rng.standard_normal(c.shape)
    A_ = A_ub + eps * rng.standard_normal(A_ub.shape)
    b_ = b_ub + eps * rng.standard_normal(b_ub.shape)

    # Scale columns to unit-ish size to help numerics
    scale = np.maximum(1.0, np.linalg.norm(A_, ord=np.inf, axis=0))
    A_ = A_ / scale
    c_ = c_ / scale

    return c_, A_, b_, bounds

def schedule_grid_lp(
    rewards: np.ndarray,
    mem_cost: np.ndarray,
    time_cost: np.ndarray,
    mem_limit: float,
    time_limit: float,
    seed: Optional[int] = None,
    method: Optional[str] = "highs",
):
    """
    Solve a resource-constrained scheduling LP:

        maximize   sum_i log(rewards_i) * x_i
        subject to sum_i mem_cost_i  * x_i <= mem_limit
                   sum_i time_cost_i * x_i <= time_limit
                   0 <= x_i <= 1

    Implemented as a minimization (SciPy) by negating the objective.

    Returns
    -------
    res : OptimizeResult
        SciPy result with extra fields:
          - res.objective : float (maximized objective value)
          - res.selected  : np.ndarray, decision vector (shape (n,))
    """
    rewards = np.asarray(rewards, dtype=float)
    mem_cost = np.asarray(mem_cost, dtype=float)
    time_cost = np.asarray(time_cost, dtype=float)

    if np.any(rewards <= 0):
        raise ValueError("All rewards must be > 0 for log-reward objective.")

    n = rewards.size
    # Minimize -sum log(r_i) x_i  <=>  maximize sum log(r_i) x_i
    c = -np.log(rewards)

    A_ub = np.vstack([mem_cost, time_cost])
    b_ub = np.array([mem_limit, time_limit], dtype=float)
    bounds = [(0.0, 1.0)] * n

    c_s, A_s, b_s, bounds_s = _smoothed_inputs(c, A_ub, b_ub, bounds, seed)
    chosen_method = _normalize_method(method)

    try:
        res = linprog(
            c_s, A_ub=A_s, b_ub=b_s, A_eq=None, b_eq=None,
            bounds=bounds_s, method=chosen_method, options={"presolve": True}
        )
    except ValueError:
        # Fallback for older SciPy builds
        res = linprog(
            c_s, A_ub=A_s, b_ub=b_s, A_eq=None, b_eq=None,
            bounds=bounds_s, method="highs", options={"presolve": True}
        )

    # Attach maximized objective
    try:
        objective = -float(res.fun) if res.success else np.nan
    except Exception:
        objective = np.nan
    res.objective = objective
    res["objective"] = objective

    # Attach selected vector (solution); keep as float in [0,1]
    try:
        selected = np.asarray(res.x, dtype=float).reshape(-1)
    except Exception:
        selected = np.full(n, np.nan)
    res.selected = selected
    res["selected"] = selected

    return res
