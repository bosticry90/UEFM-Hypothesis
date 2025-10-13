# toe/lp_orchestration.py
"""
LP orchestration utilities using simplex/HiGHS with smoothed-analysis friendly
preprocessing (scaling + tiny jitter). Great for:
- scheduling simulation grids,
- GPU/CPU time & VRAM knapsacks,
- budgeted experiment selection.

Requires SciPy (HiGHS). Falls back cleanly if SciPy missing.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any
import numpy as np

try:
    from scipy.optimize import linprog
    _SCIPY_OK = True
except Exception:
    _SCIPY_OK = False


@dataclass
class LPScheduleResult:
    ok: bool
    selected: np.ndarray          # 0/1 selection (relaxed) or fractional if needed
    objective: float
    meta: Dict[str, Any]


def _scale_and_jitter(A: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-9, seed: int = 0):
    rng = np.random.default_rng(seed)
    # scale columns of A and c to unit norm to avoid degeneracy
    scale = np.linalg.norm(A, axis=0)
    scale[scale == 0] = 1.0
    A2 = A / scale
    c2 = c / np.maximum(scale, 1e-12)
    # tiny noise (smoothed analysis friendly)
    A2 = A2 + eps * rng.standard_normal(A2.shape)
    b2 = b + eps * rng.standard_normal(b.shape)
    c2 = c2 + eps * rng.standard_normal(c2.shape)
    return A2, b2, c2, scale


def schedule_grid_lp(
    rewards: np.ndarray,
    mem_costs: np.ndarray,
    time_costs: np.ndarray,
    mem_limit: float,
    time_limit: float,
    seed: int = 0,
) -> LPScheduleResult:
    """
    LP:
      maximize c^T x
      s.t.  mem_costs^T x <= mem_limit
            time_costs^T x <= time_limit
            0 <= x_i <= 1

    Returns fractional x; round/greedy postprocessing can be added by caller.
    """
    n = len(rewards)
    if not _SCIPY_OK:
        # Trivial greedy fallback
        idx = np.argsort(-rewards / (time_costs + 1e-9))
        x = np.zeros(n)
        mem = 0.0; t = 0.0; obj = 0.0
        for i in idx:
            if mem + mem_costs[i] <= mem_limit and t + time_costs[i] <= time_limit:
                x[i] = 1.0
                mem += mem_costs[i]; t += time_costs[i]; obj += float(rewards[i])
        return LPScheduleResult(ok=False, selected=x, objective=obj, meta={"reason": "scipy_missing"})

    # Build A, b, c for linprog in standard min form (so negate rewards)
    A = np.vstack([mem_costs, time_costs])
    b = np.array([mem_limit, time_limit], dtype=float)
    c = -np.asarray(rewards, dtype=float)

    A2, b2, c2, scale = _scale_and_jitter(A, b, c, seed=seed)

    bounds = [(0.0, 1.0)] * n
    res = linprog(
        c=c2,
        A_ub=A2,
        b_ub=b2,
        bounds=bounds,
        method="highs-simplex",
        options={"presolve": True},
    )
    if not res.success:
        return LPScheduleResult(ok=False, selected=np.zeros(n), objective=0.0, meta={"status": res.status, "message": res.message})

    x = np.clip(res.x, 0.0, 1.0)
    obj = float(np.dot(rewards, x))
    return LPScheduleResult(ok=True, selected=x, objective=obj, meta={"status": res.status})
