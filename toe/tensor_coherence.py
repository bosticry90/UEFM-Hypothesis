# toe/tensor_coherence.py
"""
Tensor-network acceleration (Phase 3):
- Optional TensorLy/CuPy backend for tensor-train (TT) style compression.
- Fast, low-rank surrogate for coherence-like integrals on scalar fields:
    I[phi] ≈ ∫ ( |∇phi|^2 + V(phi) ) dx
- Falls back to NumPy if TensorLy isn't available (keeps tests green).

This is a *numerical aide* for UEFM-style high-dimensional integrals,
inspired by THOR-like TT compression. It’s lightweight on purpose.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Dict, Any

import numpy as np

# -------- optional deps (TensorLy, CuPy) ----------
_TL_OK = False
try:
    import tensorly as tl
    from tensorly.decomposition import tensor_train
    _TL_OK = True
except Exception:
    tl = None  # type: ignore

try:
    import cupy as cp  # optional GPU
    _CUPY_OK = True
except Exception:
    cp = None  # type: ignore
    _CUPY_OK = False


@dataclass
class TTCoherenceResult:
    ok: bool
    backend: str
    ranks: Tuple[int, ...]
    integral_value: float
    approx_error: Optional[float]
    meta: Dict[str, Any]


def _finite_difference_grad_sq(phi: np.ndarray) -> float:
    """Simple ∑ |∇phi|^2 with central differences (periodic)."""
    if phi.ndim == 0:
        return 0.0
    grad_sq = 0.0
    for axis in range(phi.ndim):
        fwd = np.roll(phi, -1, axis=axis)
        bwd = np.roll(phi, +1, axis=axis)
        d = (fwd - bwd) / 2.0
        grad_sq += np.vdot(d, d).real
    return float(grad_sq)


def _potential_term(phi: np.ndarray, V: Optional[Callable[[np.ndarray], np.ndarray]]) -> float:
    if V is None:
        return 0.0
    Vphi = V(phi)
    return float(np.sum(Vphi))


def coherence_integral_numpy(
    phi: np.ndarray,
    V: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> float:
    """Plain NumPy version: I = ∑ (|∇phi|^2 + V(phi))."""
    return _finite_difference_grad_sq(phi) + _potential_term(phi, V)


def coherence_integral_tt(
    phi: np.ndarray,
    max_rank: int = 8,
    backend: str = "numpy",     # "numpy" | "cupy" | "auto"
    V: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    estimate_error: bool = True,
) -> TTCoherenceResult:
    """
    Tensor-train compressed surrogate. If TensorLy is missing, we fall back
    to NumPy and report ok=False (so tests/demos still run).
    """
    if not _TL_OK:
        val = coherence_integral_numpy(phi, V=V)
        return TTCoherenceResult(
            ok=False,
            backend="numpy",
            ranks=(1,),
            integral_value=val,
            approx_error=None,
            meta={"reason": "tensorly_not_available"},
        )

    # Select backend
    if backend == "auto":
        chosen = "cupy" if _CUPY_OK else "numpy"
    else:
        chosen = backend
    tl.set_backend(chosen)

    # Move data to chosen backend
    x = phi
    if chosen == "cupy":
        x_t = tl.tensor(cp.asarray(x))
    else:
        x_t = tl.tensor(np.asarray(x))

    # TT compress phi and (optionally) V(phi)
    tt_phi = tensor_train(x_t, rank=max_rank)

    # Build |∇phi|^2 approximately:
    # We reconstruct a moderate-resolution proxy for gradients by partial to_dense().
    # For high-D, you’d keep contractions in TT form; here we keep it simple.
    phi_approx = tl.to_numpy(tl.tt_to_tensor(tt_phi))
    grad_sq = _finite_difference_grad_sq(phi_approx)

    Vterm = 0.0
    if V is not None:
        if chosen == "cupy":
            Vterm = float(cp.asnumpy(cp.array(V(phi_approx))).sum())
        else:
            Vterm = float(np.array(V(phi_approx)).sum())

    val = float(grad_sq + Vterm)

    err = None
    if estimate_error:
        # crude error estimate vs. full NumPy
        full = coherence_integral_numpy(phi, V=V)
        err = abs(val - full)

    # ranks from tt_phi are (r0, r1, ..., rN); tensorly exposes as a list of cores
    ranks = tuple(tt_phi.rank)

    return TTCoherenceResult(
        ok=True,
        backend=chosen,
        ranks=ranks,
        integral_value=val,
        approx_error=err,
        meta={},
    )
