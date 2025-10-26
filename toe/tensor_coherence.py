# toe/tensor_coherence.py
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Optional, Sequence, Union

import numpy as np

# --- Optional tensorly backend -----------------------------------------------
try:
    import tensorly as tl
    from tensorly.decomposition import tensor_train
    TENSORLY_AVAILABLE = True
except Exception:  # pragma: no cover
    tl = None
    tensor_train = None
    TENSORLY_AVAILABLE = False


# --- Result object returned by coherence_integral_tt --------------------------
@dataclass
class TTResult:
    ok: bool                      # True if tensor backend used
    integral_value: float         # Coherence computed from TT (or dense fallback)
    approx_error: Optional[float] # Relative reconstruction error (Fro), if estimated
    rank_used: Optional[int]      # Requested/used max TT rank
    backend: str                  # 'numpy', 'pytorch', etc. or 'none' on fallback


# --- Utilities ----------------------------------------------------------------
def _grad_sqr(
    phi: np.ndarray,
    dx: Union[float, Sequence[float]],
) -> np.ndarray:
    """
    Sum of squared gradients over all axes with central differences.

    Supports:
      - scalar dx: same spacing in every dimension
      - tuple/list dx: per-axis spacing matching phi.ndim
    """
    if isinstance(dx, (tuple, list, np.ndarray)):
        if len(dx) != phi.ndim:
            raise ValueError("dx length must match phi.ndim")
        grads = np.gradient(phi, *dx, edge_order=2)
    else:
        if dx <= 0:
            raise ValueError("dx must be positive.")
        grads = np.gradient(phi, dx, edge_order=2)

    g2 = np.zeros_like(phi, dtype=float)
    for g in grads:
        g2 += np.real(g * np.conj(g))
    return g2


def _volume_element(
    dx: Union[float, Sequence[float]],
    ndim: int,
) -> float:
    if isinstance(dx, (tuple, list, np.ndarray)):
        if len(dx) != ndim:
            raise ValueError("dx length must match phi.ndim")
        vol = 1.0
        for d in dx:
            vol *= float(d)
        return vol
    return float(dx) ** int(ndim)


def _potential_eval(phi: np.ndarray, potential: Optional[Callable[[np.ndarray], np.ndarray]]) -> np.ndarray:
    if potential is None:
        return np.zeros_like(phi, dtype=float)
    out = potential(phi)
    return np.asarray(out, dtype=float)


# --- Public: dense coherence integral ----------------------------------------
def coherence_integral_numpy(
    phi: np.ndarray,
    *,
    dx: Union[float, Sequence[float]] = 1.0,
    potential: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    V: Optional[Callable[[np.ndarray], np.ndarray]] = None,  # legacy alias
) -> float:
    """
    Dense coherence-like integral:
        C[phi] = ∫ (|∇phi|^2 + V(phi)) d^n x
    where dx can be scalar or per-axis spacings.
    """
    if not isinstance(phi, np.ndarray):
        phi = np.asarray(phi)
    pot = potential if potential is not None else V
    g2 = _grad_sqr(phi, dx)
    Vval = _potential_eval(phi, pot)
    dV = _volume_element(dx, phi.ndim)
    C = float(np.sum(g2 + Vval) * dV)
    return C


# --- Public: tensor (TT) coherence integral ----------------------------------
def coherence_integral_tt(
    phi: np.ndarray,
    *,
    max_rank: int = 8,
    backend: str = "numpy",
    potential: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    V: Optional[Callable[[np.ndarray], np.ndarray]] = None,  # legacy alias
    dx: Union[float, Sequence[float]] = 1.0,
    estimate_error: bool = True,
) -> TTResult:
    """
    Compute coherence via a TT approximation (if tensorly available).
    Falls back to dense result when tensorly is missing or fails.
    """
    try:
        if not TENSORLY_AVAILABLE:
            C = coherence_integral_numpy(phi, dx=dx, potential=potential, V=V)
            return TTResult(ok=False, integral_value=C, approx_error=None, rank_used=None, backend="none")

        tl.set_backend(backend)
        phi_tl = tl.tensor(np.asarray(phi))
        tt = tensor_train(phi_tl, rank=max_rank)
        phi_rec = tl.to_numpy(tl.tt_to_tensor(tt))
        C_tt = coherence_integral_numpy(phi_rec, dx=dx, potential=potential, V=V)

        err = None
        if estimate_error:
            num = np.linalg.norm(phi_rec - np.asarray(phi))
            den = np.linalg.norm(phi) + 1e-16
            err = float(num / den)

        return TTResult(ok=True, integral_value=float(C_tt), approx_error=err, rank_used=int(max_rank), backend=backend)

    except Exception:
        C = coherence_integral_numpy(phi, dx=dx, potential=potential, V=V)
        return TTResult(ok=False, integral_value=C, approx_error=None, rank_used=None, backend="none")


# --- Helpers ------------------------------------------------------------------
def reconstruct_tt(
    phi: np.ndarray,
    *,
    max_rank: int = 8,
    backend: str = "numpy",
) -> np.ndarray:
    """Return a TT-reconstructed dense approximation of `phi` (or `phi` if backend unavailable)."""
    if not TENSORLY_AVAILABLE:
        return np.asarray(phi)
    tl.set_backend(backend)
    tt = tensor_train(tl.tensor(np.asarray(phi)), rank=max_rank)
    return tl.to_numpy(tl.tt_to_tensor(tt))


def as_tt(
    phi: np.ndarray,
    *,
    max_rank: int = 8,
    rank: Optional[int] = None,      # allow rank=... alias in tests
    backend: str = "numpy",
) -> SimpleNamespace:
    """
    Build a TT representation (if available) and its dense reconstruction.

    Returns SimpleNamespace(
        ok: bool, backend: str, rank: int,
        tt: tensorly TT object or None,
        recon: np.ndarray (reconstruction or original array on fallback)
    )
    """
    if rank is not None:
        max_rank = int(rank)

    if not TENSORLY_AVAILABLE:
        return SimpleNamespace(ok=False, backend="none", rank=None, tt=None, recon=np.asarray(phi))

    tl.set_backend(backend)
    tt = tensor_train(tl.tensor(np.asarray(phi)), rank=max_rank)
    recon = tl.to_numpy(tl.tt_to_tensor(tt))
    return SimpleNamespace(ok=True, backend=backend, rank=int(max_rank), tt=tt, recon=recon)


def reconstruct(obj) -> np.ndarray:
    """
    Backwards-compatible helper used in tests:
      - If passed the object returned by `as_tt`, return its `.recon`.
      - If passed a raw ndarray, return it.
      - If passed an object with `.tt` and tensorly is available, reconstruct from TT.
    """
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "recon"):
        return np.asarray(obj.recon)
    if hasattr(obj, "tt") and TENSORLY_AVAILABLE and obj.tt is not None:
        return tl.to_numpy(tl.tt_to_tensor(obj.tt))
    raise TypeError("Unsupported object for reconstruct(). Expected ndarray or result of as_tt().")


# --- Friendly wrappers expected by tests --------------------------------------
def coherence_dense(
    phi: np.ndarray,
    dx: Union[float, Sequence[float]],
    potential: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    V: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> float:
    """Positional-friendly wrapper used by tests."""
    return coherence_integral_numpy(phi, dx=dx, potential=potential, V=V)


def coherence_tt(
    obj: Union[np.ndarray, SimpleNamespace],
    dx: Union[float, Sequence[float]] = 1.0,
    *,
    max_rank: int = 8,
    backend: str = "numpy",
    potential: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    V: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    estimate_error: bool = True,
) -> float:
    """
    Convenience wrapper that accepts either:
      - a raw ndarray `phi` (performs a fresh TT via `coherence_integral_tt`)
      - the result of `as_tt(phi, ...)` (uses its dense reconstruction directly)

    Returns a **float** (the integral value), matching test expectations.
    """
    if hasattr(obj, "recon"):  # result of as_tt(...)
        phi_rec = reconstruct(obj)
        C = coherence_integral_numpy(phi_rec, dx=dx, potential=potential, V=V)
        return float(C)

    # Otherwise assume raw ndarray
    res = coherence_integral_tt(
        np.asarray(obj),
        max_rank=max_rank,
        backend=backend,
        potential=potential,
        V=V,
        dx=dx,
        estimate_error=estimate_error,
    )
    return float(res.integral_value)


__all__ = [
    "TENSORLY_AVAILABLE",
    "TTResult",
    "coherence_integral_numpy",
    "coherence_integral_tt",
    "coherence_dense",
    "coherence_tt",
    "reconstruct_tt",
    "reconstruct",
    "as_tt",
]
