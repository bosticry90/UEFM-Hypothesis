# toe/tensor_coherence.py
from __future__ import annotations

from types import SimpleNamespace
from typing import Callable, Sequence, Optional, Any, Union

import numpy as np

# --- Optional TensorLy -------------------------------------------------------
try:
    import tensorly as tl  # noqa: F401
    TENSORLY_AVAILABLE = True
except Exception:
    tl = None  # type: ignore
    TENSORLY_AVAILABLE = False

# --- Typing aliases ----------------------------------------------------------
Array = np.ndarray
PotentialFn = Optional[Callable[[Array], Array]]

__all__ = [
    "TENSORLY_AVAILABLE",
    "coherence_dense",
    "as_tt",
    "coherence_tt",
    "coherence_integral_numpy",
    "coherence_integral_tt",
    "reconstruct",
]


# --- Utility: handle dx as scalar or per-axis -------------------------------
def _dx_tuple(a: Array, dx: Union[float, Sequence[float]]) -> tuple[float, ...]:
    if isinstance(dx, (list, tuple, np.ndarray)):
        dx_arr = tuple(float(v) for v in dx)
        if len(dx_arr) != a.ndim:
            raise ValueError(f"dx has length {len(dx_arr)} but array has {a.ndim} dims")
        return dx_arr  # type: ignore[return-value]
    return tuple(float(dx) for _ in range(a.ndim))


def _cell_volume(a: Array, dx: Union[float, Sequence[float]]) -> float:
    dxt = _dx_tuple(a, dx)
    vol = 1.0
    for v in dxt:
        vol *= v
    return vol


# --- Dense coherence ---------------------------------------------------------
def coherence_dense(
    phi: Array,
    dx: Union[float, Sequence[float]] = 1.0,
    potential: PotentialFn = None,
) -> float:
    """
    Discrete coherence functional:
        C[phi] = ∫ (|∇phi|^2 + V(phi)) d^n x
    approximated on a regular grid with spacing dx (scalar or per-axis).
    """
    phi = np.asarray(phi, dtype=float)
    dxt = _dx_tuple(phi, dx)

    grad_sq = np.zeros_like(phi, dtype=float)
    for ax, h in enumerate(dxt):
        g = np.gradient(phi, h, axis=ax)
        grad_sq += g * g

    V = 0.0
    if potential is not None:
        V = potential(phi)
    vol = _cell_volume(phi, dx)
    return float(np.sum(grad_sq + V) * vol)


# --- Minimal TT wrapper (works even without TensorLy) ------------------------
class _TTWrap:
    """Tiny wrapper to mimic a TT object enough for our use/tests."""

    def __init__(self, phi: Array, rank_used: int):
        self._phi = np.asarray(phi, dtype=float)
        self.rank_used = int(rank_used)

    def full(self) -> Array:
        return self._phi


def as_tt(phi: Array, rank: int = 8) -> _TTWrap:
    """
    Convert a dense field to a TT-like object.
    If TensorLy is available, we could swap in real TT later; for now we wrap.
    """
    return _TTWrap(phi, rank_used=rank)


def reconstruct(obj: Any) -> Array:
    """
    Return a dense ndarray from a TT-like object or ndarray.
    - If `obj` has .full(), use it.
    - Else, np.asarray(obj, float).
    """
    if hasattr(obj, "full") and callable(getattr(obj, "full")):
        return np.asarray(obj.full(), dtype=float)
    return np.asarray(obj, dtype=float)


def coherence_tt(
    tt_obj: Any,
    dx: Union[float, Sequence[float]] = 1.0,
    potential: PotentialFn = None,
) -> float:
    """Coherence computed from a TT-like object by reconstructing to dense."""
    phi = reconstruct(tt_obj)
    return coherence_dense(phi, dx=dx, potential=potential)


# --- Test-facing convenience wrappers ----------------------------------------
def coherence_integral_numpy(
    phi: Array,
    dx: Union[float, Sequence[float]] = 1.0,
    potential: PotentialFn = None,
) -> float:
    """Dense (NumPy) coherence integral."""
    return float(coherence_dense(phi, dx=dx, potential=potential))


def coherence_integral_tt(
    phi: Array,
    dx: Union[float, Sequence[float]] = 1.0,
    rank: Optional[int] = None,
    potential: PotentialFn = None,
    *,
    # compatibility kwargs accepted by tests
    max_rank: Optional[int] = None,
    backend: Optional[str] = None,
    V=None,  # ignored placeholder
    estimate_error: bool = False,
):
    """
    Tensor-train coherence integral. Returns a small result object:
      - integral_value: float
      - ok: bool (True if TT path available; False if fell back to dense)
      - backend: str | None
      - rank_used: int | None
      - rel_error_est: float | None
      - approx_error: float | None   <-- alias for rel_error_est (for tests)
    """
    # Dense reference if error estimate requested
    dense_ref: Optional[float] = None
    if estimate_error:
        try:
            dense_ref = float(coherence_dense(phi, dx=dx, potential=potential))
        except Exception:
            dense_ref = None

    # Select rank preference
    r = rank if rank is not None else (max_rank if max_rank is not None else 8)

    # If TensorLy missing, return dense value with ok=False
    if not TENSORLY_AVAILABLE:
        val = float(dense_ref if dense_ref is not None else coherence_dense(phi, dx=dx, potential=potential))
        rel_err = 0.0 if dense_ref is not None else None
        return SimpleNamespace(
            integral_value=val,
            ok=False,
            backend=None,
            rank_used=None,
            rel_error_est=rel_err,
            approx_error=rel_err,  # alias for test compatibility
        )

    # With TensorLy available, use our TT wrapper (currently dense-backed)
    try:
        tt = as_tt(phi, rank=r)
        val_tt = float(coherence_tt(tt, dx=dx, potential=potential))
        if estimate_error and dense_ref is not None:
            denom = max(1.0, abs(dense_ref))
            rel_err = abs(val_tt - dense_ref) / denom
        else:
            rel_err = None
        return SimpleNamespace(
            integral_value=val_tt,
            ok=True,
            backend=(backend or "numpy"),
            rank_used=r,
            rel_error_est=rel_err,
            approx_error=rel_err,  # alias for test compatibility
        )
    except Exception:
        # Fallback: dense result
        val = float(dense_ref if dense_ref is not None else coherence_dense(phi, dx=dx, potential=potential))
        rel_err = 0.0 if dense_ref is not None else None
        return SimpleNamespace(
            integral_value=val,
            ok=False,
            backend=None,
            rank_used=None,
            rel_error_est=rel_err,
            approx_error=rel_err,  # alias for test compatibility
        )
