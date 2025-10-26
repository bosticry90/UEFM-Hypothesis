# toe/tensor_tt_core.py
"""
Tensor-Train (TT-SVD) utilities for UEFM / ToE Phase 3.1
Provides real low-rank decomposition, reconstruction, and
rank-scheduling by tolerance.
"""
from __future__ import annotations
import numpy as np
from types import SimpleNamespace
from typing import Sequence, Optional, Callable, Any, Union

try:
    import tensorly as tl
    from tensorly.decomposition import tensor_train
    TENSORLY_AVAILABLE = True
except Exception:
    tl = None  # type: ignore
    tensor_train = None  # type: ignore
    TENSORLY_AVAILABLE = False


Array = np.ndarray
PotentialFn = Optional[Callable[[Array], Array]]


# ---------- Dense coherence (reuse core math) ------------------------------
def _dx_tuple(a: Array, dx: Union[float, Sequence[float]]) -> tuple[float, ...]:
    if isinstance(dx, (list, tuple, np.ndarray)):
        dx_arr = tuple(float(v) for v in dx)
        if len(dx_arr) != a.ndim:
            raise ValueError(f"dx has length {len(dx_arr)} but array has {a.ndim} dims")
        return dx_arr
    return tuple(float(dx) for _ in range(a.ndim))


def _cell_volume(a: Array, dx: Union[float, Sequence[float]]) -> float:
    vol = 1.0
    for v in _dx_tuple(a, dx):
        vol *= v
    return vol


def coherence_dense(phi: Array, dx=1.0, potential: PotentialFn = None) -> float:
    phi = np.asarray(phi, float)
    grad_sq = np.zeros_like(phi)
    for ax, h in enumerate(_dx_tuple(phi, dx)):
        g = np.gradient(phi, h, axis=ax)
        grad_sq += g * g
    V = 0.0
    if potential is not None:
        V = potential(phi)
    return float(np.sum(grad_sq + V) * _cell_volume(phi, dx))


# ---------- TT-SVD Compression ---------------------------------------------
def tt_svd(phi: Array, rank: int = 8) -> Any:
    """Return TensorLy TT decomposition (dense fallback if unavailable)."""
    if not TENSORLY_AVAILABLE:
        return SimpleNamespace(core=phi, full=lambda: phi, rank=[phi.shape])
    phi_tl = tl.tensor(phi, dtype=float)
    tt = tensor_train(phi_tl, rank=rank)
    return tt


def reconstruct(tt_obj: Any) -> Array:
    """Reconstruct dense ndarray from a TensorLy TT or fallback."""
    if hasattr(tt_obj, "core"):  # fallback SimpleNamespace
        return np.asarray(tt_obj.core, float)
    if hasattr(tl, "tt_to_tensor"):
        return np.asarray(tl.tt_to_tensor(tt_obj), float)
    if hasattr(tt_obj, "full"):
        return np.asarray(tt_obj.full(), float)
    return np.asarray(tt_obj, float)


def coherence_tt(tt_obj: Any, dx=1.0, potential: PotentialFn = None) -> float:
    phi = reconstruct(tt_obj)
    return coherence_dense(phi, dx=dx, potential=potential)


# ---------- Rank-scheduling -------------------------------------------------
def rank_sweep(
    phi: Array,
    dx: Union[float, Sequence[float]] = 1.0,
    ranks: Sequence[int] = (2, 4, 8, 12, 16, 24),
    potential: PotentialFn = None,
) -> list[tuple[int, float, float]]:
    """
    Returns list of (rank, rel_error, coherence).
    """
    ref = coherence_dense(phi, dx=dx, potential=potential)
    out = []
    for r in ranks:
        tt = tt_svd(phi, rank=r)
        coh = coherence_tt(tt, dx=dx, potential=potential)
        rel = abs(coh - ref) / max(1.0, abs(ref))
        out.append((r, rel, coh))
    return out


def find_min_rank_for_tol(
    phi: Array,
    dx: Union[float, Sequence[float]] = 1.0,
    target_rel_err: float = 1e-3,
    ranks: Sequence[int] = (2, 4, 8, 12, 16, 24, 32),
    potential: PotentialFn = None,
) -> SimpleNamespace:
    """Increase rank until relative error ≤ target_rel_err."""
    ref = coherence_dense(phi, dx=dx, potential=potential)
    for r in ranks:
        tt = tt_svd(phi, rank=r)
        coh = coherence_tt(tt, dx=dx, potential=potential)
        rel = abs(coh - ref) / max(1.0, abs(ref))
        if rel <= target_rel_err:
            return SimpleNamespace(rank_used=r, rel_error=rel, integral_value=coh, reference=ref)
    # if not achieved
    return SimpleNamespace(rank_used=ranks[-1], rel_error=rel, integral_value=coh, reference=ref)
