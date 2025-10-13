"""
Lightweight tensor utilities with an optional TensorLy backend.

Goals:
- Provide a stable API even if the TensorLy package is *not* installed.
- Keep computations tiny/fast for unit tests.
- Allow "TT-like" compression when TensorLy is present, but fall back to a
  no-op wrapper that simply stores the dense array.

Public API
----------
- is_tensorly_available() -> bool
- tt_from_grid(array: np.ndarray, rank: int = 8)
- tt_to_dense(tt) -> np.ndarray
- reconstruction_error(tt, original: np.ndarray) -> float
- tensor_coherence(tt, dx: float = 1.0, potential: callable | None = None) -> float

Notes
-----
- If TensorLy is available, we use a true tensor-train (TT) decomposition.
  Otherwise, we store the array densely and reconstruct exactly.
- The "coherence" functional here is a simple, general proxy:
      C = ∑_axes ||∂_axis φ||_2^2 * dx
  If `potential` is provided, we add ∑ V(φ) * dx.
  This is meant to be a quick diagnostic, not a physical law.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

try:
    import tensorly as tl
    from tensorly.decomposition import tensor_train as tl_tt_decompose
    from tensorly.tt_tensor import tt_to_tensor as tl_tt_to_tensor
    _TLY_OK = True
except Exception:
    _TLY_OK = False


def is_tensorly_available() -> bool:
    """Return True if a working TensorLy backend is available."""
    return _TLY_OK


# ---------------------------
# Internal simple TT wrapper
# ---------------------------

@dataclass
class _DenseTT:
    """Fallback wrapper when TensorLy is unavailable (stores dense array)."""
    array: np.ndarray
    ranks: int

    def to_dense(self) -> np.ndarray:
        return np.asarray(self.array, dtype=float)


@dataclass
class _TensorLyTT:
    """TensorLy-backed TT object."""
    cores: list
    ranks: int

    def to_dense(self) -> np.ndarray:
        dense = tl_tt_to_tensor(self.cores)
        # Cast to numpy array if TensorLy uses another backend:
        return np.asarray(dense)


# ---------------------------
# Public API
# ---------------------------

def tt_from_grid(array: np.ndarray, rank: int = 8):
    """
    Build a TT (or TT-like) representation from a dense ndarray.

    Parameters
    ----------
    array : np.ndarray
        Input dense tensor (any shape).
    rank : int, default 8
        Target TT-rank if TensorLy is available. If not, stored as dense.

    Returns
    -------
    TT-like object with a .to_dense() method (see classes above).
    """
    arr = np.asarray(array, dtype=float)

    # Edge-case: scalars, vectors → treat as at least 1-D
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim == 1:
        # TensorLy handles vectors, but no need to overcomplicate for tests
        if _TLY_OK:
            # Make it a 2-mode tensor: (n, 1) to go through TT machinery
            shaped = arr.reshape(arr.shape[0], 1)
            tt = tl_tt_decompose(shaped, rank=min(rank, max(2, shaped.shape[0])))
            return _TensorLyTT(cores=tt, ranks=rank)
        return _DenseTT(array=arr, ranks=rank)

    if _TLY_OK:
        try:
            # Limit rank to something sensible for tiny test tensors
            target_rank = int(max(2, min(rank, 16)))
            tt = tl_tt_decompose(arr, rank=target_rank)
            return _TensorLyTT(cores=tt, ranks=target_rank)
        except Exception:
            # Fall back to dense on any backend hiccup
            return _DenseTT(array=arr, ranks=rank)
    else:
        return _DenseTT(array=arr, ranks=rank)


def tt_to_dense(tt) -> np.ndarray:
    """Reconstruct a dense array from a TT-like object created by tt_from_grid."""
    if hasattr(tt, "to_dense"):
        return tt.to_dense()
    # Very defensive: if someone passes a numpy array by mistake
    return np.asarray(tt, dtype=float)


def reconstruction_error(tt, original: np.ndarray) -> float:
    """
    Return Frobenius norm of reconstruction error between TT and the original.
    If we are in dense fallback mode, this will be ~0 (exact).
    """
    recon = tt_to_dense(tt)
    orig = np.asarray(original, dtype=float)
    if recon.shape != orig.shape:
        return float("inf")
    return float(np.linalg.norm(recon - orig))


def _grad_l2_sq(phi: np.ndarray, dx: float) -> float:
    """Sum of L2 norms of discrete gradients along each axis."""
    if phi.ndim == 0:
        return 0.0
    total = 0.0
    for ax in range(phi.ndim):
        g = np.diff(phi, axis=ax) / float(dx)
        total += float(np.sum(g * g))
    return total


def tensor_coherence(
    tt,
    dx: float = 1.0,
    potential: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> float:
    """
    Coherence proxy:
        C[φ] = ∑_axes ||∂_axis φ||_2^2 * dx  +  ∑ V(φ) * dx     (if potential provided)

    Parameters
    ----------
    tt : TT-like object from `tt_from_grid`
    dx : float
        Grid spacing (assumed isotropic).
    potential : callable or None
        If provided, must map an ndarray φ -> ndarray of same shape, elementwise V(φ).

    Returns
    -------
    float
    """
    phi = tt_to_dense(tt)
    val = _grad_l2_sq(phi, dx=dx)
    if potential is not None:
        V = potential(phi)
        val += float(np.sum(V) * dx)
    return float(val)
