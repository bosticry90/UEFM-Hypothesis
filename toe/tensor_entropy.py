# toe/tensor_entropy.py
from __future__ import annotations
import math
from typing import Iterable, Tuple, List
import numpy as np

__all__ = [
    "entanglement_spectra_by_cuts",
    "schmidt_entropy_by_cuts",
    "tensor_rank_entropy",
    "effective_ranks_by_cuts",
]

def _as_unit_vector(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=complex).ravel()
    n2 = np.vdot(x, x).real
    if n2 <= 0:
        raise ValueError("State has zero norm.")
    return x / math.sqrt(n2)

def _infer_local_dim(total_dim: int) -> Tuple[int, int]:
    """
    Try to infer (d, N) s.t. total_dim = d**N with small d (2..6).
    Falls back to (total_dim, 1) if no perfect power is found.
    """
    for d in range(2, 7):
        N = round(math.log(total_dim, d))
        if d**N == total_dim:
            return d, N
    return total_dim, 1

def entanglement_spectra_by_cuts(
    psi: np.ndarray,
    local_dim: int | None = None,
    n_sites: int | None = None,
) -> List[np.ndarray]:
    """
    Compute Schmidt singular values across all contiguous bipartitions
    for a 1D chain. Input is a pure state vector psi (flattened).
    Returns a list [s_1, s_2, ..., s_{N-1}] where s_k are singular values
    of the reshaped matrix (d^k) x (d^{N-k}). psi is normalized internally.
    """
    v = _as_unit_vector(psi)
    D = v.size

    if local_dim is None or n_sites is None:
        d, N = _infer_local_dim(D)
    else:
        d, N = int(local_dim), int(n_sites)
        if d**N != D:
            raise ValueError(f"Expected d**N == len(psi), got {d}^{N} != {D}")

    out: List[np.ndarray] = []
    for k in range(1, N):  # cuts between sites
        left = d**k
        right = D // left
        M = v.reshape(left, right)
        # econ SVD for speed; we only need singular values
        s = np.linalg.svd(M, compute_uv=False)
        out.append(s)
    return out

def schmidt_entropy_by_cuts(
    psi: np.ndarray,
    local_dim: int | None = None,
    n_sites: int | None = None,
    base: float = math.e,
) -> np.ndarray:
    """
    Von Neumann entanglement entropy S_k = -sum p_i log(p_i) at each cut,
    with p_i = s_i^2 / (sum s_i^2). Returns array of length N-1.

    Parameters
    ----------
    base : log base (math.e for natural log; 2.0 for bits).
    """
    spectra = entanglement_spectra_by_cuts(psi, local_dim, n_sites)
    ent = []
    for s in spectra:
        p = (s**2) / np.sum(s**2)
        # numerical guard
        p = p[np.where(p > 0)]
        if p.size == 0:
            ent.append(0.0)
        else:
            ent.append(float(-np.sum(p * (np.log(p) / np.log(base)))))
    return np.asarray(ent, dtype=float)

def tensor_rank_entropy(
    psi: np.ndarray,
    local_dim: int | None = None,
    n_sites: int | None = None,
    base: float = math.e,
    reduction: str = "mean",
) -> float:
    """
    A single-number tensor-coherence diagnostic:
    take the entanglement entropy over all cuts and reduce by 'mean' or 'max'.
    """
    ent = schmidt_entropy_by_cuts(psi, local_dim, n_sites, base=base)
    if ent.size == 0:
        return 0.0
    if reduction == "mean":
        return float(np.mean(ent))
    elif reduction == "max":
        return float(np.max(ent))
    else:
        raise ValueError("reduction must be 'mean' or 'max'")

def effective_ranks_by_cuts(
    psi: np.ndarray,
    local_dim: int | None = None,
    n_sites: int | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Effective Schmidt rank per cut: minimal r such that sum_{i<=r} s_i^2
    captures (1 - eps) of total weight. Returns array length N-1.
    """
    spectra = entanglement_spectra_by_cuts(psi, local_dim, n_sites)
    ranks = []
    for s in spectra:
        w = s**2
        w = w / np.sum(w)
        c = np.cumsum(np.sort(w)[::-1])
        r = int(np.searchsorted(c, 1.0 - eps) + 1)
        ranks.append(r)
    return np.asarray(ranks, dtype=int)
