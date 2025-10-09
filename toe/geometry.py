from __future__ import annotations
from typing import Iterable, Set, Union
import numpy as np


# -------------------------
# Dual-use helper (two APIs)
# -------------------------
# 1) entanglement_distance_1d(i, j, N, periodic=True) -> int (graph distance)
# 2) entanglement_distance_1d(separation_edges=..., bond_dim=...) -> float (toy entropy = sep * ln chi)
def entanglement_distance_1d(*args, **kwargs):
    if "separation_edges" in kwargs:
        sep = int(kwargs["separation_edges"])
        chi = int(kwargs["bond_dim"])
        return float(sep * np.log(chi))
    # legacy graph-distance form
    if len(args) < 3:
        raise TypeError("Provide either (i, j, N, periodic=True) or named (separation_edges=..., bond_dim=...).")
    i, j, N = int(args[0]), int(args[1]), int(args[2])
    periodic = True if len(args) < 4 else bool(args[3])
    d = abs(i - j)
    if periodic:
        d = min(d, N - d)
    return d


def _to_set(region: Iterable[int]) -> Set[int]:
    return set(int(i) for i in region)


# --------------------------------------------------------
# wedge_reconstructable_1d: supports TWO calling patterns:
# --------------------------------------------------------
# A) Toy code-distance proxy:
#       wedge_reconstructable_1d(boundary_size=..., erased=..., code_distance_edges=...)
#    -> bool, True iff erased < code_distance_edges (QEC-style threshold).
#
# B) Geometric 1D wedge rule:
#       wedge_reconstructable_1d(A, bulk, N, periodic=True)
#    -> bool or set[int], based on dist(x, A) < dist(x, A^c)
def wedge_reconstructable_1d(*args, **kwargs):
    # --- (A) QEC-style proxy signature ---
    if "boundary_size" in kwargs or "erased" in kwargs or "code_distance_edges" in kwargs:
        boundary_size = int(kwargs.get("boundary_size", 0))
        erased = int(kwargs.get("erased", 0))
        d_edges = int(kwargs.get("code_distance_edges", 0))
        if boundary_size <= 0 or d_edges <= 0:
            return False
        # Recoverable if erasures are strictly below the code distance.
        # (Matches the unit test that expects True for erased=1, distance=2)
        return erased < d_edges

    # --- (B) Geometric signature: (A, bulk, N, periodic=True) ---
    if len(args) < 3:
        raise TypeError(
            "Call as wedge_reconstructable_1d(boundary_size=..., erased=..., code_distance_edges=...) "
            "or wedge_reconstructable_1d(A, bulk, N, periodic=True)."
        )

    A, bulk, N = args[0], args[1], int(args[2])
    periodic = True if len(args) < 4 else bool(args[3])

    Aset = _to_set(A)
    Ac = set(range(N)) - Aset

    def dist_to_set(x: int, S: Set[int]) -> int:
        if not S:
            return np.iinfo(np.int32).max
        return min(entanglement_distance_1d(x, s, N, periodic) for s in S)

    def rec_one(x: int) -> bool:
        dA = dist_to_set(x, Aset)
        dAc = dist_to_set(x, Ac)
        return dA < dAc

    if isinstance(bulk, int):
        return rec_one(int(bulk))
    return {int(x) for x in bulk if rec_one(int(x))}


def minimal_cut_length_1d(region_size: int, bond_dim: int) -> float:
    """
    Test helper: S(A) = |γ_A| * log(chi) for a contiguous region on a 1D ring.
    On a ring, any nontrivial proper subregion has |γ_A| = 2.
    Uses natural log (nats) because tests compare with np.log().
    """
    m = int(region_size)
    if m <= 0:
        return 0.0
    cut = 2
    return float(cut * np.log(int(bond_dim)))
