
import numpy as np

def bipartite_entropy_from_singular_values(sv, base=2.0):
    sv = np.asarray(sv, dtype=float)
    p = sv**2 / np.sum(sv**2)
    p = p[p>0]
    return -np.sum(p * (np.log(p) / np.log(base)))

def minimal_cut_length_1d(region_size: int, bond_dim: int) -> float:
    chi = float(bond_dim)
    return 2.0 * np.log(chi)

def entanglement_distance_1d(separation_edges: int, bond_dim: int) -> float:
    return separation_edges * np.log(float(bond_dim))

def wedge_reconstructable_1d(boundary_size: int, erased: int, code_distance_edges: int) -> bool:
    return erased < code_distance_edges
