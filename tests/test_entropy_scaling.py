# tests/test_entropy_scaling.py
import numpy as np
import networkx as nx
from toe.substrate import Substrate
from toe.geometry import minimal_cut_length_1d

def test_path_vs_proxy():
    N, chi, m = 48, 3, 10
    S = Substrate.line(N, d=2, bond_dim=chi)
    s_bits = S.entropy_of(range(m)).s_bits
    expected = minimal_cut_length_1d(region_size=m, bond_dim=chi)  # open chain proxy
    assert np.isclose(s_bits, expected)

def test_ring_two_cuts():
    N, chi, m = 64, 4, 12
    S = Substrate.ring(N, d=2, bond_dim=chi)
    s_bits = S.entropy_of(range(m)).s_bits
    # ring → typically 2 cut edges when 0<m<N
    assert np.isclose(s_bits, 2*np.log2(chi), atol=1e-12)

def test_random_graph_monotone_in_chi():
    n = 40
    G = nx.fast_gnp_random_graph(n, 0.06, seed=7)
    A = range(8)
    S2 = Substrate(G, d=2, bond_dim=2)
    S4 = Substrate(G, d=2, bond_dim=4)
    s2 = S2.entropy_of(A).s_bits
    s4 = S4.entropy_of(A).s_bits
    assert s4 >= s2 - 1e-12
