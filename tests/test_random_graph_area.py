import numpy as np
import networkx as nx
from toe.substrate import Substrate

def test_random_graph_area_proxy_sanity():
    rng = np.random.default_rng(3)
    n = 24
    p = 0.12
    G = nx.erdos_renyi_graph(n, p, seed=3)
    S = Substrate(G, bond_dim=3)
    # pick small region
    A = set(range(5))
    # minimal cut bits should be non-negative and not exceed degree(A)
    res = S.entropy_of(A).s_bits
    assert res >= 0
    degA = sum(1 for u,v in G.edges if (u in A) ^ (v in A))
    assert res <= degA * np.log2(3)
