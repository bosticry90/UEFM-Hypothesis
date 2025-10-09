from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Set
import numpy as np
import networkx as nx


@dataclass(frozen=True)
class CausalCone:
    nodes: Set[int]


@dataclass(frozen=True)
class EntropyResult:
    region: Set[int]
    minimal_cut_size: int
    bond_dim: int
    s_nats: float  # natural log base (nats)
    s_bits: float  # log2 base (bits)


class Substrate:
    """
    Minimal substrate with:
      - an undirected graph G
      - uniform bond dimension chi
      - helpers: graph distance, node degree, causal cone, entropy proxy
    """

    def __init__(self, G: nx.Graph, d: int = 2):
        if d < 2:
            raise ValueError("bond dimension d must be >= 2")
        self.G = G
        self.d = int(d)
        self._n = self.G.number_of_nodes()

    # --- constructors expected by tests ---
    @classmethod
    def line(cls, n: int, d: int = 2) -> "Substrate":
        """Open chain 0--1--...--(n-1)."""
        if n <= 0:
            raise ValueError("n must be positive")
        G = nx.path_graph(n)
        return cls(G, d=d)

    # --- helpers used by tests ---
    def graph_distance(self, i: int, j: int) -> int:
        """Unweighted shortest-path distance on G."""
        return nx.shortest_path_length(self.G, int(i), int(j))

    def degree(self, i: int) -> int:
        """Degree of node i (for a path: ends=1, interior=2)."""
        return int(self.G.degree[int(i)])

    def neighbors(self, i: int) -> Set[int]:
        return set(self.G.neighbors(int(i)))

    def causal_cone(self, seed: int, depth: int) -> CausalCone:
        visited = {int(seed)}
        frontier = {int(seed)}
        for _ in range(int(depth)):
            nxt = set().union(*[self.neighbors(v) for v in frontier]) if frontier else set()
            nxt -= visited
            visited |= nxt
            frontier = nxt
            if not frontier:
                break
        return CausalCone(nodes=visited)

    # simple entropy proxy on a region = |cut| * log(d)
    def minimal_cut_size(self, region: Iterable[int]) -> int:
        A = set(int(x) for x in region)
        if not A or len(A) == self._n:
            return 0
        # count edges crossing (A, Ac):
        cut = 0
        for u, v in self.G.edges():
            if (u in A) ^ (v in A):
                cut += 1
        return cut

    def entropy_of(self, region: Iterable[int]) -> EntropyResult:
        k = self.minimal_cut_size(region)
        s_nats = k * float(np.log(self.d))
        s_bits = k * float(np.log2(self.d))
        return EntropyResult(set(int(x) for x in region), k, self.d, s_nats, s_bits)
