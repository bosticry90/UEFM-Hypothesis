# toe/substrate.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, Any, Dict, Tuple, Set, Optional
import numpy as np
import networkx as nx

LocalRule = Callable[[np.ndarray], np.ndarray]

@dataclass(frozen=True)
class CausalCone:
    """Set of nodes possibly influenced by a site after t steps."""
    nodes: Set[int]

@dataclass(frozen=True)
class EntropyResult:
    """Simple placeholder return for region entropy / minimal cut."""
    region: Set[int]
    minimal_cut_size: int
    bond_dim: int
    s_bits: float

class Substrate:
    """
    Minimal TN/QEC/QCA substrate scaffold:
    - Graph G: nodes are sites, edges are entanglement links
    - bond_dim χ on each edge (uniform for now)
    - step(): applies a local reversible update (QCA-like) to a register
    - entropy_of(): area-law proxy via minimal cut
    """

    def __init__(self, G: nx.Graph | None = None, bond_dim: int = 2, d: int = 2):
        self.G = G if G is not None else nx.path_graph(8)
        self.bond_dim = int(bond_dim)
        self.d = int(d)
        if self.bond_dim < 2:
            raise ValueError("bond_dim must be ≥ 2")
        # placeholder register
        self.register = np.zeros((self.G.number_of_nodes(),), dtype=np.complex128)

    # ---------- constructors ----------
    @classmethod
    def line(cls, n: int, d: int = 2, bond_dim: int = 2):
        """Open chain of n sites."""
        G = nx.path_graph(int(n))
        return cls(G=G, bond_dim=bond_dim, d=d)

    @classmethod
    def ring(cls, n: int, d: int = 2, bond_dim: int = 2):
        """Ring (cycle) of n sites."""
        G = nx.cycle_graph(int(n))
        return cls(G=G, bond_dim=bond_dim, d=d)

    @classmethod
    def from_graph(cls, G: nx.Graph, d: int = 2, bond_dim: int = 2):
        return cls(G=G.copy(), bond_dim=bond_dim, d=d)

    # ---------- Locality / causality helpers ----------
    def neighbors(self, i: int) -> Set[int]:
        return set(self.G.neighbors(i))

    def degree(self, i: int) -> int:
        return int(self.G.degree[i])

    def graph_distance(self, i: int, j: int) -> int:
        return int(nx.shortest_path_length(self.G, int(i), int(j)))

    def causal_cone(self, seed: int, depth: int) -> CausalCone:
        nodes = {seed}
        frontier = {seed}
        for _ in range(int(depth)):
            nxt = set().union(*[self.neighbors(v) for v in frontier]) if frontier else set()
            frontier = nxt - nodes
            nodes |= frontier
            if not frontier:
                break
        return CausalCone(nodes)

    # ---------- QCA-like local update (toy) ----------
    def step(self, rule: LocalRule, sites: Optional[Iterable[int]] = None) -> None:
        if sites is None:
            sites = range(self.G.number_of_nodes())
        new_reg = self.register.copy()
        for i in sites:
            nbrs = sorted(self.neighbors(i))
            patch_idx = [i] + nbrs
            patch = self.register[patch_idx]
            out = rule(patch)
            if np.ndim(out) == 0:
                new_reg[i] = out
            else:
                new_reg[i] = np.asarray(out).ravel()[0]
        self.register = new_reg

    # ---------- Minimal cut / entropy proxy ----------
    def minimal_cut_size(self, region: Iterable[int]) -> int:
        A = set(region)
        if not A or A == set(self.G.nodes):
            return 0
        H = nx.DiGraph()
        for u, v in self.G.edges:
            H.add_edge(u, v, capacity=1.0)
            H.add_edge(v, u, capacity=1.0)
        SRC, SNK = "_src", "_snk"
        for a in A:
            H.add_edge(SRC, a, capacity=float("inf"))
        for b in set(self.G.nodes) - A:
            H.add_edge(b, SNK, capacity=float("inf"))
        cut_value, _ = nx.minimum_cut(H, SRC, SNK)
        return int(cut_value)

    def entropy_of(self, region: Iterable[int]) -> EntropyResult:
        k = self.minimal_cut_size(region)
        s_bits = k * np.log2(self.bond_dim)
        return EntropyResult(region=set(region), minimal_cut_size=k, bond_dim=self.bond_dim, s_bits=float(s_bits))
