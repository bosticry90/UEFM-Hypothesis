
import numpy as np

class Substrate:
    """Finite quantum substrate on a simple graph (adjacency) with on-site qudits."""

    def __init__(self, n_sites: int, d: int, edges):
        self.n = int(n_sites)
        self.d = int(d)
        self.edges = set(tuple(sorted(e)) for e in edges)
        self.adj = {i: set() for i in range(self.n)}
        for i, j in self.edges:
            self.adj[i].add(j); self.adj[j].add(i)

    @classmethod
    def line(cls, n_sites: int, d: int = 2):
        edges = [(i, i+1) for i in range(n_sites-1)]
        return cls(n_sites, d, edges)

    def degree(self, i: int) -> int:
        return len(self.adj[i])

    def neighbors(self, i: int):
        return sorted(self.adj[i])

    def graph_distance(self, a: int, b: int) -> int:
        if a==b: return 0
        from collections import deque
        seen={a}; q=deque([(a,0)])
        while q:
            u, d = q.popleft()
            for v in self.adj[u]:
                if v==b: return d+1
                if v not in seen:
                    seen.add(v); q.append((v, d+1))
        return np.inf
