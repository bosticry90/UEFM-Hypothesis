# examples/entropy_scaling_suite.py
from __future__ import annotations
import itertools as it
import numpy as np
import networkx as nx
from pathlib import Path
from toe.substrate import Substrate

def scan_path(N=64, chis=(2,3,4), regions=(2,4,8,16,32), outdir=Path("outputs")):
    outdir.mkdir(parents=True, exist_ok=True)
    for chi in chis:
        S = Substrate.line(N, d=2, bond_dim=chi)
        rows = []
        for m in regions:
            s_bits = S.entropy_of(range(m)).s_bits
            rows.append((N, chi, m, s_bits))
        p = outdir / f"entropy_path_N{N}_chi{chi}.tsv"
        np.savetxt(p, np.array(rows, float), fmt="%.6f", delimiter="\t",
                   header="N\tchi\tregion_size\ts_bits")
        print("Wrote", p)

def scan_ring(N=64, chis=(2,3,4), regions=(2,4,8,16,32), outdir=Path("outputs")):
    outdir.mkdir(parents=True, exist_ok=True)
    for chi in chis:
        S = Substrate.ring(N, d=2, bond_dim=chi)
        rows = []
        for m in regions:
            s_bits = S.entropy_of(range(m)).s_bits
            rows.append((N, chi, m, s_bits))
        p = outdir / f"entropy_ring_N{N}_chi{chi}.tsv"
        np.savetxt(p, np.array(rows, float), fmt="%.6f", delimiter="\t",
                   header="N\tchi\tregion_size\ts_bits")
        print("Wrote", p)

def scan_random_graphs(n=64, pvals=(0.02, 0.05, 0.08), chis=(2,3), samples=5, A_sizes=(4,8,12,16), outdir=Path("outputs")):
    outdir.mkdir(parents=True, exist_ok=True)
    for chi in chis:
        rows = []
        for p in pvals:
            for s in range(samples):
                G = nx.fast_gnp_random_graph(n, p, seed=1000 + s)
                S = Substrate(G, d=2, bond_dim=chi)
                for m in A_sizes:
                    A = range(m)
                    s_bits = S.entropy_of(A).s_bits
                    rows.append((n, chi, p, s, m, s_bits))
        pth = outdir / f"entropy_er_random_n{n}_chi{chi}.tsv"
        np.savetxt(pth, np.array(rows, float), fmt="%.6f", delimiter="\t",
                   header="n\tchi\tp\tsample\tregion_size\ts_bits")
        print("Wrote", pth)

def main():
    scan_path()
    scan_ring()
    scan_random_graphs()

if __name__ == "__main__":
    main()
