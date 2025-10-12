# examples/random_graph_area_law.py
import numpy as np
import networkx as nx
from pathlib import Path
from toe.substrate import Substrate

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

def random_connected_gnp(n, p, seed=0):
    rng = np.random.default_rng(seed)
    while True:
        G = nx.gnp_random_graph(n, p, seed=int(rng.integers(1, 1_000_000)))
        if nx.is_connected(G):
            return G

def scan(n_list=(32, 64, 96), p_list=(0.05, 0.08, 0.12), chi_list=(2,3,4)):
    lines = ["n p chi region_frac cut_size s_bits"]
    for n in n_list:
        for p in p_list:
            G = random_connected_gnp(n, p, seed=42+n)
            for chi in chi_list:
                S = Substrate.from_graph(G, d=2, bond_dim=chi)
                for frac in np.linspace(0.05, 0.5, 10):
                    m = max(1, int(frac * n))
                    A = set(range(m))  # contiguous label set as a proxy; labels are arbitrary
                    cut = S.minimal_cut_size(A)
                    s_bits = cut * np.log2(chi)
                    lines.append(f"{n} {p:.3f} {chi} {frac:.3f} {cut} {s_bits:.6f}")
    return lines

def main():
    lines = scan()
    out = OUT / "random_graph_area_law.tsv"
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
