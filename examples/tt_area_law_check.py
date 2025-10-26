# examples/tt_area_law_check.py
"""
Area-law proxy test: TT rank vs number of interfaces.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from toe.tensor_tt_core import tt_svd

def make_field(n=64, n_interfaces=1):
    x = np.linspace(-3, 3, n)
    phi = np.zeros_like(x)
    splits = np.linspace(0, n, n_interfaces + 2, dtype=int)[1:-1]
    sign = 1
    start = 0
    for s in list(splits) + [n]:
        phi[start:s] = sign * np.tanh(x[start:s])
        sign *= -1
        start = s
    return phi

records = []
for m in [1, 2, 4, 8]:
    phi = make_field(128, m)
    # ensure tensor has ≥2 dims so TT-SVD is defined
    phi_2d = phi.reshape(16, 8)
    tt = tt_svd(phi_2d, rank=8)
    ranks = getattr(tt, "rank", [1])
    max_r = np.max(ranks)
    records.append(dict(n_interfaces=m, max_rank=max_r))

df = pd.DataFrame(records)
df.to_csv("outputs/tt_area_law.tsv", sep="\t", index=False)

plt.figure()
plt.plot(df.n_interfaces, df.max_rank, "o-")
plt.xlabel("Interfaces (area proxy)")
plt.ylabel("Max TT rank")
plt.title("Area-law check via TT ranks")
plt.tight_layout()
plt.savefig("outputs/figs/rank_vs_cut.png", dpi=150)
print("Wrote outputs/tt_area_law.tsv and outputs/figs/rank_vs_cut.png")
