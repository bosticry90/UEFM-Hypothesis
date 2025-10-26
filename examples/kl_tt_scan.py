# examples/kl_tt_scan.py
"""
Random small codes: compare TT ranks of states that
pass vs fail Knill–Laflamme condition (toy version).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from toe.tensor_tt_core import tt_svd

def kl_check(state, noise_strength=0.05):
    pert = state + noise_strength * np.random.randn(*state.shape)
    return np.allclose(np.dot(state.conj(), pert), np.dot(state.conj(), state), atol=0.1)

rows = []
for trial in range(20):
    psi = np.random.randn(64)
    psi /= np.linalg.norm(psi)
    ok = kl_check(psi, noise_strength=np.random.uniform(0.02, 0.2))
    tt = tt_svd(psi.reshape(8, 8), rank=8)
    ranks = getattr(tt, "rank", [1])
    rows.append(dict(trial=trial, kl_ok=ok, max_rank=np.max(ranks)))

df = pd.DataFrame(rows)
df.to_csv("outputs/kl_tt_scan.tsv", sep="\t", index=False)
plt.figure()
for key, sub in df.groupby("kl_ok"):
    plt.hist(sub.max_rank, bins=np.arange(0, 20, 2), alpha=0.6, label=f"KL={key}")
plt.legend(); plt.xlabel("Max TT rank"); plt.ylabel("count")
plt.title("KL–TT compressibility")
plt.tight_layout()
plt.savefig("outputs/figs/kl_tt_scan.png", dpi=150)
print("Wrote outputs/kl_tt_scan.tsv and outputs/figs/kl_tt_scan.png")
