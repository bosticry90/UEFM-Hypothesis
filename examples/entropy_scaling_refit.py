#!/usr/bin/env python3
"""
Phase 3.5 — Entropy/area-law cross-scaling refit
Refits log(rank) vs log(boundary) from existing tt_area_law.tsv
(compatible with columns you've produced: 'n_interfaces' and 'max_rank').

Outputs:
  - outputs/phase3p5_entropy_fit.tsv
  - outputs/figs/tt_entropy_refit.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IN = "outputs/tt_area_law.tsv"
OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

def main():
    if not os.path.exists(IN):
        raise FileNotFoundError(f"Missing {IN}. Run examples/tt_area_law_check.py first.")
    df = pd.read_csv(IN, sep=None, engine="python")

    # Column resolution
    b_col = None
    for c in ["cut","boundary","boundary_size","b","bond_cut","size","n_interfaces"]:
        if c in df.columns:
            b_col = c; break
    r_col = None
    for c in ["rank_max","rank","chi_max","chi","tt_rank","r","max_rank"]:
        if c in df.columns:
            r_col = c; break
    if b_col is None or r_col is None:
        raise ValueError(f"{IN} missing boundary or rank columns. Found: {list(df.columns)}")

    df = df[[b_col, r_col]].dropna()
    df = df[(df[b_col] > 0) & (df[r_col] > 0)]
    x = np.log(df[b_col].to_numpy(dtype=float))
    y = np.log(df[r_col].to_numpy(dtype=float))

    A = np.vstack([x, np.ones_like(x)]).T
    beta, a0 = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = a0 + beta*x
    ss_res = float(np.sum((y-yhat)**2))
    ss_tot = float(np.sum((y-np.mean(y))**2))
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else float("nan")

    pd.DataFrame([dict(beta_hat=float(beta), intercept=float(a0), r2=r2,
                       n=int(len(x)), boundary_col=b_col, rank_col=r_col)]).to_csv(
        os.path.join(OUT_DIR, "phase3p5_entropy_fit.tsv"), sep="\t", index=False
    )

    # Plot
    plt.figure(figsize=(5.5,4))
    plt.scatter(x, y, s=16, alpha=0.7, label="data")
    xs = np.linspace(x.min(), x.max(), 200)
    plt.plot(xs, a0 + beta*xs, label=fr"fit: $\beta={beta:.3f}$, $R^2={r2:.3f}$")
    plt.xlabel(f"log({b_col})"); plt.ylabel(f"log({r_col})"); plt.legend()
    out_png = os.path.join(FIG_DIR, "tt_entropy_refit.png")
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"Wrote {os.path.join(OUT_DIR, 'phase3p5_entropy_fit.tsv')} and {out_png}")

if __name__ == "__main__":
    main()
