#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fit Lieb–Robinson radius decay vs noise:
Assumes R(T,p) ≈ exp(-α p T).  For each p>0, fit ln R vs T; slope = -α p ⇒ α = -slope/p.
Inputs : outputs/lr_noise_fit.tsv    (from examples/lr_noise_sweep.py)
Outputs: outputs/phase3_lr_fit.tsv
         outputs/figs/lr_noise_alpha_fit.png
"""
from __future__ import annotations
import os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

IN = Path("outputs/lr_noise_fit.tsv")
OUT_TSV = Path("outputs/phase3_lr_fit.tsv")
OUT_FIG = Path("outputs/figs/lr_noise_alpha_fit.png")

def _ensure_dirs():
    OUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

def main():
    if not IN.exists():
        raise FileNotFoundError(f"Missing {IN}. Run: python examples/lr_noise_sweep.py")

    df = pd.read_csv(IN, sep="\t")
    # Be tolerant to different column namings
    # Expect columns like: p, T, R (or 'radius', 'R_mean'); take the column that exists.
    Tcol = "T" if "T" in df.columns else ("steps" if "steps" in df.columns else None)
    Rcand = [c for c in ("R","radius","R_mean","R_med") if c in df.columns]
    if Tcol is None or not Rcand or "p" not in df.columns:
        raise ValueError(f"{IN} must contain columns p, T/steps and R/radius/R_mean/R_med")
    Rcol = Rcand[0]

    rows = []
    plt.figure()
    for p, grp in df.groupby("p"):
        if p == 0 or p == 0.0:
            continue
        T = grp[Tcol].to_numpy()
        R = grp[Rcol].to_numpy()
        # keep positive R
        mask = R > 0
        T, R = T[mask], R[mask]
        if len(T) < 2: 
            continue
        y = np.log(R)
        # linear fit y = a + b T
        A = np.vstack([np.ones_like(T), T]).T
        b0, b1 = np.linalg.lstsq(A, y, rcond=None)[0]  # y ≈ b0 + b1 T
        alpha_p = -b1 / float(p)
        # stderr of slope
        residuals = y - (b0 + b1*T)
        s2 = (residuals @ residuals) / max(1, len(T)-2)
        var_b1 = s2 / ( (T - T.mean())**2 ).sum()
        stderr = math.sqrt(max(var_b1, 0.0)) / abs(p) if var_b1 == var_b1 else np.nan

        rows.append({"p": p, "alpha_p": alpha_p, "slope": b1, "stderr_alpha": stderr})

        # plot
        Tgrid = np.linspace(T.min(), T.max(), 100)
        plt.plot(T, R, "o", label=f"p={p:.2g}")
        plt.plot(Tgrid, np.exp(b0 + b1*Tgrid), "-")

    if not rows:
        raise RuntimeError("No usable (p>0) groups found for fitting α.")

    fit_df = pd.DataFrame(rows).sort_values("p")
    # robust aggregate
    alpha_med = fit_df["alpha_p"].median()
    alpha_avg = fit_df["alpha_p"].mean()
    fit_df.attrs["alpha_hat_med"] = float(alpha_med)
    fit_df.attrs["alpha_hat_avg"] = float(alpha_avg)

    # write TSV (append header row with aggregate as comments)
    _ensure_dirs()
    with OUT_TSV.open("w", encoding="utf-8") as f:
        f.write(f"# alpha_hat_med\t{alpha_med:.6g}\n")
        f.write(f"# alpha_hat_avg\t{alpha_avg:.6g}\n")
        fit_df.to_csv(f, sep="\t", index=False)

    plt.yscale("log")
    plt.xlabel("Steps T")
    plt.ylabel("Lieb–Robinson radius R(T,p)")
    plt.title(f"LR decay vs noise: α_med={alpha_med:.3g}, α_avg={alpha_avg:.3g}")
    plt.legend(loc="best", ncols=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=180)
    print(f"Wrote {OUT_TSV} and {OUT_FIG}")
    print(f"alpha_hat_med={alpha_med:.6g} alpha_hat_avg={alpha_avg:.6g}")

if __name__ == "__main__":
    main()
