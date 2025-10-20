# examples/fit_tt_area_beta.py
# Robust fitter for "area-law" style rank ~ boundary^beta on log-log scale.
# Accepts multiple column name aliases so different generators work.

from __future__ import annotations
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IN = os.path.join("outputs", "tt_area_law.tsv")
OUT_TSV = os.path.join("outputs", "phase3_tt_area_fit.tsv")
OUT_PNG = os.path.join("outputs", "figs", "tt_area_beta_fit.png")

BOUNDARY_ALIASES = ["cut", "boundary", "boundary_size", "b", "bond_cut", "size", "n_interfaces"]
RANK_ALIASES = ["rank_max", "rank", "chi_max", "chi", "tt_rank", "r", "max_rank"]


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        key = name.lower()
        if key in cols_lower:
            return cols_lower[key]
    return None


def _safe_log(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mask = x > 0
    if not np.all(mask):
        # keep only positive entries (log defined)
        x = x[mask]
    return np.log(x), mask if x.size else (x, mask)


def fit_beta(boundary: np.ndarray, rank: np.ndarray):
    # Keep only entries with boundary>0 and rank>0
    boundary = np.asarray(boundary, dtype=float)
    rank = np.asarray(rank, dtype=float)
    mask = (boundary > 0) & (rank > 0) & np.isfinite(boundary) & np.isfinite(rank)
    if mask.sum() < 2:
        raise ValueError("Not enough positive/finite samples for a log-log fit.")
    b = boundary[mask]
    r = rank[mask]
    logb = np.log(b)
    logr = np.log(r)

    # Linear fit: logr = beta * logb + c
    beta, intercept = np.polyfit(logb, logr, 1)
    # R^2
    pred = beta * logb + intercept
    ss_res = np.sum((logr - pred) ** 2)
    ss_tot = np.sum((logr - np.mean(logr)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(beta), float(intercept), float(r2), (logb, logr, pred)


def main():
    if not os.path.exists(IN):
        raise FileNotFoundError(f"{IN} not found. Run examples/tt_area_law_check.py first.")

    os.makedirs(os.path.dirname(OUT_TSV), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)

    df = pd.read_csv(IN, sep=None, engine="python")
    b_col = _pick_col(df, BOUNDARY_ALIASES)
    r_col = _pick_col(df, RANK_ALIASES)

    if b_col is None or r_col is None:
        raise ValueError(
            f"{IN} must contain a boundary-size column (one of {BOUNDARY_ALIASES}) "
            f"and a rank column (one of {RANK_ALIASES}). Found columns: {list(df.columns)}"
        )

    boundary = df[b_col].to_numpy()
    rank = df[r_col].to_numpy()

    beta, intercept, r2, (logb, logr, pred) = fit_beta(boundary, rank)

    # Save TSV summary
    out_df = pd.DataFrame(
        {
            "beta_hat": [beta],
            "intercept": [intercept],
            "r2": [r2],
            "n_samples": [int(len(logb))],
            "boundary_col": [b_col],
            "rank_col": [r_col],
        }
    )
    out_df.to_csv(OUT_TSV, index=False, sep="\t")

    # Plot
    plt.figure()
    plt.scatter(logb, logr, s=24, label="data")
    # For a clean line, sort by logb
    order = np.argsort(logb)
    plt.plot(logb[order], pred[order], label=f"fit: β={beta:.3f}, $R^2$={r2:.3f}")
    plt.xlabel("log(boundary size)")
    plt.ylabel("log(rank)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=160)
    plt.close()

    print(
        f"Wrote {OUT_TSV} and {OUT_PNG}\n"
        f"beta_hat={beta:.6f}  r2={r2:.3f}  (using boundary='{b_col}', rank='{r_col}')"
    )


if __name__ == "__main__":
    main()
