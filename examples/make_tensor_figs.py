# examples/make_tensor_figs.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IN_TSV  = os.path.join("outputs", "uefm_tt_prototype.tsv")
OUT_DIR = os.path.join("outputs", "figs")
os.makedirs(OUT_DIR, exist_ok=True)

def _read_tsv(path: str) -> pd.DataFrame:
    """
    Robustly read the prototype TSV (handles comma/whitespace/TSV).
    Expected columns (case-insensitive subset ok):
      rank, rel_l2, coh, coh_relerr
    """
    # Try pandas auto-delimiter
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        # Fallback: whitespace
        df = pd.read_csv(path, delim_whitespace=True, engine="python")

    # Normalize expected column names
    rename_map = {}
    for c in df.columns:
        lc = c.strip().lower()
        if lc in {"rank"}:
            rename_map[c] = "rank"
        elif lc in {"rel_l2", "rell2", "rel_l2_error", "reconstruction_error"}:
            rename_map[c] = "rel_l2"
        elif lc in {"coh", "coherence"}:
            rename_map[c] = "coh"
        elif lc in {"coh_relerr", "coherence_relerr", "coh_rel_error"}:
            rename_map[c] = "coh_relerr"
    df = df.rename(columns=rename_map)

    # Basic sanity
    needed = {"rank", "rel_l2"}
    if not needed.issubset(set(df.columns)):
        raise ValueError(f"Input file missing required columns {needed}. Found: {df.columns.tolist()}")

    # Sort by rank just in case
    df = df.sort_values("rank").reset_index(drop=True)
    return df

def _save(fig, path_png: str, path_svg: str):
    fig.tight_layout()
    fig.savefig(path_png, dpi=160, bbox_inches="tight")
    fig.savefig(path_svg, bbox_inches="tight")
    plt.close(fig)

def main():
    df = _read_tsv(IN_TSV)

    # Figure 1: Rank vs Reconstruction Error (||φ - TT|| / ||φ||)
    fig1 = plt.figure(figsize=(5.2, 3.4))
    ax1 = fig1.add_subplot(111)
    ax1.plot(df["rank"].to_numpy(), df["rel_l2"].to_numpy(), marker="o")
    ax1.set_xlabel("TT rank")
    ax1.set_ylabel("Relative reconstruction error (Frobenius)")
    ax1.set_title("Tensor-Train rank vs reconstruction error")
    ax1.grid(True, alpha=0.25)
    _save(fig1,
          os.path.join(OUT_DIR, "tensor_rank_vs_relL2.png"),
          os.path.join(OUT_DIR, "tensor_rank_vs_relL2.svg"))

    # Figure 2: Rank vs Coherence relative error, if available
    if "coh_relerr" in df.columns:
        fig2 = plt.figure(figsize=(5.2, 3.4))
        ax2 = fig2.add_subplot(111)
        ax2.plot(df["rank"].to_numpy(), df["coh_relerr"].to_numpy(), marker="o")
        ax2.set_xlabel("TT rank")
        ax2.set_ylabel("Relative error in coherence functional")
        ax2.set_title("Rank vs coherence functional error")
        ax2.grid(True, alpha=0.25)
        _save(fig2,
              os.path.join(OUT_DIR, "tensor_rank_vs_cohRelErr.png"),
              os.path.join(OUT_DIR, "tensor_rank_vs_cohRelErr.svg"))
        print("Wrote figures:",
              os.path.join(OUT_DIR, "tensor_rank_vs_relL2.png"),
              os.path.join(OUT_DIR, "tensor_rank_vs_cohRelErr.png"))
    else:
        print("Wrote figure:",
              os.path.join(OUT_DIR, "tensor_rank_vs_relL2.png"),
              "(coh_relerr column not found; skipped second figure)")

if __name__ == "__main__":
    main()
