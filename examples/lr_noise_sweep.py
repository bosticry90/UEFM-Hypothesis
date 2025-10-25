# examples/lr_noise_sweep.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from toe.qca import SplitStepQCA1D, energy_conservation_proxy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = OUT / "figs"
OUT.mkdir(exist_ok=True, parents=True)
FIG.mkdir(exist_ok=True, parents=True)

def drift_proxy(qca: SplitStepQCA1D, psi0: np.ndarray, steps: int, p: float) -> float:
    """Max absolute norm drift over the trajectory vs t=0 baseline."""
    norms = energy_conservation_proxy(qca, psi0, steps=steps, p=p)
    return float(np.max(np.abs(norms - norms[0])))

def main():
    # Noise grid + robust averaging across multiple random seeds.
    p_grid = [0.00, 0.05, 0.10, 0.20]
    n_sites = 6
    steps = 10

    # Use several independent QCA seeds and different coin angles per seed to reduce variance.
    seeds = list(range(1, 11))  # 10 seeds for stability without being too slow

    rows = []
    for p in p_grid:
        drifts = []
        for s in seeds:
            qca = SplitStepQCA1D(n_sites=n_sites, d=2, seed=s)
            psi0 = np.zeros(2**n_sites, dtype=complex); psi0[0] = 1.0
            dval = drift_proxy(qca, psi0, steps=steps, p=p)
            drifts.append(dval)
        drifts = np.asarray(drifts, dtype=float)
        rows.append({
            "p": p,
            "drift_mean": float(np.mean(drifts)),
            "drift_std": float(np.std(drifts, ddof=1) if drifts.size > 1 else 0.0),
            "drift_median": float(np.median(drifts)),
            "drift_q25": float(np.quantile(drifts, 0.25)),
            "drift_q75": float(np.quantile(drifts, 0.75)),
            "n_seeds": len(seeds),
            "steps": steps,
            "n_sites": n_sites,
        })

    df = pd.DataFrame(rows)
    out_tsv = OUT / "lr_noise_fit.tsv"
    df.to_csv(out_tsv, index=False)
    print(f"Wrote {out_tsv}")

    # Plot mean with error bars (std). Keep simple so CI/phones are happy.
    plt.figure()
    plt.errorbar(df["p"].values, df["drift_mean"].values, yerr=df["drift_std"].values, fmt="o-", capsize=3)
    plt.xlabel("noise probability p")
    plt.ylabel("norm drift (mean ± std)")
    plt.title("Noise → Drift (seed-averaged)")
    fig_path = FIG / "lr_noise_fit.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    print(f"Wrote {fig_path}")

if __name__ == "__main__":
    main()
