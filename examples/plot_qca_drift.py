# examples/plot_qca_drift.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

IN = "outputs/qca_drift_grid.tsv"

def load_grid(path=IN):
    # columns: theta1 theta2 steps max_norm_drift
    data = np.loadtxt(path, skiprows=1)
    thetas1 = np.unique(data[:, 0])
    thetas2 = np.unique(data[:, 1])
    steps   = np.unique(data[:, 2]).astype(int)
    # shape: (len(steps), len(thetas1), len(thetas2))
    grids = {}
    for s in steps:
        sub = data[data[:, 2] == s]
        Z = np.zeros((len(thetas1), len(thetas2)))
        for t1_idx, t1 in enumerate(thetas1):
            row = sub[sub[:, 0] == t1]
            # row columns: theta1, theta2, steps, drift
            Z[t1_idx, :] = row[:, 3]
        grids[s] = (thetas1, thetas2, Z)
    return grids

def main():
    Path("outputs").mkdir(exist_ok=True)
    grids = load_grid(IN)
    for s, (t1, t2, Z) in grids.items():
        plt.figure()
        im = plt.imshow(
            Z.T, origin="lower",
            extent=(t1.min(), t1.max(), t2.min(), t2.max()),
            aspect="auto"
        )
        plt.colorbar(im, label="max norm drift")
        plt.xlabel(r"$\theta_1$")
        plt.ylabel(r"$\theta_2$")
        plt.title(f"Split-step QCA max norm drift (steps={s})")
        out = f"outputs/qca_drift_heatmap_steps_{s}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=180)
        print(f"Wrote {out}")

if __name__ == "__main__":
    main()
