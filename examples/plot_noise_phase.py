# examples/plot_noise_phase.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    inp = Path("outputs/qca_noise_grid.tsv")
    if not inp.exists():
        raise SystemExit(f"Missing {inp}. Run examples/qca_noise_grid.py first.")

    rows = [r.strip().split("\t") for r in inp.read_text(encoding="utf-8").splitlines() if r.strip()]
    header, data = rows[0], rows[1:]
    steps = np.array([int(r[0]) for r in data])
    ps    = np.array([float(r[1]) for r in data])
    drift = np.array([float(r[2]) for r in data])

    steps_uniq = sorted(set(steps))
    p_uniq = sorted(set(ps))

    # Make output dir
    figs = Path("outputs/figs"); figs.mkdir(parents=True, exist_ok=True)

    # Line plot: drift vs p, one curve per steps
    plt.figure()
    for T in steps_uniq:
        mask = steps == T
        pvals = [x for _, x in sorted(zip(ps[mask], ps[mask]))]
        dvals = [x for _, x in sorted(zip(ps[mask], drift[mask]))]
        plt.plot(pvals, dvals, marker="o", label=f"steps={T}")
    plt.xlabel("noise probability p")
    plt.ylabel("max norm drift")
    plt.title("QCA noise robustness")
    plt.legend()
    png = figs / "noise_phase.png"
    svg = figs / "noise_phase.svg"
    plt.tight_layout()
    plt.savefig(png, dpi=160)
    plt.savefig(svg)
    print(f"Saved {png} and {svg}")

if __name__ == "__main__":
    main()
