import numpy as np
from pathlib import Path
from toe import SplitStepQCA1D, energy_conservation_proxy

def main(N=32, steps_list=(10, 50, 100), seeds=(0, 1, 2)):
    """
    Sweep over coin angles (theta1, theta2) and measure the
    maximum L2-norm drift of the split-step QCA.
    Results are written to outputs/qca_drift_grid.tsv
    """
    thetas = np.linspace(0.0, 0.6, 7)
    rows = []

    for theta1 in thetas:
        for theta2 in thetas:
            for steps in steps_list:
                drifts = []
                for seed in seeds:
                    qca = SplitStepQCA1D(N, theta1=theta1, theta2=theta2)
                    rng = np.random.default_rng(seed)

                    # localized spinor excitation at center
                    psi0 = np.zeros((N, 2), dtype=complex)
                    psi0[N // 2, 0] = 1.0

                    drift = energy_conservation_proxy(qca, psi0, steps=steps)
                    drifts.append(drift)

                rows.append((theta1, theta2, steps, float(np.max(drifts))))

    Path("outputs").mkdir(exist_ok=True)
    out = np.array(rows, dtype=float)
    np.savetxt(
        "outputs/qca_drift_grid.tsv",
        out,
        fmt="%.6e",
        header="theta1 theta2 steps max_norm_drift"
    )
    print("Wrote outputs/qca_drift_grid.tsv")

if __name__ == "__main__":
    main()
