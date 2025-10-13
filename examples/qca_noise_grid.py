# examples/qca_noise_grid.py
import numpy as np
from toe.qca import SplitStepQCA1D, energy_conservation_proxy
from pathlib import Path

def main():
    out = Path("outputs"); out.mkdir(exist_ok=True)
    qca = SplitStepQCA1D(n_sites=8, d=2, seed=5)
    psi0 = np.zeros(2**8, dtype=complex); psi0[0] = 1.0

    steps_list = [8, 16, 32]
    p_list = [0.0, 0.02, 0.05, 0.1, 0.2]

    rows = ["steps\tp\tmax_drift"]
    for T in steps_list:
        for p in p_list:
            norms = energy_conservation_proxy(qca, psi0, steps=T, p=p)
            drift = np.max(np.abs(norms - norms[0]))
            rows.append(f"{T}\t{p}\t{drift:.6e}")

    out_file = out / "qca_noise_grid.tsv"
    out_file.write_text("\n".join(rows), encoding="utf-8")
    print(f"Wrote {out_file}")

if __name__ == "__main__":
    main()
