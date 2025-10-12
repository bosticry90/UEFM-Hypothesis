# examples/lr_operator_spread.py
from __future__ import annotations
import numpy as np
from pathlib import Path
from toe.qca import SplitStepQCA1D

def lightcone_radius(qca: SplitStepQCA1D, steps: int) -> int:
    # Cheap proxy: each step expands by at most 1 site (local update),
    # so radius ≤ steps. We also offer qca.lightcone_radius if present.
    if hasattr(qca, "lightcone_radius"):
        return int(qca.lightcone_radius(steps))
    return int(steps)

def main(N=200, steps_list=(5,10,15,20), theta=(0.25,0.33)):
    qca = SplitStepQCA1D(n_sites=N, d=2, seed=42, theta1=theta[0], theta2=theta[1])
    rows = []
    for T in steps_list:
        R = lightcone_radius(qca, T)
        rows.append((N, T, R, R/T))
    out = Path("outputs/lr_radius.tsv")
    np.savetxt(out, np.array(rows, float), fmt="%.6f", delimiter="\t",
               header="N\tsteps\tradius\tv_eff")
    print("Wrote", out)

if __name__ == "__main__":
    main()
