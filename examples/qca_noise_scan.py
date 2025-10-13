import numpy as np
import pandas as pd
from toe.qca import SplitStepQCA1D, energy_conservation_proxy

def main():
    n_sites = 8
    d = 2
    steps = 20
    seed = 7
    qca = SplitStepQCA1D(n_sites=n_sites, d=d, seed=seed)
    psi0 = np.zeros(2**n_sites, dtype=complex)
    psi0[0] = 1.0

    ps = np.linspace(0, 0.2, 11)
    records = []
    for p in ps:
        norms = energy_conservation_proxy(qca, psi0, steps=steps, p=p)
        drift = np.abs(norms - norms[0]).mean()
        records.append({"p": p, "mean_drift": drift})

    df = pd.DataFrame(records)
    df.to_csv("outputs/noise_drift.tsv", sep="\t", index=False)
    print("Wrote outputs/noise_drift.tsv")

if __name__ == "__main__":
    main()
