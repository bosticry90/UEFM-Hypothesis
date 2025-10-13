# examples/tensor_coherence_benchmark.py
import numpy as np
from toe.qca import SplitStepQCA1D, energy_conservation_proxy, apply_qca_steps
from toe.tensor_entropy import tensor_rank_entropy, schmidt_entropy_by_cuts
from pathlib import Path

def main():
    out_path = Path("outputs")
    out_path.mkdir(exist_ok=True)
    tsv = out_path / "tensor_coherence_timeseries.tsv"

    N = 16
    qca = SplitStepQCA1D(n_sites=N, d=2, seed=5, theta1=0.22, theta2=0.31)
    psi0 = np.zeros(2**N, dtype=complex); psi0[0] = 1.0  # product state

    rows = []
    T = 24
    psi = psi0.copy()
    for t in range(T+1):
        # tensor entropies (natural log)
        S_mean = tensor_rank_entropy(psi, local_dim=2, n_sites=N, base=np.e, reduction="mean")
        S_max  = tensor_rank_entropy(psi, local_dim=2, n_sites=N, base=np.e, reduction="max")
        cuts   = schmidt_entropy_by_cuts(psi, local_dim=2, n_sites=N, base=np.e)
        rows.append((t, S_mean, S_max, np.mean(cuts[:N//2])))

        # advance
        if t < T:
            psi = apply_qca_steps(qca, psi, steps=1)

    # save
    with tsv.open("w", encoding="utf-8") as f:
        f.write("t\tS_mean\tS_max\tS_halfmean\n")
        for r in rows:
            f.write(f"{r[0]}\t{r[1]:.9f}\t{r[2]:.9f}\t{r[3]:.9f}\n")

    print(f"Wrote {tsv}")

if __name__ == "__main__":
    main()
