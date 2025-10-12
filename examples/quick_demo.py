# examples/quick_demo.py

import numpy as np

from toe import (
    Substrate,
    SplitStepQCA1D,
    energy_conservation_proxy,
    random_isometry,
    erasure_errors,
    knill_laflamme_checks,
)

def main():
    # --- 1) Entropy proxy on a ring (TN/QEC picture) ---
    S = Substrate.ring(n=12, d=4)  # 12-site ring, bond dimension χ = 4
    region = set(range(3))          # take a boundary arc of length 3
    ent = S.entropy_of(region)
    print(f"Entropy proxy S(A) bits: {ent.s_bits}")

    # --- 2) QCA norm conservation (unitarity sanity check) ---
    qca = SplitStepQCA1D(n_sites=6, d=2, seed=42)
    psi0 = np.zeros(2**6, dtype=complex); psi0[0] = 1.0
    norms = energy_conservation_proxy(psi0, qca, steps=50)
    max_drift = float(np.max(np.abs(norms - norms[0])))
    print(f"Max norm drift over 50 steps: {max_drift:.3e}")

    # --- 3) Simple QEC KL check with erasure-like errors ---
    V = random_isometry(d_logical=2, d_physical=8, seed=7)  # encode 1 logical qubit into 3 physical qubits
    E = erasure_errors(n_physical=3, k=1, local_dim=2)      # up to 1-site erasures on 3-qubit register
    ok, C = knill_laflamme_checks(V, E, atol=1e-6)          # C is the (a,b) overlap matrix
    offdiag = np.max(np.abs(C - np.diag(np.diag(C))))
    print(f"KL holds? {ok} | max off-diagonal overlap: {offdiag:.3e}")

if __name__ == "__main__":
    main()
