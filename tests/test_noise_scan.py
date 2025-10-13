import numpy as np
from toe.qca import SplitStepQCA1D, energy_conservation_proxy

def test_noise_drift_monotonic():
    qca = SplitStepQCA1D(n_sites=6, d=2, seed=1)
    psi0 = np.zeros(2**6, dtype=complex)
    psi0[0] = 1.0
    drifts = []
    for p in [0.0, 0.05, 0.1, 0.2]:
        norms = energy_conservation_proxy(qca, psi0, steps=5, p=p)
        drifts.append(np.mean(np.abs(norms - norms[0])))
    assert drifts[0] < drifts[-1], "Drift should increase with noise."
