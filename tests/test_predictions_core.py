# tests/test_predictions_core.py
import numpy as np
from toe.qca import SplitStepQCA1D, energy_conservation_proxy

def test_lr_lightcone_bound():
    N, T = 101, 20
    qca = SplitStepQCA1D(n_sites=N, d=2, seed=11, theta1=0.3, theta2=0.34)
    radius = qca.lightcone_radius(steps=T)
    v_eff = radius / float(T)
    assert v_eff <= 1.05  # small numerical slack

def test_noise_drift_monotone_smallgrid():
    qca = SplitStepQCA1D(n_sites=6, d=2, seed=1)
    psi0 = np.zeros(2**6, dtype=complex); psi0[0] = 1.0
    drifts = []
    for p in [0.0, 0.05, 0.1, 0.2]:
        norms = energy_conservation_proxy(qca, psi0, steps=6, p=p)
        drift = float(np.max(np.abs(norms - norms[0])))
        drifts.append(drift)
    for a, b in zip(drifts, drifts[1:]):
        assert b >= a - 1e-12
