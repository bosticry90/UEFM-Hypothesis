import numpy as np
from toe.qca import SplitStepQCA1D, energy_conservation_proxy

def test_lr_velocity_small():
    # small N to keep state vectors tiny
    qca = SplitStepQCA1D(n_sites=8, d=2, seed=7, theta1=0.21, theta2=0.33)
    psi0 = np.zeros(2**8, dtype=complex); psi0[0] = 1.0
    drift = energy_conservation_proxy(qca, psi0, steps=12)
    assert drift < 1e-12
