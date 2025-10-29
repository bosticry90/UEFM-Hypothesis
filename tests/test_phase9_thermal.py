# tests/test_phase9_thermal.py

import numpy as np
from examples.phase9_thermal_gl import SimParams, simulate, sweep, _step_tdgl

def test_kinetic_monotone_vs_sigma():
    rows = sweep(sigmas=(0.0, 0.02, 0.05, 0.1))
    Ks = [r["final_kinetic"] for r in rows]
    assert Ks[1] >= Ks[0] * 0.98
    assert Ks[2] >= Ks[1] * 0.98
    assert Ks[3] >= Ks[2] * 0.98

def test_coherence_decreases_with_sigma():
    rows = sweep(sigmas=(0.0, 0.02, 0.05, 0.1))
    Cs = [r["final_coherence"] for r in rows]
    assert Cs[1] <= Cs[0] * 1.02
    assert Cs[2] <= Cs[1] * 1.02
    assert Cs[3] <= Cs[2] * 1.02

def test_fdt_increment_variance_scaling():
    # “Noise-only”: zero deterministic damping, keep gamma_noise in FDT factor.
    base = SimParams(steps=1, dt=8e-4, gamma=0.0, gamma_noise=0.5, sigma=0.05, seed=42)
    dx = base.dx()
    rng = np.random.default_rng(base.seed)
    psi0 = np.ones(base.shape, dtype=np.complex128)

    samples = []
    for _ in range(64):
        psi1 = _step_tdgl(psi0, base, dx, rng)
        dpsi = psi1 - psi0
        samples.append(np.var(dpsi.real) + np.var(dpsi.imag))
    var_emp = float(np.mean(samples))

    T = base.sigma ** 2
    var_th = 4.0 * base.gamma_noise * T * base.dt  # real+imag
    ratio = var_emp / (var_th + 1e-16)
    assert 0.2 <= ratio <= 5.0
