# tests/test_phase5_soliton.py
# Lightweight unit tests for Phase 5 diagnostics (NumPy-only).

import numpy as np

from examples.phase5_soliton_scan import (
    _dx_for,
    soliton_sech2,
    quartic_potential,
    energy,
    momentum_like,
    perturbation_scan,
    refinement_scan,
)

def test_energy_positive_and_finite():
    shape = (32, 24)
    dx = _dx_for(shape)
    phi = soliton_sech2(shape, dx, k=0.6)
    V = quartic_potential(m2=1.0, lam=0.5)
    E = energy(phi, dx, V)
    assert np.isfinite(E)
    assert E > 0.0

def test_momentum_like_near_zero_for_symmetric():
    shape = (32, 24)
    dx = _dx_for(shape)
    phi = soliton_sech2(shape, dx, k=0.6)
    V = quartic_potential(1.0, 0.5)
    E = energy(phi, dx, V)
    Px, Py = momentum_like(phi, dx)
    # Symmetric bump -> |P| should be tiny relative to energy scale
    Pnorm = (Px**2 + Py**2) ** 0.5
    assert Pnorm < 1e-3 * max(E, 1e-9)

def test_small_perturbations_raise_energy_on_average():
    rng = np.random.default_rng(123)
    shape = (32, 24)
    dx = _dx_for(shape)
    phi = soliton_sech2(shape, dx, k=0.6)
    V = quartic_potential(1.0, 0.5)
    perts = perturbation_scan(phi, dx, V, rng, eps_list=(0.01, 0.02, 0.05))
    # At least two out of three should increase energy
    increases = sum(1 for p in perts if p.dE_over_E > -1e-6)
    assert increases >= 2

def test_energy_converges_with_grid_refinement():
    ref = refinement_scan(extent=(10.0, 10.0), shapes=((24,16), (32,24), (48,32)))
    # Energies should approach a limit; check monotone-ish trend:
    E_coarse, E_mid, E_fine = [r.energy for r in ref]
    # Require finer grid close to mid (relative improvement)
    assert abs(E_fine - E_mid) <= 0.05 * max(E_mid, 1e-6)
