import numpy as np
import pytest

from toe.tensor_tools import (
    is_tensorly_available,
    tt_from_grid,
    tt_to_dense,
    reconstruction_error,
    tensor_coherence,
)

def test_round_trip_identity_small():
    phi = np.arange(12, dtype=float).reshape(3, 4)
    tt = tt_from_grid(phi, rank=4)
    rec = tt_to_dense(tt)
    assert rec.shape == phi.shape
    # In dense fallback, error is ~0; with TT it should be small for tiny tensors.
    err = reconstruction_error(tt, phi)
    assert err <= 1e-6

def test_coherence_matches_dense_wrap():
    # Smooth bump in 2D to avoid noisy gradients
    nx, ny = 16, 12
    x = np.linspace(-2.0, 2.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    phi = np.exp(-X**2 - 2*Y**2)

    tt = tt_from_grid(phi, rank=6)
    # Dense wrap coherence (building a "TT" from the dense again)
    coh_dense = tensor_coherence(tt_from_grid(phi, rank=6), dx=1.0)
    coh_tt = tensor_coherence(tt, dx=1.0)
    # They should be very close (identical in the dense fallback)
    assert np.isfinite(coh_dense) and np.isfinite(coh_tt)
    assert np.allclose(coh_dense, coh_tt, rtol=1e-6, atol=1e-6)

def test_error_nonincreasing_with_rank():
    # With a smooth 3D field, reconstruction error should not grow as rank increases.
    n = 8
    i = np.linspace(-1, 1, n)
    X, Y, Z = np.meshgrid(i, i, i, indexing="ij")
    phi = np.exp(- (X**2 + 0.5*Y**2 + 2*Z**2))

    # Try a few ranks (small to keep runtime negligible)
    ranks = [2, 4, 8]
    errs = []
    for r in ranks:
        tt = tt_from_grid(phi, rank=r)
        errs.append(reconstruction_error(tt, phi))

    # Monotone nonincreasing: err[r_k+1] <= err[r_k] (within numerical slack)
    assert errs[1] <= errs[0] + 1e-9
    assert errs[2] <= errs[1] + 1e-9

def test_potential_term_adds_energy():
    # φ in [0, 1]; quartic potential should be nonnegative and add to coherence.
    phi = np.linspace(0, 1, 64).reshape(8, 8)
    tt = tt_from_grid(phi, rank=4)
    base = tensor_coherence(tt, dx=1.0)
    lam = 0.2
    V = lambda arr: lam * (arr ** 4)
    with_V = tensor_coherence(tt, dx=1.0, potential=V)
    assert with_V >= base
