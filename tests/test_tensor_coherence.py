# tests/test_tensor_coherence.py
import numpy as np
import pytest

from toe.tensor_coherence import (
    TENSORLY_AVAILABLE,
    as_tt,
    reconstruct,
    coherence_dense,
    coherence_tt,
)

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _make_field(shape=(64, 16)):
    # smooth, separable-ish test field
    x = np.linspace(0, 2*np.pi, shape[0], endpoint=True)
    y = np.linspace(0, 1.0,      shape[1], endpoint=True)
    X, Y = np.meshgrid(x, y, indexing="ij")
    return np.cos(2.0*X) * np.exp(-Y)


def _quartic(V0=1.0):
    return lambda phi: V0 * (phi**4) / 4.0


def test_dense_coherence_basic():
    phi = _make_field((32, 16))
    C = coherence_dense(phi, dx=(2*np.pi/31, 1.0/15))
    assert np.isfinite(C)
    assert C > 0.0


@pytest.mark.skipif(not TENSORLY_AVAILABLE, reason="TensorLy not installed")
def test_tt_matches_dense_no_potential():
    phi = _make_field((32, 16))
    dx = (2*np.pi/31, 1.0/15)

    tt = as_tt(phi, rank=8)
    C_dense = coherence_dense(phi, dx)
    C_tt = coherence_tt(tt, dx)

    # Should match closely (exact for our modest grid & smooth field)
    assert np.isclose(C_dense, C_tt, rtol=1e-8, atol=1e-10)


@pytest.mark.skipif(not TENSORLY_AVAILABLE, reason="TensorLy not installed")
def test_tt_matches_dense_with_quartic():
    phi = _make_field((32, 16))
    dx = (2*np.pi/31, 1.0/15)
    V = _quartic(1.5)

    tt = as_tt(phi, rank=12)
    C_dense = coherence_dense(phi, dx, potential=V)
    C_tt = coherence_tt(tt, dx, potential=V)

    assert np.isclose(C_dense, C_tt, rtol=1e-8, atol=1e-10)


@pytest.mark.skipif(not TENSORLY_AVAILABLE, reason="TensorLy not installed")
def test_rank_sensitivity_monotone_error():
    phi = _make_field((32, 16))

    def rel_err(rank):
        tt = as_tt(phi, rank=rank)
        rec = reconstruct(tt)
        num = np.linalg.norm(rec - phi)
        den = np.linalg.norm(phi)
        return float(num / den)

    ranks = [2, 4, 8, 12, 16]
    errs = [rel_err(r) for r in ranks]

    # The error should be non-increasing as rank increases (allow tiny noise)
    for e1, e2 in zip(errs[:-1], errs[1:]):
        assert e2 <= e1 + 1e-12
