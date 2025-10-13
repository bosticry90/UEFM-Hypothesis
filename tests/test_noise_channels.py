# tests/test_noise_channels.py
import numpy as np
from toe.qca import SplitStepQCA1D, energy_conservation_proxy
from toe.noise import bit_flip, phase_flip, depolarizing, apply_channel

def test_noise_helpers_norm_preserved():
    psi = np.zeros(2**6, dtype=complex); psi[0] = 1.0
    for fn in (bit_flip, phase_flip, depolarizing):
        out = fn(psi, p=0.3, rng=np.random.default_rng(1))
        assert np.isclose(np.vdot(out, out).real, 1.0, atol=1e-12)

def test_apply_channel_switchboard():
    psi = np.zeros(2**4, dtype=complex); psi[0] = 1.0
    for ch in ("bitflip", "phaseflip", "depolarizing"):
        out = apply_channel(psi, p=0.2, channel=ch, rng=np.random.default_rng(0))
        assert out.shape == psi.shape

def test_qca_drift_monotone_in_p():
    qca = SplitStepQCA1D(n_sites=6, d=2, seed=2)
    psi0 = np.zeros(2**6, dtype=complex); psi0[0] = 1.0
    # reuse the proxy you already have; we’ll pass noise via p
    drifts = []
    for p in [0.0, 0.05, 0.1, 0.2]:
        norms = energy_conservation_proxy(qca, psi0, steps=8, p=p)
        drift = np.max(np.abs(norms - norms[0]))
        drifts.append(drift)
    # weakly monotone (allow equal within numerical jitter)
    assert np.all(np.diff(drifts) >= -1e-14)
