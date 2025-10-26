import numpy as np
import pytest

from examples.phase6_scattering import quick_demo, simulate, SimParams

def test_energy_near_conserved_fast():
    res = quick_demo(return_data=True)
    e0 = res["energy_start"]
    e1 = res["energy_end"]
    drift = abs(e1 - e0) / (abs(e0) + 1e-12)
    assert drift < 0.06  # loose but meaningful for explicit Euler & CI speed

def test_coherence_dip_and_recovery_fast():
    res = quick_demo(return_data=True)
    C = np.array(res["coherence_series"], dtype=float)
    # dip during interaction
    assert C.min() < C[0] * 0.995
    # partial recovery (not necessarily full)
    assert C[-1] > C.min() * 1.01

@pytest.mark.parametrize("k0", [2.0, 3.0, 4.0])
def test_velocity_dependence_monotone_dip(k0):
    # coherence dip magnitude should weakly increase with |k0|
    p_lo = SimParams(shape=(24, 16), steps=180, dt=1.2e-3, k0=2.0, dtheta=0.0, amp=1.0, sigma=0.25, sep=1.0)
    p_hi = SimParams(shape=(24, 16), steps=180, dt=1.2e-3, k0=4.0, dtheta=0.0, amp=1.0, sigma=0.25, sep=1.0)
    r_lo = simulate(p_lo); r_hi = simulate(p_hi)
    C_lo = np.array(r_lo["coherence_series"]); C_hi = np.array(r_hi["coherence_series"])
    dip_lo = C_lo[0] - C_lo.min()
    dip_hi = C_hi[0] - C_hi.min()
    assert dip_hi >= 0.9 * dip_lo  # weak monotonicity (loose)
