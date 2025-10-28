# tests/test_phase7_coupled.py
import numpy as np
from examples.phase7_coupled_scattering import CoupledParams, split_step_coupled

def test_energy_conservation_tight():
    # Strang+subcycling should keep drift ~ machine for this short run
    p = CoupledParams(steps=220, dt=8e-4, g12=0.5, k0=3.0, shape=(32,24))
    r = split_step_coupled(p)
    E = np.array(r["energy_series"], float)
    drift = abs(E[-1]-E[0])/(abs(E[0])+1e-12)
    assert drift < 1.0e-6

def test_coherence_dip_present():
    p = CoupledParams(steps=220, dt=8e-4, g12=0.4, k0=3.0, shape=(32,24))
    r = split_step_coupled(p)
    C = np.array(r["coherence_series"], float)
    assert C.min() < C[0]*0.99  # noticeable dip

def test_cross_freq_increases_with_g12():
    # Overlap-phase oscillation rate should (weakly) increase with g12
    p_lo = CoupledParams(g12=0.1, steps=260, dt=8e-4, k0=3.0, shape=(32,24))
    p_hi = CoupledParams(g12=0.6, steps=260, dt=8e-4, k0=3.0, shape=(32,24))
    r_lo = split_step_coupled(p_lo)
    r_hi = split_step_coupled(p_hi)

    def freq_proxy(c):
        z = np.asarray(c, float) - np.mean(c)
        s = np.sign(z); s[s == 0] = 1
        return (np.count_nonzero(np.diff(s))/2)/(len(c)/1.0)

    f_lo = freq_proxy(r_lo["cross_series"])
    f_hi = freq_proxy(r_hi["cross_series"])
    assert f_hi >= 0.9*f_lo and f_hi > f_lo*1.05
