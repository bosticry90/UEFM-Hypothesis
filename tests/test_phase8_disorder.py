# tests/test_phase8_disorder.py
import numpy as np
from examples.phase8_disorder_scan import SimParams, simulate, sweep

def test_energy_drift_loose_bound():
    out = sweep()
    med = np.median([r["energy_drift"] for r in out])
    assert med < 0.05  # loose but meaningful for explicit Euler + renorm

def test_coherence_monotone_decrease():
    out = sweep()
    Ws = [r["W"] for r in out]
    C  = [r["final_coherence"] for r in out]
    # Coherence should weakly decrease with stronger disorder
    for i in range(1, len(Ws)):
        assert C[i] <= C[i-1] + 1e-10

def test_ipr_monotone_increase():
    out = sweep()
    Ws = [r["W"] for r in out]
    I  = [r["final_ipr"] for r in out]
    # Localization proxy (IPR) should weakly increase with stronger disorder
    for i in range(1, len(Ws)):
        assert I[i] >= I[i-1] - 1e-10

def test_single_sim_outputs_shape():
    p = SimParams(W=0.3, steps=48)
    r = simulate(p)
    assert "energy_series" in r and "coherence_series" in r and "ipr_series" in r
    assert isinstance(r["final_coherence"], float)
    assert isinstance(r["final_ipr"], float)
    assert len(r["energy_series"]) >= 2
