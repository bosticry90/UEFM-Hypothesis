import numpy as np

# repo root import enabled by tests/conftest.py
from examples.phase10_kibble_zurek import KZParams, simulate, sweep

def test_defects_decrease_with_slower_quench():
    rows = sweep(tau_list=(8.0, 16.0, 32.0, 64.0), base=KZParams(steps=1200, dt=8e-4))
    taus = [r["tau_Q"] for r in rows]
    defs = [r["final_defects"] for r in rows]
    # Weak KZ expectation: slower quench -> fewer defects (allow ties)
    for i in range(1, len(taus)):
        assert defs[i] <= defs[i-1]

def test_sim_returns_energy_series_and_counts():
    r = simulate(KZParams(steps=200, dt=8e-4))
    assert "energy_series" in r and isinstance(r["final_defects"], int)
    # energy finite; non-negative typical for our potential
    e = np.array(r["energy_series"], float)
    assert np.all(np.isfinite(e)) and e[-1] > -1e-6

def test_defect_detection_nonzero_when_quench_fast():
    # Very fast quench should yield at least some vortices
    r = simulate(KZParams(tau_Q=6.0, steps=900, dt=8e-4))
    assert r["final_defects"] >= 1
