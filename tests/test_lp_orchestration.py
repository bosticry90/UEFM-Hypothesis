# tests/test_lp_orchestration.py
import numpy as np
from toe.lp_orchestration import schedule_grid_lp

def test_lp_basic():
    rng = np.random.default_rng(1)
    n = 10
    r = rng.uniform(1, 3, size=n)
    m = rng.uniform(0.5, 1.2, size=n)
    t = rng.uniform(0.2, 0.9, size=n)
    res = schedule_grid_lp(r, m, t, mem_limit=4.0, time_limit=3.0, seed=1)
    assert res.objective >= 0.0
    assert res.selected.shape == (n,)
