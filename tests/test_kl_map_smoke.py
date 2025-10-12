# tests/test_kl_map_smoke.py
import numpy as np
from toe.qec import random_isometry, erasure_errors, knill_laflamme_checks

def test_small_map_smoke():
    V = random_isometry(d_in=2, d_out=4, seed=3)
    errs = erasure_errors(n_physical=2, k=1, local_dim=2)
    ok, C = knill_laflamme_checks(V, errs, atol=1e-8)
    assert isinstance(ok, bool)
    assert C.shape == (len(errs), len(errs))
