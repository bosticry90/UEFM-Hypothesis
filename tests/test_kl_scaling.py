import numpy as np
from toe.qec import random_isometry, erasure_errors, knill_laflamme_checks

def test_kl_small_scaling():
    rng = np.random.default_rng(4)
    # vary small logical/physical dims
    for d_log, d_phys in [(2,4), (2,5), (3,6)]:
        V = random_isometry(d_in=d_log, d_out=d_phys, seed=5)
        errs = erasure_errors(n_physical=2, k=1, local_dim=2)
        ok, C = knill_laflamme_checks(V, errs, atol=1e-6)
        # Not guaranteed true; just ensure numeric pipeline is stable & bounded
        assert np.isfinite(C).all()
