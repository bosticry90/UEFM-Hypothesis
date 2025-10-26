# tests/test_adversarial_invariance.py
import numpy as np
from toe.tensor_coherence import coherence_integral_numpy, coherence_integral_tt, TTResult

def _mk(n=48, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(-2, 2, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    phi = np.tanh(1.5 * X) * np.exp(-(X**2 + 0.7 * Y**2))
    phi += 0.01 * np.cos(0.9 * X) * np.sin(0.8 * Y)
    return phi

def test_invariance_small_rot():
    phi = _mk()
    dense0 = coherence_integral_numpy(phi)
    tt0 = coherence_integral_tt(phi, max_rank=6, backend="numpy", estimate_error=False)
    tt0_val = tt0.integral_value if isinstance(tt0, TTResult) else float(tt0)

    phi_r = np.rot90(phi, k=1)
    dense1 = coherence_integral_numpy(phi_r)
    tt1 = coherence_integral_tt(phi_r, max_rank=6, backend="numpy", estimate_error=False)
    tt1_val = tt1.integral_value if isinstance(tt1, TTResult) else float(tt1)

    rel_dense = abs(dense1 - dense0) / (abs(dense0) + 1e-12)
    rel_tt = abs(tt1_val - tt0_val) / (abs(tt0_val) + 1e-12)

    # Loose bounds: we just want to catch egregious regressions/artifacts
    assert rel_dense < 0.1
    assert rel_tt < 0.1
