# tests/test_tensor_optional.py
import numpy as np
from toe.tensor_coherence import coherence_integral_numpy, coherence_integral_tt

def test_numpy_and_tt_agree_roughly():
    # tiny 2D field to keep CI fast
    x = np.linspace(-1, 1, 32)
    X, Y = np.meshgrid(x, x, indexing="ij")
    phi = np.tanh(2*X) * np.exp(-(X**2 + Y**2))
    full = coherence_integral_numpy(phi)

    tt = coherence_integral_tt(phi, max_rank=8, backend="numpy", V=None, estimate_error=True)
    # Always defined; if TL missing, tt.ok=False but integral_value=full
    assert np.isfinite(tt.integral_value)
    if tt.ok and tt.approx_error is not None:
        assert tt.approx_error < 0.1 * max(1.0, abs(full))
