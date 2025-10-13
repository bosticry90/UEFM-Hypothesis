# tests/test_tensor_entropy.py
import numpy as np
from toe.tensor_entropy import tensor_rank_entropy, schmidt_entropy_by_cuts
from toe.qca import SplitStepQCA1D, apply_qca_steps

def test_tensor_entropy_product_vs_entangled():
    N = 10
    psi0 = np.zeros(2**N, dtype=complex); psi0[0] = 1.0  # product
    S0 = tensor_rank_entropy(psi0, local_dim=2, n_sites=N, base=np.e, reduction="mean")
    assert np.isclose(S0, 0.0)

    qca = SplitStepQCA1D(n_sites=N, d=2, seed=3, theta1=0.2, theta2=0.3)
    psi1 = apply_qca_steps(qca, psi0, steps=4)
    S1 = tensor_rank_entropy(psi1, local_dim=2, n_sites=N, base=np.e, reduction="mean")
    assert S1 > 0.0

def test_schmidt_cut_count():
    N = 8
    psi0 = np.zeros(2**N, dtype=complex); psi0[0] = 1.0
    cuts = schmidt_entropy_by_cuts(psi0, local_dim=2, n_sites=N, base=np.e)
    assert cuts.shape == (N-1,)
    assert np.allclose(cuts, 0.0)
