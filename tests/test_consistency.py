
import numpy as np
from toe.substrate import Substrate
from toe.qec import random_isometry, knill_laflamme_checks, erasure_errors
from toe.qca import SplitStepQCA1D, norm, energy_conservation_proxy
from toe.geometry import minimal_cut_length_1d, entanglement_distance_1d, wedge_reconstructable_1d

def test_substrate_basic():
    S = Substrate.line(6, d=2)
    assert S.graph_distance(0,5) == 5
    assert S.degree(0) == 1 and S.degree(3) == 2

def test_qec_kl_sanity():
    V = random_isometry(d_logical=2, d_physical=8, seed=1)
    E = erasure_errors(n_physical=3)
    ok, C = knill_laflamme_checks(V, E, atol=1e-6)
    assert isinstance(ok, bool)
    assert C.shape == (len(E), len(E))

def test_qca_norm_conservation():
    qca = SplitStepQCA1D(n_sites=6, d=2, seed=3)
    psi0 = np.zeros(2**6, dtype=complex); psi0[0] = 1.0
    norms = energy_conservation_proxy(psi0, qca, steps=10)
    assert np.allclose(norms, norms[0], atol=1e-10)

def test_geometry_toys():
    S = minimal_cut_length_1d(region_size=3, bond_dim=4)
    assert np.isclose(S, 2*np.log(4))
    d = entanglement_distance_1d(separation_edges=5, bond_dim=3)
    assert np.isclose(d, 5*np.log(3))
    assert wedge_reconstructable_1d(boundary_size=10, erased=1, code_distance_edges=2) is True
