# tests/test_meta_axioms.py
import numpy as np
from toe.qca import SplitStepQCA1D
from toe.substrate import Substrate
from toe.geometry import minimal_cut_length_1d

def test_locality_lightcone_upper_bound():
    # Large lattice but avoid allocating 2**N vectors.
    N = 101
    qca = SplitStepQCA1D(n_sites=N, d=2, seed=11, theta1=0.3, theta2=0.34)

    # One update expands the causal cone by at most 1 site → radius <= steps.
    T = 20
    radius = qca.lightcone_radius(steps=T)
    assert radius <= T

def test_area_law_proxy_on_path():
    S = Substrate.line(32, d=2, bond_dim=3)
    m = 8
    Sbits = S.entropy_of(range(m)).s_bits
    # Expect our proxy based on natural log:
    assert np.isclose(Sbits, minimal_cut_length_1d(region_size=m, bond_dim=3))
