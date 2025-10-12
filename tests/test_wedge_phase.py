import numpy as np
from toe.geometry import wedge_reconstructable_1d

def test_wedge_curve_monotone():
    # For fixed code distance, more erasure should never improve reconstructability
    dist = 4
    vals = [wedge_reconstructable_1d(boundary_size=40, erased=e, code_distance_edges=dist)
            for e in range(0, 7)]
    # once False, remain False
    first_false = next((i for i,v in enumerate(vals) if not v), None)
    if first_false is not None:
        assert all(v is False for v in vals[first_false:])
