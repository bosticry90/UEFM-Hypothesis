# examples/tensor_rank_scheduler_demo.py
"""
Automatic rank scheduling demonstration.
"""
import numpy as np
from toe.tensor_tt_core import find_min_rank_for_tol

x = np.linspace(-3, 3, 64)
X, Y = np.meshgrid(x, x, indexing="ij")
phi = np.tanh(2 * X) * np.exp(-(X**2 + Y**2))

res = find_min_rank_for_tol(phi, target_rel_err=1e-3)
print(f"Target rel_err=1e-3  ->  rank={res.rank_used}, achieved err={res.rel_error:.2e}, C={res.integral_value:.3f}")
