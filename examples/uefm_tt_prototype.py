"""
UEFM Tensor-Train Prototype
---------------------------
Tiny demo that:
1) Builds a simple 2-D field φ(x, t) (Gaussian pulse drifting in time).
2) Constructs a TT representation (if TensorLy present; else dense wrapper).
3) Computes a simple coherence proxy from the TT and from the dense field.
4) Prints reconstruction error and coherence values.

Safe to run on modest hardware; does not require TensorLy.
"""

from __future__ import annotations
import numpy as np
from toe.tensor_tools import (
    is_tensorly_available,
    tt_from_grid,
    tt_to_dense,
    reconstruction_error,
    tensor_coherence,
)

def gaussian(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def main():
    # Small grid to keep runtime/TTS tiny
    nx, nt = 64, 16
    x = np.linspace(-3.0, 3.0, nx)
    t = np.linspace(0.0, 1.0, nt)

    # Simple drifting Gaussian φ(x, t) = exp(-((x - v t)^2)/(2σ^2))
    v = 1.0
    sigma = 0.6
    phi = np.zeros((nx, nt), dtype=float)
    for j, tj in enumerate(t):
        phi[:, j] = gaussian(x, mu=v * tj, sigma=sigma)

    # Build TT (or dense fallback) and compute diagnostics
    tt = tt_from_grid(phi, rank=8)
    phi_rec = tt_to_dense(tt)
    err = reconstruction_error(tt, phi)

    coh_dense = tensor_coherence(tt_from_grid(phi, rank=8), dx=1.0)
    coh_tt = tensor_coherence(tt, dx=1.0)

    print(f"TensorLy available: {is_tensorly_available()}")
    print(f"φ shape: {phi.shape}")
    print(f"Reconstruction error (Fro norm): {err:.3e}")
    print(f"Coherence (dense wrap): {coh_dense:.6e}")
    print(f"Coherence (TT object):  {coh_tt:.6e}")

    # Optional: a toy potential V(φ) = λ φ^4
    lam = 0.1
    V = lambda arr: lam * (arr ** 4)
    coh_with_V = tensor_coherence(tt, dx=1.0, potential=V)
    print(f"Coherence with quartic potential: {coh_with_V:.6e}")

if __name__ == "__main__":
    main()
