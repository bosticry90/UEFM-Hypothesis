# examples/phase5_soliton_scan.py
# Phase 5: Energy functional reconstruction & soliton diagnostics
#
# This module is self-contained (NumPy-only) and fast for CI.
# It provides:
#   - A simple localized “soliton-like” profile (sech ⊗ sech)
#   - Energy functional E[phi] = ∫ (|∇phi|^2 + V(phi)) dV with quartic V
#   - A momentum-like parity diagnostic (vanishes for symmetric states)
#   - Perturbation scans (energy should increase under small perturbations)
#   - Grid-refinement consistency (energy converges as resolution improves)
#
# Command-line use (from repo root):
#   python examples/phase5_soliton_scan.py
# -> writes outputs/phase5_soliton.tsv and outputs/figs/phase5_soliton.png

from __future__ import annotations
import os
import json
import math
from dataclasses import dataclass
from typing import Callable, Sequence, Tuple, Dict, Any

import numpy as np

# ---------- Utilities ----------

def _ensure_dirs():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("outputs/figs", exist_ok=True)

def _dx_for(shape: Tuple[int, int], extent: Tuple[float, float]=(10.0, 10.0)) -> Tuple[float, float]:
    """Uniform per-axis spacing to keep a fixed box size across resolutions."""
    nx, ny = shape
    Lx, Ly = extent
    return (Lx/(nx-1), Ly/(ny-1))

def _central_diff(arr: np.ndarray, axis: int, dx: float) -> np.ndarray:
    """Central differences with first-order one-sided at boundaries (fast & simple)."""
    g = np.zeros_like(arr, dtype=float)
    slc = [slice(None)]*arr.ndim

    # interior
    slc_c = slc.copy(); slc_c[axis] = slice(1, -1)
    slc_m = slc.copy(); slc_m[axis] = slice(0, -2)
    slc_p = slc.copy(); slc_p[axis] = slice(2, None)
    g[tuple(slc_c)] = (arr[tuple(slc_p)] - arr[tuple(slc_m)])/(2.0*dx)

    # left boundary (forward)
    sl0 = slc.copy(); sl0[axis] = 0
    sl1 = slc.copy(); sl1[axis] = 1
    g[tuple(sl0)] = (arr[tuple(sl1)] - arr[tuple(sl0)])/dx

    # right boundary (backward)
    slm1 = slc.copy(); slm1[axis] = -1
    slm2 = slc.copy(); slm2[axis] = -2
    g[tuple(slm1)] = (arr[tuple(slm1)] - arr[tuple(slm2)])/dx
    return g

def _grad_sqr(phi: np.ndarray, dx: Sequence[float]) -> np.ndarray:
    """Sum of squared gradients |∇phi|^2 with per-axis spacings."""
    assert len(dx) == phi.ndim, "dx length must match phi.ndim"
    g2 = np.zeros_like(phi, dtype=float)
    for ax, h in enumerate(dx):
        g = _central_diff(phi, ax, float(h))
        g2 += g*g
    return g2

# ---------- Profiles & potentials ----------

def sech(x: np.ndarray) -> np.ndarray:
    return 1.0/np.cosh(x)

def soliton_sech2(shape: Tuple[int, int], dx: Tuple[float, float], k: float=0.6) -> np.ndarray:
    """2D separable localized bump: phi(x,y) = A * sech(k x) * sech(k y)."""
    nx, ny = shape
    hx, hy = dx
    x = (np.arange(nx) - (nx-1)/2.0)*hx
    y = (np.arange(ny) - (ny-1)/2.0)*hy
    X, Y = np.meshgrid(x, y, indexing="ij")
    A = 1.0
    return A * sech(k*X) * sech(k*Y)

def quartic_potential(m2: float=1.0, lam: float=0.5) -> Callable[[np.ndarray], np.ndarray]:
    """V(phi) = 0.5*m2*phi^2 + 0.25*lam*phi^4  (positive definite for m2,lam>0)."""
    def V(phi: np.ndarray) -> np.ndarray:
        return 0.5*m2*(phi**2) + 0.25*lam*(phi**4)
    return V

# ---------- Energetics & diagnostics ----------

def energy(phi: np.ndarray, dx: Tuple[float, float], potential: Callable[[np.ndarray], np.ndarray]) -> float:
    """E = ∫ (|∇phi|^2 + V(phi)) dV."""
    hx, hy = dx
    vol = hx*hy
    g2 = _grad_sqr(phi, dx)
    dens = g2 + potential(phi)
    return float(np.sum(dens)*vol)

def momentum_like(phi: np.ndarray, dx: Tuple[float, float]) -> Tuple[float, float]:
    """
    Parity/momentum-like proxy (no dynamics): P = ∫ phi * ∇phi dV.
    For symmetric localized bumps, this should be ~ (0,0).
    """
    hx, hy = dx
    vol = hx*hy
    gx = _central_diff(phi, 0, hx)
    gy = _central_diff(phi, 1, hy)
    Px = float(np.sum(phi*gx)*vol)
    Py = float(np.sum(phi*gy)*vol)
    return Px, Py

# ---------- Scans ----------

@dataclass
class PerturbationResult:
    eps: float
    dE_over_E: float

def perturbation_scan(phi: np.ndarray,
                      dx: Tuple[float, float],
                      potential: Callable[[np.ndarray], np.ndarray],
                      rng: np.random.Generator,
                      eps_list: Sequence[float]=(0.01, 0.02, 0.05)) -> Sequence[PerturbationResult]:
    """Add small noise perturbations; energy should typically increase."""
    E0 = energy(phi, dx, potential)
    out = []
    for eps in eps_list:
        noise = rng.normal(0.0, 1.0, size=phi.shape)
        # Shape noise to be smooth-ish (1 pass of nearest-neighbor averaging)
        noise_smooth = 0.25*(np.roll(noise, 1, axis=0)+np.roll(noise, -1, axis=0)
                             + np.roll(noise, 1, axis=1)+np.roll(noise, -1, axis=1))
        phip = phi + eps*noise_smooth
        E1 = energy(phip, dx, potential)
        out.append(PerturbationResult(eps=float(eps), dE_over_E=float((E1-E0)/max(E0, 1e-15))))
    return out

@dataclass
class RefinementResult:
    shape: Tuple[int,int]
    energy: float

def refinement_scan(extent: Tuple[float,float]=(10.0,10.0),
                    shapes: Sequence[Tuple[int,int]]=((24,16),(32,24),(48,32)),
                    k: float=0.6,
                    potential: Callable[[np.ndarray], np.ndarray]=quartic_potential()) -> Sequence[RefinementResult]:
    """Compute energies on progressively finer grids of the same physical box."""
    outs = []
    for shape in shapes:
        dx = _dx_for(shape, extent=extent)
        phi = soliton_sech2(shape, dx, k=k)
        E = energy(phi, dx, potential)
        outs.append(RefinementResult(shape=shape, energy=E))
    return outs

# ---------- CLI driver ----------

def main() -> None:
    _ensure_dirs()
    rng = np.random.default_rng(5)

    # Base configuration
    shape = (32, 24)
    dx = _dx_for(shape)
    phi = soliton_sech2(shape, dx, k=0.6)
    V = quartic_potential(m2=1.0, lam=0.5)

    # Core diagnostics
    E = energy(phi, dx, V)
    Px, Py = momentum_like(phi, dx)

    perts = perturbation_scan(phi, dx, V, rng, eps_list=(0.01, 0.02, 0.05))
    ref = refinement_scan()

    # Write TSV & figure (figure optional; keep tiny)
    tsv = "outputs/phase5_soliton.tsv"
    with open(tsv, "w", encoding="utf-8") as f:
        f.write("section\tkey\tvalue\n")
        f.write(f"base\tenergy\t{E:.12g}\n")
        f.write(f"base\tPx\t{Px:.12g}\n")
        f.write(f"base\tPy\t{Py:.12g}\n")
        for r in perts:
            f.write(f"perturbation\teps={r.eps:.3g}\t{r.dE_over_E:.12g}\n")
        for r in ref:
            f.write(f"refinement\tshape={r.shape}\t{r.energy:.12g}\n")

    # Minimal plot (no heavy styling to keep CI fast)
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6, 3))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.set_title("Soliton profile")
        im = ax1.imshow(phi, origin="lower", aspect="auto")
        fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.set_title("Perturbation ΔE/E")
        xs = [p.eps for p in perts]
        ys = [max(p.dE_over_E, 0.0) for p in perts]
        ax2.plot(xs, ys, "o-")
        ax2.set_xlabel("ε")
        ax2.set_ylabel("ΔE/E")
        fig.tight_layout()
        figpath = "outputs/figs/phase5_soliton.png"
        fig.savefig(figpath, dpi=120)
    except Exception:
        figpath = None  # plotting is optional for CI

    summary: Dict[str, Any] = {
        "energy": E,
        "momentum_like": {"Px": Px, "Py": Py},
        "perturbations": [{"eps": p.eps, "dE_over_E": p.dE_over_E} for p in perts],
        "refinement": [{"shape": r.shape, "energy": r.energy} for r in ref],
        "tsv": tsv,
        "figure": figpath,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
