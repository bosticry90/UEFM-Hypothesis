# examples/gauge_symmetry_scan.py
"""
Phase 4: U(1) gauge-symmetry probes.

Exposes small, testable building blocks used by the unit tests:
  - _dx_for(shape)
  - _make_complex_field(shape, seed)
  - _global_phase_invariance(phi, dx, thetas=None)  -> dict(score=...)
  - _minimal_coupling_consistency(phi, dx)          -> dict(score=..., E_plain=..., E_cov=...)

When run as a script, writes:
  - outputs/phase4_gauge.tsv
  - outputs/figs/gauge_invariance.png
and prints a JSON summary.
"""

from __future__ import annotations

import os
import json
from typing import Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------
# Utilities
# ---------------------------

def _dx_for(shape: Tuple[int, int]) -> Tuple[float, float]:
    """Default grid spacing used across tests/examples."""
    nx, ny = shape
    # 2π-wide domain in x, unit domain in y
    return (2.0 * np.pi / (nx - 1), 1.0 / (ny - 1))


def _periodic_roll(a: np.ndarray, shift: int, axis: int) -> np.ndarray:
    """Periodic roll helper."""
    return np.roll(a, shift=shift, axis=axis)


def _central_diff_periodic(a: np.ndarray, dx: float, axis: int) -> np.ndarray:
    """Second-order central difference with periodic boundary conditions."""
    return (_periodic_roll(a, -1, axis) - _periodic_roll(a, 1, axis)) / (2.0 * dx)


def _grad(phi: np.ndarray, dx: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Periodic central-difference gradient for complex-valued 2D fields."""
    dx0, dx1 = dx
    gx = _central_diff_periodic(phi, dx0, axis=0)
    gy = _central_diff_periodic(phi, dx1, axis=1)
    return gx, gy


def _kinetic_energy(phi: np.ndarray, dx: Tuple[float, float]) -> np.ndarray:
    """|∇phi|^2 with periodic central differences (returns the field of densities)."""
    gx, gy = _grad(phi, dx)
    return (gx * np.conj(gx) + gy * np.conj(gy)).real


def _make_complex_field(shape: Tuple[int, int], seed: int = 0) -> np.ndarray:
    """Smooth complex test field on a periodic grid."""
    rng = np.random.default_rng(seed)
    nx, ny = shape
    x = np.linspace(0.0, 2.0 * np.pi, nx, endpoint=True)
    y = np.linspace(0.0, 1.0, ny, endpoint=True)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Smooth amplitude + smooth phase
    amp = np.exp(-((X - np.pi) ** 2) / (2 * 0.7**2)) * (0.6 + 0.4 * np.cos(2 * np.pi * Y))
    phase = 0.8 * np.sin(X) * np.cos(2 * np.pi * Y) + 0.1 * rng.standard_normal(shape)
    phi = amp * np.exp(1j * phase)
    return phi.astype(np.complex128, copy=False)


# ---------------------------
# Gauge probes used by tests
# ---------------------------

def _global_phase_invariance(
    phi: np.ndarray,
    dx: Tuple[float, float],
    thetas: Iterable[float] | None = None,
) -> dict:
    """
    Energy invariance under global U(1) rotations: phi -> e^{iθ} phi.
    Returns dict with 'score' in [0, 1], where 1 = perfectly invariant.

    If `thetas` is provided, average the score over all θ values.
    """
    if thetas is None:
        thetas = (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 2.0 * np.pi)

    E0 = float(np.mean(_kinetic_energy(phi, dx)))
    denom = abs(E0) + 1e-12

    rel_devs = []
    scores = []
    for th in thetas:
        rotated = phi * np.exp(1j * float(th))
        E1 = float(np.mean(_kinetic_energy(rotated, dx)))
        rel = abs(E1 - E0) / denom
        rel_devs.append(rel)
        scores.append(max(0.0, 1.0 - rel))

    return {"score": float(np.mean(scores)), "max_rel": float(np.max(rel_devs))}


def _minimal_coupling_consistency(
    phi: np.ndarray,
    dx: Tuple[float, float],
) -> dict:
    """
    Weak-field consistency check for minimal coupling.

    Construct a small, smooth gauge potential A = ∇χ (pure gauge), then verify:
      E_plain(phi) ~ E_cov(e^{iχ} phi; A)

    Discretization is periodic & uses central differences to reduce boundary error.
    Returns a dict with the scalar 'score' in [0,1], 1 = perfect consistency.
    """
    # Build small, smooth gauge scalar χ
    nx, ny = phi.shape
    x = np.linspace(0.0, 2.0 * np.pi, nx, endpoint=True)
    y = np.linspace(0.0, 1.0, ny, endpoint=True)
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Very weak, smooth χ to stay in the linear regime and minimize stencil error
    eta = 2.0e-4
    chi = eta * (np.cos(X) + 0.5 * np.cos(2.0 * np.pi * Y))

    # Pure gauge: A = ∇χ
    Ax, Ay = _grad(chi, dx)

    # Gauge-transform the field and apply covariant derivative
    phi_g = phi * np.exp(1j * chi)

    # Covariant gradient (∇ - i A) phi_g
    dphi_x, dphi_y = _grad(phi_g, dx)
    cov_x = dphi_x - 1j * Ax * phi_g
    cov_y = dphi_y - 1j * Ay * phi_g

    E_plain = float(np.mean(_kinetic_energy(phi, dx)))
    E_cov = float(np.mean((cov_x * np.conj(cov_x) + cov_y * np.conj(cov_y)).real))

    rel = abs(E_cov - E_plain) / (abs(E_plain) + 1e-12)
    score = max(0.0, 1.0 - rel)
    return {"score": float(score), "E_plain": E_plain, "E_cov": E_cov}


# ---------------------------
# Script: generate report
# ---------------------------

def _ensure_dirs() -> Tuple[str, str]:
    out_dir = os.path.join("outputs")
    figs_dir = os.path.join(out_dir, "figs")
    os.makedirs(figs_dir, exist_ok=True)
    return out_dir, figs_dir


def _make_report() -> dict:
    shape = (64, 48)
    dx = _dx_for(shape)
    phi = _make_complex_field(shape, seed=42)

    g = _global_phase_invariance(
        phi, dx, thetas=np.linspace(0.0, 2.0 * np.pi, 9, endpoint=True)
    )
    cov = _minimal_coupling_consistency(phi, dx)

    out_dir, figs_dir = _ensure_dirs()

    # Save a small TSV
    tsv_path = os.path.join(out_dir, "phase4_gauge.tsv")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("metric\tvalue\n")
        f.write(f"global_u1_score\t{g['score']:.6f}\n")
        f.write(f"global_u1_max_rel\t{g['max_rel']:.12e}\n")
        f.write(f"covariant_score\t{cov['score']:.6f}\n")
        f.write(f"E_plain\t{cov['E_plain']:.12e}\n")
        f.write(f"E_cov\t{cov['E_cov']:.12e}\n")

    # Quick visual: histogram of kinetic energy density before/after
    fig_path = os.path.join(figs_dir, "gauge_invariance.png")
    E0_field = _kinetic_energy(phi, dx)
    chi_small = 2.0e-4 * (np.cos(np.linspace(0, 2*np.pi, shape[0]))[:, None]
                          + 0.5 * np.cos(2.0*np.pi * np.linspace(0, 1, shape[1])[None, :]))
    phi_g = phi * np.exp(1j * chi_small)
    Ax, Ay = _grad(chi_small, dx)
    dphi_x, dphi_y = _grad(phi_g, dx)
    cov_x = dphi_x - 1j * Ax * phi_g
    cov_y = dphi_y - 1j * Ay * phi_g
    E_cov_field = (cov_x * np.conj(cov_x) + cov_y * np.conj(cov_y)).real

    plt.figure(figsize=(7.0, 4.0))
    plt.hist(E0_field.ravel(), bins=60, alpha=0.6, label="|∇φ|²", density=True)
    plt.hist(E_cov_field.ravel(), bins=60, alpha=0.6, label="|(∇-iA)φ_g|²", density=True)
    plt.xlabel("energy density")
    plt.ylabel("pdf")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=140)
    plt.close()

    return {
        "global_u1_score": float(g["score"]),
        "covariant_score": float(cov["score"]),
        "tsv": tsv_path,
        "figure": fig_path,
    }


def main() -> None:
    report = _make_report()
    print("Phase 4 gauge scan complete.")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
