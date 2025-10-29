# examples/phase9_thermal_gl.py
# Finite-T complex TDGL (overdamped, Model A–like) with Langevin noise.
# Exposes: SimParams, simulate, sweep. Also writes TSV + figure when run.

from __future__ import annotations
import json, math, pathlib
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt

# --------------------------- I/O utils -----------------------------------

def _ensure_dirs():
    pathlib.Path("outputs/figs").mkdir(parents=True, exist_ok=True)
    pathlib.Path("outputs").mkdir(parents=True, exist_ok=True)

# --------------------------- Numerics -------------------------------------

def _laplacian_periodic(z: np.ndarray, dx: Tuple[float, float]) -> np.ndarray:
    zx_p = np.roll(z, -1, axis=0); zx_m = np.roll(z, 1, axis=0)
    zy_p = np.roll(z, -1, axis=1); zy_m = np.roll(z, 1, axis=1)
    return (zx_p + zx_m - 2.0 * z) / (dx[0] ** 2) + (zy_p + zy_m - 2.0 * z) / (dx[1] ** 2)

def _order_coherence(psi: np.ndarray) -> float:
    # Global order parameter (sensitive to dephasing); should DROP with noise.
    return float(np.abs(np.mean(psi)) ** 2)

def _kinetic_energy(psi: np.ndarray, dx: Tuple[float, float]) -> float:
    psix = (np.roll(psi, -1, 0) - np.roll(psi, 1, 0)) / (2 * dx[0])
    psiy = (np.roll(psi, -1, 1) - np.roll(psi, 1, 1)) / (2 * dx[1])
    kin = 0.5 * (np.abs(psix) ** 2 + np.abs(psiy) ** 2)
    return float(np.trapezoid(np.trapezoid(kin.real, dx=dx[1], axis=1), dx=dx[0], axis=0))

def _energy(psi: np.ndarray, dx: Tuple[float, float], alpha: float, g: float) -> float:
    psix = (np.roll(psi, -1, 0) - np.roll(psi, 1, 0)) / (2 * dx[0])
    psiy = (np.roll(psi, -1, 1) - np.roll(psi, 1, 1)) / (2 * dx[1])
    grad = np.abs(psix) ** 2 + np.abs(psiy) ** 2
    pot = 0.5 * g * (np.abs(psi) ** 4)
    dens = 0.5 * alpha * grad + pot
    return float(np.trapezoid(np.trapezoid(dens.real, dx=dx[1], axis=1), dx=dx[0], axis=0))

# --------------------------- Params & Sim ---------------------------------

@dataclass
class SimParams:
    shape: Tuple[int, int] = (24, 16)
    Lx: float = 2 * np.pi
    Ly: float = 2.0
    steps: int = 240
    dt: float = 6e-4
    alpha: float = 1.0       # stiffness
    g: float = 0.2           # quartic
    gamma: float = 0.1       # deterministic damping
    sigma: float = 0.0       # √T
    k0: float = 0.0          # start near uniform phase (maximizes |⟨ψ⟩|)
    seed: int = 0
    # Independent damping for the noise (for FDT test with “noise-only”)
    gamma_noise: float | None = field(default=None)

    def __post_init__(self):
        if self.gamma_noise is None:
            self.gamma_noise = self.gamma

    def dx(self) -> Tuple[float, float]:
        nx, ny = self.shape
        return (self.Lx / (nx - 1), self.Ly / (ny - 1))

def _init_field(p: SimParams) -> np.ndarray:
    nx, ny = p.shape
    x = np.linspace(0, p.Lx, nx)
    X = np.tile(x[:, None], (1, ny))
    rng = np.random.default_rng(p.seed)
    # small complex jitter to break symmetry a bit
    noise = 0.01 * (rng.standard_normal(p.shape) + 1j * rng.standard_normal(p.shape))
    psi0 = (1.0 + noise) * np.exp(1j * (p.k0 * X))
    return psi0.astype(np.complex128)

def _step_tdgl(psi: np.ndarray, p: SimParams, dx: Tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    # Overdamped TDGL: dψ/dt = +α ∇²ψ − g |ψ|² ψ − γ ψ + η
    lap = _laplacian_periodic(psi, dx)
    det = p.alpha * lap - p.g * (np.abs(psi) ** 2) * psi - p.gamma * psi

    if p.sigma > 0.0:
        # Langevin η with <ηη*> ∝ 2 γ_noise T. Implemented via sqrt(dt)*η_scaled.
        eta_r = rng.standard_normal(psi.shape)
        eta_i = rng.standard_normal(psi.shape)
        # Choose scaling so Var(Δψ_real) ≈ 2 γ_noise T dt (per component); we add real+imag later in test.
        # Δψ_noise = sqrt(dt) * sigma * sqrt(2 γ_noise) * ξ
        eta = (eta_r + 1j * eta_i) * (p.sigma * math.sqrt(2.0 * p.gamma_noise))
        incr = p.dt * det + math.sqrt(p.dt) * eta
    else:
        incr = p.dt * det

    return psi + incr

def simulate(p: SimParams) -> Dict[str, Any]:
    rng = np.random.default_rng(p.seed)
    dx = p.dx()
    psi = _init_field(p)

    E_series: List[float] = []
    C_series: List[float] = []
    K_series: List[float] = []

    for _ in range(p.steps):
        psi = _step_tdgl(psi, p, dx, rng)
        if not np.all(np.isfinite(psi)):
            break
        E_series.append(_energy(psi, dx, p.alpha, p.g))
        C_series.append(_order_coherence(psi))
        K_series.append(_kinetic_energy(psi, dx))

    # smooth “final” values by averaging last 10% to reduce jitter for CI
    tail = max(1, len(C_series) // 10)
    C_final = float(np.mean(C_series[-tail:])) if C_series else np.nan
    K_final = float(np.mean(K_series[-tail:])) if K_series else np.nan

    E0 = E_series[0] if E_series else np.nan
    E1 = E_series[-1] if E_series else np.nan
    drift = float(abs(E1 - E0) / (abs(E0) + 1e-12)) if np.isfinite(E0) and np.isfinite(E1) else np.inf

    return {
        "params": p.__dict__,
        "energy_series": E_series,
        "coherence_series": C_series,
        "kinetic_series": K_series,
        "energy_start": E0,
        "energy_end": E1,
        "energy_drift": drift,
        "final_coherence": C_final,
        "final_kinetic": K_final,
    }

def sweep(sigmas=(0.0, 0.02, 0.05, 0.1)) -> List[Dict[str, Any]]:
    out = []
    for s in sigmas:
        # Parameters tuned so: coherence ↓ with σ, kinetic ↑ with σ.
        p = SimParams(
            sigma=float(s),
            seed=1234,
            steps=240,
            dt=6e-4,
            gamma=0.1,
            alpha=1.0,
            g=0.2,
            k0=0.0,
            shape=(24, 16),
        )
        out.append({"sigma": s, **simulate(p)})
    return out

# --------------------------- CLI: TSV + Plot ------------------------------

def _save_tsv(rows: List[Dict[str, Any]], path: str):
    keys = ["sigma", "energy_drift", "final_coherence", "final_kinetic"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\t".join(keys) + "\n")
        for r in rows:
            vals = [r.get(k, np.nan) for k in keys]
            f.write("\t".join(str(v) for v in vals) + "\n")

def _plot(rows: List[Dict[str, Any]], fig_path: str):
    sig = np.array([r["sigma"] for r in rows], float)
    coh = np.array([r["final_coherence"] for r in rows], float)
    kin = np.array([r["final_kinetic"] for r in rows], float)

    plt.figure(figsize=(6.0, 4.0), dpi=110)
    plt.plot(sig, coh, "o-", label="final coherence (|⟨ψ⟩|²)")
    plt.plot(sig, kin, "s-", label="final kinetic")
    plt.xlabel("sigma (√T)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

if __name__ == "__main__":
    _ensure_dirs()
    rows = sweep()
    tsv = "outputs/phase9_thermal.tsv"
    fig = "outputs/figs/phase9_trends.png"
    _save_tsv(rows, tsv)
    _plot(rows, fig)

    summary = {
        "tsv": tsv,
        "figure": fig,
        "n_cases": len(rows),
        "energy_drift_median": float(np.median([r["energy_drift"] for r in rows])),
        "final_coherence_by_sigma": {str(r["sigma"]): r["final_coherence"] for r in rows},
        "final_kinetic_by_sigma": {str(r["sigma"]): r["final_kinetic"] for r in rows},
    }
    print(json.dumps(summary, indent=2))
