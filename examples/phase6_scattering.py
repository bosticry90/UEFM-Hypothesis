#!/usr/bin/env python3
"""
Phase 6: Two-soliton (two-packet) scattering prototype (stable version).

Model (toy NLS):
    i ∂ψ/∂t = -α ∇² ψ + g |ψ|² ψ
Periodic in x∈[0,2π], y∈[0,1].  Time stepping: Strang split-step Fourier (unitary).

Artifacts:
  - TSV:     outputs/phase6_scattering.tsv
  - FIGURE:  outputs/figs/phase6_coherence_energy.png
"""

from __future__ import annotations
import json, os
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from toe.tensor_coherence import coherence_integral_numpy as coherence_dense

# ----------------------------- grid helpers ----------------------------------
def _domain(shape: Tuple[int,int]) -> Tuple[np.ndarray,np.ndarray,Tuple[float,float],Tuple[float,float]]:
    nx, ny = map(int, shape)
    Lx, Ly = 2*np.pi, 1.0
    x = np.linspace(0, Lx, nx, endpoint=True)
    y = np.linspace(0, Ly, ny, endpoint=True)
    dx = (Lx/(nx-1), Ly/(ny-1))
    # spectral wave-numbers consistent with periodic grid spacing
    kx = 2*np.pi*np.fft.fftfreq(nx, d=dx[0])
    ky = 2*np.pi*np.fft.fftfreq(ny, d=dx[1])
    return x, y, dx, (kx, ky)

def _lap_mult(alpha: float, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k2 = KX**2 + KY**2
    return np.exp(-1j * alpha * k2)  # for one unit of time; will be powered by dt

def _energy(psi: np.ndarray, dx: Tuple[float,float], alpha: float, g: float) -> float:
    # spectral gradient energy: ∫ α |∇ψ|^2 + (g/2)|ψ|^4
    nx, ny = psi.shape
    kx = 2*np.pi*np.fft.fftfreq(nx, d=dx[0])
    ky = 2*np.pi*np.fft.fftfreq(ny, d=dx[1])
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    grad2 = (np.fft.ifft2(1j*KX*np.fft.fft2(psi)))**2 + (np.fft.ifft2(1j*KY*np.fft.fft2(psi)))**2
    grad2 = np.real(np.conj(grad2) + 0*grad2)  # ensure real
    # |∇ψ|^2 = |∂x ψ|^2 + |∂y ψ|^2
    gx = np.fft.ifft2(1j*KX*np.fft.fft2(psi))
    gy = np.fft.ifft2(1j*KY*np.fft.fft2(psi))
    grad_sq = np.abs(gx)**2 + np.abs(gy)**2
    pot = 0.5*g*np.abs(psi)**4
    dA = dx[0]*dx[1]
    return float(dA*np.sum(alpha*grad_sq + pot))

def _momentum_like(psi: np.ndarray, dx: Tuple[float,float]) -> Tuple[float,float]:
    nx, ny = psi.shape
    kx = 2*np.pi*np.fft.fftfreq(nx, d=dx[0])
    ky = 2*np.pi*np.fft.fftfreq(ny, d=dx[1])
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    gx = np.fft.ifft2(1j*KX*np.fft.fft2(psi))
    gy = np.fft.ifft2(1j*KY*np.fft.fft2(psi))
    Px = float(dx[0]*dx[1]*np.sum(np.imag(np.conj(psi)*gx)))
    Py = float(dx[0]*dx[1]*np.sum(np.imag(np.conj(psi)*gy)))
    return Px, Py

def _coherence(psi: np.ndarray, dx: Tuple[float,float]) -> float:
    return float(coherence_dense(np.abs(psi), dx=dx))

# ------------------------------ params/init ----------------------------------
@dataclass
class SimParams:
    shape: Tuple[int,int] = (32,24)
    alpha: float = 0.5
    g: float = 0.6          # gentler nonlinearity to avoid stiffness
    dt: float = 6e-4        # smaller stable step (unitary scheme)
    steps: int = 800
    amp: float = 0.9
    sigma: float = 0.22
    k0: float = 3.5
    sep: float = 1.2
    dtheta: float = np.pi/2
    seed: int = 0

def _init_two_packets(p: SimParams, dx: Tuple[float,float]) -> Tuple[np.ndarray,Tuple[float,float],float]:
    nx, ny = p.shape
    Lx, Ly = 2*np.pi, 1.0
    x = np.linspace(0, Lx, nx, endpoint=True)
    y = np.linspace(0, Ly, ny, endpoint=True)
    X, Y = np.meshgrid(x, y, indexing="ij")

    x1 = (np.pi - p.sep/2.0) % (2*np.pi)
    x2 = (np.pi + p.sep/2.0) % (2*np.pi)
    y0 = 0.5

    r1 = np.exp(-((X-x1)**2 + (Y-y0)**2)/(2*p.sigma**2))
    r2 = np.exp(-((X-x2)**2 + (Y-y0)**2)/(2*p.sigma**2))

    phase1 = np.exp(1j*(+p.k0)*(X-x1))
    phase2 = np.exp(1j*(-p.k0)*(X-x2) + 1j*p.dtheta)
    psi0 = p.amp*(r1*phase1 + r2*phase2)

    # mild rescaling to keep |psi| within a safe envelope
    maxabs = np.max(np.abs(psi0))
    if maxabs > 1.5:
        psi0 = psi0*(1.5/maxabs)

    P0 = _momentum_like(psi0, dx)
    E0 = _energy(psi0, dx, p.alpha, p.g)
    return psi0, P0, E0

# -------------------------- split-step integrator ----------------------------
def _linear_step_fft(psi: np.ndarray, dt: float, alpha: float, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    # ψ(t+dt) = F^{-1}[exp(-i α k^2 dt) F[ψ(t)]]
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    k2 = KX**2 + KY**2
    phase = np.exp(-1j*alpha*k2*dt)
    return np.fft.ifft2(phase*np.fft.fft2(psi))

def _nonlinear_step(psi: np.ndarray, dt: float, g: float) -> np.ndarray:
    # exact local phase rotation: ψ -> exp(-i g |ψ|^2 dt) ψ
    return np.exp(-1j*g*np.abs(psi)**2*dt)*psi

def _step_strang(psi: np.ndarray, p: SimParams, kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    half = 0.5*p.dt
    psi = _nonlinear_step(psi, half, p.g)
    psi = _linear_step_fft(psi, p.dt, p.alpha, kx, ky)
    psi = _nonlinear_step(psi, half, p.g)
    # tiny soft cap to avoid numerical overflow if any
    amp = np.abs(psi)
    cap = 10.0
    mask = amp > cap
    if np.any(mask):
        psi[mask] *= (cap/amp[mask])
    return psi

# --------------------------------- runs --------------------------------------
def simulate(p: SimParams) -> Dict[str,object]:
    x, y, dx, (kx, ky) = _domain(p.shape)
    psi, P0, E0 = _init_two_packets(p, dx)

    times: List[float] = []
    energies: List[float] = []
    coherences: List[float] = []

    for t in range(p.steps):
        if t % 10 == 0:
            times.append(t*p.dt)
            energies.append(_energy(psi, dx, p.alpha, p.g))
            coherences.append(_coherence(psi, dx))
        psi = _step_strang(psi, p, kx, ky)

    E1 = _energy(psi, dx, p.alpha, p.g)
    C_series = np.array(coherences, float)
    return {
        "params": asdict(p),
        "energy_start": float(energies[0]),
        "energy_end": float(E1),
        "energy_rel_drift": float((E1-energies[0])/(abs(energies[0])+1e-12)),
        "coherence_series": C_series.tolist(),
        "energy_series": np.array(energies, float).tolist(),
        "time": np.array(times, float).tolist(),
        "momentum_like": {"Px": float(P0[0]), "Py": float(P0[1])},
    }

def run_sweep() -> Dict[str,object]:
    os.makedirs("outputs/figs", exist_ok=True)
    records = []
    k0_list = [2.0, 3.0, 4.0]
    dtheta_list = [0.0, np.pi/2, np.pi]
    for k0 in k0_list:
        for dth in dtheta_list:
            p = SimParams(k0=k0, dtheta=dth, steps=600, dt=6e-4, amp=0.9, sigma=0.22, sep=1.2, g=0.6)
            res = simulate(p)
            C = np.array(res["coherence_series"])
            records.append({
                "k0": k0, "dtheta": float(dth),
                "E0": res["energy_start"], "E1": res["energy_end"],
                "dE_over_E": res["energy_rel_drift"],
                "C_start": float(C[0]), "C_min": float(C.min()), "C_end": float(C[-1]),
            })
    df = pd.DataFrame.from_records(records)
    tsv_path = "outputs/phase6_scattering.tsv"
    df.to_csv(tsv_path, sep="\t", index=False)

    # demo plot
    p_demo = SimParams(k0=4.0, dtheta=np.pi/2, steps=800, dt=6e-4, amp=0.9, sigma=0.22, sep=1.2, g=0.6)
    demo = simulate(p_demo)
    t = np.array(demo["time"]); C = np.array(demo["coherence_series"]); E = np.array(demo["energy_series"])
    fig_path = "outputs/figs/phase6_coherence_energy.png"
    plt.figure(figsize=(6.4,3.4))
    plt.plot(t, (C-C[0])/(abs(C[0])+1e-12), label="rel. coherence")
    plt.plot(t, (E-E[0])/(abs(E[0])+1e-12), label="rel. energy")
    plt.axhline(0.0, ls="--", lw=0.8); plt.xlabel("time"); plt.ylabel("relative change")
    plt.legend(); plt.tight_layout(); plt.savefig(fig_path, dpi=150); plt.close()

    summary = {
        "tsv": tsv_path, "figure": fig_path, "n_cases": int(len(records)),
        "energy_drift_median": float(np.median(np.abs(df["dE_over_E"]))),
        "coherence_dip_median": float(np.median(df["C_start"] - df["C_min"])),
    }
    print(json.dumps(summary, indent=2))
    return summary

def quick_demo(return_data: bool=False):
    p = SimParams(shape=(24,16), steps=240, dt=7e-4, k0=3.0, dtheta=np.pi/2, amp=0.9, sigma=0.25, sep=1.0, g=0.6)
    res = simulate(p)
    if return_data:
        return res
    os.makedirs("outputs/figs", exist_ok=True)
    t = np.array(res["time"]); C = np.array(res["coherence_series"]); E = np.array(res["energy_series"])
    plt.figure(figsize=(6.0,3.2))
    plt.plot(t, C/(abs(C[0])+1e-12), label="coherence / C0")
    plt.plot(t, E/(abs(E[0])+1e-12), label="energy / E0")
    plt.legend(); plt.tight_layout(); plt.savefig("outputs/figs/phase6_quick_demo.png", dpi=130); plt.close()

if __name__ == "__main__":
    os.makedirs("outputs/figs", exist_ok=True)
    run_sweep()
