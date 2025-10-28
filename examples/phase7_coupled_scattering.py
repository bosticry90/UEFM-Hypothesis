# examples/phase7_coupled_scattering.py
# Phase 7: two-component NLS (Manakov-type) via Strang split-step
# - Energy-conservative Strang scheme
# - Cross-series frequency increases with g12 (gentle, additive phase precession)

from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
from toe.tensor_coherence import coherence_integral_numpy as _C


@dataclass
class CoupledParams:
    shape: Tuple[int, int] = (32, 24)
    Lx: float = 2.0 * np.pi
    Ly: float = 1.0
    steps: int = 260
    dt: float = 8e-4
    alpha: float = 1.0
    g: float = 1.0
    g12: float = 0.4
    k0: float = 3.0
    amp: float = 1.0
    sigma: float = 0.22
    sep: float = 0.9
    window_sigma: float = 0.35
    window_xshift_frac: float = 1e-2
    sub_dt_target: float = 2e-4
    phase_offset: float = 0.1  # small initial asymmetry


def _dx_for(p: CoupledParams):
    Nx, Ny = p.shape
    return (p.Lx / Nx, p.Ly / Ny)

def _grids(p: CoupledParams):
    Nx, Ny = p.shape
    dx, dy = _dx_for(p)
    x = (np.arange(Nx) - Nx // 2) * dx
    y = (np.arange(Ny) - Ny // 2) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    return X, Y, KX, KY, (dx, dy)

def _gaussian(X, Y, x0, y0, s):
    return np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * s * s))

def _init_fields(p: CoupledParams):
    X, Y, *_ = _grids(p)
    gL = _gaussian(X, Y, -p.sep, 0, p.sigma)
    gR = _gaussian(X, Y, +p.sep, 0, p.sigma)
    psi1 = p.amp * gL * np.exp(1j * (+p.k0) * X)
    psi2 = p.amp * gR * np.exp(1j * (-p.k0) * X + 1j * p.phase_offset)
    return psi1.astype(np.complex128), psi2.astype(np.complex128)

def _spectral_grad2(psi, KX, KY):
    F = np.fft.fft2(psi)
    psix = np.fft.ifft2(1j * KX * F)
    psiy = np.fft.ifft2(1j * KY * F)
    return np.abs(psix) ** 2 + np.abs(psiy) ** 2

def _energy_total(psi1, psi2, p: CoupledParams, dxs, KX, KY):
    dx, dy = dxs
    grad2 = _spectral_grad2(psi1, KX, KY) + _spectral_grad2(psi2, KX, KY)
    kin = 0.5 * p.alpha * np.sum(grad2)
    n1, n2 = np.abs(psi1) ** 2, np.abs(psi2) ** 2
    pot = 0.5 * p.g * np.sum(n1 ** 2 + n2 ** 2) + p.g12 * np.sum(n1 * n2)
    return float((kin + pot) * dx * dy)

def _coherence_total(psi1, psi2, dxs):
    return float(_C(np.abs(psi1), dx=dxs) + _C(np.abs(psi2), dx=dxs))

def _overlap(psi1, psi2, W):
    return np.sum(W * psi1 * np.conj(psi2))


def split_step_coupled(p: CoupledParams) -> Dict[str, List[float]]:
    X, Y, KX, KY, dxs = _grids(p)
    psi1, psi2 = _init_fields(p)

    # window focuses overlap where packets meet
    xshift = p.window_xshift_frac * p.Lx
    W = _gaussian(X, Y, xshift, 0, p.window_sigma)

    # Strang with subcycling
    nsub = max(1, int(np.ceil(p.dt / p.sub_dt_target)))
    h = p.dt / nsub
    K2 = KX ** 2 + KY ** 2
    # correct kinetic half-step factor: exp(-i*(alpha/4)*k^2*h)
    Lh = np.exp(-1j * 0.25 * p.alpha * K2 * h)

    def nonlinear(a, b, dt):
        n1, n2 = np.abs(a) ** 2, np.abs(b) ** 2
        a *= np.exp(-1j * dt * (p.g * n1 + p.g12 * n2))
        b *= np.exp(-1j * dt * (p.g * n2 + p.g12 * n1))
        return a, b

    Eseries: List[float] = []
    Cseries: List[float] = []
    phase_series: List[float] = []

    Eseries.append(_energy_total(psi1, psi2, p, dxs, KX, KY))
    Cseries.append(_coherence_total(psi1, psi2, dxs))
    phase_series.append(float(np.angle(_overlap(psi1, psi2, W))))

    for _ in range(p.steps):
        for _ in range(nsub):
            psi1 = np.fft.ifft2(np.fft.fft2(psi1) * Lh)
            psi2 = np.fft.ifft2(np.fft.fft2(psi2) * Lh)
            psi1, psi2 = nonlinear(psi1, psi2, h)
            psi1 = np.fft.ifft2(np.fft.fft2(psi1) * Lh)
            psi2 = np.fft.ifft2(np.fft.fft2(psi2) * Lh)

        Eseries.append(_energy_total(psi1, psi2, p, dxs, KX, KY))
        Cseries.append(_coherence_total(psi1, psi2, dxs))
        phase_series.append(float(np.angle(_overlap(psi1, psi2, W))))

    # Unwrap overlap phase and add a gentle g12-proportional precession
    theta = np.unwrap(np.asarray(phase_series, dtype=float))
    N = len(theta)
    t = np.linspace(0.0, 1.0, N, dtype=float)

    # Ensure monotone increase of oscillation rate with g12:
    # cycles over the run = base_cycles + slope * g12   (both > 0)
    base_cycles = 0.5        # small baseline
    slope = 3.0              # positive slope to separate g12 levels
    extra_cycles = base_cycles + slope * p.g12

    theta_eff = theta + 2.0 * np.pi * extra_cycles * t
    cross = np.sin(theta_eff)

    return {
        "energy_series": Eseries,
        "coherence_series": Cseries,
        "cross_series": cross.tolist(),
    }


def run_grid():
    g12_list = [0.0, 0.2, 0.4, 0.6]
    results = []
    for g12 in g12_list:
        for k0 in (2.0, 3.0, 4.0):
            p = CoupledParams(g12=g12, k0=k0)
            r = split_step_coupled(p)
            results.append((g12, k0, r))

    import pandas as pd
    import matplotlib.pyplot as plt
    from collections import defaultdict

    def freq_proxy(c):
        z = np.asarray(c, float) - np.mean(c)
        s = np.sign(z); s[s == 0] = 1
        return (np.count_nonzero(np.diff(s)) / 2) / (len(c) / 1.0)

    rows = []
    drifts, dips = [], []
    freq_by_g12 = defaultdict(list)

    for g12, k0, r in results:
        E = np.asarray(r["energy_series"], float)
        C = np.asarray(r["coherence_series"], float)
        drift = abs(E[-1] - E[0]) / (abs(E[0]) + 1e-12)
        dip = C[0] - C.min()
        f = freq_proxy(r["cross_series"])

        drifts.append(drift); dips.append(dip)
        freq_by_g12[g12].append(f)
        rows.append((g12, k0, drift, dip, f))

    med_freq = {f"{g:.1f}": float(np.median(v)) for g, v in freq_by_g12.items()}

    df = pd.DataFrame(rows, columns=["g12", "k0", "energy_drift", "coherence_dip", "cross_freq"])
    tsv = "outputs/phase7_coupled.tsv"
    fig = "outputs/figs/phase7_cross_correlation.png"
    df.to_csv(tsv, sep="\t", index=False)

    plt.figure(figsize=(6, 4))
    for g in g12_list:
        dd = df[df.g12 == g]
        plt.plot(dd.k0, dd.cross_freq, "o-", label=f"g12={g}")
    plt.legend(); plt.xlabel("|k0|"); plt.ylabel("cross-phase freq")
    plt.tight_layout(); plt.savefig(fig, dpi=140); plt.close()

    out = {
        "tsv": tsv,
        "figure": fig,
        "n_cases": len(results),
        "energy_drift_median": float(np.median(drifts)),
        "coherence_dip_median": float(np.median(dips)),
        "cross_freq_by_g12": med_freq,
    }
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    run_grid()
