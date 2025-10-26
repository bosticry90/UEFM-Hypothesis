#!/usr/bin/env python3
"""
Phase 3.5 — Curvature/metric proxy vs coherence
Builds a 2D synthetic field, computes |∇φ|^2, a simple curvature proxy
K := -∇^2 log(1 + λ |∇φ|^2), and correlates K with coherence density.

Outputs:
  - outputs/phase3p5_curvature.tsv
  - outputs/figs/curvature_vs_coherence.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from toe.tensor_coherence import coherence_integral_numpy
except Exception:
    coherence_integral_numpy = None

OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

def make_field(n=192, L=6.0, kx=2.0, sigma=1.6):
    x = np.linspace(-L, L, n)
    y = np.linspace(-L, L, n)
    X, Y = np.meshgrid(x, y, indexing="ij")
    phi = np.tanh(kx * X) * np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    return x, y, phi

def grad2(phi, dx):
    gx, gy = np.gradient(phi, dx, dx, edge_order=2)
    return gx, gy, gx**2 + gy**2

def laplacian(f, dx):
    dxx = (np.roll(f, -1, axis=0) - 2*f + np.roll(f, 1, axis=0)) / (dx**2)
    dyy = (np.roll(f, -1, axis=1) - 2*f + np.roll(f, 1, axis=1)) / (dx**2)
    return dxx + dyy

def main():
    x, y, phi = make_field()
    dx = abs(x[1]-x[0])

    gx, gy, g2 = grad2(phi, dx)
    lam = 0.75
    log_term = np.log1p(lam * g2)
    K = -laplacian(log_term, dx)  # simple scalar “curvature” proxy

    coh = None
    if coherence_integral_numpy is not None:
        # Use coherence density proxy locally: |∇φ|^2 + V(φ) with V=0 (density only)
        # We also compute the global integral for reference.
        coh_int = coherence_integral_numpy(phi, dx=dx)
        coh = g2  # local map

    # Summaries
    stats = dict(
        g2_min=float(np.min(g2)), g2_max=float(np.max(g2)), g2_mean=float(np.mean(g2)),
        K_min=float(np.min(K)),   K_max=float(np.max(K)),   K_mean=float(np.mean(K)),
        lam=float(lam),
        dx=float(dx),
        coherence_integral=float(coh_int) if coherence_integral_numpy is not None else float("nan"),
        corr_K_g2=float(np.corrcoef(K.ravel(), g2.ravel())[0,1])
    )
    pd.DataFrame([stats]).to_csv(os.path.join(OUT_DIR, "phase3p5_curvature.tsv"), sep="\t", index=False)

    # Figure
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    im0 = axs[0].imshow(phi.T, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()])
    axs[0].set_title("Field $\\phi(x,y)$"); fig.colorbar(im0, ax=axs[0], fraction=0.046)
    im1 = axs[1].imshow(g2.T, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()])
    axs[1].set_title(r"$|\nabla \phi|^2$"); fig.colorbar(im1, ax=axs[1], fraction=0.046)
    im2 = axs[2].imshow(K.T, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()])
    axs[2].set_title(r"$K = -\nabla^2 \log(1+\lambda|\nabla\phi|^2)$"); fig.colorbar(im2, ax=axs[2], fraction=0.046)

    out_png = os.path.join(FIG_DIR, "curvature_vs_coherence.png")
    plt.savefig(out_png, dpi=160)
    print(f"Wrote {os.path.join(OUT_DIR, 'phase3p5_curvature.tsv')} and {out_png}")

if __name__ == "__main__":
    main()
