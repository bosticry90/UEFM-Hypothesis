#!/usr/bin/env python3
"""
Phase 3.5 — Superfluid-style mapping
Construct a complex field ψ = A(x,y) e^{iθ(x,y)}, compare kinetic density |∇ψ|^2
to A^2 |∇θ|^2 + |∇A|^2 (Madelung identity proxy). Quantify RMSE and correlation.

Outputs:
  - outputs/phase3p5_superfluid.tsv
  - outputs/figs/superfluid_map.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

def make_complex_field(n=192, L=6.0, kx=1.2, ky=0.6, sigma=2.0):
    x = np.linspace(-L, L, n); y = np.linspace(-L, L, n)
    X, Y = np.meshgrid(x, y, indexing="ij")
    A = np.exp(-(X**2 + Y**2)/(2*sigma**2)) * (1 + 0.2*np.cos(0.8*X)*np.sin(0.7*Y))
    theta = kx*X + ky*Y + 0.25*np.sin(0.9*X)*np.cos(0.6*Y)
    psi = A * np.exp(1j*theta)
    return x, y, psi, A, theta

def grad(f, dx):
    return np.gradient(f, dx, dx, edge_order=2)

def main():
    x, y, psi, A, theta = make_complex_field()
    dx = x[1]-x[0]

    # numerical gradients
    Ax, Ay = grad(A, dx)
    thx, thy = grad(theta, dx)
    # ∇ψ via product rule
    psix = (Ax + 1j*A*thx) * np.exp(1j*theta)
    psiy = (Ay + 1j*A*thy) * np.exp(1j*theta)
    kin = np.abs(psix)**2 + np.abs(psiy)**2                # |∇ψ|^2
    kin_pred = (A**2) * (thx**2 + thy**2) + (Ax**2 + Ay**2)

    # metrics
    mask = A > (0.02*np.max(A))  # avoid divide-by-noise in tails
    diff = kin[mask] - kin_pred[mask]
    rmse = float(np.sqrt(np.mean(diff**2)))
    rel = float(np.mean(np.abs(diff) / (1e-12 + kin[mask])))
    corr = float(np.corrcoef(kin[mask].ravel(), kin_pred[mask].ravel())[0,1])

    pd.DataFrame([dict(rmse=rmse, rel_mae=rel, corr=corr, dx=float(dx))]).to_csv(
        os.path.join(OUT_DIR, "phase3p5_superfluid.tsv"), sep="\t", index=False
    )

    fig, axs = plt.subplots(1, 3, figsize=(12,4), constrained_layout=True)
    im0 = axs[0].imshow(A.T, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()])
    axs[0].set_title("Amplitude $A$"); plt.colorbar(im0, ax=axs[0], fraction=0.046)
    im1 = axs[1].imshow(kin.T, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()])
    axs[1].set_title(r"Kinetic density $|\nabla \psi|^2$"); plt.colorbar(im1, ax=axs[1], fraction=0.046)
    im2 = axs[2].imshow(kin_pred.T, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()])
    axs[2].set_title(r"$A^2|\nabla\theta|^2 + |\nabla A|^2$"); plt.colorbar(im2, ax=axs[2], fraction=0.046)

    out_png = os.path.join(FIG_DIR, "superfluid_map.png")
    plt.savefig(out_png, dpi=160)
    print(f"Wrote {os.path.join(OUT_DIR, 'phase3p5_superfluid.tsv')} and {out_png}")

if __name__ == "__main__":
    main()
