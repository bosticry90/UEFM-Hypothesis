#!/usr/bin/env python3
"""
Phase 3.5 — QCA → continuum bridge via step-size refinement
Implements a standard 1D split-step quantum walk (two-component spinor).
We check convergence by halving the effective "time step" (angles) and
doubling steps while keeping total time fixed. The L2 difference between
coarse and fine evolutions should shrink (continuum-like limit).

Outputs:
  - outputs/phase3p5_qca_bridge.tsv
  - outputs/figs/qca_bridge_profiles.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

def coin_rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s],[s, c]], dtype=complex)

def shift_lr(psi):
    # psi: shape (2, N) with components [L,R]
    L, R = psi[0], psi[1]
    Lp = np.roll(L, +1)   # move L component left→ index+1 (visual convention)
    Rp = np.roll(R, -1)   # move R component right→ index-1
    return np.vstack([Lp, Rp])

def split_step_step(psi, th1, th2):
    # U = S_R C(th2) S_L C(th1)
    C1, C2 = coin_rot(th1), coin_rot(th2)
    psi = C1 @ psi
    # left shift acts on left component
    L, R = psi[0], psi[1]
    L = np.roll(L, +1)
    psi = np.vstack([L, R])
    psi = C2 @ psi
    # right shift acts on right component
    L, R = psi[0], psi[1]
    R = np.roll(R, -1)
    psi = np.vstack([L, R])
    return psi

def evolve(N=512, steps=300, th1=0.18, th2=0.21, seed=0):
    rng = np.random.default_rng(seed)
    psi = np.zeros((2, N), dtype=complex)
    # localized initial spinor
    psi[:, N//2] = np.array([1.0, 0.0], dtype=complex)
    psi /= np.linalg.norm(psi)
    for _ in range(steps):
        psi = split_step_step(psi, th1, th2)
    return psi

def l2_distance(a, b):
    return float(np.linalg.norm(a - b))

def main():
    N = 512
    T = 300
    th1, th2 = 0.18, 0.21

    # Coarse evolution: angles (th1, th2), steps=T
    psi_c = evolve(N=N, steps=T, th1=th1, th2=th2, seed=0)

    # Fine evolution: half angles, double steps keeps approximate total "time"
    psi_f = evolve(N=N, steps=2*T, th1=0.5*th1, th2=0.5*th2, seed=0)

    # Normalize (unitary, but be safe numerically)
    psi_c /= np.linalg.norm(psi_c); psi_f /= np.linalg.norm(psi_f)

    dist = l2_distance(psi_c.ravel(), psi_f.ravel())
    dens_c = np.sum(np.abs(psi_c)**2, axis=0)
    dens_f = np.sum(np.abs(psi_f)**2, axis=0)

    # Save metrics
    pd.DataFrame([dict(N=N, T=T, th1=th1, th2=th2, l2_coarse_vs_fine=dist)]).to_csv(
        os.path.join(OUT_DIR, "phase3p5_qca_bridge.tsv"), sep="\t", index=False
    )

    # Plot profiles
    x = np.arange(N) - N//2
    plt.figure(figsize=(10,4))
    plt.plot(x, dens_c, label="coarse")
    plt.plot(x, dens_f, "--", label="fine (half-angle, double-steps)")
    plt.xlabel("site"); plt.ylabel("density"); plt.legend(); plt.title("QCA step-refinement bridge")
    out_png = os.path.join(FIG_DIR, "qca_bridge_profiles.png")
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"Wrote {os.path.join(OUT_DIR, 'phase3p5_qca_bridge.tsv')} and {out_png}\nL2 distance = {dist:.3e}")

if __name__ == "__main__":
    main()
