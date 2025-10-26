# examples/plot_tensor_vs_dense.py
import os
import numpy as np
import matplotlib.pyplot as plt

from toe.tensor_coherence import (
    TENSORLY_AVAILABLE,
    as_tt,
    reconstruct,
    coherence_dense,
    coherence_tt,
)


def main():
    # Make sure outputs dirs exist
    out_dir = os.path.join("outputs", "figs")
    os.makedirs(out_dir, exist_ok=True)

    # Small/moderate grid so dense recon is cheap
    shape = (48, 24)
    x = np.linspace(0, 2*np.pi, shape[0], endpoint=True)
    y = np.linspace(0, 1.0,      shape[1], endpoint=True)
    X, Y = np.meshgrid(x, y, indexing="ij")
    phi = np.cos(1.7*X) * np.exp(-0.8*Y)
    dx = (x[1] - x[0], y[1] - y[0])

    # Dense coherence for reference
    C_dense = coherence_dense(phi, dx)

    ranks = [2, 4, 6, 8, 12, 16, 24]
    rec_errs = []
    coh_errs = []

    if not TENSORLY_AVAILABLE:
        print("TensorLy not available; only dense value will be shown.")
    else:
        for r in ranks:
            tt = as_tt(phi, rank=r)
            rec = reconstruct(tt)
            rec_err = np.linalg.norm(rec - phi) / (np.linalg.norm(phi) + 1e-16)
            C_tt = coherence_tt(tt, dx)
            coh_err = abs(C_tt - C_dense) / (abs(C_dense) + 1e-16)

            rec_errs.append(rec_err)
            coh_errs.append(coh_err)
            print(f"rank={r:2d} | rel_recon_err={rec_err:.3e} | rel_coh_err={coh_err:.3e}")

    # Plot
    plt.figure()
    if TENSORLY_AVAILABLE and rec_errs:
        plt.plot(ranks, rec_errs, marker="o", label="TT reconstruction error")
        plt.plot(ranks, coh_errs, marker="s", label="TT coherence error")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("TT rank")
    plt.ylabel("Relative error")
    plt.title("Tensor vs Dense: reconstruction & coherence error")
    plt.legend()
    out_png = os.path.join(out_dir, "tensor_vs_dense.png")
    plt.savefig(out_png, bbox_inches="tight", dpi=150)
    print(f"Saved figure to {out_png}")

    # Also print the dense reference so it's visible in logs
    print(f"Dense coherence (reference): {C_dense:.6e}")


if __name__ == "__main__":
    main()
