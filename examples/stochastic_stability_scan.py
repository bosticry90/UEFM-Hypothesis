#!/usr/bin/env python3
"""
Phase 3.5 — Stochastic robustness of coherence (signature-agnostic)
Perturb a base field with (i) potential weight λ, and (ii) additive noise ε,
and evaluate the coherence integral:
    C = ∫ (|∇φ|^2 + V(φ)) dx dy
Outputs:
  - outputs/phase3p5_stochastic.tsv
  - outputs/figs/stochastic_stability.png
"""

import os
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figs")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ---------- Local fallbacks ----------
def _local_potential_quartic(lam: float):
    def V(phi):
        return 0.5 * (phi**2) + lam * (phi**4) / 4.0
    return V

def _local_grad_integral(phi: np.ndarray, dx: float) -> float:
    gx, gy = np.gradient(phi, dx, dx, edge_order=2)
    return float(np.sum(gx**2 + gy**2) * dx * dx)

def _local_coherence(phi: np.ndarray, dx: float, V=None) -> float:
    val = _local_grad_integral(phi, dx)
    if V is not None:
        val += float(np.sum(V(phi)) * dx * dx)
    return val

# ---------- Try project implementation, else fall back ----------
_project_coh = None
_project_pot_quartic = None
try:
    from toe.tensor_coherence import coherence_integral_numpy as _c
    _project_coh = _c
    try:
        from toe.tensor_coherence import potential_quartic as _pq
        _project_pot_quartic = _pq
    except Exception:
        _project_pot_quartic = None
except Exception:
    _project_coh = None
    _project_pot_quartic = None

def coherence_integral(phi: np.ndarray, dx: float, V=None) -> float:
    """
    Wrapper that calls project coherence if available.
    - If project function supports V, passes it.
    - If not, computes grad term via project function and adds potential manually.
    - Otherwise uses local implementation.
    """
    if _project_coh is None:
        return _local_coherence(phi, dx, V)

    sig = inspect.signature(_project_coh)
    if "V" in sig.parameters:
        # project API supports V
        return float(_project_coh(phi, dx=dx, V=V))
    else:
        # project API is grad-only: add potential separately
        grad_val = float(_project_coh(phi, dx=dx))
        pot_val = 0.0 if V is None else float(np.sum(V(phi)) * dx * dx)
        return grad_val + pot_val

def potential_quartic(lam: float):
    if _project_pot_quartic is not None:
        return _project_pot_quartic(lam)
    return _local_potential_quartic(lam)

# ---------- Field and experiment ----------
def base_field(n=256, L=8.0):
    x = np.linspace(-L, L, n); y = np.linspace(-L, L, n)
    X, Y = np.meshgrid(x, y, indexing="ij")
    # smooth localized kink-like texture
    phi = np.tanh(0.8*X) * np.exp(-(X**2 + 0.5*Y**2)/3.0)
    return x, y, phi

def main():
    x, y, phi0 = base_field()
    dx = float(x[1] - x[0])
    rng = np.random.default_rng(0)

    lam_list = [0.0, 0.2, 0.5, 1.0]
    eps_list = [0.0, 1e-4, 5e-4, 1e-3, 2e-3]

    rows = []
    for lam in lam_list:
        V = potential_quartic(lam) if lam > 0 else None
        for eps in eps_list:
            noise = eps * rng.standard_normal(phi0.shape)
            phi = phi0 + noise
            C = coherence_integral(phi, dx=dx, V=V)
            rows.append(dict(lam=lam, eps=eps, coherence=float(C)))

    df = pd.DataFrame(rows)
    # Normalize per-λ to show relative drift with noise
    base = df[df["eps"] == 0].set_index("lam")["coherence"]
    df["coh_rel"] = df.apply(lambda r: r["coherence"] / (base.get(r["lam"], r["coherence"]) + 1e-12), axis=1)

    out_tsv = os.path.join(OUT_DIR, "phase3p5_stochastic.tsv")
    df.to_csv(out_tsv, sep="\t", index=False)

    # Plot
    plt.figure(figsize=(6.6, 4.2))
    for lam, sub in df.groupby("lam"):
        sub = sub.sort_values("eps")
        plt.plot(sub["eps"], sub["coh_rel"], "o-", label=f"λ={lam}")
    plt.xlabel("noise ε")
    plt.ylabel("relative coherence")
    plt.title("Stochastic stability of coherence")
    plt.legend()
    out_png = os.path.join(FIG_DIR, "stochastic_stability.png")
    plt.savefig(out_png, dpi=160, bbox_inches="tight")
    print(f"Wrote {out_tsv} and {out_png}")

if __name__ == "__main__":
    main()
