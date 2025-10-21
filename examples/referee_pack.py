# examples/referee_pack.py
# Referee validation: invariance, noise monotonicity, and locality.
# Writes TSVs, figures, and a LaTeX summary snippet.

from __future__ import annotations
import os
from pathlib import Path
import types
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from toe.tensor_coherence import (
    coherence_integral_numpy,
    coherence_integral_tt,
)
from toe.qca import SplitStepQCA1D, energy_conservation_proxy

# ------------------------- helpers -------------------------

def ensure_dirs():
    Path("outputs/figs").mkdir(parents=True, exist_ok=True)
    Path("docs/auto").mkdir(parents=True, exist_ok=True)

def tt_value(res) -> tuple[float, bool, float | None]:
    """Make coherence_integral_tt() return shape-agnostic.
    Returns (value, ok, approx_rel_error)."""
    # new API returns a SimpleNamespace with fields
    if isinstance(res, types.SimpleNamespace):
        val = float(getattr(res, "integral_value", np.nan))
        ok = bool(getattr(res, "ok", False))
        err = getattr(res, "approx_rel_error", None)
        if err is not None:
            err = float(err)
        return val, ok, err
    # fallback: older APIs might return a bare float
    try:
        return float(res), False, None
    except Exception:
        return np.nan, False, None

def rel_change(a: float, b: float) -> float:
    denom = max(1.0, abs(a))
    return abs(a - b) / denom

# ------------------------- 1) invariance checks -------------------------

def run_invariance_suite():
    # small 2D field
    x = np.linspace(-1.0, 1.0, 64)
    X, Y = np.meshgrid(x, x, indexing="ij")
    phi = np.tanh(2 * X) * np.exp(-(X**2 + Y**2))

    # baseline (dense + TT)
    dense0 = coherence_integral_numpy(phi)  # dx default = 1.0
    tt0_val, tt0_ok, tt0_err = tt_value(
        coherence_integral_tt(phi, max_rank=8, backend="numpy", V=None, estimate_error=True)
    )

    # transformed variants (no external deps)
    variants = {
        "transpose": phi.T,
        "flipud": np.flipud(phi),
        "fliplr": np.fliplr(phi),
        "roll_xy": np.roll(np.roll(phi, 1, axis=0), -2, axis=1),
        "transpose_roll": np.roll(phi.T, 3, axis=1),
    }

    rows = []
    for name, ph in variants.items():
        d = coherence_integral_numpy(ph)
        t_val, t_ok, t_err = tt_value(
            coherence_integral_tt(ph, max_rank=8, backend="numpy", V=None, estimate_error=True)
        )
        rows.append(
            dict(
                transform=name,
                dense=d,
                tt=t_val,
                dense_rel_change=rel_change(d, dense0),
                tt_rel_change=rel_change(t_val, tt0_val),
                tt_ok=int(t_ok),
                tt_rel_err=(np.nan if t_err is None else t_err),
            )
        )

    df = pd.DataFrame(rows)
    df.to_csv("outputs/referee_invariance.tsv", sep="\t", index=False)

    # plots
    plt.figure()
    plt.bar(df["transform"], df["dense_rel_change"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Relative change vs. baseline (dense)")
    plt.title("Referee: Invariance (Dense)")
    plt.tight_layout()
    plt.savefig("outputs/figs/referee_invariance_dense.png", dpi=160)
    plt.close()

    plt.figure()
    plt.bar(df["transform"], df["tt_rel_change"])
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Relative change vs. baseline (TT)")
    plt.title("Referee: Invariance (TT)")
    plt.tight_layout()
    plt.savefig("outputs/figs/referee_invariance_tt.png", dpi=160)
    plt.close()

    # summary numbers
    med_dense = float(np.median(df["dense_rel_change"].to_numpy()))
    med_tt = float(np.median(df["tt_rel_change"].to_numpy()))

    return {
        "dense_med_rel": med_dense,
        "tt_med_rel": med_tt,
        "dense0": float(dense0),
        "tt0": float(tt0_val),
    }

# ------------------------- 2) noise monotonicity -------------------------

def run_noise_monotonicity():
    qca = SplitStepQCA1D(n_sites=8, d=2, seed=2)
    psi0 = np.zeros(2**8, dtype=complex)
    psi0[0] = 1.0

    ps = [0.0, 0.05, 0.1, 0.2, 0.3]
    drifts = []
    for p in ps:
        norms = energy_conservation_proxy(qca, psi0, steps=10, p=p)
        drift = float(np.max(np.abs(norms - norms[0])))
        drifts.append(drift)

    df = pd.DataFrame(dict(p=ps, drift=drifts))
    df.to_csv("outputs/referee_noise.tsv", sep="\t", index=False)

    # plot
    plt.figure()
    plt.plot(ps, drifts, "o-")
    plt.xlabel("Noise p")
    plt.ylabel("Max |norm - norm0|")
    plt.title("Referee: Noise Monotonicity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/figs/referee_noise.png", dpi=160)
    plt.close()

    # monotonic score: fraction of successive non-decreasing steps
    diffs = np.diff(drifts)
    mono_frac = float(np.mean(diffs >= -1e-12))
    return {"mono_frac": mono_frac, "drift0": drifts[0], "drift_last": drifts[-1]}

# ------------------------- 3) locality / lightcone -------------------------

def run_lightcone():
    # Larger N, moderate T
    N = 101
    T = 20
    qca = SplitStepQCA1D(n_sites=N, d=2, seed=11, theta1=0.3, theta2=0.34)
    radius = int(qca.lightcone_radius(steps=T))
    v_eff = radius / T

    df = pd.DataFrame(dict(T=[T], radius=[radius], v_eff=[v_eff]))
    df.to_csv("outputs/referee_lightcone.tsv", sep="\t", index=False)

    # bar plot
    plt.figure()
    plt.bar(["radius", "T"], [radius, T])
    plt.title(f"Referee: Lightcone Radius (T={T}, v_eff={v_eff:.2f})")
    plt.tight_layout()
    plt.savefig("outputs/figs/referee_lightcone.png", dpi=160)
    plt.close()

    return {"T": T, "radius": radius, "v_eff": float(v_eff)}

# ------------------------- LaTeX summary -------------------------

def write_latex_summary(inv, noise, lc):
    tex = rf"""
% Auto-generated by examples/referee_pack.py
\newcommand{{\RefereeInvDense}}{{{inv['dense_med_rel']:.2e}}}
\newcommand{{\RefereeInvTT}}{{{inv['tt_med_rel']:.2e}}}
\newcommand{{\RefereeNoiseMono}}{{{noise['mono_frac']:.2f}}}
\newcommand{{\RefereeVEff}}{{{lc['v_eff']:.2f}}}

\begin{{table}}[h]
\centering
\begin{{tabular}}{{l c}}
\toprule
Metric & Value \\
\midrule
Median rel. change (dense) & \RefereeInvDense \\
Median rel. change (TT)    & \RefereeInvTT \\
Noise monotonicity fraction & \RefereeNoiseMono \\
Effective locality speed $v_\mathrm{{eff}}$ & \RefereeVEff \\
\bottomrule
\end{{tabular}}
\caption{{Referee validation summary: invariance, noise monotonicity, and lightcone locality.}}
\end{{table}}
""".lstrip()
    with open("docs/auto/referee_summary.tex", "w", encoding="utf-8") as f:
        f.write(tex)

# ------------------------- main -------------------------

if __name__ == "__main__":
    ensure_dirs()
    inv = run_invariance_suite()
    noise = run_noise_monotonicity()
    lc = run_lightcone()
    write_latex_summary(inv, noise, lc)

    print("Referee pack complete.")
    print(f" invariance med rel changes: dense={inv['dense_med_rel']:.2e}, tt={inv['tt_med_rel']:.2e}")
    print(f" noise monotonicity fraction: {noise['mono_frac']:.2f}")
    print(f" v_eff (radius/T): {lc['v_eff']:.2f}")
