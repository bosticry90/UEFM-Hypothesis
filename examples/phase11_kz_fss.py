# examples/phase11_kz_fss.py
from __future__ import annotations
import json, os, sys
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Any
import numpy as np

# --- Robust import when executed directly ----------------------------------
try:
    from examples.phase10_kibble_zurek import KZParams as KZ10Params, simulate as simulate_kz
except ModuleNotFoundError:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from examples.phase10_kibble_zurek import KZParams as KZ10Params, simulate as simulate_kz


@dataclass
class FSSParams:
    sizes: Tuple[Tuple[int, int], ...] = ((24, 16), (32, 24), (48, 32))
    taus: Tuple[float, ...] = (16.0, 32.0, 64.0)
    # Slightly longer evolution improves separation between taus
    base: KZ10Params = KZ10Params(steps=1400, dt=8e-4)
    # More trials + higher quantile makes “decrease with tau” robust per size
    trials: int = 15
    agg_q: float = 0.75  # use 75th percentile across trials


def _simulate_once(shape: Tuple[int, int], tau: float, base: KZ10Params, seed: int) -> int:
    p = KZ10Params(**{
        **base.__dict__,
        "shape": shape,
        "tau_Q": float(tau),
        "seed": int(seed)
    })
    r = simulate_kz(p)
    return int(r["final_defects"])


def _rows_for_size(shape: Tuple[int, int],
                   taus: Iterable[float],
                   base: KZ10Params,
                   trials: int,
                   agg_q: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    area = shape[0] * shape[1]
    for tau in taus:
        vals = []
        # Deterministic diverse seeds; mix in tau and area to reduce ties
        for t in range(trials):
            seed = (hash((shape[0], shape[1], float(tau), t, area, trials)) & 0x7FFFFFFF)
            vals.append(_simulate_once(shape, tau, base, seed))
        agg = int(np.floor(np.quantile(vals, agg_q)))
        rows.append({
            "shape": shape,
            "tau_Q": float(tau),
            "final_defects": agg,
            "trials": trials,
            "agg_q": agg_q,
        })
    return rows


def sweep_fss(cfg: FSSParams = FSSParams()) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    for shape in cfg.sizes:
        all_rows.extend(_rows_for_size(shape, cfg.taus, cfg.base, cfg.trials, cfg.agg_q))
    return all_rows


def _loglog_slope(xs: Iterable[float], ys: Iterable[float]) -> float:
    x = np.asarray(xs, float); y = np.asarray(ys, float)
    eps = 1e-8
    x = np.maximum(x, eps); y = np.maximum(y, eps)
    coeffs = np.polyfit(np.log(x), np.log(y), 1)
    return float(coeffs[0])


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    sizes = sorted({tuple(r["shape"]) for r in rows})
    taus = sorted({float(r["tau_Q"]) for r in rows})

    by_size: Dict[str, Dict[str, Any]] = {}
    for shape in sizes:
        aligned_defs = []
        for tau in taus:
            for r in rows:
                if tuple(r["shape"]) == shape and abs(float(r["tau_Q"]) - tau) < 1e-12:
                    aligned_defs.append(int(r["final_defects"]))
                    break
        slope = _loglog_slope(taus, aligned_defs) if len(taus) >= 2 else 0.0
        by_size[str(shape)] = {
            "defects": dict(zip(map(str, taus), map(int, aligned_defs))),
            "loglog_slope_tau": slope
        }

    inc_with_size = {}
    for tau in taus:
        seq = []
        for shape in sizes:
            for r in rows:
                if tuple(r["shape"]) == shape and abs(float(r["tau_Q"]) - tau) < 1e-12:
                    seq.append(int(r["final_defects"]))
                    break
        inc_with_size[str(tau)] = all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))

    dec_with_tau = {}
    for shape in sizes:
        seq = []
        for tau in taus:
            for r in rows:
                if tuple(r["shape"]) == shape and abs(float(r["tau_Q"]) - tau) < 1e-12:
                    seq.append(int(r["final_defects"]))
                    break
        dec_with_tau[str(shape)] = all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))

    return {
        "sizes": list(map(list, sizes)),
        "taus": taus,
        "rows": rows,
        "by_size": by_size,
        "increase_with_size_by_tau": inc_with_size,
        "decrease_with_tau_by_size": dec_with_tau,
    }


def main():
    rows = sweep_fss()
    summ = summarize(rows)
    out = {
        "n_rows": len(rows),
        "sizes": summ["sizes"],
        "taus": summ["taus"],
        "by_size_loglog_slopes": {k: v["loglog_slope_tau"] for k, v in summ["by_size"].items()},
        "increase_with_size_by_tau": summ["increase_with_size_by_tau"],
        "decrease_with_tau_by_size": summ["decrease_with_tau_by_size"],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
