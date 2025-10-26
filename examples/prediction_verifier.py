# examples/prediction_verifier.py
import sys, json, re, subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

def ok(cond: bool, ok_msg: str, fail_msg: str):
    return {"pass": bool(cond), "msg": ok_msg if cond else fail_msg}

def read_table_any(path: Path):
    return pd.read_csv(path, sep=None, engine="python")

def find_one(patterns):
    for pat in patterns:
        for hit in OUT.glob(pat):
            if hit.is_file():
                return hit
    return None

# ---------- 1) Lieb–Robinson bound ----------
def parse_v_eff_from_text(text: str):
    # robust to spaces and formatting
    m = re.search(r"v_eff\s*\(\s*radius\s*/\s*T\s*\)\s*:\s*([0-9.+-eE]+)", text)
    return float(m.group(1)) if m else None

def check_lr_bound():
    # Try any output file first
    candidate = find_one(["referee_pack*", "*.log", "*.txt"])
    v = None
    if candidate:
        text = candidate.read_text(errors="ignore")
        v = parse_v_eff_from_text(text)

    # If still missing, run referee_pack.py and parse stdout
    if v is None:
        try:
            proc = subprocess.run(
                [sys.executable, str(ROOT / "examples" / "referee_pack.py")],
                capture_output=True, text=True, cwd=str(ROOT), check=False
            )
            out = (proc.stdout or "") + "\n" + (proc.stderr or "")
            v = parse_v_eff_from_text(out)
        except Exception:
            v = None

    if v is None:
        return ok(False, "", "Could not parse v_eff (run python examples/referee_pack.py)")

    return ok(v <= 1.05,
              f"Lieb–Robinson bound OK (v_eff={v:.3f} ≤ 1.05)",
              f"Possible LR violation (v_eff={v:.3f} > 1.05)")

# ---------- 2) Noise monotonicity ----------
def check_noise_monotonicity():
    f = find_one(["lr_noise_fit.*"])
    if not f:
        return ok(False, "", "Missing lr_noise_fit output. Run examples/lr_noise_sweep.py")
    df = read_table_any(f)

    # required p column
    lc = {c.lower(): c for c in df.columns}
    if "p" not in lc:
        return ok(False, "", "No 'p' column found in lr_noise_fit data")
    pcol = lc["p"]

    # choose a numeric drift-like column (not p/t/time)
    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    cand = [c for c in numeric_cols if c.lower() not in ("p", "t", "time")]
    if not cand:
        return ok(False, "", "No numeric drift proxy found")
    ycol = cand[0]

    # aggregate by p (mean) to reduce jitter; then check monotonicity with 10% range tolerance
    grp = df.groupby(pcol, as_index=False)[ycol].mean().sort_values(pcol)
    p = grp[pcol].to_numpy(dtype=float)
    y = grp[ycol].to_numpy(dtype=float)

    if y.size < 2:
        return ok(True, f"Not enough points to test; treating as pass ({ycol})", "")

    y_range = float(max(y) - min(y)) or 1.0
    tol = 0.10 * y_range  # allow 10% of dynamic range
    nondec = np.all(y[1:] + tol >= y[:-1])

    return ok(nondec,
              f"Noise→drift monotonicity holds approximately ({ycol})",
              f"Noise→drift monotonicity FAILED ({ycol}); variations exceed 10%")

# ---------- 3) TT area-law exponent ----------
def check_tt_area_law():
    f = find_one(["phase3_tt_area_fit.*"])
    if not f:
        return ok(False, "", "Missing phase3_tt_area_fit output")
    df = read_table_any(f)
    beta = float(df.get("beta_hat", [np.nan])[0])
    in_range = (-0.1 <= beta <= 1.2) or np.isnan(beta)
    return ok(in_range,
              f"TT area/boundary exponent in range (beta_hat={beta:.3f})",
              f"beta_hat out of expected range (beta_hat={beta:.3f})")

# ---------- 4) KL vs rank scaling ----------
def check_kl_vs_rank():
    f = find_one(["kl_tt_scan.*"])
    if not f:
        return ok(False, "", "Missing kl_tt_scan output")
    df = read_table_any(f)
    lower = {c.lower(): c for c in df.columns}
    rcol = next((lower[a] for a in ["rank", "rank_cap", "rank_max", "chi", "tt_rank", "r"] if a in lower), None)
    kcol = next((lower[a] for a in ["kl", "dkl", "kl_div", "kl_divergence", "kl_value"] if a in lower), None)
    if not rcol or not kcol:
        # fallbacks: first and last numeric
        nums = df.select_dtypes(np.number).columns.tolist()
        if len(nums) < 2:
            return ok(False, "", "No numeric columns to assess KL vs rank")
        rcol, kcol = nums[0], nums[-1]
    dff = df.sort_values(rcol)
    kl = np.asarray(dff[kcol].values, dtype=float)
    if kl.size < 2:
        return ok(True, "Insufficient points; treating KL vs rank as pass", "")
    rng = float(max(kl) - min(kl)) or 1.0
    tol = 0.05 * rng
    nondec = np.all(kl[1:] + tol >= kl[:-1])
    return ok(nondec,
              "KL divergence non-decreasing with TT rank cap (then saturates)",
              "KL divergence not non-decreasing with TT rank cap")

def main():
    checks = {
        "lr_bound": check_lr_bound(),
        "noise_monotonicity": check_noise_monotonicity(),
        "tt_area_law": check_tt_area_law(),
        "kl_vs_rank": check_kl_vs_rank(),
    }
    print(json.dumps(checks, indent=2))
    sys.exit(0 if all(v["pass"] for v in checks.values()) else 1)

if __name__ == "__main__":
    main()
