#!/usr/bin/env python
"""
Project sanity check for ToE (Phases 1–3.5)

- Verifies required folders/files
- Re-generates key outputs (idempotent)
- Runs unit tests
- Checks prediction summary (all green)
- Prints a compact status report and exits non-zero on failure
"""

from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

REQ_DIRS = [
    "docs", "examples", "outputs", "outputs/figs", "tests", "toe", ".github/workflows"
]
REQ_FILES = [
    "pyproject.toml",
    "requirements.txt",
    "README.md",
    ".gitignore",
    "docs/white_paper_skeleton.tex",
    ".github/workflows/ci.yml",
]

EXAMPLE_CMDS = [
    [PY, "examples/referee_pack.py"],
    [PY, "examples/lr_noise_sweep.py"],
    [PY, "examples/kl_tt_scan.py"],
    # prediction_verifier writes to stdout; we also save JSON file
]

PREDICTION_JSON = ROOT / "outputs" / "prediction_summary.json"

def run(cmd, cwd=ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)

def ensure_paths() -> Dict[str, Any]:
    missing = []
    for d in REQ_DIRS:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    for f in REQ_FILES:
        if not (ROOT / f).exists():
            missing.append(f)
    return {"ok": len(missing) == 0, "missing": missing}

def regenerate_outputs() -> Dict[str, Any]:
    logs = []
    ok = True
    for cmd in EXAMPLE_CMDS:
        p = run(cmd)
        logs.append({"cmd": " ".join(cmd), "rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr})
        ok &= (p.returncode == 0)

    # Run prediction_verifier and save JSON to outputs/prediction_summary.json
    p = run([PY, "examples/prediction_verifier.py"])
    logs.append({"cmd": f"{PY} examples/prediction_verifier.py", "rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr})
    try:
        data = json.loads(p.stdout.strip())
        PREDICTION_JSON.parent.mkdir(parents=True, exist_ok=True)
        PREDICTION_JSON.write_text(json.dumps(data, indent=2))
        ok &= True
    except Exception as e:
        ok = False
        logs.append({"error": f"Failed to parse prediction_verifier output: {e}"})
    return {"ok": ok, "logs": logs}

def run_tests() -> Dict[str, Any]:
    p = run([PY, "-m", "pytest", "-q"])
    return {"ok": p.returncode == 0, "rc": p.returncode, "stdout": p.stdout, "stderr": p.stderr}

def check_prediction_summary() -> Dict[str, Any]:
    if not PREDICTION_JSON.exists():
        return {"ok": False, "error": f"Missing {PREDICTION_JSON}"}
    data = json.loads(PREDICTION_JSON.read_text())
    failures = {k: v for k, v in data.items() if not (isinstance(v, dict) and v.get("pass", False))}
    return {"ok": len(failures) == 0, "failures": failures, "summary": data}

def main() -> int:
    report: Dict[str, Any] = {}

    # 1) Paths
    paths = ensure_paths()
    report["paths"] = paths
    if not paths["ok"]:
        print("[FAIL] Missing required files:", ", ".join(paths["missing"]))
        print(json.dumps(report, indent=2))
        return 2

    # 2) Regenerate outputs
    regen = regenerate_outputs()
    report["regenerate"] = {"ok": regen["ok"]}
    if not regen["ok"]:
        print("[FAIL] Example generation encountered errors.")
        print(json.dumps(report, indent=2))
        return 3

    # 3) Unit tests
    tests = run_tests()
    report["tests"] = {"ok": tests["ok"], "rc": tests["rc"]}
    if not tests["ok"]:
        print("[FAIL] Pytest failures.")
        # Keep logs compact
        print(tests["stdout"])
        print(tests["stderr"], file=sys.stderr)
        print(json.dumps(report, indent=2))
        return 4

    # 4) Prediction summary
    pred = check_prediction_summary()
    report["predictions"] = pred
    if not pred["ok"]:
        print("[FAIL] Prediction checks failed:")
        print(json.dumps(pred.get("failures", {}), indent=2))
        print(json.dumps(report, indent=2))
        return 5

    # Success
    print("[OK] Project verified (Phases 1–3.5).")
    # Compact one-line recap
    ps = pred["summary"]
    recap = {k: ("PASS" if ps[k]["pass"] else "FAIL") for k in ps}
    print("Prediction recap:", recap)
    return 0

if __name__ == "__main__":
    sys.exit(main())
