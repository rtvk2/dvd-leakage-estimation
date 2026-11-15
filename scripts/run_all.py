"""Regenerate every result of the project from the model file by running scripts in order.

    python scripts/run_all.py              # all scripts, in order
    python scripts/run_all.py 04 05 06     # only some of them (inputs must already exist)

Each script is self-contained and uses fixed random seeds, so reruns reproduce files in
data/, models/, netlists/ and results/ (given same environment/model file/ngspice).
Needs ngspice (natively or through WSL on Windows), numpy, pandas, matplotlib,
scikit-learn, lightgbm, torch.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = {
    "01": "01_gen_data.py",
    "02": "02_check_data.py",
    "03": "03_train.py",
    "04": "04_gate_spice.py",
    "05": "05_gate_ml.py",
    "06": "06_circuit.py",
    "07": "07_error_analysis.py",
}


def run(key):
    name = SCRIPTS[key]
    t0 = time.time()
    cmd = [sys.executable, str(ROOT / "scripts" / name)]
    r = subprocess.run(cmd, cwd=ROOT)
    print(f"{name}: {'ok' if r.returncode == 0 else 'FAILED'} after {(time.time() - t0) / 60:.1f} min", flush=True)
    return r.returncode


if __name__ == "__main__":
    keys = sys.argv[1:] or list(SCRIPTS)
    unknown = [k for k in keys if k not in SCRIPTS]
    if unknown:
        raise SystemExit(f"unknown script number(s) {unknown}; choose from {sorted(SCRIPTS)}")
    for k in keys:
        if run(k) != 0:
            raise SystemExit(f"stopped: script {SCRIPTS[k]} failed")
