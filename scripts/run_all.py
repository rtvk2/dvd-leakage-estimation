"""Regenerate every result of the project from the model file: run the notebooks in order.

    python scripts/run_all.py              # all notebooks, in order (about 2 h; notebook 03 dominates)
    python scripts/run_all.py 04 05 06     # only some of them (their inputs must already exist)

Each notebook is self-contained and uses fixed random seeds (data 20261105 + case, corners 20261105 + 100, data split
20261105, model fits seed 0), so a rerun reproduces the files in data/, models/, netlists/ and results/.
Needs ngspice (natively or through WSL on Windows), numpy, pandas, matplotlib, scikit-learn, lightgbm, torch, jupyter.
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = {
    "01": "01_gen_data",        # four single-transistor datasets, example netlists
    "02": "02_check_data",      # Kirchhoff check and trend plots
    "03": "03_train",           # split, four model types, comparison, test set, saved models
    "04": "04_gate_spice",      # gate netlists, 200 corners, gate-level SPICE reference
    "05": "05_gate_ml",         # gate-level ML prediction, results table and parity plot
    "06": "06_circuit",         # 3-bit CLA: netlist, SPICE reference, hierarchical ML prediction
    "07": "07_error_analysis",  # where the errors are and why
}


def run(key):
    name = NOTEBOOKS[key]
    t0 = time.time()
    cmd = [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
           "--ExecutePreprocessor.timeout=-1", f"{name}.ipynb"]
    r = subprocess.run(cmd, cwd=ROOT / "notebooks")
    print(f"{name}: {'ok' if r.returncode == 0 else 'FAILED'} after {(time.time() - t0) / 60:.1f} min", flush=True)
    return r.returncode


if __name__ == "__main__":
    keys = sys.argv[1:] or list(NOTEBOOKS)
    unknown = [k for k in keys if k not in NOTEBOOKS]
    if unknown:
        raise SystemExit(f"unknown notebook number(s) {unknown}; choose from {sorted(NOTEBOOKS)}")
    for k in keys:
        if run(k) != 0:
            raise SystemExit(f"stopped: notebook {NOTEBOOKS[k]} failed")
