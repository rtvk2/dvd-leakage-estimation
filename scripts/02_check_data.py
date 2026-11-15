"""02_check_data.py - self-contained regeneration script.

Run from project root: python scripts/02_check_data.py
Produces artifacts in data/, models/, netlists/, results/ with fixed seeds.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results"
RESULT_DIR.mkdir(exist_ok=True)

CASES = ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]
CURRENTS = ["id", "ig", "is", "ib"]
missing = [c for c in CASES if not (DATA_DIR / f"{c}.csv").exists()]
if missing:
    raise FileNotFoundError(f"missing datasets {missing}: run scripts/01_gen_data.py first")
data = {c: pd.read_csv(DATA_DIR / f"{c}.csv") for c in CASES}
{c: d.shape for c, d in data.items()}

TOL = 1e-14
rows = []
for c, d in data.items():
    total = d[CURRENTS].sum(axis=1).abs()
    biggest = d[CURRENTS].abs().max(axis=1)
    rel = (total / biggest)[biggest > 1e-12]
    rows.append(dict(case=c, rows=len(d), max_abs_sum_A=total.max(), median_abs_sum_A=total.median(),
                     max_rel_sum=rel.max(), rows_above_tol=int((total > TOL).sum()), passes=bool((total <= TOL).all())))
kirchhoff = pd.DataFrame(rows)
kirchhoff.to_csv(RESULT_DIR / "kirchhoff_check.csv", index=False)
print(f"criterion: |sum of the four terminal currents| <= {TOL:g} A for every row")
kirchhoff

fig, ax = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
for a, (c, d) in zip(ax, data.items()):
    total = d[CURRENTS].sum(axis=1).abs().clip(lower=1e-24)
    biggest = d[CURRENTS].abs().max(axis=1)
    sel = np.random.default_rng(0).choice(len(d), size=2500, replace=False)
    a.scatter(biggest.to_numpy()[sel], total.to_numpy()[sel], s=3, alpha=0.3, color="#4C78A8")
    a.axhline(TOL, color="#E45756", lw=1, label=f"{TOL:g} A")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_title(c); a.set_xlabel("largest terminal current [A]")
ax[0].set_ylabel("|id + ig + is + ib| [A]")
ax[0].legend()
fig.tight_layout()
fig.savefig(RESULT_DIR / "kirchhoff_check.png", dpi=110)
plt.close("all")

def binned(x, y, bins=20):
    """Median and 10-90 % band of y in equal-width bins of x."""
    x, y = np.asarray(x), np.asarray(y)
    edges = np.linspace(x.min(), x.max(), bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, bins - 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    stats = np.array([np.percentile(y[idx == k], [50, 10, 90]) if (idx == k).sum() > 5 else [np.nan] * 3 for k in range(bins)])
    return centers, stats[:, 0], stats[:, 1], stats[:, 2]

def thin(n, k=2500, seed=0):
    """Indices of at most k points to draw (plots only; every statistic uses all rows)."""
    return np.random.default_rng(seed).choice(n, size=min(n, k), replace=False)

def trend_panel(ax, x, y, xlabel, title, color):
    x = np.asarray(x)
    ly = np.log10(np.abs(np.asarray(y)))
    sel = thin(len(x))
    ax.scatter(x[sel], ly[sel], s=3, alpha=0.25, color=color)
    cx, med, lo, hi = binned(x, ly)
    ax.fill_between(cx, lo, hi, color="black", alpha=0.12)
    ax.plot(cx, med, color="black", lw=2)
    ax.set_xlabel(xlabel, fontsize=10); ax.set_ylabel("log10 |I_leak| [A]", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=9)

def off_samples(case):
    d = data[case]
    return d[(d.vd - d.vs).abs() >= 0.5 * d.vdd]

off = {c: off_samples(c) for c in ("nmos_off", "pmos_off")}
{c: len(d) for c, d in off.items()}

fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
for a, (c, color) in zip(ax, [("nmos_off", "#4C78A8"), ("pmos_off", "#F58518")]):
    trend_panel(a, off[c].temp, off[c].id, "temperature [degC]", f"{c}: OFF current vs temperature", color)
    a.set_ylabel("log10 |id| [A]")
fig.tight_layout()
fig.savefig(RESULT_DIR / "trend_off_vs_temp.png", dpi=110)
plt.close("all")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, (c, color) in zip(ax, [("nmos_off", "#4C78A8"), ("pmos_off", "#F58518")]):
    trend_panel(a, off[c].vth0.abs(), off[c].id, "|vth0| [V]", f"{c}: OFF current vs |vth0|", color)
    a.set_ylabel("log10 |id| [A]")
fig.tight_layout()
fig.savefig(RESULT_DIR / "trend_off_vs_vth0.png", dpi=110)
plt.close("all")

fig, ax = plt.subplots(1, 4, figsize=(16, 3.8), sharey=False)
for a, (c, color) in zip(ax, zip(CASES, ["#4C78A8", "#72B7B2", "#F58518", "#E45756"])):
    trend_panel(a, data[c].toxe * 1e9, data[c].ig, "toxe [nm]", f"{c}: |ig| vs toxe", color)
ax[0].set_ylabel("log10 |ig| [A]")
fig.tight_layout()
fig.savefig(RESULT_DIR / "trend_ig_vs_toxe.png", dpi=110)
plt.close("all")

checks = []
for c in ("nmos_off", "pmos_off"):
    checks.append((c, "temperature", off[c].temp, off[c].id, +1))
    checks.append((c, "|vth0|", off[c].vth0.abs(), off[c].id, -1))
for c in CASES:
    checks.append((c, "toxe -> |ig|", data[c].toxe, data[c].ig, -1))
rows = []
for c, name, x, y, expected in checks:
    rho = spearmanr(x, np.log10(np.abs(y))).statistic
    rows.append(dict(case=c, relation=name, spearman=round(float(rho), 3), expected_sign="+" if expected > 0 else "-",
                     as_expected=bool(np.sign(rho) == expected)))
trends = pd.DataFrame(rows)
trends.to_csv(RESULT_DIR / "trend_check.csv", index=False)
trends
