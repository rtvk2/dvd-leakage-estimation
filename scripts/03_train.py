"""03_train.py - self-contained regeneration script.

Run from project root: python scripts/03_train.py
Produces artifacts in data/, models/, netlists/, results/ with fixed seeds.
"""

# notebook-compat display shim
try:
    from IPython.display import display  # type: ignore
except Exception:
    def display(obj=None, *args, **kwargs):
        if obj is None:
            return
        try:
            print(obj.to_string())
        except Exception:
            try:
                print(obj.data if hasattr(obj, 'data') else obj)
            except Exception:
                print(obj)

import re
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"
MODEL_FILE = ROOT / "Model Files-20261105" / "models" / "16nm_HP.pm"
LDIFF = 32e-9                               # AD = AS = W * LDIFF, as in the datasets

CASES = {"nmos_on": "nmos", "nmos_off": "nmos", "pmos_on": "pmos", "pmos_off": "pmos"}
FEATS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", "vth0", "toxe", "ndep", "u0", "xj"]
TERMS = ["id", "ig", "is", "ib"]            # terminal currents flowing into the device
SPLIT = (0.70, 0.15, 0.15)                  # train / validation / test
SPLIT_SEED = 20261105
I0 = 1e-14                                  # scale of the signed-log transform [A]; chosen in section 12
REL_FLOOR = 1e-15                           # relative errors are only meaningful above the solver floor [A]
SMALL_VDS = 5e-3                            # |Vds| below this is the "small-Vds" region [V]
KB_Q = 8.617333262e-5                       # k/q [V/K]

data = {c: pd.read_csv(DATA_DIR / f"{c}.csv") for c in CASES}
{c: d.shape for c, d in data.items()}

def split_indices(n, seed=SPLIT_SEED):
    perm = np.random.default_rng(seed).permutation(n)
    n_tr, n_va = int(SPLIT[0] * n), int(SPLIT[1] * n)
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]

splits = {c: split_indices(len(d)) for c, d in data.items()}
tr, va, te = splits["nmos_on"]
print("rows per split:", len(tr), len(va), len(te), "| disjoint:", len(set(tr) | set(va) | set(te)) == len(data["nmos_on"]))

def canonicalize(df, device):
    """Swap drain and source where needed so that Vds >= 0 (NMOS) or Vsd >= 0 (PMOS); id and is swap with them."""
    d = df.copy()
    swap = (d["vd"] < d["vs"]) if device == "nmos" else (d["vd"] > d["vs"])
    d.loc[swap, ["vd", "vs"]] = d.loc[swap, ["vs", "vd"]].to_numpy()
    d.loc[swap, ["id", "is"]] = d.loc[swap, ["is", "id"]].to_numpy()
    return d, swap

canon, swapped = {}, {}
for c, dev in CASES.items():
    canon[c], swapped[c] = canonicalize(data[c], dev)
print("fraction of rows swapped:", {c: round(float(s.mean()), 3) for c, s in swapped.items()})

# canonicalising twice changes nothing, and the orientation is now fixed
again, _ = canonicalize(canon["nmos_on"], "nmos")
print("idempotent:", bool(again.equals(canon["nmos_on"])))

sign = pd.DataFrame({c: {t: f"{(d.iloc[splits[c][0]][t] > 0).mean():.3f}" for t in TERMS} for c, d in canon.items()})
print("fraction of positive values after canonicalisation (training rows only):")
sign

SPICE_OPTIONS = ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9"
VDD = 0.9
BIAS = {  # (vg, vb) per case
    "nmos_on": (VDD, 0.0), "nmos_off": (0.0, 0.0), "pmos_on": (0.0, VDD), "pmos_off": (VDD, VDD)}

def run_ngspice(netlist, name="symmetry.cir"):
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    shutil.copyfile(MODEL_FILE, build / MODEL_FILE.name)
    (build / name).write_text(netlist, newline="\n")
    if shutil.which("ngspice"):
        return subprocess.run(["ngspice", "-b", name], cwd=build, capture_output=True, text=True).stdout
    p = build.resolve()
    wsl = "/mnt/" + p.drive.rstrip(":").lower() + "/" + "/".join(p.parts[1:])
    r = subprocess.run(["wsl", "-e", "bash", "-c", f"cd '{wsl}' && ngspice -b {name} 2>/dev/null"], capture_output=True, text=True)
    return r.stdout.replace("\x00", "")

def symmetry_netlist():
    lines = ["* drain/source symmetry check", f".include {MODEL_FILE.name}", SPICE_OPTIONS]
    runs = []
    for c, dev in CASES.items():
        vg, vb = BIAS[c]
        for k, (vd, vs) in enumerate([(0.5, 0.2), (0.2, 0.5)]):
            t = f"{c}_{k}"
            w = 32e-9
            lines.append(f"M{t} d{t} g{t} s{t} b{t} {dev} W={w:.6g} L=16n AD={w * LDIFF:.6g} AS={w * LDIFF:.6g}")
            lines += [f"Vd{t} d{t} 0 {vd}", f"Vg{t} g{t} 0 {vg}", f"Vs{t} s{t} 0 {vs}", f"Vb{t} b{t} 0 {vb}"]
            runs.append(t)
    lines += [".control", "set numdgt = 15", "set temp = 27", "op"]
    lines += ["print " + " ".join(f"i(v{x}{t})" for x in "dgsb") for t in runs]
    lines += [".endc", ".end", ""]
    return "\n".join(lines), runs

try:
    text, runs = symmetry_netlist()
    out = run_ngspice(text)
    vals = [-float(v) for v in re.findall(r"^i\(v[dgsb]\w*\) = (\S+)$", out, re.M)]
    assert len(vals) == 4 * len(runs), "ngspice did not return all results"
    cur = {t: vals[4 * k:4 * k + 4] for k, t in enumerate(runs)}
    rows = []
    for c in CASES:
        a, b = cur[f"{c}_0"], cur[f"{c}_1"]            # (id, ig, is, ib) before and after swapping drain and source
        rows.append(dict(case=c, id_vs_is_swapped=abs(a[0] - b[2]) / abs(a[0]), is_vs_id_swapped=abs(a[2] - b[0]) / abs(a[2]),
                         ig=abs(a[1] - b[1]) / abs(a[1]), ib=abs(a[3] - b[3]) / abs(a[3])))
    sym = pd.DataFrame(rows).set_index("case")
    print("relative mismatch between the two orientations (0 = perfectly symmetric):")
    display(sym.style.format("{:.1e}"))
except Exception as e:                                  # ngspice or the model file not available
    print("symmetry check skipped:", e)

def vds_factor(vds, temp_c):
    """1 - exp(-|Vds| / vT) with vT = kT/q; expm1 keeps it accurate for tiny Vds."""
    vt = KB_Q * (np.asarray(temp_c) + 273.15)
    return np.clip(-np.expm1(-np.abs(vds) / vt), 1e-12, None)

def encode(currents, vds, temp_c, vds_aware=True, i0=None):
    """Currents (n, 4) in the order id, ig, is, ib -> transformed targets (n, 4)."""
    i0 = I0 if i0 is None else i0
    cur = np.asarray(currents, dtype=float).copy()
    if vds_aware:
        f = vds_factor(vds, temp_c)
        cur[:, 0] /= f
        cur[:, 2] /= f
    return np.arcsinh(cur / i0)

def decode(y, vds, temp_c, vds_aware=True, i0=None):
    i0 = I0 if i0 is None else i0
    cur = i0 * np.sinh(np.asarray(y, dtype=float))
    if vds_aware:
        f = vds_factor(vds, temp_c)
        cur[:, 0] *= f
        cur[:, 2] *= f
    return cur

rows = []
for c, dev in CASES.items():
    d = canon[c]
    vds = (d["vd"] - d["vs"]).abs().to_numpy()
    cur = d[TERMS].to_numpy()
    for aware in (False, True):
        back = decode(encode(cur, vds, d["temp"].to_numpy(), aware), vds, d["temp"].to_numpy(), aware)
        m = np.abs(cur) > 0
        rows.append(dict(case=c, vds_aware=aware, max_roundtrip_rel_error=float(np.max(np.abs(back[m] - cur[m]) / np.abs(cur[m])))))
pd.DataFrame(rows)

fig, ax = plt.subplots(2, 4, figsize=(16, 6.5), sharex=True)
slopes = []
for k, (c, dev) in enumerate(CASES.items()):
    d = canon[c].iloc[splits[c][0]]                      # training rows only: nothing here may depend on validation or test rows
    vds = (d["vd"] - d["vs"]).abs().to_numpy()
    f = vds_factor(vds, d["temp"].to_numpy())
    raw, scaled = np.log10(np.abs(d["id"].to_numpy())), np.log10(np.abs(d["id"].to_numpy()) / f)
    sel = np.random.default_rng(0).choice(len(d), size=3000, replace=False)
    for row, (y, label) in enumerate([(raw, "log10 |id|"), (scaled, "log10 (|id| / factor)")]):
        ax[row, k].scatter(np.log10(vds[sel]), y[sel], s=3, alpha=0.25, color=["#4C78A8", "#72B7B2", "#F58518", "#E45756"][k])
        ax[row, k].set_ylabel(label)
    ax[1, k].set_xlabel("log10 |Vds| [V]")
    ax[0, k].set_title(c)
    small = vds < 10e-3
    slopes.append(dict(case=c, rows_below_10mV=int(small.sum()),
                       slope_raw=np.polyfit(np.log10(vds[small]), raw[small], 1)[0],
                       slope_with_factor=np.polyfit(np.log10(vds[small]), scaled[small], 1)[0]))
fig.tight_layout()
fig.savefig(RESULT_DIR / "vds_aware_target.png", dpi=110)
plt.close("all")
print("slope of log10|id| vs log10|Vds| for |Vds| < 10 mV (1 = proportional to Vds, 0 = independent of Vds):")
pd.DataFrame(slopes).set_index("case")

def make_features(d, device):
    sgn = 1.0 if device == "nmos" else -1.0
    vt = KB_Q * (d["temp"] + 273.15)
    vgs, vds, vbs, vgd = sgn * (d["vg"] - d["vs"]), sgn * (d["vd"] - d["vs"]), sgn * (d["vb"] - d["vs"]), sgn * (d["vg"] - d["vd"])
    X = d[FEATS].copy()
    X["inv_vt"] = 1.0 / vt
    X["vgs"], X["vds"], X["vbs"], X["vgd"] = vgs, vds, vbs, vgd
    X["vov_vt"] = (vgs - sgn * d["vth0"]) / vt          # (Vgs - Vth)/vT; sgn*vth0 is the threshold magnitude for both types
    X["vds_vt"] = vds / vt
    X["log_w"] = np.log(d["w"])
    return X

X_nmos = make_features(canon["nmos_on"], "nmos")
X_pmos = make_features(canon["pmos_on"], "pmos")
print(f"{X_nmos.shape[1]} features per row:", list(X_nmos.columns))
print("Vds >= 0 after canonicalisation (NMOS, PMOS):", bool((X_nmos["vds"] >= 0).all()), bool((X_pmos["vds"] >= 0).all()))
X_nmos.describe().T[["min", "max"]]

def score(true, pred, vds, i0=None):
    """Per-terminal metrics; true and pred are (n, 4) currents in the order id, ig, is, ib."""
    i0 = I0 if i0 is None else i0
    true, pred, vds = np.asarray(true), np.asarray(pred), np.asarray(vds)
    rows = []
    for k, t in enumerate(TERMS):
        yt, yp = np.arcsinh(true[:, k] / i0), np.arcsinh(pred[:, k] / i0)
        r2_log = 1 - ((yt - yp) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()
        r2_lin = 1 - ((true[:, k] - pred[:, k]) ** 2).sum() / ((true[:, k] - true[:, k].mean()) ** 2).sum()
        m = np.abs(true[:, k]) >= REL_FLOOR
        rel = np.abs(pred[m, k] - true[m, k]) / np.abs(true[m, k])
        ms = m & (vds < SMALL_VDS)
        rel_s = np.abs(pred[ms, k] - true[ms, k]) / np.abs(true[ms, k])
        rows.append(dict(terminal=t, r2_log=r2_log, r2_lin=r2_lin, median_rel=np.median(rel), p95_rel=np.percentile(rel, 95), n_used=int(m.sum()), frac_below_floor=float(1 - m.mean()),
                         median_rel_small_vds=np.median(rel_s) if len(rel_s) else np.nan,
                         p95_rel_small_vds=np.percentile(rel_s, 95) if len(rel_s) else np.nan, n_small_vds=int(ms.sum())))
    return pd.DataFrame(rows).set_index("terminal")

# the metrics must behave on known inputs: a perfect prediction scores 0 error, a +1 % prediction scores exactly 1 %
va_idx = splits["nmos_on"][1]
d = canon["nmos_on"].iloc[va_idx]
true = d[TERMS].to_numpy()
vds = (d["vd"] - d["vs"]).abs().to_numpy()
perfect, off_by_1pct = score(true, true, vds), score(true, true * 1.01, vds)
print("perfect prediction: max median error =", perfect["median_rel"].max(), "| R2 (log) =", perfect["r2_log"].min())
print("+1 % prediction:   median error =", off_by_1pct["median_rel"].round(6).tolist(), "(expected 0.01)")
off_by_1pct

from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def prepare(case, vds_aware, physics, i0=None):
    """Standardised inputs X and targets Y for one case and one option set (scalers fitted on the training rows)."""
    i0 = I0 if i0 is None else i0
    d = canon[case]
    X = make_features(d, CASES[case])
    if not physics:
        X = X[FEATS]
    vds = (d["vd"] - d["vs"]).abs().to_numpy()
    temp = d["temp"].to_numpy()
    cur = d[TERMS].to_numpy()
    y = encode(cur, vds, temp, vds_aware, i0)
    tr, va, te = splits[case]
    sx, sy = StandardScaler().fit(X.iloc[tr]), StandardScaler().fit(y[tr])
    return dict(case=case, vds_aware=vds_aware, physics=physics, i0=i0, X=sx.transform(X), Y=sy.transform(y), sx=sx, sy=sy,
                vds=vds, temp=temp, cur=cur, tr=tr, va=va, te=te, columns=list(X.columns))

def to_currents(P, prep, idx):
    """Standardised model output for the rows idx -> currents (n, 4) in A (canonical orientation)."""
    return decode(prep["sy"].inverse_transform(P), prep["vds"][idx], prep["temp"][idx], prep["vds_aware"], prep["i0"])

# checks: scalers are fitted on the training rows, and the standardised targets decode back to the SPICE currents
for aware in (False, True):
    p = prepare("nmos_on", aware, True)
    back = to_currents(p["Y"][p["va"]], p, p["va"])
    m = np.abs(p["cur"][p["va"]]) > 0
    err = np.max(np.abs(back[m] - p["cur"][p["va"]][m]) / np.abs(p["cur"][p["va"]][m]))
    print(f"vds_aware={aware}: train mean {p['X'][p['tr']].mean():+.1e}, train std {p['X'][p['tr']].std(axis=0).mean():.3f}, "
          f"{p['X'].shape[1]} features, decode round trip max rel. error {err:.1e}")

from sklearn.linear_model import RidgeCV
import lightgbm as lgb
import time

RIDGE_ALPHAS = np.logspace(-6, 3, 19)
MODEL_SEED = 0                              # seed of every model fit in this script (the data split uses SPLIT_SEED)

class PolyRidge:
    """Polynomial features of the standardised inputs + ridge regression; the penalty is chosen per terminal by
    leave-one-out cross-validation on the training rows (no validation rows involved)."""

    def __init__(self, degree):
        self.degree = degree

    def fit(self, X, Y, X_val=None, Y_val=None):
        self.poly = PolynomialFeatures(self.degree, include_bias=False)
        self.ridge = RidgeCV(alphas=RIDGE_ALPHAS, alpha_per_target=True).fit(self.poly.fit_transform(X), Y)
        self.setting = f"degree={self.degree}, alpha={[float(f'{a:.3g}') for a in self.ridge.alpha_]}"
        return self

    def predict(self, X):
        return self.ridge.predict(self.poly.transform(X))

class GradientBoosting:
    """One LightGBM regressor per terminal; early stopping on the validation rows chooses the number of trees."""

    def __init__(self, max_trees=3000, learning_rate=0.05):
        self.max_trees, self.learning_rate = max_trees, learning_rate

    def fit(self, X, Y, X_val, Y_val):
        self.models = []
        for k in range(Y.shape[1]):
            m = lgb.LGBMRegressor(n_estimators=self.max_trees, learning_rate=self.learning_rate, num_leaves=31, min_child_samples=10,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.9, random_state=MODEL_SEED, deterministic=True,
                                  force_row_wise=True, n_jobs=8, verbosity=-1)
            m.fit(X, Y[:, k], eval_set=[(X_val, Y_val[:, k])], callbacks=[lgb.early_stopping(100, verbose=False)])
            self.models.append(m)
        self.setting = f"trees={[m.best_iteration_ for m in self.models]}"
        return self

    def predict(self, X):
        return np.column_stack([m.predict(X) for m in self.models])

def evaluate(model, prep, family):
    """Validation-set metrics of a fitted model, one row of the comparison table."""
    va = prep["va"]
    s = score(prep["cur"][va], to_currents(model.predict(prep["X"][va]), prep, va), prep["vds"][va], prep["i0"])
    row = dict(case=prep["case"], family=family, vds_aware=prep["vds_aware"], physics=prep["physics"], setting=model.setting,
               r2_log=s["r2_log"].mean(), r2_lin=s["r2_lin"].mean(), median_rel=s["median_rel"].mean(), p95_rel=s["p95_rel"].mean(),
               median_rel_small_vds=s["median_rel_small_vds"].mean())
    metrics = ("r2_log", "r2_lin", "median_rel", "p95_rel", "median_rel_small_vds", "n_small_vds", "n_used", "frac_below_floor")
    row.update({f"{m}_{t}": s.loc[t, m] for t in TERMS for m in metrics})      # all metrics per terminal
    return row

OPTIONS = [(a, p) for a in (False, True) for p in (False, True)]
prep_cache = {(c, a, p): prepare(c, a, p) for c in CASES for a, p in OPTIONS}
rows, best = [], {}       # best[(case, family)] = (median_rel, fitted model, option set)

def register(family, prep, model):
    """Evaluate a fitted model on the validation rows, add a row to the table, and keep the model if it is the best of its family and case so far."""
    row = evaluate(model, prep, family)
    rows.append(row)
    key = (prep["case"], family)
    if key not in best or row["median_rel"] < best[key][0]:
        best[key] = (row["median_rel"], model, (prep["vds_aware"], prep["physics"]))
    return row

def run(family, case, vds_aware, physics, model):
    prep = prep_cache[(case, vds_aware, physics)]
    model.fit(prep["X"][prep["tr"]], prep["Y"][prep["tr"]], prep["X"][prep["va"]], prep["Y"][prep["va"]])
    return register(family, prep, model)

t0 = time.time()
for case in CASES:
    for aware, physics in OPTIONS:
        for degree in ((2, 3) if physics else (2, 3, 4)):
            run("poly", case, aware, physics, PolyRidge(degree))
        print(f"  {case} vds_aware={aware} physics={physics}: {time.time() - t0:.0f} s")
poly_table = pd.DataFrame(rows)
poly_table.groupby(["case", "vds_aware", "physics"])["median_rel"].min().unstack(["vds_aware", "physics"]).round(3)

# repeatability: the same fit twice must give identical predictions
prep = prep_cache[("nmos_on", True, True)]
p1 = GradientBoosting(max_trees=200).fit(prep["X"][prep["tr"]], prep["Y"][prep["tr"]], prep["X"][prep["va"]], prep["Y"][prep["va"]]).predict(prep["X"][prep["va"]])
p2 = GradientBoosting(max_trees=200).fit(prep["X"][prep["tr"]], prep["Y"][prep["tr"]], prep["X"][prep["va"]], prep["Y"][prep["va"]]).predict(prep["X"][prep["va"]])
print("two identical fits give identical predictions:", bool(np.array_equal(p1, p2)))

t0 = time.time()
for case in CASES:
    for aware, physics in OPTIONS:
        run("lightgbm", case, aware, physics, GradientBoosting())
    print(f"{case}: done after {time.time() - t0:.0f} s")
baselines = pd.DataFrame(rows)
baselines.to_csv(RESULT_DIR / "baselines_validation.csv", index=False)
baselines[baselines.family == "lightgbm"].groupby(["case", "vds_aware", "physics"])["median_rel"].min().unstack(["vds_aware", "physics"]).round(3)

pick = baselines.loc[baselines.groupby(["case", "family"])["median_rel"].idxmin()].set_index(["case", "family"])
pick_r2 = baselines.loc[baselines.groupby(["case", "family"])["r2_log"].idxmax()].set_index(["case", "family"])
pick["same_choice_by_r2"] = [(pick.loc[k, "vds_aware"], pick.loc[k, "physics"]) == (pick_r2.loc[k, "vds_aware"], pick_r2.loc[k, "physics"]) for k in pick.index]
shown = pick[["vds_aware", "physics", "median_rel", "p95_rel", "r2_log", "median_rel_id", "median_rel_ig", "median_rel_is", "median_rel_ib", "same_choice_by_r2", "setting"]]
shown.round({"median_rel": 4, "p95_rel": 3, "r2_log": 4, "median_rel_id": 4, "median_rel_ig": 4, "median_rel_is": 4, "median_rel_ib": 4})

table4 = pick[["r2_log", "median_rel", "p95_rel"]].unstack("family").swaplevel(axis=1).sort_index(axis=1)
display(table4.round(4))
floor_share = pd.DataFrame({c: {t: baselines.loc[baselines.case == c, f"frac_below_floor_{t}"].iloc[0] for t in TERMS} for c in CASES}).T
print("share of the validation rows left out of the relative errors (|I| < 1e-15 A), in %:")
(floor_share * 100).round(1)

# effect of the two options: best median error of each family for each option set (log scale; lower is better)
COL = {"poly": "#2a78d6", "lightgbm": "#eb6834"}
LABEL = {"poly": "polynomial ridge", "lightgbm": "LightGBM"}
labels = ["raw inputs", "+ physics", "Vds-aware", "Vds-aware\n+ physics"]
fig, ax = plt.subplots(1, 4, figsize=(16, 3.9), sharey=True)
for a, case in zip(ax, CASES):
    for fam, dx in (("poly", -0.1), ("lightgbm", 0.1)):
        vals = [baselines[(baselines.case == case) & (baselines.family == fam) & (baselines.vds_aware == aw) & (baselines.physics == ph)]["median_rel"].min()
                for aw, ph in OPTIONS]
        a.plot(np.arange(4) + dx, vals, "o", ms=9, color=COL[fam], mec="white", mew=1.5, label=LABEL[fam])
    a.set_yscale("log"); a.set_title(case); a.set_xticks(range(4)); a.set_xticklabels(labels, fontsize=8); a.set_xlim(-0.5, 3.5)
    a.grid(axis="y", color="0.9", lw=0.8); a.set_axisbelow(True)
    a.spines[["top", "right"]].set_visible(False)
ax[0].set_ylabel("median relative error\n(mean of the four terminals)")
handles, names = ax[0].get_legend_handles_labels()
fig.legend(handles, names, loc="upper center", ncol=2, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(RESULT_DIR / "baseline_options.png", dpi=110)
plt.close("all")

QUANTUM = 2.0 ** -51                       # 4.44e-16 A
res = []
for c, d in data.items():
    for t in TERMS:
        x = np.abs(d[t].to_numpy())
        k = x / QUANTUM
        used = x >= REL_FLOOR
        res.append(dict(case=c, terminal=t, on_grid=float((np.abs(k - np.round(k)) < 1e-3).mean()),
                        below_25_quanta=float((x[used] < 25 * QUANTUM).mean())))
res = pd.DataFrame(res)
print("share of the values that are exact multiples of 2^-51 A, in % (a value picked at random passes this test with probability 0.2 %):")
display((res.pivot(index="case", columns="terminal", values="on_grid")[TERMS] * 100).round(1))
print("share of the rows used for the relative errors with |I| < 25 quanta = 1.1e-14 A, in % (rounding alone costs more than 1 % there; it applies to the terminals above that are on the grid):")
(res.pivot(index="case", columns="terminal", values="below_25_quanta")[TERMS] * 100).round(1)

import copy

import torch
import torch.nn as nn
from joblib import Parallel, delayed

class PlainMLP(nn.Module):
    """`depth` hidden layers of `width` units with SiLU activations."""

    def __init__(self, d_in, d_out, width=256, depth=3):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        layers.append(nn.Linear(d, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ResidualMLP(nn.Module):
    """Input layer, residual blocks h <- SiLU(h + Linear(SiLU(Linear(h)))), linear output layer."""

    def __init__(self, d_in, d_out, width=128, blocks=4):
        super().__init__()
        self.inp = nn.Linear(d_in, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width)) for _ in range(blocks)])
        self.act = nn.SiLU()
        self.out = nn.Linear(width, d_out)

    def forward(self, x):
        h = self.act(self.inp(x))
        for block in self.blocks:
            h = self.act(h + block(h))
        return self.out(h)

class TorchModel:
    """MLP ("mlp") or residual MLP ("resnn") with AdamW and a one-cycle schedule; the epoch with the lowest validation MSE is kept."""

    def __init__(self, kind, loss="mse", epochs=600, lr=3e-3, batch=256, weight_decay=1e-5, threads=3):
        self.kind, self.loss, self.epochs, self.lr, self.batch, self.weight_decay, self.threads = kind, loss, epochs, lr, batch, weight_decay, threads

    def fit(self, X, Y, X_val, Y_val):
        torch.set_num_threads(self.threads)
        torch.manual_seed(MODEL_SEED)
        Xt, Yt = torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)
        Xv, Yv = torch.tensor(X_val, dtype=torch.float32), torch.tensor(Y_val, dtype=torch.float32)
        net = PlainMLP(Xt.shape[1], Yt.shape[1]) if self.kind == "mlp" else ResidualMLP(Xt.shape[1], Yt.shape[1])
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        steps = int(np.ceil(len(Xt) / self.batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=self.epochs * steps, pct_start=0.1, final_div_factor=1e3)
        loss_fn = nn.MSELoss() if self.loss == "mse" else nn.SmoothL1Loss(beta=0.05)
        best_mse, best_state, best_epoch = np.inf, None, 0
        for epoch in range(self.epochs):
            net.train()
            perm = torch.randperm(len(Xt))
            for i in range(0, len(Xt), self.batch):
                idx = perm[i:i + self.batch]
                opt.zero_grad()
                loss_fn(net(Xt[idx]), Yt[idx]).backward()
                opt.step()
                sched.step()
            net.eval()
            with torch.no_grad():
                mse = float(((net(Xv) - Yv) ** 2).mean())
            if mse < best_mse:
                best_mse, best_state, best_epoch = mse, copy.deepcopy(net.state_dict()), epoch + 1
        net.load_state_dict(best_state)
        net.eval()
        self.net = net
        self.setting = f"{self.kind}, {self.loss} loss, {self.epochs} epochs (best epoch {best_epoch})"
        return self

    def predict(self, X):
        with torch.no_grad():
            return self.net(torch.tensor(X, dtype=torch.float32)).numpy()

def fit_job(model, prep):
    return model.fit(prep["X"][prep["tr"]], prep["Y"][prep["tr"]], prep["X"][prep["va"]], prep["Y"][prep["va"]])

def fit_all(jobs, n_jobs=4):
    """jobs: list of (model, prep); trained in parallel worker processes; returns the fitted models in the same order."""
    t0, out = time.time(), []
    for k, model in enumerate(Parallel(n_jobs=n_jobs, return_as="generator")(delayed(fit_job)(m, p) for m, p in jobs)):
        out.append(model)
        print(f"  {k + 1}/{len(jobs)} trained after {time.time() - t0:.0f} s")
    return out

# repeatability: the same training twice must give identical predictions
prep = prep_cache[("nmos_on", True, True)]
twice = [TorchModel("resnn", epochs=5).fit(prep["X"][prep["tr"]], prep["Y"][prep["tr"]], prep["X"][prep["va"]], prep["Y"][prep["va"]]).predict(prep["X"][prep["va"]]) for _ in range(2)]
print("two identical trainings give identical predictions:", bool(np.array_equal(*twice)))

I0_VALUES = [1e-16, 1e-15, 1e-14, 1e-13]
study_prep = {(c, i0): prepare(c, False, True, i0=i0) for c in CASES for i0 in I0_VALUES}
keys = list(study_prep)
study_models = fit_all([(TorchModel("resnn", epochs=300), study_prep[k]) for k in keys])
study = []
for k, model in zip(keys, study_models):
    r = evaluate(model, study_prep[k], "resnn")
    study.append(dict(case=k[0], i0=k[1], median_rel=r["median_rel"], p95_rel=r["p95_rel"]))
study = pd.DataFrame(study)
study.to_csv(RESULT_DIR / "i0_study.csv", index=False)
print("median relative error (mean of the four terminals) on the validation rows:")
display(study.pivot(index="case", columns="i0", values="median_rel").round(4))
print("95th-percentile relative error:")
study.pivot(index="case", columns="i0", values="p95_rel").round(3)

grid = [(case, kind, aware, loss) for case in CASES for kind in ("mlp", "resnn") for aware in (False, True) for loss in ("mse", "huber")]
nn_models = fit_all([(TorchModel(kind, loss), prep_cache[(case, aware, True)]) for case, kind, aware, loss in grid])
for (case, kind, aware, loss), model in zip(grid, nn_models):
    register(kind, prep_cache[(case, aware, True)], model)
results = pd.DataFrame(rows)
nets = results[results.family.isin(["mlp", "resnn"])].copy()
nets["loss"] = nets.setting.str.extract(r", (\w+) loss")[0]
print("median relative error (mean of the four terminals) on the validation rows, by case, network, Vds-aware target and loss:")
nets.pivot_table(index=["case", "family"], columns=["vds_aware", "loss"], values="median_rel").round(4)

FAMILIES = ["poly", "lightgbm", "mlp", "resnn"]
NAMES = {"poly": "polynomial ridge", "lightgbm": "LightGBM", "mlp": "MLP", "resnn": "residual MLP"}
results = pd.DataFrame(rows)
results.to_csv(RESULT_DIR / "model_comparison_all_runs.csv", index=False)
pick = results.loc[results.groupby(["case", "family"])["median_rel"].idxmin()].set_index(["case", "family"])
table4 = pd.DataFrame({(NAMES[f], name): pick.xs(f, level="family")[col] * scale for f in FAMILIES
                       for name, col, scale in (("R2", "r2_log", 1), ("median error [%]", "median_rel", 100), ("95th pct error [%]", "p95_rel", 100))}).loc[list(CASES)]
table4.to_csv(RESULT_DIR / "model_comparison_table.csv")
display(table4.round({c: 4 if c[1] == "R2" else 2 for c in table4.columns}))
winner = pick.groupby("case")["median_rel"].idxmin().map(lambda k: k[1])
print("best model type per case on the validation rows:", {c: NAMES[f] for c, f in winner.items()})

# per-terminal detail of the best setting of every model type and case
detail = pick[["vds_aware", "physics", "median_rel", "median_rel_id", "median_rel_ig", "median_rel_is", "median_rel_ib", "p95_rel", "r2_log", "setting"]]
detail.round({"median_rel": 4, "median_rel_id": 4, "median_rel_ig": 4, "median_rel_is": 4, "median_rel_ib": 4, "p95_rel": 3, "r2_log": 5})

# best median error of each model type and case (log scale; lower is better)
COL = {"poly": "#2a78d6", "lightgbm": "#eb6834", "mlp": "#1baf7a", "resnn": "#6250d6"}
MARK = {"poly": "o", "lightgbm": "s", "mlp": "^", "resnn": "D"}
fig, ax = plt.subplots(figsize=(8, 4))
for j, f in enumerate(FAMILIES):
    ax.plot(np.arange(4) + (j - 1.5) * 0.12, [pick.loc[(c, f), "median_rel"] * 100 for c in CASES], MARK[f], ms=9, color=COL[f], mec="white", mew=1.5, label=NAMES[f])
ax.set_yscale("log"); ax.set_xticks(range(4)); ax.set_xticklabels(list(CASES)); ax.set_xlim(-0.5, 3.5)
ax.set_ylabel("median relative error [%]\n(mean of the four terminals)")
ax.axhline(1.0, color="0.6", lw=1, ls="--"); ax.text(3.45, 1.05, "1 %", ha="right", va="bottom", color="0.4", fontsize=8)
ax.grid(axis="y", color="0.9", lw=0.8); ax.set_axisbelow(True); ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12), fontsize=9)
fig.tight_layout()
fig.savefig(RESULT_DIR / "model_comparison.png", dpi=110)
plt.close("all")

chosen = {case: best[(case, winner[case])] for case in CASES}          # (validation median error, fitted model, (vds_aware, physics))
print({case: (NAMES[winner[case]], f"vds_aware={chosen[case][2][0]}", chosen[case][1].setting) for case in CASES})
assert all(winner[case] in ("mlp", "resnn") for case in CASES), "the saving code below handles the neural models only"

def test_report(model, prep):
    """Per-terminal test-set metrics of a fitted model: score() of section 5 plus the mean relative error."""
    te = prep["te"]
    true, pred = prep["cur"][te], to_currents(model.predict(prep["X"][te]), prep, te)
    s = score(true, pred, prep["vds"][te], prep["i0"])
    for k, t in enumerate(TERMS):
        m = np.abs(true[:, k]) >= REL_FLOOR
        s.loc[t, "mean_rel"] = np.mean(np.abs(pred[m, k] - true[m, k]) / np.abs(true[m, k]))
    return s, true, pred

test, test_true, test_pred = {}, {}, {}
for case in CASES:
    prep = prep_cache[(case, *chosen[case][2])]
    test[case], test_true[case], test_pred[case] = test_report(chosen[case][1], prep)

summary = pd.DataFrame({case: dict(model=NAMES[winner[case]],
                                   val_median=chosen[case][0] * 100,
                                   test_median=test[case]["median_rel"].mean() * 100,
                                   test_p95=test[case]["p95_rel"].mean() * 100,
                                   test_mean=test[case]["mean_rel"].mean() * 100,
                                   test_r2_log=test[case]["r2_log"].mean(),
                                   test_r2_lin=test[case]["r2_lin"].mean()) for case in CASES}).T
per_terminal = pd.concat({case: test[case][["median_rel", "p95_rel", "mean_rel", "r2_log", "r2_lin", "median_rel_small_vds", "n_small_vds", "n_used", "frac_below_floor"]]
                          for case in CASES}, names=["case"])
summary.to_csv(RESULT_DIR / "chosen_models_test.csv")
per_terminal.to_csv(RESULT_DIR / "chosen_models_test_per_terminal.csv")
print("test rows (1,500 per case); errors in %, each the mean over the four terminals; the validation median of the same model is shown for comparison:")
display(summary.astype({c: float for c in summary.columns if c != "model"}).round({"val_median": 2, "test_median": 2, "test_p95": 2, "test_mean": 2, "test_r2_log": 4, "test_r2_lin": 4}))

print("per terminal, test rows (errors in %):")
pt = per_terminal.copy()
for c in ("median_rel", "p95_rel", "mean_rel", "median_rel_small_vds", "frac_below_floor"):
    pt[c] = pt[c] * 100
pt.round({"median_rel": 2, "p95_rel": 2, "mean_rel": 2, "r2_log": 4, "r2_lin": 4, "median_rel_small_vds": 2, "frac_below_floor": 1})

from matplotlib.ticker import FuncFormatter

def sym_ticks(lo, hi, max_ticks=6):
    """Tick positions for a symmetric-log axis: powers of ten above I0 on either side of zero, thinned to about max_ticks (and zero)."""
    k0 = int(np.log10(I0)) + 1                       # the decade at I0 itself is left out: it would sit on top of the label 0
    pos = [10.0 ** k for k in range(k0, int(np.floor(np.log10(hi))) + 1)] if hi > I0 else []
    neg = [-10.0 ** k for k in range(int(np.floor(np.log10(-lo))), k0 - 1, -1)] if lo < -I0 else []
    step = int(np.ceil((len(pos) + len(neg) + 1) / max_ticks))
    return neg[::step] + [0.0] + pos[::-1][::step][::-1]

def pred_vs_spice(case):
    true, pred = test_true[case], test_pred[case]
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.2))
    for k, (a, t) in enumerate(zip(ax, TERMS)):
        lo, hi = min(true[:, k].min(), pred[:, k].min()), max(true[:, k].max(), pred[:, k].max())
        a.set_xscale("symlog", linthresh=I0); a.set_yscale("symlog", linthresh=I0)
        a.set_xticks(sym_ticks(lo, hi)); a.set_yticks(sym_ticks(lo, hi))
        a.xaxis.set_minor_locator(plt.NullLocator()); a.yaxis.set_minor_locator(plt.NullLocator())
        a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:.0e}".replace("e-0", "e-").replace("e+0", "e")))
        a.yaxis.set_major_formatter(a.xaxis.get_major_formatter())
        a.plot([lo, hi], [lo, hi], color="0.6", lw=1, ls="--", zorder=1)
        a.scatter(true[:, k], pred[:, k], s=6, alpha=0.35, color=COL["resnn"] if winner[case] == "resnn" else COL["mlp"], linewidths=0, zorder=2)
        a.set_title(f"{t}   median error {test[case].loc[t, 'median_rel'] * 100:.2f} %", fontsize=10)
        a.set_xlabel(f"SPICE {t} [A]"); a.set_ylabel(f"predicted {t} [A]")
        a.grid(color="0.92", lw=0.8); a.set_axisbelow(True); a.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{case}: {NAMES[winner[case]]}, test rows", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULT_DIR / f"pred_vs_spice_{case}.png", dpi=110)
    plt.close("all")
for case in CASES:
    pred_vs_spice(case)

edges = np.array([1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1.0])
names_bins = ["1e-15..1e-14", "1e-14..1e-13", "1e-13..1e-12", "1e-12..1e-11", "1e-11..1e-10", "1e-10..1e-9", "1e-9..1e-8", "1e-8..1e-7", ">1e-7"]
tab = []
for case in CASES:
    true, pred = test_true[case], test_pred[case]
    for k, t in enumerate(TERMS):
        a = np.abs(true[:, k]); m = a >= REL_FLOOR
        rel = np.abs(pred[:, k] - true[:, k]) / np.where(a > 0, a, np.nan)
        which = np.digitize(a, edges) - 1
        for b, nm in enumerate(names_bins):
            sel = m & (which == b)
            if sel.sum() >= 10:
                tab.append(dict(case=case, terminal=t, bin=nm, n=int(sel.sum()), median=np.median(rel[sel]) * 100, p95=np.percentile(rel[sel], 95) * 100))
bins = pd.DataFrame(tab)
bins.to_csv(RESULT_DIR / "chosen_models_test_by_size.csv", index=False)
print("median relative error [%] on the test rows by |I| bin (bins with at least 10 rows):")
bins.pivot_table(index=["case", "terminal"], columns="bin", values="median")[[b for b in names_bins if b in set(bins["bin"])]].round(2)

import json

def export_model(case, path):
    model, (aware, physics) = chosen[case][1], chosen[case][2]
    prep = prep_cache[(case, aware, physics)]
    meta = dict(case=case, device=CASES[case], kind=winner[case], vds_aware=bool(aware), physics=bool(physics), i0=float(prep["i0"]),
                columns=prep["columns"], feats=FEATS, terms=TERMS, setting=model.setting)
    arrays = {"meta": np.array(json.dumps(meta)), "x_mean": prep["sx"].mean_, "x_scale": prep["sx"].scale_,
              "y_mean": prep["sy"].mean_, "y_scale": prep["sy"].scale_}
    for name, p in model.net.state_dict().items():
        arrays["w." + name] = p.numpy().astype(np.float64)
    np.savez(path, **arrays)

class SavedModel:
    """Callable model loaded from models/<case>.npz: raw 12 columns in, currents (id, ig, is, ib) out (numpy only)."""

    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(z["meta"]))
        self.x_mean, self.x_scale, self.y_mean, self.y_scale = z["x_mean"], z["x_scale"], z["y_mean"], z["y_scale"]
        self.w = {k[2:]: z[k] for k in z.files if k.startswith("w.")}

    @staticmethod
    def _silu(x):
        return x / (1.0 + np.exp(-x))

    def _forward(self, x):
        w, kind = self.w, self.meta["kind"]
        if kind == "mlp":
            n = len([k for k in w if k.endswith(".weight")])
            idx = sorted(int(k.split(".")[1]) for k in w if k.endswith(".weight"))
            for j, i in enumerate(idx):
                x = x @ w[f"net.{i}.weight"].T + w[f"net.{i}.bias"]
                if j < n - 1:
                    x = self._silu(x)
            return x
        h = self._silu(x @ w["inp.weight"].T + w["inp.bias"])
        for b in range(len([k for k in w if k.startswith("blocks.") and k.endswith(".0.weight")])):
            z = self._silu(h @ w[f"blocks.{b}.0.weight"].T + w[f"blocks.{b}.0.bias"]) @ w[f"blocks.{b}.2.weight"].T + w[f"blocks.{b}.2.bias"]
            h = self._silu(h + z)
        return h @ w["out.weight"].T + w["out.bias"]

    def __call__(self, raw):
        m, d = self.meta, pd.DataFrame(raw)[FEATS].astype(float).reset_index(drop=True)
        swap = (d["vd"] < d["vs"]) if m["device"] == "nmos" else (d["vd"] > d["vs"])
        d.loc[swap, ["vd", "vs"]] = d.loc[swap, ["vs", "vd"]].to_numpy()
        X = make_features(d, m["device"])[m["columns"]].to_numpy()
        y = self._forward((X - self.x_mean) / self.x_scale) * self.y_scale + self.y_mean
        vds = (d["vd"] - d["vs"]).abs().to_numpy()
        cur = decode(y, vds, d["temp"].to_numpy(), m["vds_aware"], m["i0"])
        cur[swap.to_numpy()] = cur[swap.to_numpy()][:, [2, 1, 0, 3]]            # id and is trade places back
        return cur

MODEL_DIR.mkdir(exist_ok=True)
for case in CASES:
    export_model(case, MODEL_DIR / f"{case}.npz")
print({p.name: f"{p.stat().st_size / 1024:.0f} kB" for p in sorted(MODEL_DIR.glob("*.npz"))})

rows_check = []
for case in CASES:
    sm = SavedModel(MODEL_DIR / f"{case}.npz")
    te = splits[case][2]
    raw_rows = data[case].iloc[te]
    out = sm(raw_rows[FEATS])
    # bring the output into the canonical orientation again (as the data were scored) and compare
    swap = swapped[case].to_numpy()[te]
    out_canon = out.copy()
    out_canon[swap] = out_canon[swap][:, [2, 1, 0, 3]]
    ref = test_pred[case]
    m = np.abs(ref) > 1e-15
    s = score(test_true[case], out_canon, prep_cache[(case, *chosen[case][2])]["vds"][te])
    rows_check.append(dict(case=case, max_rel_diff_vs_torch=float(np.max(np.abs(out_canon[m] - ref[m]) / np.abs(ref[m]))),
                           median_err_saved=s["median_rel"].mean() * 100, median_err_section14=test[case]["median_rel"].mean() * 100,
                           p95_err_saved=s["p95_rel"].mean() * 100, p95_err_section14=test[case]["p95_rel"].mean() * 100))
check = pd.DataFrame(rows_check).set_index("case")
assert (check["max_rel_diff_vs_torch"] < 1e-3).all(), "saved model disagrees with the trained model"
check.round({"max_rel_diff_vs_torch": 8, "median_err_saved": 4, "median_err_section14": 4, "p95_err_saved": 3, "p95_err_section14": 3})

REFINE_EPOCHS = 600
EXTRA_CASES = ["nmos_off", "pmos_off"]
N_HOLD = 1000
extra = {c: pd.read_csv(DATA_DIR / f"extra_{c}.csv") for c in EXTRA_CASES}
print("extra samples:", {c: len(e) for c, e in extra.items()}, "| of which held out as the rail validation set:", N_HOLD)

def refine_arrays(case):
    aware, physics = chosen[case][2]
    prep = prep_cache[(case, aware, physics)]
    dev = CASES[case]
    ex_c, _ = canonicalize(extra[case], dev)
    X = prep["sx"].transform(make_features(ex_c, dev)[prep["columns"]])
    Y = prep["sy"].transform(encode(ex_c[TERMS].to_numpy(), (ex_c["vd"] - ex_c["vs"]).abs().to_numpy(), ex_c["temp"].to_numpy(), aware, prep["i0"]))
    hold, add = np.arange(len(ex_c) - N_HOLD, len(ex_c)), np.arange(0, len(ex_c) - N_HOLD)
    return dict(Xtr=np.vstack([prep["X"][prep["tr"]], X[add]]), Ytr=np.vstack([prep["Y"][prep["tr"]], Y[add]]),
                Xva=np.vstack([prep["X"][prep["va"]], X[hold]]), Yva=np.vstack([prep["Y"][prep["va"]], Y[hold]]), hold=extra[case].iloc[hold].reset_index(drop=True))

def fit_arrays(kind, loss, a):
    return TorchModel(kind, loss, epochs=REFINE_EPOCHS).fit(a["Xtr"], a["Ytr"], a["Xva"], a["Yva"])

arrays = {c: refine_arrays(c) for c in EXTRA_CASES}
t0 = time.time()
fitted = Parallel(n_jobs=len(EXTRA_CASES))(delayed(fit_arrays)(winner[c], chosen[c][1].loss, arrays[c]) for c in EXTRA_CASES)
refined = dict(zip(EXTRA_CASES, fitted))
print(f"{len(refined)} networks trained on {len(arrays[EXTRA_CASES[0]]['Xtr'])} rows each in {time.time() - t0:.0f} s:", {c: m.setting for c, m in refined.items()})

BUILD_REFINED = ROOT / "build" / "refined"
BUILD_REFINED.mkdir(parents=True, exist_ok=True)
kept = {c: chosen[c] for c in EXTRA_CASES}
for c in EXTRA_CASES:                                     # export_model() takes the model from `chosen`
    chosen[c] = (kept[c][0], refined[c], kept[c][2])
    export_model(c, BUILD_REFINED / f"{c}.npz")
    chosen[c] = kept[c]

def canonical_eval(path, rows_raw, case):
    """Score a saved model on raw rows: predictions and truth in the canonical orientation, as everywhere in this notebook."""
    dev = CASES[case]
    pred = SavedModel(path)(rows_raw[FEATS])
    c, swap = canonicalize(rows_raw, dev)
    sw = swap.to_numpy()
    pred[sw] = pred[sw][:, [2, 1, 0, 3]]
    return c[TERMS].to_numpy(), pred, (c["vd"] - c["vs"]).abs().to_numpy()

sets = lambda c: {"uniform validation": data[c].iloc[splits[c][1]], "rail validation": arrays[c]["hold"], "uniform test": data[c].iloc[splits[c][2]]}
rows, adopt = [], {}
for c in EXTRA_CASES:
    for name, rr in sets(c).items():
        s_old, s_new = (score(*canonical_eval(p / f"{c}.npz", rr, c), I0) for p in (MODEL_DIR, BUILD_REFINED))
        rows.append(dict(case=c, rows=name, chosen_median=s_old["median_rel"].mean() * 100, refined_median=s_new["median_rel"].mean() * 100,
                         chosen_p95=s_old["p95_rel"].mean() * 100, refined_p95=s_new["p95_rel"].mean() * 100))
before_after = pd.DataFrame(rows)
for c in EXTRA_CASES:
    u, r = (before_after[(before_after.case == c) & (before_after.rows == name)].iloc[0] for name in ("uniform validation", "rail validation"))
    adopt[c] = bool(r["refined_median"] < r["chosen_median"] and r["refined_p95"] < r["chosen_p95"] and u["refined_median"] - u["chosen_median"] < 0.5)
before_after.to_csv(RESULT_DIR / "refinement_before_after.csv", index=False)
print("adopt the refined model (rail validation median and 95th percentile improve, uniform validation median rises by less than 0.5 points):", adopt)
before_after.round(2)

import shutil

for c in EXTRA_CASES:
    if adopt[c]:
        shutil.copyfile(BUILD_REFINED / f"{c}.npz", MODEL_DIR / f"{c}.npz")
final_source = {c: ("refined (rail samples added)" if adopt.get(c, False) else "section 14") for c in CASES}

final, rows_pt = {}, []
for c in CASES:
    sets_c = {"validation": data[c].iloc[splits[c][1]], "test": data[c].iloc[splits[c][2]]}
    out = {}
    for name, rr in sets_c.items():
        true, pred, vds = canonical_eval(MODEL_DIR / f"{c}.npz", rr, c)
        s = score(true, pred, vds, I0)
        for k, t in enumerate(TERMS):
            m = np.abs(true[:, k]) >= REL_FLOOR
            s.loc[t, "mean_rel"] = np.mean(np.abs(pred[m, k] - true[m, k]) / np.abs(true[m, k]))
        out[name] = (s, true, pred)
    test[c], test_true[c], test_pred[c] = out["test"]
    final[c] = dict(model=NAMES[winner[c]], source=final_source[c], val_median=out["validation"][0]["median_rel"].mean() * 100,
                    test_median=out["test"][0]["median_rel"].mean() * 100, test_p95=out["test"][0]["p95_rel"].mean() * 100,
                    test_mean=out["test"][0]["mean_rel"].mean() * 100, test_r2_log=out["test"][0]["r2_log"].mean(), test_r2_lin=out["test"][0]["r2_lin"].mean())
final = pd.DataFrame(final).T
final.to_csv(RESULT_DIR / "final_models_test.csv")
pd.concat({c: test[c][["median_rel", "p95_rel", "mean_rel", "r2_log", "r2_lin", "median_rel_small_vds", "n_small_vds", "n_used", "frac_below_floor"]] for c in CASES},
          names=["case"]).to_csv(RESULT_DIR / "final_models_test_per_terminal.csv")
print("final models, test rows (1,500 per case); errors in %, each the mean over the four currents:")
display(final.astype({c: float for c in final.columns if c not in ("model", "source")}).round({"val_median": 2, "test_median": 2, "test_p95": 2, "test_mean": 2, "test_r2_log": 4, "test_r2_lin": 4}))

for c in CASES:
    pred_vs_spice(c)

# the saved files are the ones that were scored: reload them once more and compare with the numbers above
chk = []
for c in CASES:
    true, pred, vds = canonical_eval(MODEL_DIR / f"{c}.npz", data[c].iloc[splits[c][2]], c)
    s = score(true, pred, vds, I0)
    chk.append(dict(case=c, median_err_reloaded=s["median_rel"].mean() * 100, median_err_final_table=final.loc[c, "test_median"], p95_reloaded=s["p95_rel"].mean() * 100, p95_final_table=final.loc[c, "test_p95"]))
chk = pd.DataFrame(chk).set_index("case")
assert np.allclose(chk["median_err_reloaded"], chk["median_err_final_table"].astype(float)) and np.allclose(chk["p95_reloaded"], chk["p95_final_table"].astype(float))
chk.round(4)
