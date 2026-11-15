"""07_error_analysis.py - self-contained regeneration script.

Run from project root: python scripts/07_error_analysis.py
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

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR, RESULT_DIR = ROOT / "data", ROOT / "results"
KEY = ["gate", "pattern", "corner"]

corners = pd.read_csv(DATA_DIR / "corners.csv")
gate = pd.read_csv(DATA_DIR / "gate_ml.csv", dtype={"pattern": str}).merge(corners.drop(columns=["vdd", "temp"]), on="corner")
dev = pd.read_csv(DATA_DIR / "gate_devices.csv", dtype={"pattern": str})
circ = pd.read_csv(DATA_DIR / "circuit_ml.csv").merge(corners.drop(columns=["vdd", "temp"]), on="corner")
gate["p_vth0_mag"], circ["p_vth0_mag"] = gate["p_vth0"].abs(), circ["p_vth0"].abs()
print("gate-level points:", len(gate), "| transistor instances:", len(dev), "| circuit points:", len(circ))

def stats(d, err="err"):
    a = d[err].abs()
    return pd.Series({"n": len(d), "bias [%]": d[err].mean() * 100, "median [%]": a.median() * 100, "95th pct [%]": a.quantile(0.95) * 100, "mean [%]": a.mean() * 100})

by_case = gate.groupby(["gate", "pattern"], sort=False).apply(stats, include_groups=False).sort_values("median [%]", ascending=False)
by_gate = gate.groupby("gate", sort=False).apply(stats, include_groups=False)
by_case.to_csv(RESULT_DIR / "gate_error_by_case.csv")
by_gate.to_csv(RESULT_DIR / "gate_error_by_gate.csv")
print("by gate type:")
display(by_gate.round(2))
print("by gate and input pattern, largest median error first:")
by_case.round(2)

FACTORS = [("temp", "temperature [degC]", [-40, -12.5, 15, 42.5, 70, 97.5, 125]), ("vdd", "VDD [V]", 3), ("n_vth0", "NMOS vth0 [V]", 3), ("p_vth0_mag", "PMOS |vth0| [V]", 3),
           ("n_toxe", "NMOS toxe [m]", 3), ("p_toxe", "PMOS toxe [m]", 3)]

def binned(df, col, bins, err="err"):
    cut = pd.cut(df[col], bins, include_lowest=True) if isinstance(bins, list) else pd.qcut(df[col], bins)
    return df.groupby(cut, observed=True).apply(stats, err=err, include_groups=False)

tabs = {name: binned(gate, col, bins) for col, name, bins in FACTORS}
for name, t in tabs.items():
    t.index = [f"{i.left:.3g} .. {i.right:.3g}" for i in t.index]
    print(name)
    display(t.round(2))
pd.concat(tabs, names=["factor", "bin"]).to_csv(RESULT_DIR / "gate_error_by_corner.csv")

COL = {"median": "#2a78d6", "p95": "#eb6834"}
fig, ax = plt.subplots(2, 3, figsize=(14, 6.5))
for a, (name, t) in zip(ax.ravel(), tabs.items()):
    x = np.arange(len(t))
    a.plot(x, t["median [%]"], "o-", color=COL["median"], ms=7, mec="white", mew=1.2, label="median |error|")
    a.plot(x, t["95th pct [%]"], "s-", color=COL["p95"], ms=7, mec="white", mew=1.2, label="95th percentile")
    a.plot(x, t["bias [%]"], "^--", color="0.4", ms=6, mec="white", mew=1.0, lw=1, label="bias")
    a.axhline(0, color="0.8", lw=0.8)
    a.set_xticks(x); a.set_xticklabels([s.replace(" .. ", "\n") for s in t.index], fontsize=7.5)
    a.set_xlabel(name); a.set_ylabel("error [%]")
    a.grid(axis="y", color="0.92", lw=0.8); a.set_axisbelow(True); a.spines[["top", "right"]].set_visible(False)
handles, labels = ax[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(RESULT_DIR / "gate_error_factors.png", dpi=110)
plt.close("all")

dv = dev.merge(gate[KEY + ["leak_total"]], on=KEY)
total = gate["leak_total"].sum()
rows = []
for case, g in dv.groupby("case"):
    for comp in ("channel", "gate", "body"):
        d = g[f"ml_{comp}"] - g[f"spice_{comp}"]
        rows.append(dict(model=case, component=comp, abs_budget=d.abs().sum() / total * 100, bias=d.sum() / total * 100, n=len(g)))
budget = pd.DataFrame(rows)
budget_abs = budget.pivot(index="model", columns="component", values="abs_budget").assign(total=lambda t: t.sum(axis=1))
budget_bias = budget.pivot(index="model", columns="component", values="bias").assign(total=lambda t: t.sum(axis=1))
pd.concat({"absolute error budget [% of the summed leakage]": budget_abs, "bias [% of the summed leakage]": budget_bias}, axis=1).to_csv(RESULT_DIR / "gate_error_budget.csv")
display(budget_abs.round(2).rename_axis("absolute error budget [% of the summed leakage]"))
display(budget_bias.round(2).rename_axis("bias [% of the summed leakage]"))
print("check: summed bias of all terms =", round(budget["bias"].sum(), 3), "% | bias of the sum of the gate leakages =", round((gate["ml_leak"].sum() - total) / total * 100, 3), "%")

share = pd.DataFrame([dict(model=case, component=comp, share=g[f"spice_{comp}"].sum() / total * 100) for case, g in dv.groupby("case") for comp in ("channel", "gate", "body")])
share = share.pivot(index="model", columns="component", values="share").assign(total=lambda t: t.sum(axis=1))
share.to_csv(RESULT_DIR / "gate_leakage_share.csv")
share.round(1)

worst = gate.reindex(gate["err"].abs().sort_values(ascending=False).index).head(10)[KEY + ["temp", "vdd", "n_vth0", "p_vth0", "leak_total", "ml_leak", "err"]].copy()
dom = []
for _, r in worst.iterrows():
    g = dv[(dv.gate == r.gate) & (dv.pattern == r.pattern) & (dv.corner == r.corner)]
    best = max(((abs(row[f"ml_{c}"] - row[f"spice_{c}"]), f"{row['device']} ({row['case']}) {c}") for _, row in g.iterrows() for c in ("channel", "gate", "body")))
    dom.append(best[1])
worst["largest term error"] = dom
worst.to_csv(RESULT_DIR / "gate_worst_points.csv", index=False)
worst.assign(err=lambda t: (t["err"] * 100).round(1), leak_total=lambda t: t["leak_total"].map("{:.2e}".format), ml_leak=lambda t: t["ml_leak"].map("{:.2e}".format))

def inv_input_rail(g, p):                         # logic value of the NAND2 / NOR2 output that drives the inverter: 1 -> VDD, 0 -> 0 V
    return (p != "11") if g == "and2" else (p == "00")

rows = []
for (g, p), d in gate[gate.gate.isin(["and2", "or2"])].groupby(["gate", "pattern"], sort=False):
    dist = (d["vdd"] - d["v_n"]) if inv_input_rail(g, p) else d["v_n"]
    rows.append({"gate": g, "pattern": p, "rail": "VDD" if inv_input_rail(g, p) else "0 V", "median distance [mV]": dist.median() * 1e3, "max distance [mV]": dist.max() * 1e3,
                 "Spearman(distance, error)": dist.corr(d["err"], method="spearman"), "Spearman(distance, temperature)": dist.corr(d["temp"], method="spearman"), "Spearman(distance, VDD)": dist.corr(d["vdd"], method="spearman"), "median signed error [%]": d["err"].median() * 100, "median |error| [%]": d["err"].abs().median() * 100})
nonrail = pd.DataFrame(rows)
nonrail.to_csv(RESULT_DIR / "gate_error_nonrail_input.csv", index=False)
nonrail.round(2)

rows = []
for col, name, bins in FACTORS[:4]:
    cut = pd.cut(circ[col], bins, include_lowest=True) if isinstance(bins, list) else pd.qcut(circ[col], bins)
    for iv, g in circ.groupby(cut, observed=True):
        rows.append({"factor": name, "bin": f"{iv.left:.3g} .. {iv.right:.3g}",
                     "ML: median |err| [%]": g["err"].abs().median() * 100, "ML: 95th pct [%]": g["err"].abs().quantile(0.95) * 100, "ML: bias [%]": g["err"].mean() * 100,
                     "floor: median |err| [%]": g["err_floor"].abs().median() * 100, "floor: bias [%]": g["err_floor"].mean() * 100,
                     "gate models: bias [%]": g["err_gates"].mean() * 100, "gate models: median |err| [%]": g["err_gates"].abs().median() * 100})
circ_tab = pd.DataFrame(rows).set_index(["factor", "bin"])
circ_tab.to_csv(RESULT_DIR / "circuit_error_by_corner.csv")
circ_tab.round(2)

bv = circ.groupby("vector").apply(lambda g: pd.Series({"a": int(g.name) & 7, "b": (int(g.name) >> 3) & 7, "cin": int(g.name) >> 6, "median |err| [%]": g["err"].abs().median() * 100,
                                                       "bias [%]": g["err"].mean() * 100, "SPICE median [A]": g["leak_total"].median()}), include_groups=False)
print("largest median error:")
display(bv.sort_values("median |err| [%]", ascending=False).head(6).round(3))
print("smallest median error:")
bv.to_csv(RESULT_DIR / "circuit_error_by_vector.csv")
bv.sort_values("median |err| [%]").head(6).round(3)
