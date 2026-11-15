"""06_circuit.py - self-contained regeneration script.

Run from project root: python scripts/06_circuit.py
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

import json
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "Model Files-20261105" / "models" / "16nm_HP.pm"
BUILD_DIR, DATA_DIR, RESULT_DIR, NETLIST_DIR = ROOT / "build", ROOT / "data", ROOT / "results", ROOT / "netlists"
LMIN = 16e-9
WMIN = LMIN
LDIFF = 2 * LMIN
PROC_PARAMS = ("vth0", "toxe", "ndep", "u0", "xj")
SPICE_OPTIONS = ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9"
NUMDGT = 15
N_WORKERS = 4                                # parallel ngspice processes: on the machine used 4 give 1.7x the throughput of one, 8 are slower than 4
CHUNK = 5                                   # corners per ngspice process (5 x 128 = 640 operating points)
REUSE_SPICE = False                         # True: reuse data/circuit_spice.csv of an earlier run (quick rerun of the ML part only)

XX = json.loads((DATA_DIR / "gate_sizing.json").read_text())["XX"]             # drive strength chosen in script 04
corners = pd.read_csv(DATA_DIR / "corners.csv")
gate_ml = pd.read_csv(DATA_DIR / "gate_ml.csv", dtype={"pattern": str})
print(corners.shape, gate_ml.shape)

INPUTS = ["a0", "a1", "a2", "b0", "b1", "b2", "c0"]
OUTPUTS = ["s0", "s1", "s2", "c3"]
CLA = []                                           # (instance, gate type, (A, B), output net), in an order that is valid for evaluation

def xor(tag, x, y, out):
    CLA.extend([(f"{tag}n1", "nand2", (x, y), f"{tag}_n1"), (f"{tag}n2", "nand2", (x, f"{tag}_n1"), f"{tag}_n2"),
                (f"{tag}n3", "nand2", (y, f"{tag}_n1"), f"{tag}_n3"), (f"{tag}n4", "nand2", (f"{tag}_n2", f"{tag}_n3"), out)])

for i in range(3):
    xor(f"px{i}", f"a{i}", f"b{i}", f"p{i}")
    CLA.append((f"g{i}", "and2", (f"a{i}", f"b{i}"), f"g{i}"))
CLA += [("pc0", "and2", ("p0", "c0"), "pc0"), ("c1", "or2", ("g0", "pc0"), "c1"),
        ("t1", "and2", ("p1", "g0"), "t1"), ("t2", "and2", ("p1", "p0"), "t2"), ("t3", "and2", ("t2", "c0"), "t3"),
        ("o1", "or2", ("g1", "t1"), "o1"), ("c2", "or2", ("o1", "t3"), "c2"),
        ("u1", "and2", ("p2", "g1"), "u1"), ("u2", "and2", ("p2", "t1"), "u2"), ("u3", "and2", ("p2", "t3"), "u3"),
        ("o2", "or2", ("g2", "u1"), "o2"), ("o3", "or2", ("o2", "u2"), "o3"), ("c3", "or2", ("o3", "u3"), "c3")]
for i in range(3):
    xor(f"sx{i}", f"p{i}", f"c{i}", f"s{i}")
FUNC = {"nand2": lambda a, b: 1 - (a & b), "nor2": lambda a, b: 1 - (a | b), "and2": lambda a, b: a & b, "or2": lambda a, b: a | b}
TRANSISTORS = {"nand2": 4, "nor2": 4, "and2": 6, "or2": 6}
counts = pd.Series([g for _, g, _, _ in CLA]).value_counts()
print("gates:", len(CLA), counts.to_dict(), "| transistors:", sum(TRANSISTORS[g] * n for g, n in counts.items()))

def vector_bits(v):
    """Input vector number 0..127 -> bits of (a0, a1, a2, b0, b1, b2, c0): a = v mod 8, b = (v div 8) mod 8, carry-in = v div 64."""
    a, b, c = v & 7, (v >> 3) & 7, v >> 6
    return [(a >> i) & 1 for i in range(3)] + [(b >> i) & 1 for i in range(3)] + [c]

def logic(v):
    """Logic value of every net for input vector v."""
    net = dict(zip(INPUTS, vector_bits(v)))
    for _, g, (x, y), out in CLA:
        net[out] = int(FUNC[g](net[x], net[y]))
    return net

# the logic simulator must be a binary adder: a + b + cin = (c3 s2 s1 s0)
bad = [v for v in range(128) if sum(logic(v)[o] << k for k, o in enumerate(OUTPUTS)) != (v & 7) + ((v >> 3) & 7) + (v >> 6)]
print("input vectors where the circuit is not a correct adder:", bad)
assert not bad

def fmt(x):
    return f"{float(x):.10g}"

SUBCKTS = """* gates (sizes as in Part B): widths are multiples of Wmin, junction areas AD = AS = W*Ldiff
.subckt inv a y vdd
MP y a vdd vdd pmos W={2*XX*Wmin} L={Lmin} AD={2*XX*Wmin*Ldiff} AS={2*XX*Wmin*Ldiff}
MN y a 0 0 nmos W={XX*Wmin} L={Lmin} AD={XX*Wmin*Ldiff} AS={XX*Wmin*Ldiff}
.ends
.subckt nand2 a b y vdd
MP1 y a vdd vdd pmos W={2*XX*Wmin} L={Lmin} AD={2*XX*Wmin*Ldiff} AS={2*XX*Wmin*Ldiff}
MP2 y b vdd vdd pmos W={2*XX*Wmin} L={Lmin} AD={2*XX*Wmin*Ldiff} AS={2*XX*Wmin*Ldiff}
MN1 y a x 0 nmos W={2*XX*Wmin} L={Lmin} AD={2*XX*Wmin*Ldiff} AS={2*XX*Wmin*Ldiff}
MN2 x b 0 0 nmos W={2*XX*Wmin} L={Lmin} AD={2*XX*Wmin*Ldiff} AS={2*XX*Wmin*Ldiff}
.ends
.subckt nor2 a b y vdd
MP1 z a vdd vdd pmos W={4*XX*Wmin} L={Lmin} AD={4*XX*Wmin*Ldiff} AS={4*XX*Wmin*Ldiff}
MP2 y b z vdd pmos W={4*XX*Wmin} L={Lmin} AD={4*XX*Wmin*Ldiff} AS={4*XX*Wmin*Ldiff}
MN1 y a 0 0 nmos W={XX*Wmin} L={Lmin} AD={XX*Wmin*Ldiff} AS={XX*Wmin*Ldiff}
MN2 y b 0 0 nmos W={XX*Wmin} L={Lmin} AD={XX*Wmin*Ldiff} AS={XX*Wmin*Ldiff}
.ends
.subckt and2 a b y vdd
X1 a b n vdd nand2
X2 n y vdd inv
.ends
.subckt or2 a b y vdd
X1 a b n vdd nor2
X2 n y vdd inv
.ends"""
SOURCES = ["vdd"] + [f"v{i}" for i in INPUTS]
PRINT = "print " + " ".join([f"i({s})" for s in SOURCES] + ["v(vdd)"] + [f"v({i})" for i in INPUTS] + [f"v({o})" for o in OUTPUTS])

def nominal_values(path, params):
    cards, cur = {}, None
    for raw in Path(path).read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("*"):
            continue
        if s.lower().startswith(".model"):
            cur = s.split()[1].lower(); cards[cur] = [s]
        elif s.startswith("+") and cur:
            cards[cur].append(s[1:])
        else:
            cur = None
    out = {}
    for name, parts in cards.items():
        text = " ".join(parts)
        out[name] = {p: float(re.search(rf"(?<![\w.]){p}\s*=\s*([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)", text, re.I).group(1)) for p in params}
    return out

NOMINAL = nominal_values(MODEL_FILE, PROC_PARAMS)
nominal_corner = pd.DataFrame([dict(corner=0, vdd=0.9, temp=27.0, **{f"n_{p}": NOMINAL["nmos"][p] for p in PROC_PARAMS}, **{f"p_{p}": NOMINAL["pmos"][p] for p in PROC_PARAMS})])

def cla_netlist(corner_df, vectors, include):
    lines = [f"* 3-bit carry-lookahead adder, {len(CLA)} gates: {len(corner_df)} corner(s) x {len(vectors)} input vector(s)",
             "* inputs a0..a2, b0..b2, c0 (carry-in); outputs s0..s2 and c3 (carry-out)", f".include {include}", SPICE_OPTIONS, "",
             f".param Lmin={fmt(LMIN)}", f".param Wmin={fmt(WMIN)}", f".param Ldiff={fmt(LDIFF)}", f".param XX={XX}", "", SUBCKTS, "",
             f"Vdd vdd 0 {fmt(corner_df.iloc[0].vdd)}"]
    lines += [f"V{i} {i} 0 0" for i in INPUTS]
    lines += [f"X{n} {x} {y} {out} vdd {g}" for n, g, (x, y), out in CLA]
    lines += ["", ".control", f"set numdgt = {NUMDGT}"]
    for c in corner_df.itertuples():
        lines.append(f"* corner {c.corner}")
        lines += [f"altermod nmos {p}={fmt(getattr(c, 'n_' + p))}" for p in PROC_PARAMS] + [f"altermod pmos {p}={fmt(getattr(c, 'p_' + p))}" for p in PROC_PARAMS]
        lines += [f"set temp = {fmt(c.temp)}", f"alter vdd = {fmt(c.vdd)}"]
        for v in vectors:
            lines.append(f"echo @@ {c.corner} {v}")
            lines += [f"alter v{i} = {fmt(bit * c.vdd)}" for i, bit in zip(INPUTS, vector_bits(v))]
            lines += ["op", PRINT]
    return "\n".join(lines + [".endc", ".end", ""])

text = cla_netlist(nominal_corner, [0], '"../Model Files-20261105/models/16nm_HP.pm"')
print("\n".join(text.splitlines()[:30]), "\n...")

def to_wsl(path):
    p = Path(path).resolve()
    return "/mnt/" + p.drive.rstrip(":").lower() + "/" + "/".join(p.parts[1:])

def run_ngspice(cir_name, cwd):
    if shutil.which("ngspice"):
        r = subprocess.run(["ngspice", "-b", str(cir_name)], cwd=cwd, capture_output=True, text=True, errors="replace")
        return r.stdout.replace("\x00", "")
    r = subprocess.run(["wsl", "-e", "bash", "-c", f"cd '{to_wsl(cwd)}' && ngspice -b '{cir_name}' 2>/dev/null"], capture_output=True, text=True, errors="replace")
    return r.stdout.replace("\x00", "")

def ngspice_version():
    r = subprocess.run(["wsl", "-e", "bash", "-c", "ngspice --version 2>&1"], capture_output=True, text=True, errors="replace") if not shutil.which("ngspice") else \
        subprocess.run(["ngspice", "--version"], capture_output=True, text=True, errors="replace")
    m = re.search(r"ngspice-\d+(?:\.\d+)?", (r.stdout + r.stderr).replace("\x00", ""))
    return m.group(0) if m else "unknown"

_LINE = re.compile(r"^(@@) (\d+) (\d+)$|^([iv])\((\w+)\) = (\S+)$")
WANT = {f"i({s})" for s in SOURCES} | {"v(vdd)"} | {f"v({i})" for i in INPUTS} | {f"v({o})" for o in OUTPUTS}

def parse_cla(out, corner_df):
    by_corner = corner_df.set_index("corner")
    blocks, cur = {}, None
    for line in out.splitlines():
        m = _LINE.match(line.strip())
        if not m:
            continue
        if m.group(1):
            cur = (int(m.group(2)), int(m.group(3))); blocks[cur] = {}
        elif cur is not None:
            blocks[cur][f"{m.group(4)}({m.group(5)})"] = float(m.group(6))
    rows = []
    for (k, v), d in blocks.items():
        if set(d) != WANT or k not in by_corner.index:
            continue
        c = by_corner.loc[k]
        bits = vector_bits(v)
        if abs(d["v(vdd)"] - c.vdd) > 1e-9 or any(abs(d[f"v({i})"] - bit * c.vdd) > 1e-9 for i, bit in zip(INPUTS, bits)):
            continue
        row = dict(corner=k, vector=v, vdd=c.vdd, temp=c.temp, i_vdd=d["i(vdd)"])
        row.update({f"i_{i}": d[f"i(v{i})"] for i in INPUTS})
        row.update({f"v_{o}": d[f"v({o})"] for o in OUTPUTS})
        rows.append(row)
    return pd.DataFrame(rows)

print("simulator:", ngspice_version())

NETLIST_DIR.mkdir(exist_ok=True)
vecs = [0, 27, 77, 127]
example = NETLIST_DIR / "test_circuit_cla3.cir"
example.write_text(cla_netlist(nominal_corner, vecs, '"../Model Files-20261105/models/16nm_HP.pm"'), newline="\n")
got = parse_cla(run_ngspice(example.name, NETLIST_DIR), nominal_corner)
assert len(got) == len(vecs), "example netlist did not simulate"
got["sum_read"] = [sum(int(getattr(r, f"v_{o}") > r.vdd / 2) << k for k, o in enumerate(OUTPUTS)) for r in got.itertuples()]
got["sum_true"] = [(v & 7) + ((v >> 3) & 7) + (v >> 6) for v in got["vector"]]
print("example netlist: output sum read from the voltages equals a + b + cin:", bool((got["sum_read"] == got["sum_true"]).all()))
got[["vector", "i_vdd", "v_s0", "v_s1", "v_s2", "v_c3", "sum_read", "sum_true"]]

def run_cla_chunk(sub, tag):
    name = f"cla_{tag}.cir"
    (BUILD_DIR / name).write_text(cla_netlist(sub, range(128), MODEL_FILE.name), newline="\n")
    return parse_cla(run_ngspice(name, BUILD_DIR), sub)

BUILD_DIR.mkdir(exist_ok=True)
shutil.copyfile(MODEL_FILE, BUILD_DIR / MODEL_FILE.name)
t0 = time.time()
if REUSE_SPICE and (DATA_DIR / "circuit_spice.csv").exists():
    raw = pd.read_csv(DATA_DIR / "circuit_spice.csv")[["corner", "vector", "vdd", "temp", "i_vdd"] + [f"i_{i}" for i in INPUTS] + [f"v_{o}" for o in OUTPUTS]]
else:
    jobs = [(corners.iloc[s:s + CHUNK], s) for s in range(0, len(corners), CHUNK)]
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        raw = pd.concat(list(ex.map(lambda j: run_cla_chunk(*j), jobs)), ignore_index=True)
seconds = time.time() - t0
have = set(zip(raw["corner"], raw["vector"]))
assert len(raw) == len(have) == 200 * 128 and have == {(k, v) for k in corners["corner"] for v in range(128)}, f"{200 * 128 - len(have)} operating points missing"
circuit = raw.sort_values(["corner", "vector"]).reset_index(drop=True)
bits = np.array([vector_bits(v) for v in circuit["vector"]])
circuit["leak_supply"] = -circuit["i_vdd"]
circuit["leak_total"] = -circuit["i_vdd"] - (bits * circuit[[f"i_{i}" for i in INPUTS]].to_numpy()).sum(axis=1)
print(f"{len(circuit)} operating points in {seconds / 60:.1f} min on {N_WORKERS} processes; none missing, none duplicated, none rejected by the parser")
circuit.head()

expect = np.array([[logic(v)[o] for o in OUTPUTS] for v in circuit["vector"]])
volts = circuit[[f"v_{o}" for o in OUTPUTS]].to_numpy()
dev_v = np.abs(volts - expect * circuit[["vdd"]].to_numpy())
print(f"1. outputs: largest deviation from the logic rail {dev_v.max() * 1e3:.1f} mV; points with a wrong output bit: {int(((volts > circuit[['vdd']].to_numpy() / 2) != expect.astype(bool)).sum())}")
print(f"2. leak_supply > 0: {bool((circuit['leak_supply'] > 0).all())}; leak_total > 0: {bool((circuit['leak_total'] > 0).all())}")
assert dev_v.max() < 0.1 and (circuit["leak_supply"] > 0).all() and (circuit["leak_total"] > 0).all()
circuit.to_csv(DATA_DIR / "circuit_spice.csv", index=False, float_format="%.12g")
circuit[["leak_supply", "leak_total"]].describe().T

gl = gate_ml.set_index(["gate", "pattern", "corner"]).sort_index()
corner_ids = corners["corner"].to_numpy()
ml_by = {k: gl.loc[k, "ml_leak"].reindex(corner_ids).to_numpy() for k in {(g, p) for g, p, _ in gl.index}}
sp_by = {k: gl.loc[k, "leak_total"].reindex(corner_ids).to_numpy() for k in ml_by}
ml_sum = np.zeros((128, len(corner_ids)))
sp_sum = np.zeros((128, len(corner_ids)))
for v in range(128):
    net = logic(v)
    for _, g, (x, y), out in CLA:
        pat = f"{net[x]}{net[y]}"
        ml_sum[v] += ml_by[(g, pat)]
        sp_sum[v] += sp_by[(g, pat)]
pos = {k: i for i, k in enumerate(corner_ids)}
idx = (circuit["vector"].to_numpy(), circuit["corner"].map(pos).to_numpy())
circuit["ml_leak"], circuit["gate_sum_spice"] = ml_sum[idx], sp_sum[idx]
circuit["err"] = circuit["ml_leak"] / circuit["leak_total"] - 1                  # hierarchical ML vs the full circuit
circuit["err_floor"] = circuit["gate_sum_spice"] / circuit["leak_total"] - 1      # composition floor: perfect gate models vs the full circuit
circuit["err_gates"] = circuit["ml_leak"] / circuit["gate_sum_spice"] - 1         # ML gate models vs the isolated-gate SPICE sum
circuit.to_csv(DATA_DIR / "circuit_ml.csv", index=False, float_format="%.12g")
ab = lambda s: f"median |error| {s.abs().median() * 100:.2f} %, 95th percentile {s.abs().quantile(0.95) * 100:.2f} %, mean |error| {s.abs().mean() * 100:.2f} %, bias {s.mean() * 100:+.2f} %"
print("hierarchical ML vs full-circuit SPICE :", ab(circuit["err"]))
print("composition floor (perfect gates)     :", ab(circuit["err_floor"]))
print("ML gate models vs isolated-gate SPICE :", ab(circuit["err_gates"]))

def metrics(g, err="err", leak="ml_leak"):
    return pd.Series({"SPICE median [A]": g["leak_total"].median(), "ML median [A]": g[leak].median(), "bias [%]": g[err].mean() * 100,
                      "median error [%]": g[err].abs().median() * 100, "95th pct error [%]": g[err].abs().quantile(0.95) * 100, "mean error [%]": g[err].abs().mean() * 100})

summary = pd.DataFrame({"hierarchical ML vs SPICE": metrics(circuit), "composition floor (perfect gate models)": metrics(circuit, "err_floor", "gate_sum_spice"),
                        "ML gate models vs isolated-gate SPICE sum": pd.Series({"bias [%]": circuit["err_gates"].mean() * 100, "median error [%]": circuit["err_gates"].abs().median() * 100,
                                                                              "95th pct error [%]": circuit["err_gates"].abs().quantile(0.95) * 100, "mean error [%]": circuit["err_gates"].abs().mean() * 100})})
summary.to_csv(RESULT_DIR / "circuit_results.csv")
by_vector = circuit.groupby("vector").apply(metrics, include_groups=False)
by_vector.insert(0, "a", [v & 7 for v in by_vector.index]); by_vector.insert(1, "b", [(v >> 3) & 7 for v in by_vector.index]); by_vector.insert(2, "cin", [v >> 6 for v in by_vector.index])
by_vector.to_csv(RESULT_DIR / "circuit_results_by_vector.csv")
print("all 25,600 points:")
display(summary.style.format("{:.3g}"))
print("per input vector (128 rows): spread of the per-vector metrics")
by_vector[["bias [%]", "median error [%]", "95th pct error [%]", "mean error [%]"]].describe().loc[["min", "50%", "max"]].round(2)

fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.2))
lo, hi = circuit["leak_total"].min() * 0.8, circuit["leak_total"].max() * 1.25
ax[0].plot([lo, hi], [lo, hi], "--", color="0.5", lw=1)
for f in (0.9, 1.1):
    ax[0].plot([lo, hi], [lo * f, hi * f], ":", color="0.7", lw=0.8)
ax[0].scatter(circuit["leak_total"], circuit["ml_leak"], s=2, alpha=0.25, color="#6250d6", linewidths=0)
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlim(lo, hi); ax[0].set_ylim(lo, hi)
ax[0].set_xlabel("SPICE circuit leakage [A]"); ax[0].set_ylabel("ML circuit leakage [A]"); ax[0].set_title("3-bit CLA: 128 vectors x 200 corners")
bins = np.arange(-40, 141, 20)
mid = (bins[:-1] + bins[1:]) / 2
cut = pd.cut(circuit["temp"], bins)
for col, lab, c, m in (("err", "hierarchical ML", "#6250d6", "D"), ("err_floor", "composition floor (perfect gate models)", "#eb6834", "s")):
    s = circuit.groupby(cut, observed=False)[col].agg(lambda x: x.median() * 100)
    ax[1].plot(mid, s.to_numpy(), m + "-", color=c, ms=7, mec="white", mew=1.2, label=lab)
ax[1].axhline(0, color="0.6", lw=0.8)
ax[1].set_xlabel("temperature [degC]"); ax[1].set_ylabel("median signed error [%]"); ax[1].legend(frameon=False)
for a in ax:
    a.grid(color="0.92", lw=0.8); a.set_axisbelow(True); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(RESULT_DIR / "circuit_parity.png", dpi=120)
plt.close("all")