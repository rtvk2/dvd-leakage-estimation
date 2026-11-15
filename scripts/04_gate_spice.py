"""04_gate_spice.py - self-contained regeneration script.

Run from project root: python scripts/04_gate_spice.py
Produces artifacts in data/, models/, netlists/, results/ with fixed seeds.
"""

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
MODEL_FILE = ROOT / "Model Files-20261105" / "models" / "16nm_HP.pm"     # assigned model file
BUILD_DIR = ROOT / "build"          # scratch space for generated netlists (git-ignored)
NETLIST_DIR = ROOT / "netlists"
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results"

VDD_NOM = 0.9
LMIN = 16e-9                        # channel length = feature size of the node
WMIN = LMIN                         # smallest allowed width (1 x Lmin); the spec allows 1 to 8 x Lmin
LDIFF = 2 * LMIN                    # diffusion length for the junction areas AD = AS = W * LDIFF (as in the datasets)
VDD_VAR = 0.10                      # VDD uniform in nominal +/- 10 %
TEMP_RANGE = (-40.0, 125.0)         # degC, uniform
PROC_VAR = 0.10                     # process parameters uniform in nominal +/- 10 %
PROC_PARAMS = ("vth0", "toxe", "ndep", "u0", "xj")

N_CORNERS = 200
CORNER_SEED = 20261105 + 100        # the datasets use 20261105 + 0..3
SPICE_OPTIONS = ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9"   # required in every netlist
NUMDGT = 15                         # printed digits; the default 6 would round the currents
N_WORKERS = 8                       # parallel ngspice processes
print(f"model {MODEL_FILE.name} | exists: {MODEL_FILE.exists()} | nominal VDD {VDD_NOM} V | L = Wmin = {LMIN * 1e9:.0f} nm | {N_CORNERS} corners")

def read_cards(path):
    """Return {card_name: text}; continuation lines ('+') are joined to their .model line."""
    cards, cur = {}, None
    for raw in Path(path).read_text().splitlines():
        s = raw.strip()
        if not s or s.startswith("*"):
            continue
        if s.lower().startswith(".model"):
            cur = s.split()[1].lower()
            cards[cur] = [s]
        elif s.startswith("+") and cur:
            cards[cur].append(s[1:])
        else:
            cur = None
    return {k: " ".join(v) for k, v in cards.items()}

_NUM = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"

def nominal_values(path, params):
    """Return {card_name: {param: value}} for the requested parameters."""
    out = {}
    for name, text in read_cards(path).items():
        vals = {}
        for p in params:
            m = re.search(rf"(?<![\w.]){p}\s*=\s*({_NUM})", text, re.I)      # (?<![\w.]) keeps 'vth0' from matching inside 'lvth0'
            if m is None:
                raise ValueError(f"parameter {p} not found in card {name}")
            vals[p] = float(m.group(1))
        out[name] = vals
    return out

NOMINAL = nominal_values(MODEL_FILE, PROC_PARAMS)

def to_wsl(path):
    p = Path(path).resolve()
    return "/mnt/" + p.drive.rstrip(":").lower() + "/" + "/".join(p.parts[1:])

def prepare_build():
    """Create the scratch directory and copy the model file there (no spaces in the path)."""
    BUILD_DIR.mkdir(exist_ok=True)
    dst = BUILD_DIR / MODEL_FILE.name
    if not dst.exists() or dst.read_bytes() != MODEL_FILE.read_bytes():
        shutil.copyfile(MODEL_FILE, dst)
    return BUILD_DIR

def run_ngspice(cir_name, cwd, merge_stderr=False):
    """Run `ngspice -b cir_name` inside cwd and return its output as text."""
    cwd = Path(cwd)
    if shutil.which("ngspice"):
        r = subprocess.run(["ngspice", "-b", str(cir_name)], cwd=cwd, capture_output=True, text=True, errors="replace")
        return (r.stdout + (r.stderr if merge_stderr else "")).replace("\x00", "")
    cmd = f"cd '{to_wsl(cwd)}' && ngspice -b '{cir_name}'" + (" 2>&1" if merge_stderr else " 2>/dev/null")
    r = subprocess.run(["wsl", "-e", "bash", "-c", cmd], capture_output=True, text=True, errors="replace")
    return r.stdout.replace("\x00", "")

def ngspice_version():
    if shutil.which("ngspice"):
        r = subprocess.run(["ngspice", "--version"], capture_output=True, text=True, errors="replace")
        text = r.stdout + r.stderr
    else:
        r = subprocess.run(["wsl", "-e", "bash", "-c", "ngspice --version 2>&1"], capture_output=True, text=True, errors="replace")
        text = r.stdout.replace("\x00", "")
    m = re.search(r"ngspice-\d+(?:\.\d+)?", text)
    return m.group(0) if m else "unknown"

print("simulator:", ngspice_version())
pd.DataFrame(NOMINAL).T

# (name, type, drain, gate, source, bulk, width as a multiple of Wmin)
XX = 2                                      # drive strength: every width below is a multiple of XX * Wmin
GATES = {
    "inv": dict(inputs=("a",), nodes=("out",), devices=[
        ("MP1", "pmos", "out", "a", "vdd", "vdd", 2 * XX), ("MN1", "nmos", "out", "a", "vss", "vss", 1 * XX)]),
    "nand2": dict(inputs=("a", "b"), nodes=("x", "out"), devices=[
        ("MP1", "pmos", "out", "a", "vdd", "vdd", 2 * XX), ("MP2", "pmos", "out", "b", "vdd", "vdd", 2 * XX),
        ("MN1", "nmos", "out", "a", "x", "vss", 2 * XX), ("MN2", "nmos", "x", "b", "vss", "vss", 2 * XX)]),
    "nor2": dict(inputs=("a", "b"), nodes=("y", "out"), devices=[
        ("MP1", "pmos", "y", "a", "vdd", "vdd", 4 * XX), ("MP2", "pmos", "out", "b", "y", "vdd", 4 * XX),
        ("MN1", "nmos", "out", "a", "vss", "vss", 1 * XX), ("MN2", "nmos", "out", "b", "vss", "vss", 1 * XX)]),
    "and2": dict(inputs=("a", "b"), nodes=("x", "n", "out"), devices=[
        ("MP1", "pmos", "n", "a", "vdd", "vdd", 2 * XX), ("MP2", "pmos", "n", "b", "vdd", "vdd", 2 * XX),
        ("MN1", "nmos", "n", "a", "x", "vss", 2 * XX), ("MN2", "nmos", "x", "b", "vss", "vss", 2 * XX),
        ("MP3", "pmos", "out", "n", "vdd", "vdd", 2 * XX), ("MN3", "nmos", "out", "n", "vss", "vss", 1 * XX)]),
    "or2": dict(inputs=("a", "b"), nodes=("y", "n", "out"), devices=[
        ("MP1", "pmos", "y", "a", "vdd", "vdd", 4 * XX), ("MP2", "pmos", "n", "b", "y", "vdd", 4 * XX),
        ("MN1", "nmos", "n", "a", "vss", "vss", 1 * XX), ("MN2", "nmos", "n", "b", "vss", "vss", 1 * XX),
        ("MP3", "pmos", "out", "n", "vdd", "vdd", 2 * XX), ("MN3", "nmos", "out", "n", "vss", "vss", 1 * XX)]),
}
LOGIC = {"inv": lambda a: 1 - a, "nand2": lambda a, b: 1 - (a & b), "nor2": lambda a, b: 1 - (a | b),
         "and2": lambda a, b: a & b, "or2": lambda a, b: a | b}
PATTERNS = {g: (["0", "1"] if len(d["inputs"]) == 1 else ["00", "01", "10", "11"]) for g, d in GATES.items()}
CASES = [(g, p) for g in GATES for p in PATTERNS[g]]
DATA_DIR.mkdir(exist_ok=True)
(DATA_DIR / "gate_sizing.json").write_text(json.dumps(dict(XX=XX, Wmin=WMIN, Lmin=LMIN, Ldiff=LDIFF)))
print(len(CASES), "gate and input cases:", ", ".join(f"{g}_{p}" for g, p in CASES))
pd.DataFrame([dict(gate=g, device=n, type=t, drain=d, gate_node=gt, source=s, bulk=b, W_nm=w * WMIN * 1e9) for g, spec in GATES.items()
              for n, t, d, gt, s, b, w in spec["devices"]]).set_index(["gate", "device"])

def fmt(x):
    """Number formatting shared by the netlists and the CSV files (10 significant digits)."""
    return f"{float(x):.10g}"

CORNER_COLS = ["corner", "vdd", "temp"] + [f"n_{p}" for p in PROC_PARAMS] + [f"p_{p}" for p in PROC_PARAMS]

def nominal_corner():
    row = dict(corner=0, vdd=VDD_NOM, temp=27.0)
    row.update({f"n_{p}": NOMINAL["nmos"][p] for p in PROC_PARAMS})
    row.update({f"p_{p}": NOMINAL["pmos"][p] for p in PROC_PARAMS})
    return pd.DataFrame([row])[CORNER_COLS]

def printed(gate):
    """Names of the vectors printed after every operating point."""
    ins = GATES[gate]["inputs"]
    return [f"i(vdd)"] + [f"i(v{i})" for i in ins] + ["i(vss)", "v(vdd)"] + [f"v({i})" for i in ins] + [f"v({n})" for n in GATES[gate]["nodes"]]

def gate_netlist(gate, corners, include):
    spec = GATES[gate]
    ins = spec["inputs"]
    lines = [
        f"* {gate.upper()} gate, {MODEL_FILE.stem}: {len(corners)} corner(s) x {len(PATTERNS[gate])} input pattern(s)",
        f"* inputs {', '.join(i.upper() for i in ins)}; recorded internal/output nodes: {', '.join(spec['nodes'])}",
        f".include {include}",
        SPICE_OPTIONS,
        "",
        f".param Lmin={fmt(LMIN)}",
        f".param Wmin={fmt(WMIN)}",
        f".param Ldiff={fmt(LDIFF)}",
        "",
        "* devices: junction areas AD = AS = W*Ldiff; NMOS bodies on vss, PMOS bodies on vdd",
    ]
    lines += [f"{n} {d} {g} {s} {b} {t} W={{{w}*Wmin}} L={{Lmin}} AD={{{w}*Wmin*Ldiff}} AS={{{w}*Wmin*Ldiff}}" for n, t, d, g, s, b, w in spec["devices"]]
    r0 = corners.iloc[0]
    lines += ["", "* supply, inputs and the ground rail (0 V source: its current closes the Kirchhoff balance)", f"Vdd vdd 0 {fmt(r0.vdd)}"]
    lines += [f"V{i} {i} 0 0" for i in ins] + ["Vss vss 0 0", "", ".control", f"set numdgt = {NUMDGT}"]
    for c in corners.itertuples():
        lines.append(f"* corner {c.corner}")
        lines += [f"altermod nmos {p}={fmt(getattr(c, 'n_' + p))}" for p in PROC_PARAMS]
        lines += [f"altermod pmos {p}={fmt(getattr(c, 'p_' + p))}" for p in PROC_PARAMS]
        lines += [f"set temp = {fmt(c.temp)}", f"alter vdd = {fmt(c.vdd)}"]
        for pat in PATTERNS[gate]:
            lines.append(f"echo @@ {c.corner} {pat}")
            lines += [f"alter v{i} = {fmt(int(bit) * c.vdd)}" for i, bit in zip(ins, pat)]
            lines += ["op", "print " + " ".join(printed(gate))]
    lines += [".endc", ".end", ""]
    return "\n".join(lines)

print(gate_netlist("nand2", nominal_corner(), '"../Model Files-20261105/models/16nm_HP.pm"'))

_LINE = re.compile(r"^(@@) (\d+) (\w+)$|^([iv])\((\w+)\) = (\S+)$")

def parse_gate(out, gate, corners):
    """One row per (corner, pattern) with the printed currents and voltages; blocks with missing vectors or wrong source voltages are dropped."""
    spec, want = GATES[gate], set(printed(gate))
    blocks, cur = {}, None
    for line in out.splitlines():
        m = _LINE.match(line.strip())
        if not m:
            continue
        if m.group(1):
            cur = (int(m.group(2)), m.group(3))
            blocks[cur] = {}
        elif cur is not None:
            blocks[cur][f"{m.group(4)}({m.group(5)})"] = float(m.group(6))
    by_corner = corners.set_index("corner")
    rows = []
    for (k, pat), d in blocks.items():
        if set(d) != want or k not in by_corner.index:
            continue
        c = by_corner.loc[k]
        if abs(d["v(vdd)"] - c.vdd) > 1e-9 or any(abs(d[f"v({i})"] - int(bit) * c.vdd) > 1e-9 for i, bit in zip(spec["inputs"], pat)):
            continue
        row = dict(gate=gate, pattern=pat, corner=k, vdd=c.vdd, temp=c.temp, a=int(pat[0]), b=int(pat[1]) if len(pat) > 1 else np.nan,
                   i_vdd=d["i(vdd)"], i_a=d["i(va)"], i_b=d.get("i(vb)", np.nan), i_vss=d["i(vss)"], v_out=d["v(out)"])
        row.update({f"v_{n}": d.get(f"v({n})", np.nan) for n in ("x", "y", "n")})
        rows.append(row)
    return pd.DataFrame(rows)

def draw_corners(n=N_CORNERS, seed=CORNER_SEED):
    rng = np.random.default_rng(seed)
    r = lambda a: np.array([float(f"{x:.10g}") for x in a])
    cols = dict(corner=np.arange(n), vdd=r(rng.uniform(VDD_NOM * (1 - VDD_VAR), VDD_NOM * (1 + VDD_VAR), n)), temp=r(rng.uniform(*TEMP_RANGE, n)))
    for dev, prefix in (("nmos", "n"), ("pmos", "p")):
        for p in PROC_PARAMS:
            cols[f"{prefix}_{p}"] = r(NOMINAL[dev][p] * rng.uniform(1 - PROC_VAR, 1 + PROC_VAR, n))
    return pd.DataFrame(cols)[CORNER_COLS]

corners = draw_corners()
DATA_DIR.mkdir(exist_ok=True)
corners.to_csv(DATA_DIR / "corners.csv", index=False, float_format="%.10g")
corners.head()

check = pd.read_csv(DATA_DIR / "corners.csv")
assert check.shape == (N_CORNERS, len(CORNER_COLS)) and check["corner"].is_unique
assert np.allclose(check.drop(columns="corner").to_numpy(), corners.drop(columns="corner").to_numpy(), rtol=1e-9, atol=0), "stored corners differ from the drawn ones"
ranges = {"vdd": (VDD_NOM * 0.9, VDD_NOM * 1.1), "temp": TEMP_RANGE}
ranges.update({f"{pre}_{p}": tuple(sorted((NOMINAL[dev][p] * 0.9, NOMINAL[dev][p] * 1.1))) for dev, pre in (("nmos", "n"), ("pmos", "p")) for p in PROC_PARAMS})   # sorted: the PMOS vth0 is negative
inside = {c: bool(corners[c].between(*ranges[c]).all()) for c in ranges}
print("all parameters inside their range:", all(inside.values()))
pn = np.corrcoef(corners[[f"n_{p}" for p in PROC_PARAMS]].to_numpy().T, corners[[f"p_{p}" for p in PROC_PARAMS]].to_numpy().T)[:5, 5:]
print("largest |correlation| between an NMOS and a PMOS parameter:", round(float(np.abs(pn).max()), 3), "(for 200 independent draws each correlation is about +/-0.07)")
corners.drop(columns="corner").describe().T[["min", "max", "mean"]].assign(nominal=[VDD_NOM, np.nan] + [NOMINAL["nmos"][p] for p in PROC_PARAMS] + [NOMINAL["pmos"][p] for p in PROC_PARAMS])

EXAMPLE_INCLUDE = f'"../{MODEL_FILE.parent.parent.name}/{MODEL_FILE.parent.name}/{MODEL_FILE.name}"'

def write_examples():
    NETLIST_DIR.mkdir(exist_ok=True)
    nom, rows = nominal_corner(), []
    for gate in GATES:
        path = NETLIST_DIR / f"gate_{gate}.cir"
        path.write_text(gate_netlist(gate, nom, EXAMPLE_INCLUDE), newline="\n")
        got = parse_gate(run_ngspice(path.name, NETLIST_DIR), gate, nom)
        if len(got) != len(PATTERNS[gate]):
            raise RuntimeError(f"{path.name}: simulation failed\n{run_ngspice(path.name, NETLIST_DIR, merge_stderr=True)[-600:]}")
        rows.append(got)
    return pd.concat(rows, ignore_index=True)

ex = write_examples()
ex["expected"] = [int(LOGIC[g](*[int(c) for c in p])) for g, p in zip(ex.gate, ex.pattern)]
ex["out_ok"] = (ex.v_out - ex.expected * ex.vdd).abs() < 0.05
ex["kirchhoff_A"] = ex[["i_vdd", "i_a", "i_b", "i_vss"]].sum(axis=1)
assert ex.out_ok.all(), "an example gate has the wrong logic value"
print(f"{len(ex)} cases simulated | all outputs have the right logic value | largest |sum of source currents| = {ex.kirchhoff_A.abs().max():.1e} A")
ex[["gate", "pattern", "v_out", "v_x", "v_y", "v_n", "i_vdd", "i_a", "i_b", "i_vss", "kirchhoff_A"]]

CHUNK = 25                                  # corners per ngspice process

def run_chunk(gate, sub, tag):
    name = f"gate_{gate}_{tag}.cir"
    (BUILD_DIR / name).write_text(gate_netlist(gate, sub, MODEL_FILE.name), newline="\n")
    return parse_gate(run_ngspice(name, BUILD_DIR), gate, sub)

def simulate_gates(corner_df, gates=None, workers=N_WORKERS):
    prepare_build()
    jobs = [(g, corner_df.iloc[s:s + CHUNK], s) for g in (gates or GATES) for s in range(0, len(corner_df), CHUNK)]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(lambda j: run_chunk(*j), jobs))
    return pd.concat(parts, ignore_index=True)

def add_leakage(df):
    df = df.copy()
    b_in, i_b = df["b"].fillna(0), df["i_b"].fillna(0)
    df["leak_supply"] = -df["i_vdd"]
    df["leak_total"] = -df["i_vdd"] - df["a"] * df["i_a"] - b_in * i_b
    df["kirchhoff_A"] = df["i_vdd"] + df["i_a"] + i_b + df["i_vss"]
    return df

t0 = time.time()
raw = simulate_gates(corners)
seconds = time.time() - t0
want = {(g, k, p) for g, p in CASES for k in corners["corner"]}
have = list(zip(raw["gate"], raw["corner"], raw["pattern"]))
assert len(have) == len(set(have)) and set(have) == want, f"missing: {sorted(want - set(have))[:5]}, duplicated: {len(have) - len(set(have))}"
order = {g: i for i, g in enumerate(GATES)}
gate_spice = add_leakage(raw.assign(_o=raw["gate"].map(order)).sort_values(["_o", "corner", "pattern"]).drop(columns="_o").reset_index(drop=True))
print(f"{len(gate_spice)} operating points in {seconds:.0f} s on {N_WORKERS} processes; none missing, none duplicated, none rejected by the parser")
gate_spice.head()

exp = np.array([int(LOGIC[g](*[int(c) for c in p])) for g, p in zip(gate_spice["gate"], gate_spice["pattern"])])
dev = (gate_spice["v_out"] - exp * gate_spice["vdd"]).abs()
print(f"1. logic: largest deviation of an output from its rail = {dev.max() * 1e3:.1f} mV (limit 50 mV) -> {'ok' if (dev < 0.05).all() else 'FAILED'}")
print(f"2. Kirchhoff: largest |sum of source currents| = {gate_spice['kirchhoff_A'].abs().max():.1e} A; "
      f"largest relative to leak_supply = {(gate_spice['kirchhoff_A'].abs() / gate_spice['leak_supply']).max():.1e}")
rng = np.random.default_rng(0)
worst = 0.0
for idx in rng.choice(len(gate_spice), size=12, replace=False):
    r = gate_spice.iloc[idx]
    solo = simulate_gates(corners[corners["corner"] == r["corner"]], gates=[r["gate"]])
    solo = solo[solo["pattern"] == r["pattern"]].iloc[0]
    for col in ("i_vdd", "i_a", "i_vss", "v_out"):
        worst = max(worst, abs(solo[col] - r[col]) / max(abs(r[col]), 1e-30))
print(f"3. a point simulated alone vs inside the batch (12 random points, 4 currents/voltages each): worst relative difference {worst:.1e}")
print(f"4. leak_supply > 0 in all points: {bool((gate_spice['leak_supply'] > 0).all())}")
assert (dev < 0.05).all() and (gate_spice["leak_supply"] > 0).all() and worst < 1e-6

gate_spice.to_csv(DATA_DIR / "gate_spice.csv", index=False, float_format="%.12g")
back = pd.read_csv(DATA_DIR / "gate_spice.csv", dtype={"pattern": str})
assert back["pattern"].equals(gate_spice["pattern"])
assert back.shape == gate_spice.shape and np.allclose(back["leak_total"], gate_spice["leak_total"], rtol=1e-9)
print(back.shape, "columns:", list(back.columns))

g = gate_spice.groupby(["gate", "pattern"], sort=False)["leak_total"]
leak_table = pd.DataFrame({"median": g.median(), "min": g.min(), "max": g.max()})
leak_table["max/min"] = leak_table["max"] / leak_table["min"]
leak_table.style.format({"median": "{:.3e}", "min": "{:.3e}", "max": "{:.3e}", "max/min": "{:.0f}"})

COLORS = {"0": "#2a78d6", "1": "#eb6834", "00": "#2a78d6", "01": "#eb6834", "10": "#1baf7a", "11": "#6250d6"}
MARKERS = {"0": "o", "1": "s", "00": "o", "01": "s", "10": "^", "11": "D"}
fig, ax = plt.subplots(1, 5, figsize=(17, 3.9), sharey=True)
for a, gate in zip(ax, GATES):
    for pat in PATTERNS[gate]:
        d = gate_spice[(gate_spice["gate"] == gate) & (gate_spice["pattern"] == pat)]
        a.plot(d["temp"], d["leak_total"], MARKERS[pat], ms=4, color=COLORS[pat], mec="white", mew=0.5, alpha=0.9, label=f"AB = {pat}" if len(pat) == 2 else f"A = {pat}")
    a.set_yscale("log"); a.set_xlabel("temperature [degC]")
    a.grid(color="0.92", lw=0.8); a.set_axisbelow(True); a.spines[["top", "right"]].set_visible(False)
    a.legend(title=gate.upper(), frameon=False, fontsize=8, title_fontsize=10, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(PATTERNS[gate]), handletextpad=0.1, columnspacing=0.8)
ax[0].set_ylabel("gate leakage [A]\n(200 corners)")
fig.tight_layout()
fig.savefig(RESULT_DIR / "gate_spice_leakage.png", dpi=110)
plt.close("all")

rel = (gate_spice["leak_total"] / gate_spice["leak_supply"] - 1).rename("rel")
diff = gate_spice.assign(rel=rel).groupby(["gate", "pattern"], sort=False)["rel"].agg(median="median", largest=lambda s: s.abs().max()) * 100
diff.round(3).rename(columns={"median": "median [%]", "largest": "largest |.| [%]"})
