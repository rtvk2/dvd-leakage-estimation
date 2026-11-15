"""01_gen_data.py - self-contained regeneration script.

Run from project root: python scripts/01_gen_data.py
Produces artifacts in data/, models/, netlists/, results/ with fixed seeds.
"""

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
pd.set_option("display.max_columns", 30)

# paths: the script lives in notebooks/; outputs go to the folders next to it
ROOT = Path(__file__).resolve().parent.parent
MODEL_FILE = ROOT / "Model Files-20261105" / "models" / "16nm_HP.pm"     # assigned model file
BUILD_DIR = ROOT / "build"          # scratch space for generated netlists (git-ignored)
NETLIST_DIR = ROOT / "netlists"
DATA_DIR = ROOT / "data"

# assignment and sweep ranges (project spec, section 4)
VDD_NOM = 0.9
LMIN = 16e-9                        # channel length = feature size of the node
WMIN = LMIN
W_RANGE = (1 * LMIN, 8 * LMIN)      # transistor width, uniform
LDIFF = 2 * LMIN                    # diffusion length for the junction areas AD = AS = W * LDIFF
VDD_VAR = 0.10                      # VDD uniform in nominal +/- 10 %
TEMP_RANGE = (-40.0, 125.0)         # degC, uniform
PROC_VAR = 0.10                     # process parameters uniform in nominal +/- 10 %
PROC_PARAMS = ("vth0", "toxe", "ndep", "u0", "xj")

N_SAMPLES = 10_000
SEED = 20261105
# gate and body voltage per case, as a fraction of the sampled VDD
CASES = {
    "nmos_on":  dict(device="nmos", vg=1.0, vb=0.0),
    "nmos_off": dict(device="nmos", vg=0.0, vb=0.0),
    "pmos_on":  dict(device="pmos", vg=0.0, vb=1.0),
    "pmos_off": dict(device="pmos", vg=1.0, vb=1.0),
}
CASE_SEEDS = {name: SEED + i for i, name in enumerate(CASES)}

SPICE_OPTIONS = ".options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9"   # required in every netlist
NUMDGT = 15                         # printed digits; the default 6 breaks the Kirchhoff check
N_WORKERS = 8                       # parallel ngspice processes
CHUNK = 250                         # samples per ngspice process
CURRENTS = ["id", "ig", "is", "ib"]
print(f"model {MODEL_FILE.name} | exists: {MODEL_FILE.exists()} | nominal VDD {VDD_NOM} V | L = Wmin = {LMIN*1e9:.0f} nm")

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
            # (?<![\w.]) keeps e.g. 'vth0' from matching inside 'lvth0' or 'wvth0'
            m = re.search(rf"(?<![\w.]){p}\s*=\s*({_NUM})", text, re.I)
            if m is None:
                raise ValueError(f"parameter {p} not found in card {name}")
            vals[p] = float(m.group(1))
        out[name] = vals
    return out

def nominals():
    return nominal_values(MODEL_FILE, PROC_PARAMS)

pd.DataFrame(nominals()).T

def to_wsl(path):
    """Convert a Windows path to its /mnt/<drive>/... form."""
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

def fmt(x):
    """Number formatting shared by the netlists and the CSV files (10 significant digits)."""
    return f"{float(x):.10g}"

def example_netlist(case, w=None, temp=27.0):
    spec = CASES[case]
    dev = spec["device"]
    w = WMIN if w is None else w
    vdd = VDD_NOM
    vd, vs = (vdd, 0.0) if dev == "nmos" else (0.0, vdd)
    vg, vb = spec["vg"] * vdd, spec["vb"] * vdd
    nom = nominals()[dev]
    inc = f"../{MODEL_FILE.parent.parent.name}/{MODEL_FILE.parent.name}/{MODEL_FILE.name}"
    name = "Mn" if dev == "nmos" else "Mp"
    lines = [
        f"* Single-transistor netlist: {case.replace('_', ' ').upper()} ({MODEL_FILE.stem}, nominal process)",
        f'.include "{inc}"',
        SPICE_OPTIONS,
        "",
        "* geometry and bias of this example (the datasets sample these and the process parameters at random)",
        f".param Lmin={fmt(LMIN)}",
        f".param Ldiff={fmt(LDIFF)}",
        f".param Wd={fmt(w)}",
        f".param vdd={fmt(vdd)}",
        "",
        "* device with junction areas AD = AS = W*Ldiff, one voltage source per terminal",
        f"{name} d g s b {dev} W={{Wd}} L={{Lmin}} AD={{Wd*Ldiff}} AS={{Wd*Ldiff}}",
        f"Vd d 0 {fmt(vd)}",
        f"Vg g 0 {fmt(vg)}",
        f"Vs s 0 {fmt(vs)}",
        f"Vb b 0 {fmt(vb)}",
        "",
        ".control",
        f"set numdgt = {NUMDGT}",
        f"set temp = {fmt(temp)}",
        "* one altermod per parameter (only the first parameter on a line is applied)",
    ]
    lines += [f"altermod {dev} {p}={fmt(nom[p])}" for p in PROC_PARAMS]
    lines += ["op", "print i(vd) i(vg) i(vs) i(vb)", ".endc", ".end", ""]
    return "\n".join(lines)

def batch_netlist(case, df):
    """Explicit script that simulates every row of df; each sample is announced with an `@@ k` marker."""
    dev = CASES[case]["device"]
    r0 = df.iloc[0]
    lines = [
        f"* single-FET batch: {case}",
        f".include {MODEL_FILE.name}",
        SPICE_OPTIONS,
        f"M1 d g s b {dev} W={fmt(r0.w)} L={fmt(LMIN)} AD={fmt(r0.w * LDIFF)} AS={fmt(r0.w * LDIFF)}",
        f"Vd d 0 {fmt(r0.vd)}",
        f"Vg g 0 {fmt(r0.vg)}",
        f"Vs s 0 {fmt(r0.vs)}",
        f"Vb b 0 {fmt(r0.vb)}",
        ".control",
        f"set numdgt = {NUMDGT}",
    ]
    for k, row in enumerate(df.itertuples(index=False)):
        lines.append(f"echo @@ {k}")
        lines += [f"altermod {dev} {p}={fmt(getattr(row, p))}" for p in PROC_PARAMS]
        lines += [
            f"set temp = {fmt(row.temp)}",
            f"alter vd = {fmt(row.vd)}",
            f"alter vg = {fmt(row.vg)}",
            f"alter vs = {fmt(row.vs)}",
            f"alter vb = {fmt(row.vb)}",
            f"alter @m1[w] = {fmt(row.w)}",
            f"alter @m1[ad] = {fmt(row.w * LDIFF)}",
            f"alter @m1[as] = {fmt(row.w * LDIFF)}",
            "op",
            "print i(vd) i(vg) i(vs) i(vb) v(d) v(g) v(s) v(b)",
        ]
    lines += [".endc", ".end", ""]
    return "\n".join(lines)

print(example_netlist("nmos_on"))

COLUMNS = ["vdd", "temp", "w", "vd", "vs", "vg", "vb", *PROC_PARAMS]

def _round(a):
    return np.array([float(f"{x:.10g}") for x in a])

def draw(case, n, rng):
    spec = CASES[case]
    nom = nominals()[spec["device"]]
    vdd = _round(rng.uniform(VDD_NOM * (1 - VDD_VAR), VDD_NOM * (1 + VDD_VAR), n))
    vd = _round(rng.uniform(0.0, 1.0, n) * vdd)
    vs = _round(rng.uniform(0.0, 1.0, n) * vdd)
    w = _round(rng.uniform(*W_RANGE, n))
    temp = _round(rng.uniform(*TEMP_RANGE, n))
    cols = dict(vdd=vdd, temp=temp, w=w, vd=vd, vs=vs, vg=_round(spec["vg"] * vdd), vb=_round(spec["vb"] * vdd))
    for p in PROC_PARAMS:
        cols[p] = _round(nom[p] * rng.uniform(1 - PROC_VAR, 1 + PROC_VAR, n))
    return pd.DataFrame(cols)[COLUMNS]

draw("nmos_on", 5, np.random.default_rng(CASE_SEEDS["nmos_on"]))

_LINE = re.compile(r"^(@@) (\d+)$|^([iv])\((\w+)\) = (\S+)$")
_BIAS = ["vd", "vg", "vs", "vb"]

def parse(out, df):
    """Terminal currents (into the device) per sample; NaN where the operating point failed."""
    n = len(df)
    res = np.full((n, 4), np.nan)
    seen, k = {}, None
    for line in out.splitlines():
        m = _LINE.match(line.strip())
        if not m:
            continue
        if m.group(1):
            k = int(m.group(2))
            seen[k] = {}
        elif k is not None:
            seen[k][f"{m.group(3)}({m.group(4)})"] = float(m.group(5))
    for k, d in seen.items():
        if k >= n or len(d) != 8:
            continue
        row = df.iloc[k]
        if all(abs(d[f"v({t})"] - row[v]) < 1e-9 for t, v in zip("dgsb", _BIAS)):
            res[k] = [-d[f"i({v})"] for v in _BIAS]       # current into the device = -I(source)
    return res

def run_chunk(case, df, tag):
    name = f"{case}_{tag}.cir"
    (BUILD_DIR / name).write_text(batch_netlist(case, df), newline="\n")
    return parse(run_ngspice(name, BUILD_DIR), df)

def simulate(case, df, workers=N_WORKERS):
    prepare_build()
    starts = list(range(0, len(df), CHUNK))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(lambda s: run_chunk(case, df.iloc[s:s + CHUNK], s), starts))
    return np.vstack(parts)

def generate(case, n=N_SAMPLES, workers=N_WORKERS):
    """n valid samples of a case with their currents; failed samples are redrawn."""
    rng = np.random.default_rng(CASE_SEEDS[case])
    df = draw(case, n, rng)
    cur = simulate(case, df, workers)
    bad = np.isnan(cur).any(axis=1)
    failed = []
    while bad.any():
        failed.append(df[bad].copy())
        new = draw(case, int(bad.sum()), rng)
        df.loc[bad, :] = new.to_numpy()
        cur[bad] = simulate(case, new, workers)
        bad = np.isnan(cur).any(axis=1)
    out = df.copy()
    out[CURRENTS] = cur
    return out, failed

def write_examples():
    NETLIST_DIR.mkdir(exist_ok=True)
    for case in CASES:
        path = NETLIST_DIR / f"single_{case}.cir"
        path.write_text(example_netlist(case), newline="\n")
        out = run_ngspice(path.name, NETLIST_DIR)
        if len(re.findall(r"^i\(v[dgsb]\) = ", out, re.M)) != 4:
            raise RuntimeError(f"{path.name}: simulation failed\n{run_ngspice(path.name, NETLIST_DIR, merge_stderr=True)[-600:]}")

def verify(case, df, n_check=5):
    """Re-simulate a few rows alone; returns the worst relative difference to the batch result."""
    rng = np.random.default_rng(0)
    worst = 0.0
    for k in rng.choice(len(df), size=min(n_check, len(df)), replace=False):
        single = run_chunk(case, df.iloc[[k]][COLUMNS], f"check{k}")[0]
        ref = df.iloc[k][CURRENTS].to_numpy(dtype=float)
        worst = max(worst, float(np.max(np.abs(single - ref) / np.maximum(np.abs(ref), 1e-30))))
    return worst

def generate_all(n=N_SAMPLES, out_dir=DATA_DIR, cases=None, check=5):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_examples()
    rows = []
    for case in cases or list(CASES):
        t0 = time.time()
        df, failed = generate(case, n)
        total = df[CURRENTS].sum(axis=1).abs()
        biggest = df[CURRENTS].abs().max(axis=1)
        diff = verify(case, df, check)
        df.to_csv(out_dir / f"{case}.csv", index=False, float_format="%.12g")
        if failed:
            pd.concat(failed).to_csv(BUILD_DIR / f"failed_{case}.csv", index=False)
        rows.append(dict(case=case, rows=len(df), seconds=round(time.time() - t0, 1),
                         redrawn_after_failed_op=sum(len(f) for f in failed),
                         kirchhoff_max_abs_sum_A=float(total.max()),
                         kirchhoff_max_rel=float((total / biggest)[biggest > 1e-12].max()),
                         single_vs_batch_max_rel_diff=diff))
    return pd.DataFrame(rows)

summary = generate_all(N_SAMPLES, DATA_DIR)
summary

data = {case: pd.read_csv(DATA_DIR / f"{case}.csv") for case in CASES}
for case, d in data.items():
    print(case, d.shape)
data["nmos_on"].head()

d = data["nmos_on"]
fig, ax = plt.subplots(2, 5, figsize=(16, 5.5))
panels = [(d.vdd, "VDD [V]"), (d.temp, "T [degC]"), (d.w * 1e9, "W [nm]"), (d.vd / d.vdd, "Vd / VDD"), (d.vs / d.vdd, "Vs / VDD"),
          (d.vth0, "vth0 [V]"), (d.toxe * 1e9, "toxe [nm]"), (d.ndep, "ndep [1/cm3]"), (d.u0, "u0 [m2/Vs]"), (d.xj * 1e9, "xj [nm]")]
for a, (x, label) in zip(ax.ravel(), panels):
    a.hist(x, bins=30, color="#4C78A8")
    a.set_xlabel(label)
fig.suptitle("Sampled inputs, NMOS ON (10,000 samples)")
fig.tight_layout()
plt.close("all")

fig, ax = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
for a, (case, d) in zip(ax, data.items()):
    for col, color in zip(CURRENTS, ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]):
        mag = d[col].abs()
        a.hist(np.log10(mag[mag > 0]), bins=60, histtype="step", color=color, label=col)
    a.set_title(case)
    a.set_xlabel("log10 |I| [A]")
ax[0].set_ylabel("samples")
ax[0].legend()
fig.tight_layout()
plt.close("all")
pd.DataFrame({case: {col: f"{(d[col] > 0).mean():.2f} positive" for col in CURRENTS} for case, d in data.items()})

N_EXTRA = 6000
EXTRA_MARGIN = 0.03                                  # fraction of VDD
EXTRA_CASES = ["nmos_off", "pmos_off"]
EXTRA_SEEDS = {c: SEED + 100 + i for i, c in enumerate(EXTRA_CASES)}

def draw_rails(case, n, rng, margin=EXTRA_MARGIN):
    spec = CASES[case]
    nom = nominals()[spec["device"]]
    vdd = _round(rng.uniform(VDD_NOM * (1 - VDD_VAR), VDD_NOM * (1 + VDD_VAR), n))

    def volt():
        kind = rng.integers(0, 3, n)                 # 0: uniform, 1: near 0 V, 2: near VDD
        frac = np.where(kind == 0, rng.uniform(0, 1, n), np.where(kind == 1, rng.uniform(0, margin, n), 1 - rng.uniform(0, margin, n)))
        return _round(frac * vdd)

    vd, vs = volt(), volt()
    cols = dict(vdd=vdd, temp=_round(rng.uniform(*TEMP_RANGE, n)), w=_round(rng.uniform(*W_RANGE, n)), vd=vd, vs=vs, vg=_round(spec["vg"] * vdd), vb=_round(spec["vb"] * vdd))
    for p in PROC_PARAMS:
        cols[p] = _round(nom[p] * rng.uniform(1 - PROC_VAR, 1 + PROC_VAR, n))
    return pd.DataFrame(cols)[COLUMNS]

def generate_extra(case, n=N_EXTRA):
    rng = np.random.default_rng(EXTRA_SEEDS[case])
    df = draw_rails(case, n, rng)
    cur = simulate(case, df)
    bad = np.isnan(cur).any(axis=1)
    redrawn = 0
    while bad.any():                                  # failed operating points are redrawn from the same stream
        redrawn += int(bad.sum())
        df.loc[bad, :] = draw_rails(case, int(bad.sum()), rng).to_numpy()
        cur[bad] = simulate(case, df[bad])
        bad = np.isnan(cur).any(axis=1)
    out = df.copy()
    out[CURRENTS] = cur
    return out, redrawn

rows = []
for case in EXTRA_CASES:
    t0 = time.time()
    ex, redrawn = generate_extra(case)
    ex.to_csv(DATA_DIR / f"extra_{case}.csv", index=False, float_format="%.12g")
    total = ex[CURRENTS].sum(axis=1).abs()
    near = lambda s: ((s / ex["vdd"] < EXTRA_MARGIN) | (s / ex["vdd"] > 1 - EXTRA_MARGIN))
    rows.append(dict(case=case, rows=len(ex), seconds=round(time.time() - t0, 1), redrawn_after_failed_op=redrawn, kirchhoff_max_abs_sum_A=float(total.max()),
                     share_vd_within_3pct_of_a_rail=float(near(ex["vd"]).mean()), share_vs_within_3pct_of_a_rail=float(near(ex["vs"]).mean()),
                     share_both_at_rails=float((near(ex["vd"]) & near(ex["vs"])).mean())))
pd.DataFrame(rows)

base = {c: pd.read_csv(DATA_DIR / f"{c}.csv") for c in EXTRA_CASES}
share = lambda d: float(((d["vd"] / d["vdd"] < EXTRA_MARGIN) | (d["vd"] / d["vdd"] > 1 - EXTRA_MARGIN)).mean())
print("share of Vd within 3 % of VDD of a rail: main datasets", {c: round(share(d), 3) for c, d in base.items()},
      "| extra samples", {c: round(share(pd.read_csv(DATA_DIR / f'extra_{c}.csv')), 3) for c in EXTRA_CASES})
