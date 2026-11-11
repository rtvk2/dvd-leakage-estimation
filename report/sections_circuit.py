# Circuit results, error analysis, limitations and appendix of the report (executed after sections_results.py by build_report.py).
# Uses the variables gm, g95 (gate-level median and 95th percentile error), gr (gate results table) of sections_results.py.

# ---------------------------------------------------------------------------------------------------------------------
H("5.1 Test-circuit results (deliverable 8)", 2)
cr = pd.read_csv(RES / "circuit_results.csv", index_col=0)
T([[idx] + [("" if pd.isna(v) else (sci(v) if "[A]" in idx else f"{v:.2f}")) for v in r.tolist()] for idx, r in cr.iterrows()],
  ["metric"] + list(cr.columns), widths=[1.6, 1.8, 1.8, 1.8], size=8, caption="Table 12. Test circuit, all 25,600 operating points (128 input vectors x 200 corners)")
cc = pd.read_csv(DATA / "circuit_ml.csv")
ce = cc["err"].abs() * 100
hm, fl, gm_ = cr["hierarchical ML vs SPICE"], cr["composition floor (perfect gate models)"], cr["ML gate models vs isolated-gate SPICE sum"]
cv = pd.read_csv(RES / "circuit_results_by_vector.csv", index_col=0)
P(f"The hierarchical prediction has a median error of **{hm['median error [%]']:.2f} %**, a 95th percentile of {hm['95th pct error [%]']:.2f} % and a mean error of {hm['mean error [%]']:.2f} % (bias {hm['bias [%]']:+.2f} %, largest {ce.max():.1f} %); "
  f"{(ce < 1).mean() * 100:.0f} % of the 25,600 points are within 1 %, {(ce < 5).mean() * 100:.0f} % within 5 % and {(ce < 10).mean() * 100:.1f} % within 10 %. "
  f"Almost all of this is the composition itself: **with perfect gate models the same method would give a median error of {fl['median error [%]']:.2f} % and a 95th percentile of {fl['95th pct error [%]']:.2f} %** (bias {fl['bias [%]']:+.2f} %: the sum of the isolated gates underestimates the circuit). "
  f"The ML gate models add little to it: against the sum of the isolated-gate SPICE leakages they have a median error of {gm_['median error [%]']:.2f} % and a 95th percentile of {gm_['95th pct error [%]']:.2f} % (bias {gm_['bias [%]']:+.2f} %). "
  f"The circuit error is therefore larger than the gate error ({gm:.2f} % median, {g95:.2f} % 95th percentile) because of what the hierarchical method leaves out, not because of the transistor models.")
F("circuit_parity.png", "Figure 6. Left: ML against SPICE leakage of the complete circuit (log axes); right: median signed error against temperature.", 6.3)
P(f"The results for each of the 128 input vectors (median, 95th-percentile and mean error and bias over the 200 corners) are in Appendix A and in `results/circuit_results_by_vector.csv`. "
  f"The vector changes the error very little (section 6.5): the median error per vector lies between {cv['median error [%]'].min():.2f} % and {cv['median error [%]'].max():.2f} %.")

# ---------------------------------------------------------------------------------------------------------------------
H("6. Error analysis")
P("Notebook 07 analyses where the errors occur, at gate level and for the test circuit: by gate and input pattern, by corner condition (temperature, VDD, `vth0`, `toxe` of NMOS and PMOS), "
  "by transistor model and leakage component, by the voltage of the gate inputs, and by input vector. It reads the results of the earlier notebooks and needs no simulator.")
H("6.1 Gate level: gate and input pattern", 2)
gcase = pd.read_csv(RES / "gate_error_by_case.csv", dtype={"pattern": str})
top3 = gcase.head(3)
P("The ranking by input pattern is in Table 10 and Figure 5. " + "; ".join(f"{GATE[r['gate']]} {r['pattern']}: median {r['median [%]']:.2f} %, 95th percentile {r['95th pct [%]']:.2f} %, bias {r['bias [%]']:+.2f} %" for _, r in top3.iterrows()) +
  f" are the three cases with the largest median error, and the least accurate case still has a median error of only {gcase['median [%]'].max():.2f} %. There is no systematic difference between the combinations in which the leakage is large and those in which it is small "
  f"(the cases span {gr['SPICE median [A]'].min():.1e} A to {gr['SPICE median [A]'].max():.1e} A in median leakage); the two largest errors belong to the two gate types that contain an internal inverter driven by a node that is not exactly at a rail (section 6.4).")
H("6.2 Gate level: corner conditions", 2)
gc = pd.read_csv(RES / "gate_error_by_corner.csv")


def bins(f):
    return gc[gc.factor == f].reset_index(drop=True)


tb, vb = bins("temperature [degC]"), bins("VDD [V]")
oth = tb.iloc[:-1]
proc = pd.concat([bins(f) for f in ("NMOS vth0 [V]", "PMOS |vth0| [V]", "NMOS toxe [m]", "PMOS toxe [m]")])
F("gate_error_factors.png", "Figure 7. Gate-level error against corner conditions (median, 95th percentile and bias; terciles for VDD and the process parameters).", 6.3)
P(f"The error depends little on the corner. It is largest at the hot end: in the bin {tb.iloc[-1]['bin']} degC the median error is {tb.iloc[-1]['median [%]']:.2f} % and the 95th percentile {tb.iloc[-1]['95th pct [%]']:.2f} %, "
  f"against a median of {oth['median [%]'].min():.2f} % to {oth['median [%]'].max():.2f} % in the other temperature bins, and in the highest VDD tercile ({vb.iloc[-1]['bin']} V: median {vb.iloc[-1]['median [%]']:.2f} %, 95th percentile {vb.iloc[-1]['95th pct [%]']:.2f} %; "
  f"the lower terciles {vb.iloc[0]['median [%]']:.2f} % and {vb.iloc[1]['median [%]']:.2f} %). The temperature dependence is explained in section 6.4 (the gate inputs of the inverter inside AND2 and OR2 move away from their rails as the temperature rises); "
  "no single cause was isolated for the VDD dependence. "
  f"For `vth0` and `toxe` of the NMOS and the PMOS there is no consistent trend (median error {proc['median [%]'].min():.2f} % to {proc['median [%]'].max():.2f} % over all terciles); a low threshold voltage, which raises the leakage, does not make the gate error larger. "
  f"The bias stays between {gc['bias [%]'].min():+.2f} % and {gc['bias [%]'].max():+.2f} % in every bin.")
H("6.3 Gate level: transistor model and leakage component", 2)
bud = pd.read_csv(RES / "gate_error_budget.csv", header=[0, 1], index_col=0)
shr = pd.read_csv(RES / "gate_leakage_share.csv", index_col=0)
ba, bb = bud.iloc[:, :4].copy(), bud.iloc[:, 4:].copy()
ba.columns, bb.columns = [c[1] for c in ba.columns], [c[1] for c in bb.columns]
T([[CASE[m], pct(shr.loc[m, "total"], 1), pct(ba.loc[m, "total"]), pct(ba.loc[m, "channel"]), pct(ba.loc[m, "gate"] + ba.loc[m, "body"], 3), f"{bb.loc[m, 'total']:+.2f}"] for m in ["nmos_on", "nmos_off", "pmos_on", "pmos_off"]],
  ["transistor model", "share of the leakage [%]", "abs. error budget [%]", "of which channel term [%]", "of which gate + body terms [%]", "bias [%]"], widths=[1.3, 1.1, 1.1, 1.2, 1.3, 0.8], size=8,
  caption="Table 13. Error budget of the gate leakage by transistor model (all 3,600 operating points; in % of the summed SPICE leakage)")
tot = ba["total"].sum()
P("With the same voltages, the ML and SPICE currents of every transistor give two values of each term of the leakage sum, so the error of the gate leakage can be traced back to a model (Table 13). "
  f"The PMOS-OFF channel term alone is {ba.loc['pmos_off', 'channel'] / tot * 100:.0f} % of the absolute error budget ({ba.loc['pmos_off', 'channel']:.2f} of {tot:.2f} percentage points) and the NMOS-OFF channel term {ba.loc['nmos_off', 'channel'] / tot * 100:.0f} %; "
  f"the OFF transistors carry {shr.loc['pmos_off', 'total'] + shr.loc['nmos_off', 'total']:.0f} % of the leakage of the gates (PMOS OFF {shr.loc['pmos_off', 'total']:.0f} %, NMOS OFF {shr.loc['nmos_off', 'total']:.0f} %), the ON transistors about {shr.loc['nmos_on', 'total'] + shr.loc['pmos_on', 'total']:.0f} %. "
  f"The gate-current and body-current terms contribute less than {max(ba['gate'].max(), ba['body'].max()) + 0.005:.2f} percentage points: the errors of the small currents `ig` and `ib`, which are among the hardest at transistor level (section 3.4), do not reach the gate leakage because these currents are tiny against the sub-threshold current. "
  f"The summed bias of all terms ({bb['total'].sum():+.2f} %) is the bias of the gate leakage; it comes mainly from the NMOS-OFF channel term. This is why the refinement of the two OFF models (section 3.5) had such an effect at gate level, and why the ON models, although the most accurate at transistor level, hardly matter here.")
H("6.4 Gate level: the worst points and gate inputs that are not at a rail", 2)
wpa = pd.read_csv(RES / "gate_worst_points.csv", dtype={"pattern": str})
wp = wpa.head(5)
T([[GATE[r["gate"]], r["pattern"], f"{r['temp']:.0f}", f"{r['vdd']:.3f}", sci(r["leak_total"]), sci(r["ml_leak"]), f"{r['err'] * 100:+.1f}", r["largest term error"]] for _, r in wp.iterrows()],
  ["gate", "input", "T [degC]", "VDD [V]", "SPICE [A]", "ML [A]", "error [%]", "term with the largest error"], widths=[0.6, 0.5, 0.7, 0.7, 0.9, 0.9, 0.8, 1.9], size=8,
  caption="Table 14. The five worst gate operating points (the full list of ten is in `results/gate_worst_points.csv`)")
nr = pd.read_csv(RES / "gate_error_nonrail_input.csv", dtype={"pattern": str})
sel = nr[nr["Spearman(distance, error)"] < -0.5]
T([[GATE[r["gate"]], r["pattern"], r["rail"], pct(r["median distance [mV]"]), pct(r["max distance [mV]"], 1), f"{r['Spearman(distance, error)']:+.2f}", f"{r['median signed error [%]']:+.2f}", pct(r["median |error| [%]"])] for _, r in nr.iterrows()],
  ["gate", "input", "inverter input should be at", "median distance [mV]", "largest distance [mV]", "Spearman (distance, error)", "median signed error [%]", "median |error| [%]"], widths=[0.55, 0.5, 0.95, 0.9, 0.9, 0.95, 0.9, 0.8], size=8,
  caption="Table 15. AND2 and OR2: distance of the inverter input from its rail (SPICE) against the error of the gate leakage")
_a = ge[(ge["gate"] == "and2") & (ge["pattern"] == "01")]
and01_cold, and01_hot = _a[_a["temp"] < 40]["err"].abs().median() * 100, _a[_a["temp"] > 90]["err"].abs().median() * 100
expl = wpa[((wpa["gate"] == "and2") & (wpa["pattern"] == "01")) | ((wpa["gate"] == "or2") & (wpa["pattern"] == "00"))]
rest = wpa.drop(expl.index)
hot = int((wpa["temp"] > 100).sum())
P(f"The ten worst points have errors between {wpa['err'].abs().min() * 100:.1f} % and {wpa['err'].abs().max() * 100:.1f} %, all of them underestimates; {hot} of them are in corners with T > 100 degC. "
  f"{len(expl)} of the ten (AND2 01 and OR2 00) have the PMOS-OFF channel term of the inverter as the term with the largest error and are explained below. "
  "The other " + str(len(rest)) + " (" + ", ".join(f"{GATE[r['gate']]} {r['pattern']} at {r['temp']:.0f} degC" for _, r in rest.iterrows()) + f") are errors of the NMOS-OFF channel term of {rest['err'].abs().min() * 100:.1f} % to {rest['err'].abs().max() * 100:.1f} %; their cause was not investigated further. "
  "For the first group the explanation is a model assumption. The ON and OFF models were trained with the gate voltage at exactly 0 or VDD, and the gate voltage of every transistor is set to the nearest rail when the models are called (as in the instructor's AND2 equation). "
  "In AND2 and OR2 the inverter is driven by the NAND2 / NOR2 output, which in SPICE sits a few millivolts away from its rail because of the leakage through the ON network. "
  "A PMOS-OFF transistor whose gate is a few millivolts below the source leaks more, exponentially, than the model for gate = VDD says. Table 15 supports this: for " + " and ".join(f"{GATE[r['gate']]} {r['pattern']}" for _, r in sel.iterrows()) +
  f" (inverter input at VDD, so that the PMOS of the inverter is the OFF transistor that carries the leakage) the distance of the input from the rail reaches {sel['max distance [mV]'].max():.0f} mV, it correlates strongly with the error (Spearman {sel['Spearman(distance, error)'].min():.2f} and {sel['Spearman(distance, error)'].max():.2f}), "
  f"and the model underestimates the leakage. The distance grows with temperature (Spearman {nr['Spearman(distance, temperature)'].min():.2f} to {nr['Spearman(distance, temperature)'].max():.2f} over the eight cases), because the leakage through the ON network grows, "
  f"and so does the error of these two cases: the median error of AND2 01 is {and01_cold:.2f} % below 40 degC and {and01_hot:.2f} % above 90 degC. The remedy would be a gate-voltage margin in the training samples (the gate voltage within a few tens of millivolts of the rail), which the project text does not require.")
H("6.5 Test circuit", 2)
ce_ = pd.read_csv(RES / "circuit_error_by_corner.csv")
T([[r["factor"], r["bin"], pct(r["ML: median |err| [%]"]), pct(r["ML: 95th pct [%]"]), f"{r['ML: bias [%]']:+.2f}", pct(r["floor: median |err| [%]"]), f"{r['floor: bias [%]']:+.2f}", pct(r["gate models: median |err| [%]"])] for _, r in ce_.iterrows()],
  ["factor", "bin", "ML: median [%]", "ML: 95th pct [%]", "ML: bias [%]", "floor: median [%]", "floor: bias [%]", "gate models: median [%]"], widths=[1.2, 1.1, 0.8, 0.8, 0.8, 0.8, 0.8, 0.9], size=8,
  caption="Table 16. Test circuit: error of the hierarchical prediction (ML), of the composition floor and of the ML gate models by corner condition (200 corners, 128 vectors)")
ct_ = ce_[ce_.factor == "temperature [degC]"].reset_index(drop=True)
cv_ = ce_[ce_.factor == "VDD [V]"].reset_index(drop=True)
cn_ = ce_[ce_.factor == "NMOS vth0 [V]"].reset_index(drop=True)
byv = pd.read_csv(RES / "circuit_error_by_vector.csv").sort_values("median |err| [%]", ascending=False).reset_index(drop=True)
P(f"**The circuit-level error is a temperature effect of the composition.** The median error of the hierarchical prediction rises from {ct_.iloc[0]['ML: median |err| [%]']:.2f} % at the cold end ({ct_.iloc[0]['bin']} degC) to {ct_.iloc[-1]['ML: median |err| [%]']:.2f} % at the hot end ({ct_.iloc[-1]['bin']} degC; 95th percentile {ct_.iloc[-1]['ML: 95th pct [%]']:.1f} %) "
  f"and its bias from {ct_.iloc[0]['ML: bias [%]']:+.2f} % to {ct_.iloc[-1]['ML: bias [%]']:+.2f} %. The composition floor follows it ({ct_.iloc[0]['floor: median |err| [%]']:.2f} % to {ct_.iloc[-1]['floor: median |err| [%]']:.2f} %, bias {ct_.iloc[0]['floor: bias [%]']:+.2f} % to {ct_.iloc[-1]['floor: bias [%]']:+.2f} %), "
  f"whereas the median error of the ML gate models stays between {ct_['gate models: median |err| [%]'].min():.2f} % and {ct_['gate models: median |err| [%]'].max():.2f} % in every temperature bin. "
  "By construction the floor contains only what is removed by isolating a gate: the real gate inputs are not exactly at 0 or VDD, and the real outputs are loaded by the inputs of the next gates (their gate-leakage currents). Both are expected to grow with the leakage current, which grows with temperature (for the inverter input of AND2 and OR2 this was measured in section 6.4). "
  f"The error is also larger at high VDD (median {cv_.iloc[-1]['ML: median |err| [%]']:.2f} % in the top tercile against {cv_.iloc[0]['ML: median |err| [%]']:.2f} % in the bottom one) and at a low NMOS threshold ({cn_.iloc[0]['ML: median |err| [%]']:.2f} % in the lowest tercile against {cn_.iloc[-1]['ML: median |err| [%]']:.2f} % in the highest), "
  "which are again the corners with the highest leakage. "
  f"By input vector the error varies little: the median error per vector is between {byv['median |err| [%]'].min():.2f} % and {byv['median |err| [%]'].max():.2f} %, "
  f"and the vector with the largest error is a = {int(byv.iloc[0]['a'])}, b = {int(byv.iloc[0]['b'])}, cin = {int(byv.iloc[0]['cin'])} ({byv.iloc[0]['median |err| [%]']:.2f} %); no single input pattern is responsible for the error.")

# ---------------------------------------------------------------------------------------------------------------------
H("7. Limitations and improvements")
B([f"**Hierarchical composition.** The circuit prediction sums isolated-gate leakages with ideal 0 / VDD inputs and no loading, as prescribed. This limits the accuracy of any gate model: the composition floor is {fl['median error [%]']:.2f} % (median) and {fl['95th pct error [%]']:.2f} % (95th percentile), "
   f"and at the hot end ({ct_.iloc[-1]['bin']} degC) {ct_.iloc[-1]['floor: median |err| [%]']:.2f} % (median) with a bias of {ct_.iloc[-1]['floor: bias [%]']:+.2f} %, so a 99 % accuracy at the circuit level is not reachable at hot corners with this composition, whatever the transistor models are. "
   "Propagating the real output voltages of each gate to the inputs of the next gates (an estimate of the node voltages, outside the prescribed method) would remove most of this error.",
   "**Gate inputs that are not at a rail.** The INV inside AND2 and OR2 is driven by an internal node that sits up to about 30 mV from its rail, while the models were trained at exact rails; the leakage of an OFF transistor depends exponentially on such a deviation (section 6.4). "
   "Training samples with the gate voltage within a few tens of millivolts of the rails, or the gate voltage as a free feature near the rails, would address it.",
   "**Training distribution against operating region.** The transistors of a gate operate at the corners of the (Vd, Vs) square at fixed gate and body voltages, whereas the datasets are uniform. The refinement of the two OFF models with 6,000 rail samples each (section 3.5) lowered the gate-level median error from "
   f"{gv.median_error_pct.iloc[1]:.2f} % to {gv.median_error_pct.iloc[2]:.2f} % and the 95th percentile from {gv.p95_error_pct.iloc[1]:.2f} % to {gv.p95_error_pct.iloc[2]:.2f} %, at the price of a slightly worse score on the uniform test samples; "
   "a dataset designed from the operating region of the circuits, or an active-learning loop on the gate-level error, would go further. The refinement was motivated by the gate-level analysis (disclosed in section 3.5); it uses no gate-level data.",
   f"**Transistor accuracy.** The PMOS-OFF model has a median test error of {float(ct.loc['pmos_off', 'test_median']):.2f} % and a 95th percentile of {float(ct.loc['pmos_off', 'test_p95']):.1f} % (the project notes name an error below 1 % as a goal); its small currents `ig` and `is` are the hardest to learn, partly because the simulator resolves currents only to 4.4e-16 A, "
   "so that relative errors of currents below about 1e-14 A are limited by the data. At |Vds| < 5 mV the PMOS-OFF channel currents have a median error of "
   + f"{pt[(pt.case == 'pmos_off') & pt.terminal.isin(['id', 'is'])].median_rel_small_vds.min() * 100:.0f} % to {pt[(pt.case == 'pmos_off') & pt.terminal.isin(['id', 'is'])].median_rel_small_vds.max() * 100:.0f} % "
   + f"(on {int(pt[(pt.case == 'pmos_off') & pt.terminal.isin(['id', 'is'])].n_small_vds.min())} to {int(pt[(pt.case == 'pmos_off') & pt.terminal.isin(['id', 'is'])].n_small_vds.max())} test samples); this does not reach the gate level because the channel term of such a transistor is weighted by its own Vds.",
   "**Definition of the leakage.** The project text does not define the gate leakage; the supply-plus-input-power definition and the component sum are used and compared (section 4.2). The supply current alone differs by more than 5 % only in NOR2 with inputs 11.",
   "**Statistics.** One data split, one seed per model and 200 corners: the numbers carry sampling noise, and no confidence intervals are given. An ensemble of seeds would reduce the variance of the OFF models.",
   "**Open points not answered by the course staff at the time of writing:** whether the adder has a carry-in (7 inputs and 128 vectors are used; the 64 vectors without carry-in are a subset), whether notebooks are accepted as the code deliverable, and the exact definition of \"mean accuracy\" (median, 95th percentile and mean relative errors are all reported)."])

# ---------------------------------------------------------------------------------------------------------------------
H("Appendix A. Test-circuit results per input vector")
T([[idx, int(r["a"]), int(r["b"]), int(r["cin"]), sci(r["SPICE median [A]"]), sci(r["ML median [A]"]), f"{r['bias [%]']:+.2f}", pct(r["median error [%]"]), pct(r["95th pct error [%]"]), pct(r["mean error [%]"])] for idx, r in cv.iterrows()],
  ["vector", "a", "b", "cin", "SPICE median [A]", "ML median [A]", "bias [%]", "median error [%]", "95th pct [%]", "mean error [%]"], widths=[0.55, 0.35, 0.35, 0.4, 0.95, 0.95, 0.65, 0.75, 0.7, 0.7], size=7,
  caption="Table A1. Hierarchical prediction against the full-circuit SPICE simulation for each input vector (vector = a + 8 b + 64 cin; 200 corners per vector)")
