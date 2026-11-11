# Results sections of the report (executed after sections.py by build_report.py). Every number is read from a file in results/ or data/.

# ---------------------------------------------------------------------------------------------------------------------
H("4.4 Gate results (deliverables 6 and 7)", 2)
gr = pd.read_csv(RES / "gate_results.csv", dtype={"pattern": str})
T([[GATE[r[0]], r[1], sci(r[2]), sci(r[3]), f"{r[4]:+.2f}", pct(r[5]), pct(r[6]), pct(r[7])] for r in gr.itertuples(index=False)],
  ["gate", "input", "SPICE median [A]", "ML median [A]", "bias [%]", "median error [%]", "95th pct error [%]", "mean error [%]"], widths=[0.7, 0.6, 1.1, 1.1, 0.8, 0.9, 1.0, 0.9], size=8,
  caption="Table 10. Gate results, 200 corners per case (bias = mean signed relative error (ML - SPICE) / SPICE)")
ge = pd.read_csv(DATA / "gate_ml.csv", dtype={"pattern": str})
gm, g95, gmean, gbias = ge["err"].abs().median() * 100, ge["err"].abs().quantile(0.95) * 100, ge["err"].abs().mean() * 100, ge["err"].mean() * 100
gv = pd.read_csv(RES / "gate_summary_versions.csv")
gk = ge["err_kcl"].abs()
mcol, pcol, ecol = "median error [%]", "95th pct error [%]", "mean error [%]"
best, worst2 = gr.sort_values(mcol).iloc[0], gr.sort_values(mcol, ascending=False).iloc[:2]
byg = pd.read_csv(RES / "gate_error_by_gate.csv", index_col=0)
P(f"Over all 3,600 operating points the median error is **{gm:.2f} %**, the 95th percentile {g95:.2f} % and the mean error {gmean:.2f} % (bias {gbias:+.2f} %, largest single error {ge['err'].abs().max() * 100:.1f} %); "
  f"the mean accuracy, read as 100 % minus the mean error, is {100 - gmean:.2f} %. Every one of the 18 cases has a median error below {np.ceil(gr[mcol].max() * 10) / 10:.1f} % and a 95th percentile below {np.ceil(gr[pcol].max() * 10) / 10:.1f} %. "
  f"The most accurate case is {GATE[best['gate']]} {best['pattern']} (median {best[mcol]:.2f} %), the least accurate are " + " and ".join(f"{GATE[r['gate']]} {r['pattern']} ({r[mcol]:.2f} %, 95th percentile {r[pcol]:.2f} %)" for _, r in worst2.iterrows()) +
  f". By gate type the median error lies between {byg['median [%]'].min():.2f} % ({GATE[byg['median [%]'].idxmin()]}) and {byg['median [%]'].max():.2f} % ({GATE[byg['median [%]'].idxmax()]}); the two gate types with an internal inverter (AND2, OR2) have the largest 95th percentiles "
  f"({byg.loc['and2', '95th pct [%]']:.2f} % and {byg.loc['or2', '95th pct [%]']:.2f} %; section 6). The Kirchhoff composition (current balance at the nodes tied to VDD) is a second, independent estimate from the same transistor currents: median {gk.median() * 100:.2f} %, 95th percentile {gk.quantile(0.95) * 100:.2f} %.")
F("gate_parity.png", "Figure 5. ML against SPICE gate leakage, 18 cases x 200 corners (log axes; identity dashed, +/- 10 % dotted).", 4.6)
T([[r.version, pct(r.median_error_pct), pct(r.p95_error_pct), pct(r.mean_error_pct), f"{r.bias_pct:+.2f}", pct(r.max_error_pct, 1)] for r in gv.itertuples()],
  ["version", "median [%]", "95th pct [%]", "mean [%]", "bias [%]", "largest [%]"], widths=[3.2, 0.8, 0.9, 0.8, 0.7, 0.8], size=8,
  caption="Table 11. Gate-level error (3,600 operating points) for the sizing choice and for the refinement of the OFF models (`results/gate_summary_versions.csv`)")
P("Table 11 shows the two decisions taken with the gate-level analysis. The drive-strength multiplier XX = 2 (instead of 1) keeps the NMOS widths of the gates inside the training range instead of exactly on its lower edge. "
  "The refinement of the two OFF models with rail samples (section 3.5) lowers the median error by a factor " + f"{gv.median_error_pct.iloc[1] / gv.median_error_pct.iloc[2]:.1f}" + " and the 95th percentile by a factor " + f"{gv.p95_error_pct.iloc[1] / gv.p95_error_pct.iloc[2]:.1f}" +
  ", although the same models are slightly worse on the uniform test samples of section 3.4. The reason is the one given in 3.5: the operating region of a gate is a small part of the (Vd, Vs) square, and this is where the uniform datasets are sparse. "
  "The same models were used for the test circuit.")

# ---------------------------------------------------------------------------------------------------------------------
H("5. Test circuit: 3-bit carry-lookahead adder (Part C)")
P("**Circuit.** A 3-bit carry-lookahead adder with carry-in, built only from the five gate types: generate `g_i = a_i b_i`, propagate `p_i = a_i xor b_i`, "
  "`c1 = g0 + p0 c0`, `c2 = g1 + p1 g0 + p1 p0 c0`, `c3 = g2 + p2 g1 + p2 p1 g0 + p2 p1 p0 c0`, `s_i = p_i xor c_i`. Every XOR is four NAND2 gates, the products are AND2 gates (shared sub-products are reused) and the sums are OR2 gates. "
  "The circuit has **40 gates (24 NAND2, 10 AND2, 6 OR2; 192 transistors)**, seven inputs (`a0..a2, b0..b2, c0`, hence 128 input vectors) and four outputs (`s0..s2` and the carry-out `c3`). "
  "A Python logic simulator of the same netlist is checked against binary addition for all 128 vectors, and the SPICE outputs are checked against it at every operating point. "
  "Every gate is a subcircuit with exactly the transistors and sizes of Part B, the process parameters of a corner are set on the shared model cards (all NMOS of the circuit share one draw, all PMOS another). Example netlist: `netlists/test_circuit_cla3.cir`.")
P("**SPICE reference.** The complete transistor-level netlist is simulated for every input vector at every one of the 200 corners of Part B: 25,600 operating points. "
  "The leakage is defined as for the gates (power drawn from the supply and from the inputs at VDD, divided by VDD). The simulation takes about 35 minutes with four parallel ngspice processes on the machine used.")
P("**ML prediction.** For each input vector the logic value at the inputs of every gate is determined by the logic simulator; the pattern `AB` of each gate is read from the logic values of its two input nets, and the leakage of that gate type, "
  "pattern and corner is taken from Part B (the gate model driven by the node voltages of the isolated gate at the same corner). The 40 gate leakages are summed. This assumes that every gate input sits at exactly 0 or VDD and ignores loading between gates, as the project text prescribes. "
  "A second sum, built from the isolated-gate **SPICE** leakages, is the **composition floor**: what the hierarchical method would give with perfect gate models. The difference between the floor and the SPICE leakage of the full circuit is the error of the composition itself, which no better transistor model can remove.")
