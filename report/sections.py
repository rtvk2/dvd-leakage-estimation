# Content of the report (executed by build_report.py, which defines H, P, B, T, F, pct, sci, CASE, GATE, RES, DATA).

# ---------------------------------------------------------------------------------------------------------------------
H("Summary")
_ct = pd.read_csv(RES / "final_models_test.csv", index_col=0).astype({"test_median": float, "test_p95": float, "test_mean": float, "test_r2_log": float})
_g = pd.read_csv(DATA / "gate_ml.csv")["err"].abs() * 100
_c = pd.read_csv(DATA / "circuit_ml.csv")
_ch, _cf = _c["err"].abs() * 100, _c["err_floor"].abs() * 100
P("Four transistor models (NMOS and PMOS, ON and OFF; residual neural networks on four 10,000-sample SPICE datasets of the 16nm_HP card) predict the four terminal currents of a transistor from its voltages, width, temperature and five process parameters. "
  "Their predictions, taken at the SPICE node voltages, are summed to the leakage of five gates (INV, NAND2, NOR2, AND2, OR2) at 200 PVT corners, and the gate leakages are summed to the leakage of a 3-bit carry-lookahead adder (40 gates, 128 input vectors). "
  "Every level is compared with a transistor-level ngspice simulation:")
T([["Transistor currents (4 models, 1,500 test samples each)", f"{_ct.test_median.min():.2f} to {_ct.test_median.max():.2f}", f"{_ct.test_p95.min():.1f} to {_ct.test_p95.max():.1f}", f"{_ct.test_mean.min():.2f} to {_ct.test_mean.max():.2f}"],
   ["Gate leakage (5 gates, 18 input cases x 200 corners = 3,600 points)", f"{_g.median():.2f}", f"{_g.quantile(0.95):.2f}", f"{_g.mean():.2f}"],
   ["3-bit CLA leakage, hierarchical (128 vectors x 200 corners = 25,600 points)", f"{_ch.median():.2f}", f"{_ch.quantile(0.95):.2f}", f"{_ch.mean():.2f}"],
   ["3-bit CLA with perfect gate models (the composition floor)", f"{_cf.median():.2f}", f"{_cf.quantile(0.95):.2f}", f"{_cf.mean():.2f}"]],
  ["level", "median error [%]", "95th percentile [%]", "mean error [%]"], widths=[3.6, 1.0, 1.1, 1.0], caption="Table 0. Summary of the results (relative error against SPICE)")
P(f"The R2 of the signed-log currents is at least {_ct.test_r2_log.min():.4f} for every transistor model. The gate-level error is below {np.ceil(_g.quantile(0.95)):.0f} % for 95 % of the points. "
  f"The circuit-level error is dominated by the composition method (ideal 0 / VDD gate inputs, no loading), which is prescribed by the project and costs a median of {_cf.median():.2f} % even with perfect gate models; it grows with temperature (sections 5.1 and 6.5). "
  "The most useful single step for the accuracy was to give the two OFF models extra training samples near the supply rails, where the gates operate (section 3.5).")

H("1. Setup")
T([["Model file", "16nm_HP: PTM high-performance 16 nm, metal gate / high-k, BSIM4 (level 54), TNOM 27 degC"],
   ["Nominal VDD", "0.9 V (the file header says 0.7 V; the assigned 0.9 V is used, so VDD is sampled between 0.81 V and 0.99 V)"],
   ["Geometry", "L = 16 nm (= Lmin); W between 1 x Lmin and 8 x Lmin, i.e. 16 nm to 128 nm"],
   ["Test circuit", "3-bit carry-lookahead adder with carry-in (7 inputs, 128 input vectors)"],
   ["Simulator", "ngspice-42, batch mode (`ngspice -b`), run through WSL on Windows 11; `.options gmin=1e-18 abstol=1e-18 reltol=1e-6 vntol=1e-9` in every netlist; `set numdgt = 15` so that currents are printed with 15 digits"],
   ["Software", "Python 3.12.8, numpy 1.26.4, pandas 2.2.3, scikit-learn 1.5.1, LightGBM 4.6.0, PyTorch 2.7.0 (CPU), matplotlib 3.9.2"]],
  ["Item", "Value"], widths=[1.3, 5.6], caption="Table 1. Assignment and tools")
P("The work is organised as seven self-contained Jupyter notebooks (`notebooks/01` to `07`), run in order by `scripts/run_all.py`; all random choices use fixed seeds. "
  "They produce the datasets (`data/`), the example netlists (`netlists/`), the four saved models (`models/`) and all tables and plots (`results/`).")

# ---------------------------------------------------------------------------------------------------------------------
H("2. Data generation")
H("2.1 Sampling", 2)
P("For each of the four cases (NMOS ON, NMOS OFF, PMOS ON, PMOS OFF) 10,000 samples were drawn with the distributions of the project text. VDD is drawn first (uniform, 0.9 V +/- 10 %); "
  "then the drain and source voltages (uniform between 0 and the sampled VDD), the width (uniform 16 nm to 128 nm), the temperature (uniform -40 to 125 degC) and the five process parameters "
  "`vth0, toxe, ndep, u0, xj` (uniform, nominal value of the model card +/- 10 %, separately for the NMOS and the PMOS card). The gate and body voltages are fixed by the case: "
  "NMOS ON (VDD, 0), NMOS OFF (0, 0), PMOS ON (0, VDD), PMOS OFF (VDD, VDD). Every parameter of every sample is drawn independently; the seeds are 20261105 + case index.")
T([["vth0 [V]", "0.47965", "-0.43121"], ["toxe [m]", "9.5e-10", "1.0e-9"], ["ndep [1/cm3]", "7.0e18", "5.5e18"], ["u0 [m2/Vs]", "0.030", "0.006"], ["xj [m]", "5.0e-9", "5.0e-9"]],
  ["parameter", "NMOS nominal", "PMOS nominal"], widths=[1.6, 1.6, 1.6], caption="Table 2. Nominal process parameters read from the model file")
H("2.2 Netlists and simulation", 2)
P("Each sample is simulated in a one-transistor netlist with **its own voltage source on every terminal** (`Vd, Vg, Vs, Vb`); the terminal currents flowing **into** the device are the negatives of the source currents. "
  "The junction areas are given explicitly, `AD = AS = W x 32 nm` (without them the OFF-state body current of the model is 12 to 23 times too large; the value of the diffusion length between 22 nm and 44 nm changes the currents by less than 0.004 %). "
  "The process parameters are set per sample with `altermod` (one `altermod` line per parameter, since only the first parameter on a line is applied), the temperature with `set temp`, the voltages, the width and the junction areas with `alter`, and an operating point (`op`) is run. "
  "250 samples go into one ngspice script and the scripts run in parallel processes (17 to 26 s per dataset on the machine used). A sample whose operating point failed would be detected (the printed node voltages must equal the applied ones) and redrawn; none failed. "
  "A sample simulated alone gives exactly the same currents as inside its batch (checked on 5 random samples per case). Example netlists: `netlists/single_<case>.cir`.")
H("2.3 Data check", 2)
k = pd.read_csv(RES / "kirchhoff_check.csv")
T([[CASE[r.case], f"{r.rows:,}", sci(r.max_abs_sum_A), sci(r.median_abs_sum_A), "yes" if r.passes else "NO"] for r in k.itertuples()],
  ["case", "rows", "max |sum of the 4 currents| [A]", "median [A]", "passes"], widths=[1.2, 0.8, 2.3, 1.3, 0.8],
  caption="Table 3. Kirchhoff check: the four terminal currents of every sample sum to zero")
P("The residual is at the level of the solver tolerance (at most 1.5e-15 A, against terminal currents of up to 3e-4 A). Three trend plots check the physics of the data "
  "(all Spearman correlations have the expected sign):")
tr = pd.read_csv(RES / "trend_check.csv")
T([[CASE[r.case] if r.case in CASE else r.case, r.relation, f"{r.spearman:+.3f}", r.expected_sign, "yes" if r.as_expected else "NO"] for r in tr.itertuples()],
  ["case", "relation", "Spearman", "expected sign", "as expected"], widths=[1.2, 1.7, 1.0, 1.1, 1.1], caption="Table 4. Trend checks")
F("trend_off_vs_temp.png", "Figure 1. OFF-state leakage current against temperature (increases with temperature).", 5.6)
F("trend_off_vs_vth0.png", "Figure 2. OFF-state leakage current against |vth0| (decreases with a higher threshold).", 5.6)
F("trend_ig_vs_toxe.png", "Figure 3. Gate current against oxide thickness (decreases with a thicker oxide).", 5.6)
P("One property of the data matters for the models: the simulator reports small currents only to a finite resolution. Many values are exact multiples of 2^-51 A = 4.4e-16 A "
  "(for example 44 % of the NMOS-ON gate currents and 43 % of the PMOS-OFF gate currents), so below about 1e-14 A a rounding of one quantum is more than 1 % of the current.")
H("2.4 Split", 2)
P("Each dataset is split at random into 70 % training (7,000), 15 % validation (1,500) and 15 % test samples (1,500) with the fixed seed 20261105. The same split is used for every model type. "
  "All choices (options, number of trees or epochs, selection of the final model) are made on the training and validation samples; the test samples are used once, for the chosen models.")

# ---------------------------------------------------------------------------------------------------------------------
H("3. Transistor models")
H("3.1 Preprocessing", 2)
B(["**Source/drain canonicalisation.** A transistor is symmetric in drain and source, so every sample is mapped to one orientation (Vds >= 0 for NMOS, Vsd >= 0 for PMOS; `id` and `is` swap with the terminals). "
   "This fixes the sign of `id` and `is` in the ON cases and removes the sign change at Vd = Vs from the learning problem. The saved models do this internally and swap the currents back.",
   "**Target transform.** The four currents span about 11 decades and some change sign, so the models predict `arcsinh(I / I0)` with `I0 = 1e-14 A` (a relative scale for large currents, linear near zero, sign kept). "
   "`I0` was chosen on the validation samples (1e-16 to 1e-13 A tried); a smaller value makes the network chase the 4.4e-16 A resolution of the data (PMOS OFF: 11.3 % median error at 1e-16 A, 2.0 % at 1e-14 A, residual MLP).",
   "**Vds-aware target (option).** The BSIM sub-threshold current contains the factor `1 - exp(-Vds / V_T)`. For `id` and `is` the current is divided by this factor before the transform, which makes the ON-state current at |Vds| of microvolts to millivolts (where the transistors inside gates operate) a smooth function. It is chosen per case on the validation samples.",
   "**Features.** The 12 raw inputs plus 8 physics features: q/kT, Vgs, Vds, Vbs, Vgd, (Vgs - vth0)/V_T, Vds/V_T and ln W (source-referenced, sign mirrored for PMOS). Inputs and targets are standardised with training-set statistics only."])
H("3.2 Model types and hyperparameters", 2)
P("One model per case, each predicting the four currents `id, ig, is, ib`. Four model types were compared:")
T([["Polynomial ridge regression", "polynomial features of the standardised inputs (degrees 2 to 4 on the 12 raw inputs, 2 to 3 on the 20 physics features, at most about 1,800 terms); ridge penalty chosen per target by leave-one-out cross-validation on the training samples"],
   ["Gradient boosting (LightGBM)", "one regressor per current, 31 leaves, learning rate 0.05, up to 3,000 trees with early stopping on the validation samples, row/column subsampling 0.8/0.9, deterministic"],
   ["MLP", "3 hidden layers of 256 units, SiLU activation; AdamW (weight decay 1e-5), batch 256, one-cycle learning rate (peak 3e-3), 600 epochs, seed 0; the epoch with the lowest validation MSE is kept"],
   ["Residual MLP (ResNN)", "input layer of 128 units, 4 residual blocks h <- SiLU(h + Linear(SiLU(Linear(h)))), linear output; same training as the MLP"]],
  ["model type", "main hyperparameters"], widths=[1.8, 5.1], caption="Table 5. Model types")
P("For each model type and case the option sets (raw or physics features, Vds-aware target or not; for the networks MSE or Huber loss) were compared on the validation samples, and the best one is reported. "
  "**Selection rule:** the lowest mean over the four currents of the median relative error `|pred - true| / |true|` (rows with |I| >= 1e-15 A, where the solver floor is not dominant; the project text does not define \"error\", this definition is used throughout).")
H("3.3 Model comparison on the validation set (deliverable 4)", 2)
mc = pd.read_csv(RES / "model_comparison_table.csv", header=[0, 1], index_col=0)
rows = []
for case in CASE:
    row = [CASE[case]]
    for m in ("polynomial ridge", "LightGBM", "MLP", "residual MLP"):
        row += [f"{mc.loc[case, (m, 'R2')]:.4f}", pct(mc.loc[case, (m, 'median error [%]')]), pct(mc.loc[case, (m, '95th pct error [%]')], 1)]
    rows.append(row)
T(rows, ["case"] + ["R2", "med. %", "p95 %"] * 4, size=8,
  caption="Table 6. Validation set. Columns per model type (in order: polynomial ridge, LightGBM, MLP, residual MLP): R2 of the signed-log currents, median and 95th-percentile relative error (mean over the four currents)")
F("model_comparison.png", "Figure 4. Median relative error of the four model types (validation set, log axis).", 5.2)
P("**Reason for the final choice.** The residual MLP has the lowest validation error in all four cases; its median error is 7 to 31 times lower than the best of the two classical models (polynomial ridge cannot follow currents that span many decades, "
  "LightGBM produces piece-wise constant predictions) and 1.35 to 1.5 times lower than the plain MLP. The final models are therefore the four residual MLPs: NMOS ON and PMOS ON with the Vds-aware target and Huber loss, "
  "NMOS OFF without the Vds-aware target (MSE loss), PMOS OFF with it (MSE loss), all with the physics features and 600 epochs.")
H("3.4 Final models on the test set (deliverable 5)", 2)
ct = pd.read_csv(RES / "final_models_test.csv", index_col=0)
T([[CASE[c], "refined (section 3.5)" if str(r.source).startswith("refined") else "as chosen", pct(r.val_median), pct(r.test_median), pct(r.test_p95), pct(r.test_mean), f"{r.test_r2_log:.4f}"] for c, r in ct.iterrows()],
  ["case (residual MLP)", "version", "validation median [%]", "test median [%]", "test 95th pct [%]", "test mean [%]", "test R2 (log)"], widths=[1.3, 1.3, 1.0, 0.9, 1.0, 0.9, 0.9],
  caption="Table 7. Final models, test set (1,500 samples per case; errors are means over the four currents; the models are scored from the saved files in `models/`)")
pt = pd.read_csv(RES / "final_models_test_per_terminal.csv")
T([[CASE[r.case], r.terminal, pct(r.median_rel * 100), pct(r.p95_rel * 100), pct(r.mean_rel * 100), pct(r.median_rel_small_vds * 100)] for r in pt.itertuples()],
  ["case", "current", "median [%]", "95th pct [%]", "mean [%]", "median, |Vds| < 5 mV [%]"], widths=[1.1, 0.8, 1.0, 1.1, 1.0, 1.7], size=8,
  caption="Table 8. Final models, test set, per current")
for case in CASE:
    F(f"pred_vs_spice_{case}.png", f"Figure: predicted against SPICE currents, {CASE[case]} (1,500 test samples, symmetric-log axes, dashed line = identity).", 6.3)
below1 = [CASE[c] for c in ct.index if float(ct.loc[c, "test_median"]) < 1]
above1 = [f"{CASE[c]} ({float(ct.loc[c, 'test_median']):.2f} %)" for c in ct.index if float(ct.loc[c, "test_median"]) >= 1]
worst = pt.assign(m=pt.median_rel * 100).sort_values("m", ascending=False).head(3)
small = pt[pt.n_small_vds >= 1]
P(f"The test errors agree with the validation errors (the test median differs from the validation median by at most {(ct.test_median.astype(float) - ct.val_median.astype(float)).abs().max():.2f} percentage points), so the choices made on the validation samples left little optimism. "
  f"The median error is below 1 % for {len(below1)} of the four cases" + (f" ({', '.join(below1)}); **{', '.join(above1)} is above it**" if above1 else "") + f"; the 95th percentile is {ct.test_p95.astype(float).min():.1f} % to {ct.test_p95.astype(float).max():.1f} % and the mean error {ct.test_mean.astype(float).min():.2f} % to {ct.test_mean.astype(float).max():.2f} % "
  "(a few samples with large relative errors, in currents close to the resolution of the data, raise the mean above the median). "
  "The R2 of the signed-log currents is at least " + f"{ct.test_r2_log.astype(float).min():.4f}" + " in every case (the slide target R2 > 0.99). "
  "The hardest currents are " + ", ".join(f"`{r.terminal}` of {CASE[r.case]} (median {r.m:.2f} %)" for r in worst.itertuples()) + ": small currents that are sums of components of opposite sign. "
  f"In the region of small |Vds| (< 5 mV), where the ON transistors of gates operate, the median error is {small[~((small.case == 'pmos_off') & small.terminal.isin(['id', 'is']))].median_rel_small_vds.min() * 100:.2f} % to "
  f"{small[~((small.case == 'pmos_off') & small.terminal.isin(['id', 'is']))].median_rel_small_vds.max() * 100:.2f} % for all currents except the channel currents of PMOS OFF, for which it is "
  f"{small[(small.case == 'pmos_off') & (small.terminal == 'id')].median_rel_small_vds.iloc[0] * 100:.1f} % (`id`) and {small[(small.case == 'pmos_off') & (small.terminal == 'is')].median_rel_small_vds.iloc[0] * 100:.1f} % (`is`) on only "
  f"{int(small[(small.case == 'pmos_off') & (small.terminal == 'id')].n_small_vds.iloc[0])} and {int(small[(small.case == 'pmos_off') & (small.terminal == 'is')].n_small_vds.iloc[0])} test samples.")
H("3.5 Refinement of the OFF models with extra samples near the rails", 2)
P("The transistors inside a gate operate at the corners of the (Vd, Vs) square: a terminal is at a supply rail or at an internal node close to one. The uniform datasets have only 6 % of their Vd values within 3 % of VDD of a rail, and the gate-level error analysis (section 6) "
  "showed that almost all gate-level error comes from the two OFF models there, because the OFF transistors carry almost all of the leakage. Notebook 01 (section 9) therefore draws **6,000 extra samples per OFF case** with the same procedure and ranges as the datasets, "
  "except that each of Vd and Vs is drawn from a mixture (1/3 uniform, 1/3 within 3 % of VDD of 0 V, 1/3 within 3 % of VDD of VDD); about two thirds of the values then lie near a rail. They satisfy the same Kirchhoff check and are used only as additional training samples. "
  "Each OFF model is retrained with exactly the settings chosen in section 3.3 (residual MLP, same target options, loss, epochs, seed and scalers) on the 7,000 uniform training rows plus 5,000 of the extra samples; the other 1,000 extra samples form a separate rail validation set. "
  "The refined model replaces the chosen one if it lowers both the median and the 95th-percentile error on the rail validation set and raises the median error on the uniform validation rows by less than 0.5 percentage points (a rule evaluated on validation data only; the test rows play no part in the decision). "
  "The idea of the refinement comes from the gate-level analysis (which showed where the error is), but no gate-level data are used to train, select or tune any model.")
rb = pd.read_csv(RES / "refinement_before_after.csv")
T([[CASE[r.case], r.rows, pct(r.chosen_median), pct(r.refined_median), pct(r.chosen_p95, 1), pct(r.refined_p95, 1)] for r in rb.itertuples()],
  ["case", "rows", "as chosen: median [%]", "refined: median [%]", "as chosen: 95th pct [%]", "refined: 95th pct [%]"], widths=[1.1, 1.5, 1.0, 1.0, 1.1, 1.0], size=8,
  caption="Table 8b. Effect of the refinement (mean over the four currents; the test rows are shown for information only)")
adopted = [CASE[c] for c in ct.index if str(ct.loc[c, "source"]).startswith("refined")]
P(("The refined model was adopted for " + (" and ".join(adopted) if adopted else "none of the cases") + "; the ON models are unchanged (their leakage is a negligible part of the gate leakage). "
   "The effect on the gate-level and the circuit-level error is shown in sections 4.4 and 5.1."))

# ---------------------------------------------------------------------------------------------------------------------
H("4. Gate-level validation (Part B)")
H("4.1 Gates, sizing and corners", 2)
P("The five gates are standard static CMOS (INV, NAND2, NOR2) with AND2 = NAND2 + INV and OR2 = NOR2 + INV, simulated as complete transistor-level netlists (one `netlists/gate_<gate>.cir` per gate). "
  "The sizes follow the convention of the course slides (inverter Wp = 2 Wn; devices in a series stack scaled by the stack height) with a drive-strength multiplier XX = 2: "
  "INV 32 nm / 64 nm (NMOS / PMOS), NAND2 64 nm / 64 nm, NOR2 32 nm / 128 nm, and the INV inside AND2 and OR2 as above; all widths are within the allowed 16 nm to 128 nm. "
  "(XX = 1 would put the NMOS of the INV and NOR2 at exactly the minimum width, the edge of the training range, where the models are least accurate; Table 11 shows the effect.) "
  "The ground rail is connected through a 0 V source so that its current can be read, which gives a Kirchhoff check for every gate. Every gate netlist contains the required `.options` line.")
P("A **corner** is one random draw of VDD (0.9 V +/- 10 %), temperature (-40 to 125 degC) and the five process parameters (nominal +/- 10 %), drawn separately for the NMOS and the PMOS card, so that all NMOS of a gate share one draw and all PMOS another. "
  "200 corners were drawn once (seed 20261105 + 100, `data/corners.csv`) and are used for all gates and for the test circuit. For each of the 18 gate and input cases (INV: A = 0, 1; the other gates: AB = 00, 01, 10, 11) every corner is simulated: 3,600 operating points. "
  "Per corner the process parameters are set on the shared model cards, per input pattern the input sources, and the supply current, the input currents and all internal and output node voltages are recorded. "
  "Checks on the reference: every output is within 30 mV of its logic rail, the four source currents sum to at most 2.0e-15 A, 12 points simulated alone reproduce the batch exactly, and the supply delivers current in every point.")
H("4.2 How the gate leakage is defined and computed", 2)
P("**SPICE reference.** The leakage current of a gate is the power drawn from the supply and from the inputs held at VDD, divided by VDD (the definition of the instructor's inverter leakage netlist: supply power plus input-source power). "
  "An input at 0 V delivers no power. By Kirchhoff's laws this is the current that flows from the VDD level to the 0 V level through the gate, and it is additive when gates drive each other. The supply current alone, `-I(VDD)`, differs from it (relative to the supply current) by less than 3.5 % except for NOR2 with inputs 11, "
  "where the supply current is the smallest of all states and the gate currents of the two inputs held at VDD are comparable to it (median difference 26 %); this is why the definition is stated explicitly.")
P("**ML estimate.** The method of the course material: the leakage is the sum of the sub-threshold, gate and body (junction) components of all transistors. "
  "For every transistor of the gate the terminal voltages (drain, gate, source, bulk) are taken from the SPICE node voltages, the transistor is sent to the ON or OFF model of its type according to its gate voltage "
  "(NMOS ON if the gate is above VDD/2, PMOS ON if it is below), together with its width, the corner's temperature, VDD and process parameters, and the model returns `id, ig, is, ib`. "
  "The currents map onto the three components: the channel terminals (`id`, `is`) are the sub-threshold current (including GIDL), `ig` the gate current and `ib` the body / junction current. "
  "With voltages measured from the source terminal, the three components of a transistor, expressed as currents at the supply voltage, are")
B(["channel: `(Vd - Vs) Id + ...` over the channel terminals divided by VDD,", "gate: `(Vg - Vs) Ig / VDD`,", "body: `(Vb - Vs) Ib / VDD`,"])
P("and the gate leakage is their sum over all transistors. Because the four currents of a transistor sum to zero, the sum of the three terms is exactly the power of the transistor, and the transistor powers add up to the power delivered by the sources (Tellegen's theorem), "
  "so with exact currents the sum equals the SPICE leakage. What this gives for the two states: **an OFF transistor contributes all three components** (its channel term is its sub-threshold current weighted by the share of the supply voltage across it, "
  "so a stack of two OFF transistors, which carry the same current and share the supply, is counted once in total); **an ON transistor contributes its gate and body leakage**, and its channel current (which is the leakage of the OFF transistor it is in series with) "
  "enters only with its own tiny Vds (microvolts to millivolts), i.e. practically not at all. The channel term of an ON transistor is nevertheless kept and not set to zero by hand, because an ON transistor can carry a real share of the supply "
  "(the upper NMOS of a NAND2 with inputs 10 has 0.25 V across it): dropping the term gives errors of 27 % to 28 % for NAND2 10 and NOR2 10 (section 4.3).")
H("4.3 Oracle check of the composition", 2)
orc = pd.read_csv(RES / "gate_oracle_check.csv", index_col=0) if have("gate_oracle_check.csv") else None
if orc is not None:
  T([[idx, sci(r["median"]), sci(r["p95"]), sci(r["largest"])] for idx, r in orc.iterrows()], ["composition (with exact SPICE transistor currents)", "median", "95th pct", "largest"],
    widths=[3.9, 1.0, 1.0, 1.0], caption="Table 9. Relative difference to the SPICE gate leakage when the transistor models are replaced by single-transistor SPICE runs (16,800 transistor instances)")
P("Before any ML number is trusted, the whole chain (voltage mapping, selection of the transistor, composition) is run with single-transistor SPICE simulations of all 16,800 transistor instances at the recorded voltages instead of the models. "
  "The component sum then reproduces the SPICE gate leakage with a largest relative difference of " + (sci(orc.loc["component sum", "largest"]) if orc is not None else "[pending]") + ", i.e. the voltage mapping and the composition are correct and **all remaining error in Table 10 is model error**. "
  "The table also shows the Kirchhoff composition (sum of the terminal currents at the nodes tied to the supply or to inputs at VDD), which is equally exact and is used as a cross-check, and the variant that drops the channel term of the ON transistors.")
