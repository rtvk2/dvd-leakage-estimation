# ML-Driven PVT Variation Aware Leakage Estimation

Digital VLSI Design course project: four machine-learning transistor models (NMOS / PMOS, ON / OFF) trained on ngspice data are used to estimate the leakage of five CMOS gates and of a 3-bit carry-lookahead adder under process, voltage and temperature variation, and every level is checked against transistor-level SPICE.

| | |
|---|---|
| Model file | 16nm_HP (PTM high performance 16 nm, BSIM4 level 54) |
| Nominal VDD | 0.9 V (sampled +/- 10 %), temperature -40 to 125 degC, five process parameters +/- 10 % |
| Test circuit | 3-bit carry-lookahead adder with carry-in (40 gates, 192 transistors, 7 inputs, 128 vectors) |
| Simulator | ngspice-42 (through WSL on Windows) |

## Results

Errors are relative errors against SPICE.

**Transistor models** (residual MLPs, 1,500 test samples per case, errors are means over the four terminal currents):

| Case | Validation median | Test median | Test 95th pct | Test mean | Test R2 (signed-log currents) |
|---|---|---|---|---|---|
| NMOS ON | 0.39 % | 0.37 % | 2.19 % | 0.73 % | 1.0000 |
| NMOS OFF (refined) | 0.74 % | 0.69 % | 4.64 % | 1.62 % | 0.9996 |
| PMOS ON | 0.24 % | 0.26 % | 1.31 % | 0.45 % | 1.0000 |
| PMOS OFF (refined) | 1.29 % | 1.28 % | 11.68 % | 3.51 % | 0.9996 |

**Gate level** (5 gates, 18 input cases x 200 corners = 3,600 points): median error **0.58 %**, 95th percentile 2.37 %, mean 0.83 %. Every case has a median error below 1 %.

**Test circuit** (128 input vectors x 200 corners = 25,600 points): hierarchical ML against the full-circuit SPICE: median **1.50 %**, 95th percentile 6.15 %, mean 2.07 %. With perfect gate models the same composition would give a median of 1.37 % and a 95th percentile of 5.25 % (loading and non-rail gate inputs are ignored, as prescribed), so most of the circuit error is the composition itself; it grows with temperature (median error 0.60 % in the coldest bin, -40 to -12.5 degC, and 3.87 % in the hottest, 97.5 to 125 degC).

The two OFF models were refined with 6,000 extra samples each near the supply rails, where the gates operate (notebook 03, section 16): the gate-level median error fell from 1.49 % to 0.58 % and the 95th percentile from 7.36 % to 2.37 %, at the price of a slightly worse score on the uniform test samples. Details, tables and the error analysis are in the report.

## Deliverables

| # | Deliverable | Where |
|---|---|---|
| 1 | Four datasets (10,000 samples each) | `data/nmos_on.csv`, `nmos_off.csv`, `pmos_on.csv`, `pmos_off.csv` (the extra rail samples used only for training the OFF refinement: `data/extra_nmos_off.csv`, `extra_pmos_off.csv`) |
| 2 | Example netlists | `netlists/` (four single transistors, five gates, `test_circuit_cla3.cir`) |
| 3 | Data check: Kirchhoff and three trend plots | `results/kirchhoff_check.*`, `results/trend_*.png`, `results/trend_check.csv` |
| 4 | Model comparison (4 model types x 4 cases) | `results/model_comparison_table.csv`, `model_comparison.png`, `baselines_validation.csv`, `model_comparison_all_runs.csv` |
| 5 | Chosen models: test errors, predicted-vs-SPICE plots, saved models | `results/final_models_test*.csv`, `results/pred_vs_spice_*.png`, `models/*.npz`, `results/refinement_before_after.csv` |
| 6 | Gate results table | `results/gate_results.csv` (data: `data/gate_spice.csv`, `data/gate_ml.csv`) |
| 7 | Gate parity plot | `results/gate_parity.png` |
| 8 | Test-circuit results and parity plot | `results/circuit_results.csv`, `circuit_results_by_vector.csv`, `circuit_parity.png` (data: `data/circuit_spice.csv`, `data/circuit_ml.csv`) |
| 9 | Code that regenerates everything | `notebooks/01` to `07`, `scripts/run_all.py` |
| 10 | Report | `report/report.docx` (built by `report/build_report.py`) |

## Reproducing the results

Each notebook is self-contained (it needs only the model file and the files written by earlier notebooks) and uses fixed seeds: data `20261105 + case`, corners `20261105 + 100`, data split `20261105`, model fits seed 0. Run them in order:

```bash
python scripts/run_all.py              # notebooks 01 to 07 in order, executed in place
python scripts/run_all.py 05 06 07     # only some of them (their inputs must exist)
python report/build_report.py          # rebuild report/report.docx from the result files
```

| Notebook | Content | Time on the machine used |
|---|---|---|
| `01_gen_data` | four single-transistor datasets, example netlists, extra rail samples for the OFF cases | 3 min |
| `02_check_data` | Kirchhoff check, three trend plots | 1 min |
| `03_train` | split, four model types, comparison, test set, refinement of the OFF models, saved models | 1.5 to 2 h |
| `04_gate_spice` | five gate netlists, 200 corners, 3,600 gate-level SPICE points | 1 min |
| `05_gate_ml` | gate leakage from the models, oracle check, gate results table, parity plot | 1 min |
| `06_circuit` | 3-bit CLA netlist, 25,600 SPICE points, hierarchical ML prediction | 35 min (4 ngspice processes) |
| `07_error_analysis` | errors by gate, input pattern, corner condition, model and component, gate-input voltage, input vector | under 1 min |

A rerun reproduces the datasets, the SPICE references and the models' scores (the four datasets, `corners.csv`, `gate_spice.csv` and `circuit_spice.csv` come out byte-identical).

## Method in short

* **Data:** one-transistor netlists with an own voltage source on every terminal; VDD, temperature, width, drain and source voltage and `vth0, toxe, ndep, u0, xj` sampled as in the project text; the terminal currents into the device are the negatives of the source currents (`set numdgt = 15`, one `altermod` per parameter, `AD = AS = W x 32 nm`).
* **Models:** the drain and source are canonicalised (Vds >= 0), the currents are predicted as `arcsinh(I / 1e-14 A)` (optionally after dividing by the sub-threshold factor `1 - exp(-Vds / V_T)`), from the 12 inputs plus 8 physics features. Polynomial ridge, LightGBM, an MLP and a residual MLP were compared on the validation set; the residual MLP is best in all four cases.
* **Gate leakage:** SPICE reference = power drawn from the supply and from the inputs held at VDD, divided by VDD. ML estimate = sum over the transistors of the channel, gate and body terms, computed from the model currents at the SPICE node voltages (by Tellegen's theorem exact with exact currents; checked with 16,800 single-transistor SPICE runs).
* **Circuit:** logic simulation gives the input pattern of each of the 40 gates; the leakages of the gates (type, pattern, corner) from the gate level are summed.

## Layout

```
data/        datasets, corners, gate and circuit SPICE results and ML predictions
models/      the four saved transistor models (numpy files, no PyTorch needed to use them)
netlists/    example netlists
notebooks/   01 to 07, executed, with outputs
results/     tables (CSV) and figures (PNG) for all deliverables
report/      report.docx and the scripts that build it
scripts/     run_all.py
```

## Requirements

ngspice (natively or through WSL on Windows) and the Python packages in `requirements.txt` (numpy, pandas, scipy, scikit-learn, LightGBM, PyTorch (CPU), matplotlib, joblib, python-docx, jupyter / nbconvert). The notebooks call `ngspice` directly when it is on the PATH and through `wsl` otherwise; the model files are expected in `Model Files-20261105/models/`.
