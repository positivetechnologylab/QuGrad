# QuGrad: Toward Quantum Data Generation for QML Utility

Code and data for:

> **Toward Quantum Data Generation for QML Utility**
> Kacem Ettahali, Tirthak Patel — Rice University — HPEC'26

## Overview

**QuGrad** is a noise-aware framework for generating synthetic quantum datasets whose **concentratable entanglement (CE)** matches a user-specified target distribution.

QuGrad trains a **low-depth parameterized ansatz** with **CMA-ES**, a derivative-free population-based evolution strategy, to minimize the **total variation distance (TVD)** between the generated and target **NZP** histograms. NZP (nonzero probability, `z = 1 − P(0ⁿ)`) is a SWAP-test-derived, measurement-efficient surrogate for CE that avoids recomputing reduced-subsystem purities inside the optimization loop. CE histograms are then used for **post-training** evaluation.

Circuits are executed through Qiskit Aer's `AerSimulator` under the empirical noise model of the 156-qubit **IBM Boston** backend; real-hardware results use the 133-qubit **IBM Torino** processor. All runs use 2048 shots.

QuGrad's baseline is [QMill](https://github.com/positivetechnologylab/QMill), which targets the same CE-distribution problem with `scipy.optimize.dual_annealing` in the ideal setting only.

---

## Installation

```bash
git clone https://github.com/positivetechnologylab/QuGrad.git
cd QuGrad
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The `qml_stuff/` folder (Table II) has its **own** `requirements.txt` — it pins a different Qiskit version that conflicts with the root environment. Use a separate venv for it.

IBM noise models are committed under `noise_models/ibm_boston/`, so the noisy results reproduce with **no IBM Quantum account**. An account is only needed to fetch fresh noise models via `ibm_noise_fetcher.py`.

---

## Repository Layout

```text
QuGrad/
├── train_qugrad.py            # ★ CMA-ES / NZP-TVD generator training
├── runs_qugrad/               # QuGrad trained params, CE values, histories (ideal + noisy)
├── runs_qmill/                # QMill baseline outputs (ideal, dual annealing)
├── qmill_noisy_eval/          # QMill evaluated under the IBM Boston noise model
├── noise_models/ibm_boston/   # IBM Boston noise model (calibrated 2025-12-31)
│
├── circuits.py                # Ansatz definitions (Five, Sixteen, Custom_One, Custom_Two)
├── circuits_layered.py        # Layered ansatz variants
├── classes.py                 # Ansatz class, SWAP test, V-SWAP test, CE binning
├── dists.py                   # Target CE distributions + TVD
├── ML.py                      # Product-state sampling, cost plumbing
├── custom_executor.py         # CustomCircuitExecutor (ideal) + NoisyCircuitExecutor
├── variety.py                 # SWAP-test diversity between generated states (see Notes)
├── ibm_noise_fetcher.py       # Fetch/cache IBM backend noise models
│
├── nzp_ce_validation.py       # Table I
├── plot_tvd_cleveland.py      # Figs 2-4
├── swap_runs.py               # Fig 5 (raw)   -> swapres.txt
├── swap_violin_plot.py        # Fig 5 (plot)
├── plotentropy.py             # Fig 6
├── plotpurity.py              # Fig 7
├── plotrealnoisyideal_qugrad.py  # Fig 8
├── qml_stuff/                 # Table II (own requirements.txt)
└── cmaes_scaling_benchmark.py # Supplementary: optimizer scaling
```

---

## Training a Generator

```bash
# Noisy training (default) under the IBM Boston noise model
python3 train_qugrad.py --ansatz Sixteen --dist Uniform \
                        --backend ibm_boston --date 2025-12-31 --max_evals 20000

# Ideal (noiseless) training
python3 train_qugrad.py --ansatz Sixteen --dist Uniform --ideal
```

Outputs land in `runs_qugrad/{ansatz}/{dist}/5/1/1/` (noisy) or
`runs_qugrad/Ideal/{ansatz}/{dist}/5/2/1/` (ideal):

| file | contents |
|---|---|
| `{ansatz}_5_1.npy` | trained parameters θ |
| `{ansatz}_5_1_x0.npy` | initial parameters |
| `{ansatz}_5_1_results.npy` | CE values of the generated states |
| `{ansatz}_5_1_history.npy` | per-generation TVD, entropy, purity |
| `{ansatz}_5_1.txt` | human-readable run summary |

Targets: `Uniform`, `Normal`, `Left_Weibull`, `Right_Weibull`, `MNIST`, `Fashion_MNIST`, `CIFAR`, `QCHEM`, `Soillow`, `Soilhigh`, `dmlow`, `dmhigh`.

---

## Reproducing the Paper

Trained outputs are committed, so every figure and table reproduces **without retraining**.

| Paper item | Command | Output |
|---|---|---|
| **Figs 2–4** — TVD vs QMill (stress / real / sensor targets) | `python3 plot_tvd_cleveland.py` | `tvd_cleveland_stress.pdf`, `tvd_cleveland_real.pdf`, `tvd_cleveland_sensors.pdf`, `tvd_results.csv` |
| **Fig 5** — SWAP-test diversity across CE bins | `python3 swap_violin_plot.py` | `swap_test_violin.pdf` (reads `swapres.txt`) |
| **Fig 6** — von Neumann entropy under noise | `python3 plotentropy.py` | `Paper Plots/entropy_comparison.pdf` |
| **Fig 7** — purity under noise | `python3 plotpurity.py` | `Paper Plots/purity_comparison.pdf` |
| **Fig 8** — real hardware (IBM Torino), MNIST target | `python3 plotrealnoisyideal_qugrad.py` | `noisyideal_qugrad.pdf` |
| **Table I** — NZP ↔ CE agreement (Pearson *r*, Spearman *ρ*) | `python3 nzp_ce_validation.py` | `Results/nzp_vs_ce_scatter.pdf`, LaTeX table |
| **Table II** — downstream QNN (RealAmplitudes, Soil Low/High) | `cd qml_stuff && python3 qml_res_table_qugrad.py` | printed table |

To regenerate Fig 5's raw numbers from scratch (slow — 1000 states × 32 ansatz/target pairs):

```bash
python3 swap_runs.py        # prints results; swapres.txt was saved from this output
python3 swap_violin_plot.py
```

**Supplementary:** `cmaes_scaling_benchmark.py` compares CMA-ES against dual annealing as the parameter count grows. It uses a *synthetic* surrogate cost (`NZP = sigmoid(s·θ/√d)·0.6`) to isolate optimizer scaling and does **not** execute quantum circuits. It backs no figure or table in the paper.

---

## Notes

**Diversity analysis (`variety.py`).** Implements the SWAP-test diversity procedure of §V.C: 1000 random product states are propagated through the trained ansatz, grouped into CE bins of width 0.02, and randomly paired within each bin, with the SWAP-test overlap `P(0)` recorded for each pair. `swap_runs.py` drives this across all ansatz/target pairs and its output is saved to `swapres.txt`, which `swap_violin_plot.py` renders as Fig 5. Because both the state sampling and the within-bin pairing are random, rerunning `swap_runs.py` reproduces the recorded statistics to within Monte-Carlo sampling noise rather than value-for-value.

**Result coverage.** The committed QuGrad results cover the ansatz/target cells reported in the paper (`Five` and `Sixteen`); the QMill baseline is complete across all four ansatze. The plotting scripts skip cells with no data, so they regenerate the published panels rather than a full 4 × 8 grid.

**Run provenance.** Every run's optimizer is recorded in its `.txt` summary. The authoritative record is `*_history.npy`.
