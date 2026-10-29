# Replication package — *Does Compression Erode Trust?*

Code, trained models and raw experimental outputs for

> A. Çakın and S. Dilek, "Does Compression Erode Trust? Calibration of Compressed Fault  
> Diagnosis Models on Edge Hardware Under Operating Point Shift," submitted to 
> *IEEE Journal of Emerging and Selected Topics in Industrial Electronics* (special section
> "Trustworthy Fault Diagnosis and Health Management for Industrial Electronics Systems"), 2026.

The study trains two compact CNNs (WDCNN, 1-D ResNet) on the Paderborn bearing data set at one operating
point, prunes them (0–85 %), deploys them with TensorRT in FP32/FP16/INT8 on a Jetson Orin Nano, and measures
accuracy, calibration (ECE, temperature scaling), agreement with the uncompressed model and split-conformal
coverage at four operating points, on bearings never seen in training.

```
 WORKSTATION (GPU)                                  JETSON ORIN NANO                      ANY COMPUTER (CPU)
 download data -> features -> train (C0)       ->   build TensorRT engines           ->   evaluate: metrics, TS, conformal
              -> prune + fine-tune                  FP32 / FP16 / INT8-src / INT8-all     report: tables, figures,
              -> bearing-level calibration models   dump logits; latency + energy          deployment rules, in-text numbers
              -> export ONNX + eval data
     runs/, export/                                      jetson_out/                           results/
```

Every metric in the paper is computed from saved **logits**. This package ships all logits, checkpoints,
ONNX models and Jetson benchmarks of the paper run, so the analysis can be reproduced without the data set
or any GPU. Training data and anything derived from it can be regenerated with the scripts below.

---

## Contents

1. [Quick start (Tier 1, ~1 minute)](#1-quick-start-tier-1-1-minute)
2. [Which file reproduces which part of the paper](#2-which-file-reproduces-which-part-of-the-paper)
3. [Three levels of replication](#3-three-levels-of-replication)
4. [Data: download the Paderborn bearing data set](#4-data-download-the-paderborn-bearing-data-set)
5. [Tier 2: rebuild features, export and Jetson results from the shipped models](#5-tier-2-rebuild-features-export-and-jetson-results-from-the-shipped-models)
6. [Tier 3: retrain everything from scratch, ablations](#6-tier-3-retrain-everything-from-scratch-ablations)
7. [Hardware and software used for the paper](#7-hardware-and-software-used-for-the-paper)
8. [Expected tolerances](#8-expected-tolerances)
9. [Repository layout and what is (not) included](#9-repository-layout-and-what-is-not-included)
10. [Pipeline reference](#10-pipeline-reference)
11. [Troubleshooting](#11-troubleshooting)
12. [Citation, licence, contact](#12-citation-licence-contact)

---

## 1. Quick start (Tier 1, ~1 minute)

Regenerates **every table, figure and in-text number** of the paper from the shipped logits.
Needs Python ≥ 3.10 and no GPU, no PyTorch, no data download.

```bash
git clone https://github.com/sepidi/TrustworthyEdgeFaultDetection.git
cd TrustworthyEdgeFaultDetection
python3 -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-analysis.txt
cp -r results results_shipped                    # keep the shipped results for comparison
bash scripts/run_analysis.sh
python tools/check_results.py results_shipped results   # must end with "ALL RESULTS REPRODUCED"
```

Output (in `results/`):

| File | Content |
|---|---|
| `tables/tab_main_resnet1d.tex`, `tab_main_wdcnn.tex`, `tab_jetson.tex`, `tab_selection.tex` | Tables III–VI of the paper (LaTeX) |
| `figures/fig_acc_ece_resnet1d.pdf`, `fig_coverage_resnet1d.pdf`, `fig_reliability_resnet1d.pdf` | Figs. 2, 3, 4 |
| `paper_numbers.md` | every number quoted in the text that is not in a table, grouped by paper section |
| `summary.md` | aggregated metrics for all models, precisions and conformal variants |
| `metrics.csv`, `cp.csv`, `selection.csv` | per fold × model × precision × condition: accuracy, ECE, NLL, Brier, agreement; conformal coverage and set size; deployment-rule choices |

`check_results.py` requires the LaTeX tables and Markdown files to be byte-identical and the CSV files to be
numerically identical (pandas/NumPy versions may print the last floating-point digit differently); PDF figures
are only checked for presence, because their bytes depend on the matplotlib version.

On Windows without bash, run the three Python steps directly:
`python -m tcfd.evaluate -c config.yaml`, `python -m tcfd.report -c config.yaml`, `python -m tcfd.paper_numbers -c config.yaml`.

---

## 2. Which file reproduces which part of the paper

| Paper | Reproduced by | Produced from |
|---|---|---|
| Table I (operating conditions) | `config.yaml` → `conditions` | — |
| Table II (bearing-wise folds) | `config.yaml` → `folds`, `train_only_bearings` | — |
| Fig. 1 (pipeline) | drawn in LaTeX (TikZ) | — |
| Fig. 2 (accuracy and ECE vs. pruning) | `results/figures/fig_acc_ece_resnet1d.pdf` | Tier 1 |
| Table III (ResNet-1D, all 20 configurations) | `results/tables/tab_main_resnet1d.tex` | Tier 1 |
| Table IV (WDCNN, all 20 configurations) | `results/tables/tab_main_wdcnn.tex` | Tier 1 |
| Fig. 3 (conformal coverage) | `results/figures/fig_coverage_resnet1d.pdf` | Tier 1 |
| Table V (Jetson latency and energy) | `results/tables/tab_jetson.tex` | Tier 1 (from `jetson_out/bench/bench.csv`) |
| Fig. 4 (reliability diagrams) | `results/figures/fig_reliability_resnet1d.pdf` | Tier 1 |
| Table VI (deployment rules) | `results/tables/tab_selection.tex` | Tier 1 |
| In-text numbers, Sections VI-A to VI-G (incl. fitted temperatures, NLL/Brier, singleton rates, APS) | `results/paper_numbers.md` | Tier 1 |
| Input-representation ablation (FFT: 0.41–0.51; no artificial-damage bearings: 0.63 / 0.55) | `results_ablation_fft/`, `results_ablation_noart/` | Tier 3 (shipped as results only) |
| Hardware and software versions (Section V) | `jetson_out/bench/env.txt`, Section 7 below | — |
| Supplementary (not in the paper): WDCNN figures, Pareto plots, INT8 calibration-data figure | `results/figures/*_wdcnn.pdf`, `fig_pareto_*.pdf`, `fig_int8_calib.pdf` | Tier 1 |

All results use **LAC** conformal scores with α = 0.1 (`analysis.score`, `analysis.alpha`); APS results are
also in `results/cp.csv`.

---

## 3. Three levels of replication

| Tier | What it reproduces | Needs | Time |
|---|---|---|---|
| **1. Re-analysis** | all tables, figures and numbers from the shipped logits | any computer, Python | ~1 min |
| **2. Re-deployment** | features, ONNX export, workstation logits and the complete Jetson stage (engines, logits, latency, energy) from the **shipped checkpoints** | data set (~17 GB), NVIDIA GPU (optional), Jetson Orin Nano | download + workstation: a few hours, mostly download and CPU-bound feature extraction; Jetson ~2–3 h unattended |
| **3. Full retraining** | everything, including training, pruning and the bearing-level calibration models; ablations | data set, NVIDIA GPU, Jetson | workstation: under 1 h per configuration on an RTX 4090 once features exist; Jetson ~2–3 h |

Times are approximate and were observed on the hardware in Section 7. A synthetic-data **self-test** (`bash tools/selftest.sh`, CPU only, ~5 min) runs every stage of Tiers 2–3
end to end on fake data; use it to check an installation before downloading the data set (Section 10).

---

## 4. Data: download the Paderborn bearing data set

The experiments use the **Paderborn University bearing data set** (KAt-DataCenter, Chair of Design and Drive
Technology). It is **not redistributed** in this package; please download it from the original source and
cite it (see below).

* Project page: <https://mb.uni-paderborn.de/en/kat/research/kat-datacenter/bearing-datacenter>
* Download page: <https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download>
* Reference: C. Lessmeier, J. K. Kimotho, D. Zimmer, W. Sextro, "Condition Monitoring of Bearing Damage in
  Electromechanical Drive Systems by Using Motor Current Signals of Electric Motors: A Benchmark Data Set
  for Data-Driven Classification," *Proc. European Conf. of the PHM Society*, vol. 3, 2016.

### 4.1 Bearings used (29 of the 32)

| Role | Bearings | Class |
|---|---|---|
| training / validation / calibration / test (real damage from accelerated lifetime tests) | K001, K002, K003, K004, K005, K006 | healthy |
| | KA04, KA15, KA16, KA22, KA30 | outer race |
| | KI04, KI14, KI16, KI17, KI18, KI21 | inner race |
| **training only** (artificial damage: EDM, drilling, electric engraving) | KA01, KA03, KA05, KA06, KA07, KA08, KA09 | outer race |
| | KI01, KI03, KI05, KI07, KI08 | inner race |

Not used: the combined-damage bearings (KB23, KB24, KB27).
Signal used: `vibration_1` (64 kHz); motor currents are not used.
Operating conditions: all four (`N15_M07_F10`, `N09_M07_F10`, `N15_M01_F10`, `N15_M07_F04`), 20 measurements of 4 s each.

### 4.2 Download

```bash
sudo apt install unrar             # or: p7zip-full, or unar   (needed to unpack the .rar archives)
bash scripts/download_pu.sh        # all 29 bearings -> data/raw/<BEARING>/*.mat
```

* Size: about **17 GB** of `.rar` archives (≈10 GB for the 17 real-damage/healthy bearings, ≈7 GB for the
  12 artificial-damage bearings), plus about the same again once extracted. Delete `data/raw/_rar/`
  afterwards to save space.
* Other location: `DEST=/big/disk/pu bash scripts/download_pu.sh`, then set `data.raw_dir` in `config.yaml`
  (or pass `--set data.raw_dir=/big/disk/pu` to every Python step).
* Subset: `BEARINGS="K001 KA04" bash scripts/download_pu.sh`.
* The script resumes interrupted downloads and skips bearings that are already extracted. Each bearing must
  report **80 .mat files** (4 conditions × 20 measurements).
* If the server URL changes: download the `.rar` file of each bearing manually from the download page into
  `data/raw/_rar/` (e.g. `data/raw/_rar/K001.rar`) and re-run the script; it then only extracts.
  The direct-download base URL can be overridden with `BASE=...`.

Expected layout:
```
data/raw/K001/N15_M07_F10_K001_1.mat … N15_M07_F04_K001_20.mat   (80 files)
data/raw/KA01/…
…
```

**Known defect in the original data:** `KA08/N15_M01_F10_KA08_2.mat` cannot be read. The preprocessing
skips it with a `[WARN missing files [2]]` message; this is expected and is how the paper's results were
produced (KA08 is a training-only bearing).

---

## 5. Tier 2: rebuild features, export and Jetson results from the shipped models

### 5.1 Workstation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124   # or the build for your CUDA / CPU
pip install -r requirements.txt
bash tools/selftest.sh                       # optional, ~5 min on CPU: must end with "SELFTEST OK"
bash scripts/download_pu.sh                  # Section 4

# keep the shipped outputs for comparison
cp -r runs runs_shipped && cp -r jetson_out jetson_out_shipped

bash scripts/run_workstation.sh
```

Because the checkpoints (`runs/*/*/model.pt`) and the bearing-level calibration logits (`runs/*/*/lobo/`)
are shipped, `run_workstation.sh` **skips training, pruning and `tcfd.lobo`** (`[skip] … exists`) and only
(1) computes the order-normalised envelope features into `data/processed/`, and (2) exports, for every fold,
the ONNX models, evaluation windows, INT8 calibration sets and workstation PyTorch logits
(`runs/*/*/logits_ws/`). Compare with the shipped logits:

```bash
python tools/compare_logits.py runs_shipped runs --pattern "*/*/logits_ws/*.npy"
```

A GPU is optional here (export runs on CPU too, more slowly).

### 5.2 Jetson Orin Nano

Prepare the board once (JetPack 6.x; TensorRT comes with JetPack — never `pip install tensorrt`):

```bash
python3 -c "import tensorrt as t; print(t.__version__)"   # must work
ls /usr/src/tensorrt/bin/trtexec                          # used for latency/energy
pip3 install -r requirements-jetson.txt                   # polygraphy, numpy<2
sudo apt install tmux
sudo nvpmodel -m 2                                        # MAXN_SUPER on our Orin Nano Super; ids differ by board/JetPack,
                                                          # list them with: sudo nvpmodel -p --verbose ; check: sudo nvpmodel -q
```

Then:

```bash
# workstation
JETSON=user@<jetson-ip> bash scripts/sync_to_jetson.sh     # copies jetson/ and export/ (~0.6 GB) to ~/tcfd

# Jetson (inside tmux, so an SSH drop does not stop it)
cd ~/tcfd && tmux new -s tcfd
python3 jetson/smoke_test.py      # ~5 min: builds fold-A dense engines, compares with PyTorch -> "SMOKE TEST PASSED"
bash jetson/run_all.sh            # smoke test, 120 engines, logits, benchmark: ~2-3 h
#   detach: Ctrl-b d     re-attach: tmux attach -t tcfd

# workstation
JETSON=user@<jetson-ip> bash scripts/sync_from_jetson.sh   # brings jetson_out/logits, bench, logs back
python tools/compare_logits.py jetson_out_shipped/logits jetson_out/logits --pattern "*/*/*/*.npy"
bash scripts/run_analysis.sh                               # Tier 1 on your own logits and benchmarks
```

`run_all.sh` runs `jetson_clocks` (needs sudo) to fix clocks, builds engines for 2 architectures × 5 pruning
ratios × 4 precisions × 3 folds, dumps logits for all evaluation sets, and benchmarks fold A (latency and
energy depend only on shapes, not weights). Energy is read from `tegrastats` (VDD_IN rail); if the energy
columns of `bench.csv` are NaN, re-run `python3 jetson/bench.py` with `sudo -E`.

Without a Jetson, `--backend ort` runs the same scripts with ONNX Runtime on the CPU (FP32 only; used by the
self-test). This checks the export but does not reproduce TensorRT numerics or Table V.

---

## 6. Tier 3: retrain everything from scratch, ablations

`config_retrain.yaml` is identical to `config.yaml` but writes to `runs_retrain/`, `export_retrain/`,
`jetson_out_retrain/` and `results_retrain/`, so the shipped outputs stay untouched:

```bash
CONFIG=config_retrain.yaml bash scripts/run_workstation.sh                 # reuses data/processed from Tier 2
EXPORT=export_retrain JETSON=user@<ip> bash scripts/sync_to_jetson.sh
#   Jetson:  EXPORT=export_retrain OUT=jetson_out_retrain bash jetson/run_all.sh
OUT=jetson_out_retrain JETSON=user@<ip> bash scripts/sync_from_jetson.sh
CONFIG=config_retrain.yaml bash scripts/run_analysis.sh                    # -> results_retrain/
```

Without a Jetson, `CONFIG=config_retrain.yaml bash scripts/run_analysis.sh` still produces every
workstation-FP32 result (`ws_fp32` rows) and uses a MACs × bytes proxy instead of measured latency.

Run one fold first to check the setup: `CONFIG=config_retrain.yaml bash scripts/run_workstation.sh --folds A`.

### Ablations (input representation, Section VI-A)

| Config | Change w.r.t. `config.yaml` | Paper number | Shipped results |
|---|---|---|---|
| `config_ablation_fft.yaml` | log-magnitude FFT of raw 64 ms windows; real-damage training bearings only; fold A | test bAcc 0.41–0.51 | `results_ablation_fft/` |
| `config_ablation_noart.yaml` | order-normalised envelope; **without** the 12 artificial-damage training bearings | dense ResNet bAcc 0.63 (C0) / 0.55 (shift) | `results_ablation_noart/` |

```bash
CONFIG=config_ablation_fft.yaml   bash scripts/run_workstation.sh --folds A && CONFIG=config_ablation_fft.yaml   bash scripts/run_analysis.sh
CONFIG=config_ablation_noart.yaml bash scripts/run_workstation.sh           && CONFIG=config_ablation_noart.yaml bash scripts/run_analysis.sh
```
These use workstation logits only (no Jetson stage). The random-forest check mentioned in the paper is in
`python tools/diagnose.py -c <config> --fold A` (outputs of the paper run: `logs/paper_run/diagnose_*.txt`).

---

## 7. Hardware and software used for the paper

| | Workstation | Jetson |
|---|---|---|
| Hardware | Intel Core i9-14900KS, NVIDIA RTX 4090 | Jetson Orin Nano Super Developer Kit (8 GB), power mode MAXN_SUPER |
| OS / platform | Linux | Jetson Linux R36.5 (JetPack 6.x), kernel 5.15.185-tegra |
| Key packages | Python 3, PyTorch 2.6.0+cu124, torch-pruning 1.6.1, ONNX opset 17 | TensorRT 10.3.0, Polygraphy 0.53.6, `trtexec` |

The full Jetson environment of the benchmark run is in `jetson_out/bench/env.txt`.
Tier 1 has also been verified to give byte-identical outputs in a fresh CPU-only environment (Python 3.13,
NumPy 2.5, pandas 3.0/3.1).

To record your own environment (recommended when reporting a replication):
```bash
bash tools/capture_env.sh workstation    # -> env/workstation.txt
bash tools/capture_env.sh jetson         # on the board -> env/jetson.txt
```

`env/requirements-workstation.txt` and `env/requirements-jetson.txt` are `pip freeze` snapshots of the exact
package versions used to produce the paper's results (a stricter, fully-pinned complement to the loose
`requirements.txt` / `requirements-jetson.txt` at the repository root, which only pin what matters for
correctness). Installing from the root files is enough for Tiers 1–3; install from the `env/` files instead
(`pip install -r env/requirements-workstation.txt`, `pip3 install -r env/requirements-jetson.txt`) if a
dependency conflict shows up and you want to match our environment exactly.

---

## 8. Expected tolerances

| Tier | What to expect |
|---|---|
| 1 | Identical outputs (checked by `tools/check_results.py`). All steps are deterministic CPU computations on the shipped logits. |
| 2, workstation | Features are deterministic. PyTorch logits on a different GPU, driver or cuDNN version may differ by floating-point noise (max \|Δlogit\| typically ≪ 0.01); label agreement with the shipped logits should be ≥ 0.999. |
| 2, Jetson | Our expectation (not yet measured on a second board): TensorRT chooses kernels (tactics) when it builds an engine, so rebuilt FP16 and especially INT8 engines can differ slightly from ours even on identical hardware. Expect label agreement with the shipped logits of ≈1.000 (FP32), ≥ 0.997 (FP16) and ≥ 0.95 (INT8) per set, and aggregated metrics within about ±0.01. A different TensorRT/JetPack version can change INT8 results more. Latency and energy depend on power mode, clocks, temperature and background load; expect a few percent run-to-run variation, more without `jetson_clocks`. |
| 3 | Seeds are fixed (`seed: 0`), but GPU training is not bit-reproducible (non-deterministic cuDNN kernels), and early stopping on three validation bearings amplifies small differences. Individual per-fold numbers can differ by about the fold-to-fold standard deviations reported in the paper (e.g. 0.02–0.14 for worst-condition coverage); the qualitative findings (compression changes 12–18 % of predictions without lowering accuracy; source-calibrated conformal sets under-cover badly; bearing-level per-condition calibration recovers most of the coverage; FP16/INT8 hardly matter) are what should be compared. |

The Jetson smoke test flags FP32 engines whose logits differ from PyTorch by more than 1e-2 or that flip a
prediction with a clear margin; flips between two almost equal logits (ties within numerical noise) are
reported but are not errors.

---

## 9. Repository layout and what is (not) included

```
config.yaml                 paper configuration (default for every script)
config_retrain.yaml         same, writing to *_retrain/ (Tier 3)
config_ablation_fft.yaml    ablation: FFT input           config_ablation_noart.yaml   ablation: no artificial bearings
tcfd/                       Python package (all workstation and analysis code, see Section 10)
jetson/                     scripts that run on the Jetson (TensorRT engines, logits, benchmark)
scripts/                    download_pu.sh, run_workstation.sh, run_analysis.sh, sync_to/from_jetson.sh
tools/                      selftest.sh, make_synthetic_pu.py, diagnose.py, check_results.py, compare_logits.py,
                            capture_env.sh, fake_precisions.py (self-test only)
runs/<fold>/<model>/        model.pt (checkpoint), info.json (params, MACs, training history),
                            logits_ws/ (PyTorch FP32 logits), lobo/ (out-of-bearing logits for bearing-level CP)
export/                     manifest.json; <fold>/onnx/*.onnx; <fold>/ref/ (PyTorch reference logits for the
                            smoke test); <fold>/data/*_y.npy, *_bearing.npy (labels and bearing ids)
jetson_out/                 logits/<fold>/<model>/<precision>/*.npy (TensorRT logits), bench/bench.csv (latency,
                            energy), bench/env.txt, bench/logs/ (trtexec + tegrastats), build/dump/bench logs
results/                    Tier 1 outputs (Section 1)
results_ablation_*/         ablation results (Section 6)
logs/paper_run/             console logs of the workstation run that produced the shipped artifacts
requirements*.txt           analysis-only / workstation / Jetson dependencies (loose pins)
env/                        workstation.txt, jetson.txt (captured versions, Section 7); requirements-workstation.txt,
                             requirements-jetson.txt (exact pip-freeze pins of the paper run)
LICENSE, CITATION.cff
SHA256SUMS                  checksums of all shipped files:  sha256sum -c SHA256SUMS  (before running anything)
```

Model names: `<arch>_p<ratio>`, e.g. `resnet1d_p85` = ResNet-1D with 85 % of the channels of every
convolution pruned. Folds `A`, `B`, `C` as in Table II. Evaluation sets: `test_C0 … test_C3`
(test bearings), `valfit_*` (validation bearings, measurements 1–10), `valcal_*` (validation bearings,
measurements 11–20).

**Not included, and why:**

| Item | Reason | How to get it |
|---|---|---|
| Raw Paderborn data (`data/raw`) | third-party data, not ours to redistribute | Section 4 |
| Features (`data/processed`), evaluation windows (`export/*/data/<set>.npy`) and INT8 calibration windows (`export/*/calib/`) | derived from the data set; several GB | Tier 2 regenerates them |
| TensorRT engines | specific to the GPU, TensorRT and JetPack version; not portable | rebuilt by `jetson/run_all.sh` |
| Checkpoints and logits of the ablations | not needed for the main results | Tier 3 |

The logs in `logs/paper_run/` were written before the folders were renamed for this package; there,
`runs_ord_art`, `export_ord_art`, `results_ord_art` correspond to `runs`, `export`, `results` here.

---

## 10. Pipeline reference

All Python steps take `-c <config>` and `--set key=value` overrides (e.g. `--set train.device=cpu`); the
workstation steps also take `--folds`, `--archs` and `--force` (redo finished work).

| Step | Command | Output |
|---|---|---|
| features | `python -m tcfd.data_pu -c config.yaml --workers 8` | `data/processed/…` |
| dense models | `python -m tcfd.train -c config.yaml` | `runs/<fold>/<arch>_p00/` |
| structured pruning + fine-tuning (DepGraph, L1) | `python -m tcfd.prune -c config.yaml` | `runs/<fold>/<arch>_p30 … _p85/` |
| bearing-level calibration models (K = 4 groups) | `python -m tcfd.lobo -c config.yaml` | `runs/<fold>/<model>/lobo/` |
| export | `python -m tcfd.export -c config.yaml` | `export/`, `runs/*/*/logits_ws/` |
| Jetson | `bash jetson/run_all.sh` (= `smoke_test.py`, `build_engines.py`, `dump_logits.py`, `bench.py`) | `jetson_out/` |
| metrics + conformal | `python -m tcfd.evaluate -c config.yaml` | `results/metrics.csv`, `results/cp.csv` |
| tables, figures, rules | `python -m tcfd.report -c config.yaml` | `results/tables`, `results/figures`, `selection.csv`, `summary.md` |
| in-text numbers | `python -m tcfd.paper_numbers -c config.yaml` | `results/paper_numbers.md` |

Main settings (`config.yaml`): 1 kHz Butterworth high-pass → Hilbert envelope → decimation ×16 to 4 kHz →
4096-sample windows (75 % overlap for training, none for evaluation) → log-magnitude spectrum resampled to
0–32 shaft orders (1024 bins); AdamW with cosine schedule and class-weighted cross-entropy, up to 30 epochs, early stopping on val-fit/C0 NLL (patience 6, from epoch 8);
pruning ratios 0.3/0.5/0.7/0.85 with 15 fine-tuning epochs; INT8 entropy calibration with 512 class-balanced
val-fit windows; conformal α = 0.1, ECE with 15 bins; deployment-rule tolerance ε = 0.02, max. set size 2.

**Self-test.** `bash tools/selftest.sh [/tmp/dir]` generates a synthetic data set with the same file
structure as Paderborn, then runs preprocessing, training, pruning, `tcfd.lobo`, export, the Jetson scripts
with ONNX Runtime, evaluation, report and paper numbers on the CPU (~5 min). It must end with
`SELFTEST OK`. It verifies the code, not the results.

---

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| `download_pu.sh`: 404 or HTML instead of `.rar` | the server URL changed: download the `.rar` files manually (Section 4.2) into `data/raw/_rar/` |
| `Install unrar, unar or p7zip-full first` | `sudo apt install unrar` (or `p7zip-full`) |
| a bearing reports fewer than 80 `.mat` files | delete its `.rar` in `data/raw/_rar/` and its folder in `data/raw/`, re-run the script |
| `[WARN missing files [2]]` for KA08/C2 | expected, see Section 4.2 |
| `ModuleNotFoundError: tcfd` | run from the repository root, or `export PYTHONPATH=$PWD` |
| `run_workstation.sh` retrains although checkpoints are shipped | the step looks for `runs/<fold>/<model>/model.pt`; check that `train.runs_dir` points to `runs` |
| CUDA out of memory | `--set train.batch_size=128` |
| Jetson: `import tensorrt` fails | use the system Python 3 of JetPack; do not create a venv without `--system-site-packages`, never pip-install TensorRT |
| Jetson: NumPy 2 errors in TensorRT | `pip3 install "numpy<2"` |
| Jetson: smoke test reports `CHECK` for FP32 | real mismatch: check TensorRT version and that `export/` was copied completely; tie-flips marked `ok` are harmless |
| `bench.csv` energy columns are NaN | run `bench.py` with `sudo -E` so `tegrastats` can read the power rails |
| latency much higher than Table V | power mode not MAXN_SUPER or `jetson_clocks` not applied; check `sudo nvpmodel -q` |
| `report`: cost uses a "MACs × bytes proxy" | no `bench.csv` found: Jetson results missing, `Lat.` column is then not in milliseconds |

---

## 12. Citation, licence, contact

Please cite the paper (see `CITATION.cff`) and the data set (Lessmeier et al., 2016) when using this package.

* Code and shipped artifacts: MIT licence (`LICENSE`).
* Paderborn data set: not included; subject to the terms of the KAt-DataCenter, Paderborn University.

Contact: Selma Dilek (selmadilek@hacettepe.edu.tr), Alperen Çakın (cakin.alperen@gmail.com),
Hacettepe University, Ankara, Türkiye.
