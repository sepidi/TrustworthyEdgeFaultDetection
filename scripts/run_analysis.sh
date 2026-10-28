#!/usr/bin/env bash
# Tier 1 of the replication: metrics, conformal variants, deployment rules, figures, LaTeX tables,
# summary.md and the in-text numbers of the paper, all from the logits shipped in runs/ and jetson_out/.
#   bash scripts/run_analysis.sh                    # config.yaml -> results/
#   CONFIG=config_ablation_noart.yaml bash scripts/run_analysis.sh
# Works with partial results (e.g., workstation logits only, or Jetson logits without bench.csv).
set -euo pipefail
cd "$(dirname "$0")/.."
CFG="${CONFIG:-config.yaml}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p logs
python -m tcfd.evaluate      -c "$CFG" 2>&1 | tee logs/05_evaluate.log
python -m tcfd.report        -c "$CFG" 2>&1 | tee logs/06_report.log
python -m tcfd.paper_numbers -c "$CFG" 2>&1 | tee logs/07_paper_numbers.log
RES=$(python -c "from tcfd.config import load_config, path; c=load_config('$CFG'); print(path(c, c['analysis']['results_dir']))")
echo "Done. See $RES/summary.md, $RES/paper_numbers.md, $RES/tables/, $RES/figures/"
