#!/bin/bash
# Closing job of the unified GNPS benchmark: merge the landed evaluation
# directories from the three benchmark jobs into the ChemAware paper table
# and six-panel figure.  The CA-1 cross-pipeline consistency gate runs inside
# the plotting code and fails closed on any cross-job disagreement.
#
# Submit with the landed run directories, e.g.:
#   sbatch --export=ALL,\
#EVAL_NOISE=data/validation/noise_gnps_article_benchmark_run_<id>/evaluation,\
#EVAL_CHEMAWARE=data/validation/GLM_chemaware_gnps_unified_benchmark/run_<id>/evaluation,\
#EVAL_PUBLIC=data/validation/GLM_chemaware_gnps_public_model_benchmark/run_<id>/evaluation \
#     tasks/run_GLM_chemaware_gnps_merged_figures.sbatch
#SBATCH --job-name=chem_gnps_fig
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/chem_gnps_fig_%j.out
#SBATCH --error=data/validation/chem_gnps_fig_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

OUT="data/validation/GLM_chemaware_gnps_unified_figures/run_${SLURM_JOB_ID}"

for name in EVAL_NOISE EVAL_CHEMAWARE EVAL_PUBLIC; do
  value="${!name:-}"
  [[ -n "$value" ]] || { echo "GLM_GNPS_FIG_MISSING_ENV: $name is not set" >&2; exit 2; }
  [[ -f "$value/report.json" ]] || { echo "GLM_GNPS_FIG_MISSING_EVAL: $value/report.json" >&2; exit 2; }
  [[ -f "$value/curves_identity_disjoint.csv.gz" ]] || { echo "GLM_GNPS_FIG_MISSING_CURVES: $value" >&2; exit 2; }
done
[[ ! -e "$OUT" ]] || { echo "Output collision: $OUT" >&2; exit 3; }

python -m py_compile tasks/plot_GLM_chemaware_gnps_unified_benchmark.py

python -u tasks/plot_GLM_chemaware_gnps_unified_benchmark.py \
  --evaluation "$EVAL_NOISE" \
  --evaluation "$EVAL_CHEMAWARE" \
  --evaluation "$EVAL_PUBLIC" \
  --baseline official_dreams \
  --output-dir "$OUT"

sha256sum "$OUT/summary_table.csv" \
  "$OUT/main_benchmark_figure.png" \
  "$OUT/main_benchmark_figure.pdf" \
  "$EVAL_NOISE/report.json" \
  "$EVAL_CHEMAWARE/report.json" \
  "$EVAL_PUBLIC/report.json" > "$OUT/SHA256SUMS.txt"

echo "GLM_GNPS_MERGED_FIGURES_COMPLETE: $OUT"
