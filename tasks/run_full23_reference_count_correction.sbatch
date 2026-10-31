#!/bin/bash
# One-parameter repair for max-over-reference opportunity bias.
# No spectrum encoding and no upstream model inference occur in this job.
#SBATCH --job-name=full23_refcount
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/full23_refcount_%j.out
#SBATCH --error=data/validation/full23_refcount_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=10

ENVEDA="data/validation/enveda180_cross_condition_benchmark_v3_20261012"
ENVEDA_PRED="data/validation/enveda_no_query_context_diagnostic_run_2413878/modeling/candidate_predictions_oof.npz"
GNPS_ID="data/validation/GLM_gnps_identity_panel_reconstruction/panel_identity_disjoint.npz"
GNPS_FORMULA="data/validation/gnps_gold_silver_10ppm_benchmark_v1/panel_formula_disjoint.npz"
GNPS_SCORES="data/validation/enveda_no_query_context_diagnostic_run_2413878/gnps_fixed_evaluation"
OUT="data/validation/full23_reference_count_correction_run_${SLURM_JOB_ID}"

required=(
  tasks/train_full23_reference_count_correction.py
  "$ENVEDA/panel_identity_disjoint.npz"
  "$ENVEDA/panel_formula_disjoint.npz"
  "$ENVEDA_PRED"
  "$GNPS_ID"
  "$GNPS_FORMULA"
  "$GNPS_SCORES/candidate_scores_identity_disjoint.npz"
  "$GNPS_SCORES/candidate_scores_formula_disjoint.npz"
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "FULL23_REFERENCE_COUNT_MISSING: $path" >&2; exit 4; }
done

python -m py_compile tasks/train_full23_reference_count_correction.py
python -u tasks/train_full23_reference_count_correction.py \
  --enveda-benchmark "$ENVEDA" \
  --enveda-predictions "$ENVEDA_PRED" \
  --gnps-identity-panel "$GNPS_ID" \
  --gnps-formula-panel "$GNPS_FORMULA" \
  --gnps-scores "$GNPS_SCORES" \
  --output "$OUT" \
  --epochs 30 --batch-queries 256 \
  --learning-rate 0.02 --weight-decay 0.001 \
  --residual-bound 0.25 --bootstrap-resamples 5000 --seed 20261031

echo "FULL23_REFERENCE_COUNT_CORRECTION_COMPLETE_NO_EXTERNAL_ACCESS: $OUT"
