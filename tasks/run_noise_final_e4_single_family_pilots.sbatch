#!/bin/bash
#SBATCH --job-name=noise_e4_single
#SBATCH --array=0-2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=data/validation/noise_final_e4_single_%A_%a.out
#SBATCH --error=data/validation/noise_final_e4_single_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

FAMILIES=(candidate_gradient acquisition_positive_gradient role_confounder)
FAMILY="${FAMILIES[$SLURM_ARRAY_TASK_ID]}"
ROOT="data/validation/g8r_noise_final_e4_single_family"
CONFIG="${FAMILY}_step_0.060_delta_0.080"
OUTPUT="$ROOT/$CONFIG/seed_20260826/fold_0"

[[ -f data/validation/g8r_noise_final_e4_target_cache/report.json ]] || {
  echo "Missing formal E4-M0 target cache" >&2; exit 3;
}
[[ ! -e "$OUTPUT" ]] || {
  echo "Refusing to overwrite E4 single-family pilot: $OUTPUT" >&2; exit 4;
}

echo "=== final noise E4 single-family $FAMILY started: $(date) ==="
python -u tasks/test_noise_final_e4_shared_adapter.py
python -u tasks/train_noise_final_e4_shared_adapter.py \
  --family "$FAMILY" \
  --outer-fold 0 --seed 20260826 --epochs 5 \
  --batch-per-family 16 --negative-count 4 \
  --target-step 0.06 --delta-bound 0.08 \
  --action-warmup-epochs 1 \
  --lambda-action 1.0 --lambda-rank 0.5 \
  --lambda-safety 4.0 --lambda-preserve 1.0 \
  --output-root "$ROOT" \
  --device cuda
python -u tasks/validate_noise_final_e4_shared_adapter.py --output-dir "$OUTPUT"
echo "=== final noise E4 single-family $FAMILY completed: $(date) ==="
