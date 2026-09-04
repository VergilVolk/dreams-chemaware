#!/bin/bash
#SBATCH --job-name=noise_final_e4m1b
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=data/validation/noise_final_e4m1b_%j.out
#SBATCH --error=data/validation/noise_final_e4m1b_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

OUTPUT="data/validation/g8r_noise_final_e4_shared_adapter_v2/step_0.060_delta_0.080/seed_20260826/fold_0"
[[ -f data/validation/g8r_noise_final_e4_target_cache/report.json ]] || {
  echo "Missing formal E4-M0 target cache" >&2; exit 3;
}
[[ ! -e "$OUTPUT" ]] || {
  echo "Refusing to overwrite E4-M1 pilot: $OUTPUT" >&2; exit 4;
}

echo "=== final noise E4-M1b gradient-balanced shared-adapter pilot started: $(date) ==="
python -u tasks/test_noise_final_e4_shared_adapter.py
python -u tasks/train_noise_final_e4_shared_adapter.py \
  --outer-fold 0 --seed 20260826 --epochs 5 \
  --batch-per-family 16 --negative-count 4 \
  --target-step 0.06 --delta-bound 0.08 \
  --action-warmup-epochs 1 \
  --lambda-action 1.0 --lambda-rank 0.5 \
  --lambda-safety 4.0 --lambda-preserve 1.0 \
  --output-root data/validation/g8r_noise_final_e4_shared_adapter_v2 \
  --device cuda
python -u tasks/validate_noise_final_e4_shared_adapter.py --output-dir "$OUTPUT"
echo "=== final noise E4-M1b gradient-balanced shared-adapter pilot completed: $(date) ==="
