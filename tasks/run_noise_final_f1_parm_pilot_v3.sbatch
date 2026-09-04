#!/bin/bash
#SBATCH --job-name=noise_f1_pv3
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=data/validation/noise_final_f1_parm_v3_%j.out
#SBATCH --error=data/validation/noise_final_f1_parm_v3_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

TOKEN_DIR=data/validation/g8r_noise_final_f1_full_tokens
OUTPUT_ROOT=data/validation/g8r_noise_final_f1_parm_v3
OUTPUT=$OUTPUT_ROOT/seed_20260825/fold_0
[[ -f "$TOKEN_DIR/report.json" ]] || {
  echo "Missing F1 full-token cache: $TOKEN_DIR/report.json" >&2
  exit 3
}
[[ ! -e "$OUTPUT" ]] || {
  echo "Refusing to overwrite F1 v3 pilot: $OUTPUT" >&2
  exit 4
}

echo "=== F1 v3 candidate-margin distillation pilot started: $(date) ==="
python -u tasks/test_noise_final_f1_parm.py
python -u tasks/train_noise_final_f1_parm.py \
  --outer-fold 0 \
  --seed 20260825 \
  --objective teacher_margin \
  --distill-negatives 4 \
  --lambda-teacher 0.10 \
  --lambda-rank 1.0 \
  --lambda-safety 4.0 \
  --lambda-preserve 1.0 \
  --output-root "$OUTPUT_ROOT" \
  --device cuda
echo "=== F1 v3 candidate-margin distillation pilot completed: $(date) ==="
