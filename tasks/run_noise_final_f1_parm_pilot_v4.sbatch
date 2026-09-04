#!/bin/bash
#SBATCH --job-name=noise_f1_pv4
#SBATCH --partition=gpu
#SBATCH --array=0-2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=data/validation/noise_final_f1_parm_v4_%A_%a.out
#SBATCH --error=data/validation/noise_final_f1_parm_v4_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

TOKEN_DIR=data/validation/g8r_noise_final_f1_full_tokens
OUTPUT_ROOT=data/validation/g8r_noise_final_f1_parm_v4
SEEDS=(20260825 20260826 20260827)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT=$OUTPUT_ROOT/seed_${SEED}/fold_0
[[ -f "$TOKEN_DIR/report.json" ]] || {
  echo "Missing F1 full-token cache: $TOKEN_DIR/report.json" >&2
  exit 3
}
[[ ! -e "$OUTPUT" ]] || {
  echo "Refusing to overwrite F1 v4 pilot: $OUTPUT" >&2
  exit 4
}

echo "=== F1 v4 corrected-only rescue/safety pilot started: $(date) ==="
python -u tasks/test_noise_final_f1_parm.py
python -u tasks/train_noise_final_f1_parm.py \
  --outer-fold 0 \
  --seed "$SEED" \
  --objective teacher_margin \
  --teacher-policy corrected_only \
  --safety-ratio 1.0 \
  --distill-negatives 4 \
  --lambda-teacher 0.10 \
  --lambda-rank 1.0 \
  --lambda-safety 4.0 \
  --lambda-preserve 1.0 \
  --output-root "$OUTPUT_ROOT" \
  --device cuda
python -u tasks/validate_noise_final_f1_parm_v4.py --run-dir "$OUTPUT"
echo "=== F1 v4 corrected-only rescue/safety pilot completed: $(date) ==="
