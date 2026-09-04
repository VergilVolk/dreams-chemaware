#!/bin/bash
#SBATCH --job-name=noise_r2_shared
#SBATCH --partition=gpu
#SBATCH --array=0-2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/noise_r2_shared_%A_%a.out
#SBATCH --error=data/validation/noise_r2_shared_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

BACKBONE_LR=(5e-7 1e-6 2e-6)
HEAD_LR=(2.5e-6 5e-6 1e-5)
I="${SLURM_ARRAY_TASK_ID}"
BLR="${BACKBONE_LR[$I]}"
HLR="${HEAD_LR[$I]}"
TAG="blr_$(printf '%.0e' "$BLR")_hlr_$(printf '%.0e' "$HLR")"
OUTPUT="data/validation/g8r_noise_final_r2_shared_encoder/$TAG/seed_20260827/fold_0"

[[ -f data/validation/g8r_noise_final_r1_privileged_teacher/report.json ]] || {
  echo "Missing formal R1 teacher" >&2; exit 3;
}
[[ -f data/validation/g8r_noise_final_r2_preflight/report.json ]] || {
  echo "Missing passing R2 preflight; submit run_noise_final_r2_preflight.sbatch first" >&2; exit 3;
}
[[ ! -e "$OUTPUT" ]] || {
  echo "Refusing to overwrite R2: $OUTPUT" >&2; exit 4;
}

echo "=== final noise R2 shared encoder $TAG started: $(date) ==="
python -u tasks/test_noise_final_r2_shared_encoder.py
python -u tasks/train_noise_final_r2_shared_encoder.py \
  --outer-fold 0 --seed 20260827 --epochs 4 \
  --batch-identities 4 --positive-spectra 4 --negative-molecules 8 \
  --backbone-lr "$BLR" --head-lr "$HLR" \
  --rank-margin 0.05 --temperature 0.10 \
  --lambda-action-rank 0.75 --lambda-transfer 0.50 \
  --lambda-specificity 0.25 --specificity-margin 0.01 \
  --lambda-robust 0.50 --lambda-protected 1.0 --lambda-preserve 5.0 \
  --device cuda --no-amp
python -u tasks/validate_noise_final_r2_shared_encoder.py --output-dir "$OUTPUT"
echo "=== final noise R2 shared encoder $TAG completed: $(date) ==="
