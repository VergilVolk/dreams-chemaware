#!/bin/bash
#SBATCH --job-name=noise_e4a_direct
#SBATCH --partition=gpu
#SBATCH --array=0-2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=18:00:00
#SBATCH --output=data/validation/noise_e4a_direct_%A_%a.out
#SBATCH --error=data/validation/noise_e4a_direct_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

POLICY=(candidate combined combined)
BLOCKS=(1 1 2)
BACKBONE_LR=(1e-6 1e-6 5e-7)
HEAD_LR=(5e-6 5e-6 5e-6)
I="${SLURM_ARRAY_TASK_ID}"
TAG="${POLICY[$I]}_errors_views2_blocks${BLOCKS[$I]}_blr_$(printf '%.0e' "${BACKBONE_LR[$I]}")_hlr_$(printf '%.0e' "${HEAD_LR[$I]}")"
OUTPUT="data/validation/g8r_noise_final_e4a_direct/$TAG/seed_20260827/fold_0"

[[ -f data/validation/g8r_noise_final_r0_faithful_s3a/report.json ]] || {
  echo "Missing formal R0 fixed-action manifest" >&2; exit 3;
}
[[ ! -e "$OUTPUT" ]] || {
  echo "Refusing to overwrite E4-A: $OUTPUT" >&2; exit 4;
}

echo "=== direct noise E4-A $TAG started: $(date) ==="
python -u tasks/test_noise_final_e4a_direct_augmentation.py
python -u tasks/train_noise_final_e4a_direct_augmentation.py \
  --policy "${POLICY[$I]}" --action-scope errors \
  --outer-fold 0 --seed 20260827 --epochs 4 \
  --batch-actions 4 --views-per-identity 2 \
  --positive-spectra 4 --negative-molecules 8 \
  --unfreeze-blocks "${BLOCKS[$I]}" \
  --backbone-lr "${BACKBONE_LR[$I]}" --head-lr "${HEAD_LR[$I]}" \
  --rank-margin 0.05 --temperature 0.10 \
  --lambda-clean-rank 1.0 --lambda-aug-rank 1.0 \
  --lambda-consistency 0.25 --lambda-margin-floor 2.0 \
  --lambda-preserve 5.0 --margin-floor-slack 0.005 \
  --safety-ratio 1.0 --device cuda --no-amp
python -u tasks/validate_noise_final_e4a_direct_augmentation.py --output-dir "$OUTPUT"
echo "=== direct noise E4-A $TAG completed: $(date) ==="
