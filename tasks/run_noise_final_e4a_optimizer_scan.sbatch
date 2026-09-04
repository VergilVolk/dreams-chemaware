#!/bin/bash
#SBATCH --job-name=noise_e4a_opt
#SBATCH --partition=gpu
#SBATCH --array=0-5
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=18:00:00
#SBATCH --output=data/validation/noise_e4a_opt_%A_%a.out
#SBATCH --error=data/validation/noise_e4a_opt_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

# One-factor-at-a-time scan around the passing all/views4 E4-A model.
# Arm 5 is the only preregistered interaction and is tested last in the array.
NAME=(low_lr high_lr clip2 clip4 epoch6 high_lr_clip2)
BACKBONE_LR=(5e-7 2e-6 1e-6 1e-6 1e-6 2e-6)
HEAD_LR=(2.5e-6 1e-5 5e-6 5e-6 5e-6 1e-5)
GRAD_CLIP=(1.0 1.0 2.0 4.0 1.0 2.0)
EPOCHS=(4 4 4 4 6 4)
I="${SLURM_ARRAY_TASK_ID}"
SUFFIX="opt_${NAME[$I]}_ep${EPOCHS[$I]}"
TAG="curriculum_all_views4_blocks1_blr_$(printf '%.0e' "${BACKBONE_LR[$I]}")_hlr_$(printf '%.0e' "${HEAD_LR[$I]}")_${SUFFIX}"
OUTPUT="data/validation/g8r_noise_final_e4a_direct/$TAG/seed_20260828/fold_0"

[[ -f data/validation/g8r_noise_final_r0_faithful_s3a/report.json ]] || {
  echo "Missing formal R0 fixed-action manifest" >&2; exit 3;
}
[[ ! -e "$OUTPUT" ]] || {
  echo "Refusing to overwrite optimizer scan result: $OUTPUT" >&2; exit 4;
}

echo "=== E4-A optimizer scan ${NAME[$I]} started: $(date) ==="
python -u tasks/test_noise_final_e4a_direct_augmentation.py
python -u tasks/train_noise_final_e4a_direct_augmentation.py \
  --policy curriculum --action-scope all \
  --outer-fold 0 --seed 20260828 --epochs "${EPOCHS[$I]}" \
  --batch-actions 4 --views-per-identity 4 \
  --positive-spectra 4 --negative-molecules 8 \
  --unfreeze-blocks 1 \
  --backbone-lr "${BACKBONE_LR[$I]}" --head-lr "${HEAD_LR[$I]}" \
  --grad-clip "${GRAD_CLIP[$I]}" --run-suffix "$SUFFIX" \
  --rank-margin 0.05 --temperature 0.10 \
  --lambda-clean-rank 1.0 --lambda-aug-rank 1.0 \
  --lambda-consistency 0.25 --lambda-margin-floor 2.0 \
  --lambda-preserve 5.0 --margin-floor-slack 0.005 \
  --safety-ratio 1.0 --device cuda --no-amp
python -u tasks/validate_noise_final_e4a_direct_augmentation.py --output-dir "$OUTPUT"

echo "=== E4-A optimizer scan ${NAME[$I]} completed: $(date) ==="
