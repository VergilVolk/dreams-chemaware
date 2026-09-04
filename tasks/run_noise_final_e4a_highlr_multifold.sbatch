#!/bin/bash
#SBATCH --job-name=noise_e4a_mf
#SBATCH --partition=gpu
#SBATCH --array=0-14
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=18:00:00
#SBATCH --output=data/validation/noise_e4a_mf_%A_%a.out
#SBATCH --error=data/validation/noise_e4a_mf_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

SEEDS=(20260828 20260829 20260830)
I="${SLURM_ARRAY_TASK_ID}"
SEED_INDEX=$((I / 5))
FOLD=$((I % 5))
SEED="${SEEDS[$SEED_INDEX]}"
SUFFIX="highlr_multifold"
TAG="curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_${SUFFIX}"
OUTPUT="data/validation/g8r_noise_final_e4a_direct/$TAG/seed_${SEED}/fold_${FOLD}"

[[ ! -e "$OUTPUT" ]] || {
  echo "Refusing to overwrite multifold result: $OUTPUT" >&2; exit 4;
}

echo "=== E4-A high-LR multifold seed=$SEED fold=$FOLD started: $(date) ==="
python -u tasks/test_noise_final_e4a_direct_augmentation.py
python -u tasks/train_noise_final_e4a_direct_augmentation.py \
  --policy curriculum --action-scope all \
  --outer-fold "$FOLD" --seed "$SEED" --epochs 4 \
  --batch-actions 4 --views-per-identity 4 \
  --positive-spectra 4 --negative-molecules 8 \
  --unfreeze-blocks 1 --backbone-lr 2e-6 --head-lr 1e-5 \
  --grad-clip 1.0 --run-suffix "$SUFFIX" \
  --rank-margin 0.05 --temperature 0.10 \
  --lambda-clean-rank 1.0 --lambda-aug-rank 1.0 \
  --lambda-consistency 0.25 --lambda-margin-floor 2.0 \
  --lambda-preserve 5.0 --margin-floor-slack 0.005 \
  --safety-ratio 1.0 --device cuda --no-amp
python -u tasks/validate_noise_final_e4a_direct_augmentation.py --output-dir "$OUTPUT"
echo "=== E4-A high-LR multifold seed=$SEED fold=$FOLD completed: $(date) ==="
