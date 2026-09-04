#!/bin/bash
#SBATCH --job-name=noise_pn_scan
#SBATCH --partition=gpu
#SBATCH --array=0-3
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=18:00:00
#SBATCH --output=data/validation/noise_pn_scan_%A_%a.out
#SBATCH --error=data/validation/noise_pn_scan_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

WEIGHTS=(0.125 0.25 0.5 1.0)
WEIGHT="${WEIGHTS[$SLURM_ARRAY_TASK_ID]}"
SEED=20260828
FOLD=0
SUFFIX="pn_scan"
TAG="curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_pnw_${WEIGHT}_pv2_${SUFFIX}"
OUTPUT="data/validation/g8r_noise_final_e4a_direct/$TAG/seed_${SEED}/fold_${FOLD}"

[[ -f data/validation/g8r_noise_final_pn_positive_manifest/report.json ]] || {
  echo "Missing frozen P-arm manifest; run run_noise_final_pn_positive_manifest.sbatch first" >&2; exit 5;
}
[[ ! -e "$OUTPUT" ]] || {
  echo "Refusing to overwrite P/N/S scan result: $OUTPUT" >&2; exit 4;
}

echo "=== P/N/S scan weight=$WEIGHT seed=$SEED fold=$FOLD started: $(date) ==="
python -u tasks/test_noise_final_e4a_direct_augmentation.py
python -u tasks/test_noise_final_pn_shared_encoder.py
python -u tasks/train_noise_final_e4a_direct_augmentation.py \
  --policy curriculum --action-scope all \
  --outer-fold "$FOLD" --seed "$SEED" --epochs 4 \
  --batch-actions 4 --views-per-identity 4 \
  --positive-spectra 4 --negative-molecules 8 \
  --positive-stream-weight "$WEIGHT" --positive-ratio 1.0 \
  --positive-views-per-identity 2 \
  --lambda-positive-rank 1.0 --lambda-positive-margin-floor 2.0 \
  --unfreeze-blocks 1 --backbone-lr 2e-6 --head-lr 1e-5 \
  --grad-clip 1.0 --run-suffix "$SUFFIX" \
  --rank-margin 0.05 --temperature 0.10 \
  --lambda-clean-rank 1.0 --lambda-aug-rank 1.0 \
  --lambda-consistency 0.25 --lambda-margin-floor 2.0 \
  --lambda-preserve 5.0 --margin-floor-slack 0.005 \
  --safety-ratio 1.0 --device cuda --no-amp
python -u tasks/validate_noise_final_e4a_direct_augmentation.py --output-dir "$OUTPUT"
echo "=== P/N/S scan weight=$WEIGHT completed: $(date) ==="
