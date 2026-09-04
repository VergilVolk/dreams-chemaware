#!/bin/bash
#SBATCH --job-name=noise_e13_smoke
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/noise_e13_smoke_%j.out
#SBATCH --error=data/validation/noise_e13_smoke_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

E8="data/validation/g8r_noise_final_e8_direct_transfer/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e8_baseline_symmetric_shared/seed_20260830/fold_0/final_shared_encoder.pt"
E12B="data/validation/g8r_noise_final_e12b_relaxed_recurrence"
ROOT_OUT="data/validation/g8r_noise_final_e13_relaxed_transfer_smoke"
SUFFIX="e13_relax_symmetric_w050_smoke"
TAG="curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_gpn_transfer_gw0.5_gv4_gtm_symmetric_grp0.5_gmax10_gscope_all_${SUFFIX}"
OUTPUT="$ROOT_OUT/$TAG/seed_20260830/fold_0"
[[ -f "$E8" ]] || { echo "Missing mature E8 checkpoint" >&2; exit 3; }
[[ -f "$E12B/report.json" ]] || { echo "Missing formal E12-B report" >&2; exit 4; }
[[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite E13 smoke: $OUTPUT" >&2; exit 5; }

echo "=== E13 relaxed-transfer end-to-end smoke started: $(date) ==="
python -u tasks/test_noise_final_e4a_direct_augmentation.py
python -u tasks/test_noise_final_e13_relaxed_transfer.py
python -u tasks/train_noise_final_e4a_direct_augmentation.py \
  --output-root "$ROOT_OUT" \
  --policy curriculum \
  --action-selection fixed \
  --action-scope all \
  --direct-transfer-mode symmetric \
  --rank-reference-mode shared \
  --outer-fold 0 \
  --seed 20260830 \
  --epochs 4 \
  --batch-actions 4 \
  --views-per-identity 4 \
  --unfreeze-blocks 1 \
  --backbone-lr 2e-6 \
  --head-lr 1e-5 \
  --rank-margin 0.05 \
  --temperature 0.10 \
  --lambda-clean-rank 1.0 \
  --lambda-aug-rank 1.0 \
  --lambda-consistency 0.25 \
  --lambda-margin-floor 2.0 \
  --lambda-preserve 5.0 \
  --margin-floor-slack 0.005 \
  --safety-ratio 1.0 \
  --safety-stream-weight 1.0 \
  --guided-noise-policy transfer \
  --guided-query-scope all \
  --guided-noise-weight 0.5 \
  --guided-noise-ratio 1.0 \
  --guided-noise-views-per-identity 4 \
  --lambda-guided-transfer 0.50 \
  --guided-transfer-mode symmetric \
  --guided-recurrence-prevalence 0.50 \
  --guided-recurrence-max-peaks 10 \
  --guided-action-authorization-dir "$E12B" \
  --guided-reference-checkpoint "$E8" \
  --run-suffix "$SUFFIX" \
  --device cuda \
  --no-amp \
  --smoke
python -u tasks/validate_noise_final_e4a_direct_augmentation.py --output-dir "$OUTPUT"
echo "=== E13 relaxed-transfer end-to-end smoke completed: $(date) ==="
