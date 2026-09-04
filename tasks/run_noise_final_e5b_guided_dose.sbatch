#!/bin/bash
#SBATCH --job-name=noise_e5b_dose
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-4
#SBATCH --output=data/validation/noise_e5b_dose_%A_%a.out
#SBATCH --error=data/validation/noise_e5b_dose_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# E5 showed that the fixed positive-guided action is learnable, but weight 1.0
# overwhelms preservation.  This array changes one factor at a time around the
# exact paired N-only control.  Transfer and joint policies are deliberately
# excluded because neither beat intensity in E5 and their inclusion would
# confound this dose/safety decision.
case "${SLURM_ARRAY_TASK_ID}" in
  0) GUIDED="none"      ; WEIGHT="0.0"  ; SAFETY="1.0" ; SUFFIX="e5b_n_only" ;;
  1) GUIDED="intensity" ; WEIGHT="0.10" ; SAFETY="1.0" ; SUFFIX="e5b_intensity_w010_s1" ;;
  2) GUIDED="intensity" ; WEIGHT="0.25" ; SAFETY="1.0" ; SUFFIX="e5b_intensity_w025_s1" ;;
  3) GUIDED="intensity" ; WEIGHT="0.50" ; SAFETY="1.0" ; SUFFIX="e5b_intensity_w050_s1" ;;
  4) GUIDED="intensity" ; WEIGHT="0.25" ; SAFETY="2.0" ; SUFFIX="e5b_intensity_w025_s2" ;;
  *) echo "Unexpected array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 3 ;;
esac

echo "=== E5-B guided dose policy=${GUIDED} weight=${WEIGHT} safety=${SAFETY} started: $(date) ==="
python -u tasks/test_noise_final_positive_guided_matrix.py
python -u tasks/test_noise_final_positive_peak_transfer.py
python -u tasks/train_noise_final_e4a_direct_augmentation.py \
  --output-root data/validation/g8r_noise_final_e5b_guided_dose \
  --policy curriculum \
  --action-scope all \
  --outer-fold 0 \
  --seed 20260828 \
  --epochs 4 \
  --batch-actions 4 \
  --views-per-identity 4 \
  --unfreeze-blocks 1 \
  --backbone-lr 2e-6 \
  --head-lr 1e-5 \
  --safety-ratio "${SAFETY}" \
  --guided-noise-policy "${GUIDED}" \
  --guided-noise-weight "${WEIGHT}" \
  --guided-noise-ratio 1.0 \
  --guided-noise-views-per-identity 2 \
  --run-suffix "${SUFFIX}" \
  --device cuda
echo "=== E5-B guided dose policy=${GUIDED} weight=${WEIGHT} safety=${SAFETY} completed: $(date) ==="
