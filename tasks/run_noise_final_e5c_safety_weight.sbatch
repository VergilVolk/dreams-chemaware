#!/bin/bash
#SBATCH --job-name=noise_e5c_safe
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-4
#SBATCH --output=data/validation/noise_e5c_safe_%A_%a.out
#SBATCH --error=data/validation/noise_e5c_safe_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# E5-B established a monotone guided-dose/preservation tradeoff and showed
# that safety-ratio changes coverage but not mean-loss strength.  E5-C is the
# minimal explicit safety-gradient scan around the best guided dose (0.10),
# plus one lower-dose arm.  The N-only arm is repeated for exact pairing.
case "${SLURM_ARRAY_TASK_ID}" in
  0) GUIDED="none"      ; GWEIGHT="0.0"  ; SWEIGHT="1.0" ; SUFFIX="e5c_n_only_sw1" ;;
  1) GUIDED="none"      ; GWEIGHT="0.0"  ; SWEIGHT="2.0" ; SUFFIX="e5c_n_only_sw2" ;;
  2) GUIDED="intensity" ; GWEIGHT="0.10" ; SWEIGHT="2.0" ; SUFFIX="e5c_iw010_sw2" ;;
  3) GUIDED="intensity" ; GWEIGHT="0.10" ; SWEIGHT="4.0" ; SUFFIX="e5c_iw010_sw4" ;;
  4) GUIDED="intensity" ; GWEIGHT="0.05" ; SWEIGHT="2.0" ; SUFFIX="e5c_iw005_sw2" ;;
  *) echo "Unexpected array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 3 ;;
esac

echo "=== E5-C policy=${GUIDED} guided_weight=${GWEIGHT} safety_weight=${SWEIGHT} started: $(date) ==="
python -u tasks/test_noise_final_e4a_direct_augmentation.py
python -u tasks/test_noise_final_positive_guided_matrix.py
python -u tasks/train_noise_final_e4a_direct_augmentation.py \
  --output-root data/validation/g8r_noise_final_e5c_safety_weight \
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
  --safety-ratio 1.0 \
  --safety-stream-weight "${SWEIGHT}" \
  --guided-noise-policy "${GUIDED}" \
  --guided-noise-weight "${GWEIGHT}" \
  --guided-noise-ratio 1.0 \
  --guided-noise-views-per-identity 2 \
  --run-suffix "${SUFFIX}" \
  --device cuda
echo "=== E5-C policy=${GUIDED} guided_weight=${GWEIGHT} safety_weight=${SWEIGHT} completed: $(date) ==="
