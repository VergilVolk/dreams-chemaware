#!/bin/bash
#SBATCH --job-name=noise_e5_guided
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=data/validation/noise_e5_guided_%A_%a.out
#SBATCH --error=data/validation/noise_e5_guided_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

case "${SLURM_ARRAY_TASK_ID}" in
  0) GUIDED="none" ;;
  1) GUIDED="transfer" ;;
  2) GUIDED="intensity" ;;
  3) GUIDED="both" ;;
  *) echo "Unexpected array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 3 ;;
esac

echo "=== E5 guided shared-embedding pilot policy=${GUIDED} started: $(date) ==="
python -u tasks/test_noise_final_positive_guided_matrix.py
python -u tasks/test_noise_final_positive_peak_transfer.py
python -u tasks/train_noise_final_e4a_direct_augmentation.py \
  --output-root data/validation/g8r_noise_final_e5_guided_shared \
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
  --guided-noise-policy "${GUIDED}" \
  --guided-noise-weight 1.0 \
  --guided-noise-ratio 1.0 \
  --guided-noise-views-per-identity 2 \
  --run-suffix "e5pilot_${GUIDED}" \
  --device cuda
echo "=== E5 guided shared-embedding pilot policy=${GUIDED} completed: $(date) ==="
