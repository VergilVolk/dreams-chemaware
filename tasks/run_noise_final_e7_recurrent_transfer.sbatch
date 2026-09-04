#!/bin/bash
#SBATCH --job-name=noise_e7_recur
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-4
#SBATCH --output=data/validation/noise_e7_recur_%A_%a.out
#SBATCH --error=data/validation/noise_e7_recur_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# Direct shared-encoder noise fine-tuning.  The only new action family is the
# fixed, real-positive recurrent_union_mix cell (dose=0.50).  It is encoded by
# the same trainable DreaMS as clean queries and references; inference remains
# clean-spectrum-only.  P2b and every downstream score are forbidden.
case "${SLURM_ARRAY_TASK_ID}" in
  0) GPOLICY="none"     ; GWEIGHT="0.0"   ; SUFFIX="e7_fixed_control" ;;
  1) GPOLICY="transfer" ; GWEIGHT="0.025" ; SUFFIX="e7_recur_w0025" ;;
  2) GPOLICY="transfer" ; GWEIGHT="0.05"  ; SUFFIX="e7_recur_w005" ;;
  3) GPOLICY="transfer" ; GWEIGHT="0.10"  ; SUFFIX="e7_recur_w010" ;;
  4) GPOLICY="transfer" ; GWEIGHT="0.20"  ; SUFFIX="e7_recur_w020" ;;
  *) echo "Unexpected array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 3 ;;
esac

echo "=== E7 recurrent transfer policy=${GPOLICY} weight=${GWEIGHT} started: $(date) ==="
python -u tasks/test_noise_final_r0_faithful_s3a.py
python -u tasks/test_noise_final_e4a_direct_augmentation.py
python -u tasks/test_noise_final_e7_recurrent_transfer.py
python -u tasks/train_noise_final_e4a_direct_augmentation.py \
  --output-root data/validation/g8r_noise_final_e7_recurrent_transfer \
  --policy curriculum \
  --action-selection fixed \
  --action-scope errors \
  --outer-fold 0 \
  --seed 20260829 \
  --epochs 4 \
  --batch-actions 4 \
  --views-per-identity 2 \
  --unfreeze-blocks 1 \
  --backbone-lr 2e-6 \
  --head-lr 1e-5 \
  --safety-ratio 1.0 \
  --safety-stream-weight 2.0 \
  --guided-noise-policy "${GPOLICY}" \
  --guided-noise-weight "${GWEIGHT}" \
  --guided-noise-ratio 1.0 \
  --guided-noise-views-per-identity 2 \
  --run-suffix "${SUFFIX}" \
  --device cuda
echo "=== E7 recurrent transfer policy=${GPOLICY} weight=${GWEIGHT} completed: $(date) ==="
