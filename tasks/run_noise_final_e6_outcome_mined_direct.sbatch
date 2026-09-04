#!/bin/bash
#SBATCH --job-name=noise_e6_mined
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=data/validation/noise_e6_mined_%A_%a.out
#SBATCH --error=data/validation/noise_e6_mined_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# E6 returns to direct raw-spectrum noise fine-tuning.  Arms 1--3 use the
# already-frozen 882-query S3A+A4 correction-mined action table only in the
# non-held formula partition.  Action outcomes select a training augmentation
# but never enter the loss, sample weight, held evaluation or inference.
case "${SLURM_ARRAY_TASK_ID}" in
  0) SELECT="fixed"         ; VIEWS="2" ; SWEIGHT="2.0" ; SUFFIX="e6_fixed_v2_sw2" ;;
  1) SELECT="fixed"         ; VIEWS="4" ; SWEIGHT="2.0" ; SUFFIX="e6_fixed_v4_sw2" ;;
  2) SELECT="fixed"         ; VIEWS="4" ; SWEIGHT="4.0" ; SUFFIX="e6_fixed_v4_sw4" ;;
  3) SELECT="outcome_mined" ; VIEWS="2" ; SWEIGHT="2.0" ; SUFFIX="e6_mined_v2_sw2" ;;
  4) SELECT="outcome_mined" ; VIEWS="4" ; SWEIGHT="2.0" ; SUFFIX="e6_mined_v4_sw2" ;;
  5) SELECT="outcome_mined" ; VIEWS="4" ; SWEIGHT="4.0" ; SUFFIX="e6_mined_v4_sw4" ;;
  *) echo "Unexpected array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 3 ;;
esac

echo "=== E6 direct outcome-mined noise selection=${SELECT} views=${VIEWS} safety=${SWEIGHT} started: $(date) ==="
python -u tasks/test_noise_final_r0_faithful_s3a.py
python -u tasks/test_noise_final_r1_privileged_teacher.py
python -u tasks/test_noise_final_e4a_direct_augmentation.py
python -u tasks/test_noise_final_e6_outcome_mined_direct.py
python -u tasks/train_noise_final_e4a_direct_augmentation.py \
  --output-root data/validation/g8r_noise_final_e6_outcome_mined_direct \
  --policy curriculum \
  --action-selection "${SELECT}" \
  --outcome-action-dir data/validation/g8r_noise_final_r1_privileged_teacher \
  --action-scope errors \
  --outer-fold 0 \
  --seed 20260828 \
  --epochs 4 \
  --batch-actions 4 \
  --views-per-identity "${VIEWS}" \
  --unfreeze-blocks 1 \
  --backbone-lr 2e-6 \
  --head-lr 1e-5 \
  --safety-ratio 1.0 \
  --safety-stream-weight "${SWEIGHT}" \
  --guided-noise-policy none \
  --guided-noise-weight 0.0 \
  --run-suffix "${SUFFIX}" \
  --device cuda
echo "=== E6 direct outcome-mined noise selection=${SELECT} views=${VIEWS} safety=${SWEIGHT} completed: $(date) ==="
