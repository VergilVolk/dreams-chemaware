#!/bin/bash
#SBATCH --job-name=noise_e8_transfer
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-6
#SBATCH --output=data/validation/noise_e8_transfer_%A_%a.out
#SBATCH --error=data/validation/noise_e8_transfer_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# E8 is a fixed causal factorial.  It does not scan learning rates or action
# weights.  All arms use the previously validated high-LR direct-noise recipe,
# the complete non-held training partition and the same seed/fold/sampler.
case "${SLURM_ARRAY_TASK_ID}" in
  0) POLICY="curriculum" ; TRANSFER="symmetric"                ; REFERENCE="shared"   ; SUFFIX="e8_baseline_symmetric_shared" ;;
  1) POLICY="curriculum" ; TRANSFER="student_action_stopgrad"  ; REFERENCE="shared"   ; SUFFIX="e8_stopgrad_shared" ;;
  2) POLICY="curriculum" ; TRANSFER="official_action"          ; REFERENCE="shared"   ; SUFFIX="e8_officialaction_shared" ;;
  3) POLICY="curriculum" ; TRANSFER="symmetric"                ; REFERENCE="official" ; SUFFIX="e8_symmetric_officialref" ;;
  4) POLICY="curriculum" ; TRANSFER="official_action"          ; REFERENCE="official" ; SUFFIX="e8_officialaction_officialref" ;;
  5) POLICY="candidate"  ; TRANSFER="official_action"          ; REFERENCE="official" ; SUFFIX="e8_candidate_terminal" ;;
  6) POLICY="combined"   ; TRANSFER="official_action"          ; REFERENCE="official" ; SUFFIX="e8_combined_terminal" ;;
  *) echo "Unexpected E8 array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 3 ;;
esac

OUTPUT="data/validation/g8r_noise_final_e8_direct_transfer/${POLICY}_all_views4_blocks1_blr_2e-06_hlr_1e-05_${SUFFIX}/seed_20260830/fold_0"
[[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite E8 result: $OUTPUT" >&2; exit 4; }

echo "=== E8 policy=${POLICY} transfer=${TRANSFER} refs=${REFERENCE} started: $(date) ==="
python -u tasks/test_noise_final_e4a_direct_augmentation.py
python -u tasks/test_noise_final_e8_direct_transfer_factorial.py
python -u tasks/train_noise_final_e4a_direct_augmentation.py \
  --output-root data/validation/g8r_noise_final_e8_direct_transfer \
  --policy "$POLICY" \
  --action-selection fixed \
  --action-scope all \
  --direct-transfer-mode "$TRANSFER" \
  --rank-reference-mode "$REFERENCE" \
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
  --guided-noise-policy none \
  --guided-noise-weight 0.0 \
  --run-suffix "$SUFFIX" \
  --device cuda \
  --no-amp
python -u tasks/validate_noise_final_e4a_direct_augmentation.py --output-dir "$OUTPUT"
echo "=== E8 policy=${POLICY} transfer=${TRANSFER} refs=${REFERENCE} completed: $(date) ==="
