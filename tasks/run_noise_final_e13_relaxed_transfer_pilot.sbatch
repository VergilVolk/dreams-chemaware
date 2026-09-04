#!/bin/bash
#SBATCH --job-name=noise_e13_relax
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=data/validation/noise_e13_relax_%A_%a.out
#SBATCH --error=data/validation/noise_e13_relax_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

E8="data/validation/g8r_noise_final_e8_direct_transfer/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e8_baseline_symmetric_shared/seed_20260830/fold_0/final_shared_encoder.pt"
E12B="data/validation/g8r_noise_final_e12b_relaxed_recurrence"
ROOT_OUT="data/validation/g8r_noise_final_e13_relaxed_transfer"
[[ -f "$E8" ]] || { echo "Missing mature E8 checkpoint" >&2; exit 3; }
[[ -f "$E12B/report.json" ]] || { echo "Missing formal E12-B report" >&2; exit 4; }

# Arm 0 is a same-executable reproduction control. Arm 1 isolates the old
# detached transfer graph. Arms 2-4 use E8's symmetric shared-gradient graph
# at three fixed weights. No post-outcome selector or P2b feature enters.
case "${SLURM_ARRAY_TASK_ID}" in
  0) GUIDED="none"     ; MODE="stopgrad" ; WEIGHT="0"    ; SUFFIX="e13_repro_control" ;;
  1) GUIDED="transfer" ; MODE="stopgrad" ; WEIGHT="0.5"  ; SUFFIX="e13_relax_stopgrad_w050" ;;
  2) GUIDED="transfer" ; MODE="symmetric"; WEIGHT="0.25" ; SUFFIX="e13_relax_symmetric_w025" ;;
  3) GUIDED="transfer" ; MODE="symmetric"; WEIGHT="0.5"  ; SUFFIX="e13_relax_symmetric_w050" ;;
  4) GUIDED="transfer" ; MODE="symmetric"; WEIGHT="1"    ; SUFFIX="e13_relax_symmetric_w100" ;;
  *) echo "Unexpected E13 array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 5 ;;
esac

if [[ "$GUIDED" == "none" ]]; then
  TAG="curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_${SUFFIX}"
  GUIDED_ARGS=(--guided-noise-policy none --guided-noise-weight 0)
else
  TAG="curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_gpn_transfer_gw${WEIGHT}_gv4_gtm_${MODE}_grp0.5_gmax10_gscope_all_${SUFFIX}"
  GUIDED_ARGS=(
    --guided-noise-policy transfer
    --guided-query-scope all
    --guided-noise-weight "$WEIGHT"
    --guided-noise-ratio 1.0
    --guided-noise-views-per-identity 4
    --lambda-guided-transfer 0.50
    --guided-transfer-mode "$MODE"
    --guided-recurrence-prevalence 0.50
    --guided-recurrence-max-peaks 10
    --guided-action-authorization-dir "$E12B"
    --guided-reference-checkpoint "$E8"
  )
fi
OUTPUT="$ROOT_OUT/$TAG/seed_20260830/fold_0"
[[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite E13 result: $OUTPUT" >&2; exit 6; }

echo "=== E13 guided=${GUIDED} mode=${MODE} weight=${WEIGHT} started: $(date) ==="
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
  "${GUIDED_ARGS[@]}" \
  --run-suffix "$SUFFIX" \
  --device cuda \
  --no-amp
python -u tasks/validate_noise_final_e4a_direct_augmentation.py --output-dir "$OUTPUT"
echo "=== E13 guided=${GUIDED} mode=${MODE} weight=${WEIGHT} completed: $(date) ==="
