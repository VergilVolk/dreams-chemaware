#!/bin/bash
#SBATCH --job-name=noise_e14_pilot
#SBATCH --partition=gpu
#SBATCH --array=0-3
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=18:00:00
#SBATCH --output=data/validation/noise_e14_pilot_%A_%a.out
#SBATCH --error=data/validation/noise_e14_pilot_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

FOLD=4
SEED=20260906
MATURE="data/validation/g8r_noise_final_e4a_direct/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_highlr_multifold/seed_20260830/fold_${FOLD}/final_shared_encoder.pt"
TEACHER_ROOT="data/validation/g8r_noise_final_e14_crossfit_teacher"
OUTPUT_ROOT="data/validation/g8r_noise_final_e14_selected_transfer"
[[ -f "$MATURE" ]] || { echo "Missing mature E4-A checkpoint" >&2; exit 3; }

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    SUFFIX="e14_continuation_control"
    GUIDED_ARGS=(--guided-noise-policy none --guided-noise-weight 0)
    TAG="curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_warm_${SUFFIX}"
    ;;
  1)
    SUFFIX="e14_selected_no_fixed_margin"
    GUIDED_ARGS=(
      --guided-noise-policy selected
      --guided-crossfit-root "$TEACHER_ROOT"
      --guided-noise-weight 1.0
      --guided-noise-ratio 1.0
      --guided-noise-views-per-identity 2
      --guided-transfer-mode symmetric
      --lambda-guided-transfer 0
      --lambda-guided-teacher-margin 0
      --guided-risk-control-ratio 0
    )
    TAG="curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_warm_gpn_selected_gw1_gv2_gtm_symmetric_grp0.67_gmax5_gscope_positive_deficit_errors_gtmargin0_gttargetabsolute_gtdf0.5_grisk0_gbal0_${SUFFIX}"
    ;;
  2)
    SUFFIX="e14_selected_conservative_delta"
    GUIDED_ARGS=(
      --guided-noise-policy selected
      --guided-crossfit-root "$TEACHER_ROOT"
      --guided-noise-weight 1.0
      --guided-noise-ratio 1.0
      --guided-noise-views-per-identity 2
      --guided-transfer-mode symmetric
      --lambda-guided-transfer 0
      --lambda-guided-teacher-margin 1.0
      --guided-teacher-target-mode delta
      --guided-teacher-delta-fraction 0.50
      --guided-teacher-delta-cap 0.20
      --guided-risk-control-ratio 0
    )
    TAG="curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_warm_gpn_selected_gw1_gv2_gtm_symmetric_grp0.67_gmax5_gscope_positive_deficit_errors_gtmargin1_gttargetdelta_gtdf0.5_grisk0_gbal0_${SUFFIX}"
    ;;
  3)
    SUFFIX="e14_selected_conservative_delta_risk"
    GUIDED_ARGS=(
      --guided-noise-policy selected
      --guided-crossfit-root "$TEACHER_ROOT"
      --guided-noise-weight 1.0
      --guided-noise-ratio 1.0
      --guided-noise-views-per-identity 2
      --guided-transfer-mode symmetric
      --lambda-guided-transfer 0
      --lambda-guided-teacher-margin 1.0
      --guided-teacher-target-mode delta
      --guided-teacher-delta-fraction 0.50
      --guided-teacher-delta-cap 0.20
      --guided-risk-control-ratio 1.0
      --guided-auto-balance
    )
    TAG="curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_warm_gpn_selected_gw1_gv2_gtm_symmetric_grp0.67_gmax5_gscope_positive_deficit_errors_gtmargin1_gttargetdelta_gtdf0.5_grisk1_gbal1_${SUFFIX}"
    ;;
  *) echo "Unexpected E14 pilot index ${SLURM_ARRAY_TASK_ID}" >&2; exit 5 ;;
esac

OUTPUT="$OUTPUT_ROOT/$TAG/seed_${SEED}/fold_${FOLD}"
[[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite E14 pilot: $OUTPUT" >&2; exit 6; }
if [[ "${SLURM_ARRAY_TASK_ID}" != 0 ]]; then
  [[ -f "$TEACHER_ROOT/fold_${FOLD}/report.json" ]] || {
    echo "Run and complete E14 teacher build before the pilot" >&2; exit 7;
  }
fi

echo "=== E14 selected-transfer arm=${SLURM_ARRAY_TASK_ID} started: $(date) ==="
python -u tasks/test_noise_final_e4a_direct_augmentation.py
python -u tasks/test_noise_final_e14_crossfit_teacher.py
python -u tasks/validate_noise_final_e14_crossfit_teacher.py \
  --output-dir "$TEACHER_ROOT/fold_${FOLD}"
python -u tasks/train_noise_final_e4a_direct_augmentation.py \
  --output-root "$OUTPUT_ROOT" \
  --initial-student-checkpoint "$MATURE" \
  --policy curriculum \
  --action-selection fixed \
  --action-scope all \
  --direct-transfer-mode symmetric \
  --rank-reference-mode shared \
  --outer-fold "$FOLD" \
  --seed "$SEED" \
  --epochs 2 \
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
echo "=== E14 selected-transfer arm=${SLURM_ARRAY_TASK_ID} completed: $(date) ==="
