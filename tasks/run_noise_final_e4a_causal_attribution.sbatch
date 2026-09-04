#!/bin/bash
#SBATCH --job-name=noise_e4a_causal
#SBATCH --partition=gpu
#SBATCH --array=0-2
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/noise_e4a_causal_%A_%a.out
#SBATCH --error=data/validation/noise_e4a_causal_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1
command -v flock >/dev/null 2>&1 || {
  echo "Required process-lock utility 'flock' is unavailable" >&2
  exit 2
}

ARMS=(clean_duplicate matched_random targeted)
ARM="${ARMS[$SLURM_ARRAY_TASK_ID]}"
SEED=20260828
FOLD=0
RUN_SUFFIX=e4a_causal_v1_20260901
RUN_ROOT=data/validation/g8r_noise_final_e4a_causal_attribution
BASE_TAG=curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05
OUTPUT="$RUN_ROOT/${BASE_TAG}_${RUN_SUFFIX}_causal_${ARM}/seed_${SEED}/fold_${FOLD}"
SUMMARY_ROOT=data/validation/g8r_noise_final_e4a_causal_attribution_summary
SUMMARY_OUTPUT="$SUMMARY_ROOT/$RUN_SUFFIX/seed_${SEED}/fold_${FOLD}"
SUMMARY_LOCK="data/validation/.${RUN_SUFFIX}_summary.lock"

[[ -f data/validation/g8r_noise_final_r0_faithful_s3a/report.json ]] || {
  echo "Missing formal R0 target/matched-control manifest" >&2
  exit 3
}
echo "=== E4-A causal attribution arm=$ARM seed=$SEED fold=$FOLD started: $(date) ==="
python -u tasks/test_noise_final_e4a_direct_augmentation.py
python -u tasks/test_noise_final_e4a_causal_attribution.py
python -u tasks/preflight_noise_final_e4a_causal_attribution.py
if [[ -d "$OUTPUT" ]]; then
  [[ -f "$OUTPUT/decision.json" && -f "$OUTPUT/final_shared_encoder.pt" && -f "$OUTPUT/held_per_query.csv.gz" ]] || {
    echo "Existing causal arm is incomplete; use a new frozen run suffix: $OUTPUT" >&2
    exit 4
  }
  echo "Reusing completed causal arm without overwrite: $OUTPUT"
else
  python -u tasks/train_noise_final_e4a_direct_augmentation.py \
    --output-root "$RUN_ROOT" \
    --causal-arm "$ARM" \
    --policy curriculum --action-selection fixed --action-scope all \
    --outer-fold "$FOLD" --formula-fold-seed 20260825 --seed "$SEED" --epochs 4 \
    --batch-actions 4 --views-per-identity 4 --error-views-per-identity 0 \
    --positive-spectra 4 --negative-molecules 8 \
    --unfreeze-blocks 1 --backbone-lr 2e-6 --head-lr 1e-5 \
    --weight-decay 1e-4 --grad-clip 1.0 --run-suffix "$RUN_SUFFIX" \
    --rank-margin 0.05 --temperature 0.10 \
    --lambda-clean-rank 1.0 --lambda-aug-rank 1.0 \
    --lambda-consistency 0.25 --lambda-margin-floor 2.0 \
    --lambda-preserve 5.0 --margin-floor-slack 0.005 \
    --safety-ratio 1.0 --safety-stream-weight 1.0 \
    --positive-stream-weight 0.0 --guided-noise-policy none \
    --direct-transfer-mode symmetric --rank-reference-mode shared \
    --device cuda --no-amp
fi
python -u tasks/validate_noise_final_e4a_direct_augmentation.py --output-dir "$OUTPUT"
echo "=== E4-A causal attribution arm=$ARM completed: $(date) ==="

# Array member 2 owns summary publication and waits briefly for shared-filesystem
# visibility of the other two completed arms.  A process-level flock is
# released automatically after cancellation or node failure.
CLEAN="$RUN_ROOT/${BASE_TAG}_${RUN_SUFFIX}_causal_clean_duplicate/seed_${SEED}/fold_${FOLD}/decision.json"
RANDOM="$RUN_ROOT/${BASE_TAG}_${RUN_SUFFIX}_causal_matched_random/seed_${SEED}/fold_${FOLD}/decision.json"
TARGET="$RUN_ROOT/${BASE_TAG}_${RUN_SUFFIX}_causal_targeted/seed_${SEED}/fold_${FOLD}/decision.json"
if [[ "$SLURM_ARRAY_TASK_ID" -eq 2 ]]; then
  for _ in $(seq 1 60); do
    [[ -f "$CLEAN" && -f "$RANDOM" && -f "$TARGET" ]] && break
    sleep 5
  done
fi
if [[ "$SLURM_ARRAY_TASK_ID" -eq 2 && -f "$CLEAN" && -f "$RANDOM" && -f "$TARGET" ]]; then
  (
    flock -n 9 || exit 0
    if [[ ! -e "$SUMMARY_OUTPUT" ]]; then
    python -u tasks/summarize_noise_final_e4a_causal_attribution.py \
      --run-root "$RUN_ROOT" --run-suffix "$RUN_SUFFIX" \
      --seed "$SEED" --outer-fold "$FOLD" \
      --bootstrap-resamples 5000 --output-dir "$SUMMARY_ROOT"
    python -u tasks/validate_noise_final_e4a_causal_attribution.py \
      --output-dir "$SUMMARY_OUTPUT"
    echo "=== E4-A causal attribution paired summary completed: $(date) ==="
    fi
  ) 9>"$SUMMARY_LOCK"
fi
if [[ "$SLURM_ARRAY_TASK_ID" -eq 2 && -e "$SUMMARY_OUTPUT" ]]; then
  python -u tasks/validate_noise_final_e4a_causal_attribution.py \
    --output-dir "$SUMMARY_OUTPUT"
fi
