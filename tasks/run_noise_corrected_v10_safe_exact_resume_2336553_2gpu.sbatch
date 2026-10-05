#!/bin/bash
#SBATCH --job-name=noise_v10_resume
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --output=data/validation/noise_v10_resume_2336553_%j.out
#SBATCH --error=data/validation/noise_v10_resume_2336553_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ALLOCATED_GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1}"
IFS=',' read -r -a ALLOCATED_GPUS <<< "$ALLOCATED_GPU_LIST"
if (( ${#ALLOCATED_GPUS[@]} != 2 )); then
  echo "Expected exactly two allocated GPUs, got CUDA_VISIBLE_DEVICES=$ALLOCATED_GPU_LIST" >&2
  exit 3
fi
GPU_ZERO="${ALLOCATED_GPUS[0]}"
GPU_ONE="${ALLOCATED_GPUS[1]}"

FOLD=0
SEED=20260913
ORIGINAL_JOB_ID=2336553
SOURCE_RUN="data/validation/noise_corrected_best_v5_canary_fold_0_run_2332784"
GRAPH="data/validation/noise_corrected_candidate_graph_v1_20260906"
EVAL_METADATA="data/validation/chemaware_corrected_candidate_manifest_v1"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
OFFICIAL="data/e1/official_embedding_slim.pt"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
INITIAL="data/validation/g8r_noise_final_e8_direct_transfer/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e8_baseline_symmetric_shared/seed_20260830/fold_${FOLD}/final_shared_encoder.pt"
LEDGER="$SOURCE_RUN/ledger"

RUN_ROOT="data/validation/.noise_corrected_v10_safe_exact_canary_fold_${FOLD}_seed_${SEED}_run_${ORIGINAL_JOB_ID}.staging"
FINAL_ROOT="data/validation/noise_corrected_v10_safe_exact_canary_fold_${FOLD}_seed_${SEED}_run_${ORIGINAL_JOB_ID}"
ARMS="$RUN_ROOT/arms"
SUMMARY="$RUN_ROOT/summary"
SNAPSHOT_ROOT="$PWD/$RUN_ROOT/source_snapshot"

[[ -d "$RUN_ROOT" ]] || {
  echo "Missing original V10 staging root; do not silently reconstruct it: $RUN_ROOT" >&2
  exit 4
}
[[ -d "$SNAPSHOT_ROOT" && -f "$SNAPSHOT_ROOT/sha256_manifest.txt" ]] || {
  echo "Missing frozen job-${ORIGINAL_JOB_ID} source snapshot" >&2
  exit 4
}
[[ ! -e "$FINAL_ROOT" ]] || {
  echo "Refusing to overwrite completed V10 result: $FINAL_ROOT" >&2
  exit 5
}

(
  cd "$SNAPSHOT_ROOT"
  sha256sum -c sha256_manifest.txt
  sha256sum -c tasks/noise_best_v10_source_manifest.sha256
)
export PYTHONPATH="$SNAPSHOT_ROOT:$SNAPSHOT_ROOT/tasks"

# This validator runs inside the allocated Slurm job. It does not load the
# checkpoint; it verifies the frozen report and the complete file digest.
arm_is_valid() {
  local ARM_DIR="$1"
  local EXPECTED_ARM="$2"
  local EXPECTED_BRANCH="$3"
  python - "$ARM_DIR" "$EXPECTED_ARM" "$EXPECTED_BRANCH" <<'PY'
import hashlib
import json
import pathlib
import sys

arm_dir = pathlib.Path(sys.argv[1])
expected_arm = sys.argv[2]
expected_branch = sys.argv[3]
checkpoint = arm_dir / "final_shared_encoder.pt"
decision = arm_dir / "decision.json"
held = arm_dir / "held_per_query.csv.gz"
if not all(path.is_file() and path.stat().st_size > 0 for path in (checkpoint, decision, held)):
    raise SystemExit(1)
report = json.loads(decision.read_text(encoding="utf-8"))
if report.get("arm") != expected_arm:
    raise SystemExit(1)
if report.get("corrective_branch_mode") != expected_branch:
    raise SystemExit(1)
if report.get("direct_contract") != "best_action_v10_safe_exact":
    raise SystemExit(1)
digest = hashlib.sha256()
with checkpoint.open("rb") as handle:
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(block)
expected_digest = report.get("provenance", {}).get("final_shared_encoder_sha256")
if not expected_digest or digest.hexdigest() != expected_digest:
    raise SystemExit(1)
PY
}

validate_existing_or_absent() {
  local OUTPUT_NAME="$1"
  local EXPECTED_ARM="$2"
  local EXPECTED_BRANCH="$3"
  local ARM_DIR="$ARMS/$OUTPUT_NAME"
  if [[ -e "$ARM_DIR" ]]; then
    if ! arm_is_valid "$ARM_DIR" "$EXPECTED_ARM" "$EXPECTED_BRANCH"; then
      echo "Existing arm is incomplete or provenance-invalid: $ARM_DIR" >&2
      exit 7
    fi
    echo "Preserving completed arm: $OUTPUT_NAME"
  else
    echo "Arm still missing and will be resumed: $OUTPUT_NAME"
  fi
}

validate_existing_or_absent scalar_transfer_only routed_direct scalar_transfer_only
[[ -d "$ARMS/scalar_transfer_only" ]] || {
  echo "The completed scalar arm from job ${ORIGINAL_JOB_ID} is missing; refusing a fake resume" >&2
  exit 7
}
validate_existing_or_absent full_action_view routed_direct full_action_view
validate_existing_or_absent matched_shuffled shuffled_action_control full_action_view
validate_existing_or_absent clean_control clean_control full_action_view

MISSING_ARMS=0
for OUTPUT_NAME in full_action_view matched_shuffled clean_control; do
  [[ -d "$ARMS/$OUTPUT_NAME" ]] || MISSING_ARMS=$((MISSING_ARMS + 1))
done
AVAILABLE_BYTES=$(df -PB1 "$RUN_ROOT" | awk 'NR == 2 {print $4}')
REQUIRED_BYTES=$((1024 * 1024 * 1024 + MISSING_ARMS * 600 * 1024 * 1024))
echo "V10 resume storage preflight: missing_arms=$MISSING_ARMS available_bytes=$AVAILABLE_BYTES required_bytes=$REQUIRED_BYTES"
df -h "$RUN_ROOT"
df -i "$RUN_ROOT"
quota -s 2>/dev/null || true
if (( AVAILABLE_BYTES < REQUIRED_BYTES )); then
  echo "Insufficient filesystem headroom for atomic V10 checkpoint writes" >&2
  exit 8
fi

COMMON=(
  --graph-dir "$GRAPH" --source-manifest-dir "$EVAL_METADATA"
  --routed-ledger-dir "$LEDGER" --data "$DATA"
  --official-checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCHITECTURE"
  --initial-student-checkpoint "$INITIAL" --outer-fold "$FOLD"
  --formula-fold-seed 20260825 --corrective-objective-mode v3_direct
  --direct-contract best_action_v10_safe_exact
  --corrective-admission strict_top1 --corrective-margin-floor 0.000005
  --corrective-gradient-locality query_action_only
  --calibration-corrective-branch-mode full_action_view
  --arm-invariant-targeted-calibration
  --inner-identity-folds 5 --inner-holdout-fold 0
  --inner-holdout-unit formula
  --development --maximum-corrective-queries 512
  --maximum-risk-queries 1024 --maximum-robust-queries 2048
  --maximum-clean-queries 4096 --outer-held-eval-queries 4096
  --epochs 1 --batch-queries 4 --protective-batch-queries 8
  --positive-references 4 --action-conditioned-positive-references 32
  --action-conditioned-negative-molecules 32 --negative-molecules 8
  --references-per-negative 2 --reference-refresh-every-epochs 0
  --n-highest-peaks 100 --unfreeze-blocks 1
  --backbone-lr 2e-6 --head-lr 1e-5 --weight-decay 1e-4
  --rank-margin 0.05 --temperature 0.10 --topk-negatives 8
  --action-safety-slack 0.005 --v3-harmful-damage-slack 0.005
  --lambda-margin-transfer 1.0 --lambda-action-rank 0.25
  --lambda-action-safety 0.25 --lambda-v3-corrective-consistency 0.25
  --lambda-v3-robust-rank 0.25 --lambda-v3-robust-floor 0.25
  --lambda-v3-harmful-rank 1.0 --lambda-v3-harmful-floor 1.0
  --margin-transfer-fraction 0.50 --margin-transfer-cap 0.10
  --transfer-target-allocation mass_neutral_monotone
  --maximum-transfer-cap-factor 2.0
  --target-v3-payload-to-transfer-ratio 0.25
  --target-v3-consistency-to-transfer-ratio 0.25
  --target-v3-robust-to-transfer-ratio 0.10
  --target-v3-harmful-to-risk-ratio 0.25
  --v3-branch-scale-cap 16 --minimum-v3-internal-combination-retention 0.50
  --minimum-v3-active-transfer-fraction 0.05
  --v3-maximum-protective-microbatches-per-step 4
  --v3-maximum-auxiliary-microbatches-per-step 4
  --v3-maximum-corrective-recycle-factor 4 --v3-corrective-recycle-full-dose
  --lambda-clean-continuation 1.0 --lambda-risk-floor 2.0
  --lambda-preserve 5.0 --risk-margin-slack 0.005
  --calibration-batches 16 --target-corrective-to-risk-ratio 1.0
  --corrective-scale-cap 256 --minimum-corrective-scale 0.5
  --target-preclip-gradient-norm 0.05 --minimum-action-retention-p10 0.50
  --minimum-optimizer-action-alignment-p10 0.05
  --minimum-optimizer-action-attributable-fraction-p10 0.25
  --minimum-corrective-direction-retention-p10 0.10
  --v4-audit-minimum-attributable-fraction 0.25
  --v4-audit-maximum-action-gain 4.0
  --materialize-optimizer-update-restoration
  --optimizer-restoration-scope safe_exact_corrective
  --reconcile-restored-adamw-first-moment
  --optimizer-restoration-minimum-risk-component-retention 0.90
  --minimum-optimizer-restoration-target-reached-fraction 1.0
  --maximum-safe-exact-update-norm-ratio 1.50
  --continue-after-signal-gate-failure
  --optimizer-attribution-steps-per-epoch 64 --progress-every-steps 25
  --maximum-clip-event-fraction 0.10 --grad-clip 1.0
  --eval-batch-size 256 --maximum-spectra-per-action-forward 512
  --bootstrap-resamples 2000 --device cuda --no-amp --seed "$SEED"
)

run_arm_async() {
  local WORKER_GPU="$1"
  local OUTPUT_NAME="$2"
  local ARM="$3"
  local BRANCH_MODE="$4"
  export CUDA_VISIBLE_DEVICES="$WORKER_GPU" OMP_NUM_THREADS=8
  exec python -u "$SNAPSHOT_ROOT/tasks/train_noise_corrected_routed_direct.py" \
    "${COMMON[@]}" --arm "$ARM" \
    --corrective-branch-mode "$BRANCH_MODE" \
    --output-dir "$ARMS/$OUTPUT_NAME"
}

PIDS=()
NAMES=()
if [[ ! -d "$ARMS/full_action_view" ]]; then
  run_arm_async "$GPU_ZERO" full_action_view routed_direct full_action_view &
  PIDS+=("$!")
  NAMES+=("full_action_view")
fi
if [[ ! -d "$ARMS/matched_shuffled" ]]; then
  run_arm_async "$GPU_ONE" matched_shuffled shuffled_action_control full_action_view &
  PIDS+=("$!")
  NAMES+=("matched_shuffled")
fi

PAIR_STATUS=0
for INDEX in "${!PIDS[@]}"; do
  if wait "${PIDS[$INDEX]}"; then
    echo "V10 resume arm completed: ${NAMES[$INDEX]}"
  else
    CURRENT_STATUS=$?
    echo "V10 resume arm failed: ${NAMES[$INDEX]} status=$CURRENT_STATUS" >&2
    PAIR_STATUS=$CURRENT_STATUS
  fi
done
(( PAIR_STATUS == 0 )) || exit 9

validate_existing_or_absent full_action_view routed_direct full_action_view
validate_existing_or_absent matched_shuffled shuffled_action_control full_action_view

if [[ ! -d "$ARMS/clean_control" ]]; then
  # Keep the main batch shell alive: run_arm_async uses exec internally and
  # therefore must always be launched in a background subshell.
  run_arm_async "$GPU_ZERO" clean_control clean_control full_action_view &
  CLEAN_PID=$!
  if wait "$CLEAN_PID"; then
    echo "V10 resume arm completed: clean_control"
  else
    CLEAN_STATUS=$?
    echo "V10 resume arm failed: clean_control status=$CLEAN_STATUS" >&2
    exit 9
  fi
fi
validate_existing_or_absent clean_control clean_control full_action_view
validate_existing_or_absent scalar_transfer_only routed_direct scalar_transfer_only

[[ ! -e "$SUMMARY" ]] || {
  echo "Refusing to overwrite an existing summary: $SUMMARY" >&2
  exit 10
}
python -u "$SNAPSHOT_ROOT/tasks/summarize_noise_corrected_v10_safe_exact_canary.py" \
  --full-action-view-dir "$ARMS/full_action_view" \
  --scalar-transfer-only-dir "$ARMS/scalar_transfer_only" \
  --matched-shuffled-dir "$ARMS/matched_shuffled" \
  --clean-control-dir "$ARMS/clean_control" \
  --bootstrap-resamples 10000 --seed "$SEED" --output-dir "$SUMMARY"

printf 'original_job_id=%s\nrecovery_job_id=%s\n' \
  "$ORIGINAL_JOB_ID" "$SLURM_JOB_ID" > "$RUN_ROOT/recovery_job.txt"
mv "$RUN_ROOT" "$FINAL_ROOT"
echo "=== V10 recovery complete: $FINAL_ROOT ==="
