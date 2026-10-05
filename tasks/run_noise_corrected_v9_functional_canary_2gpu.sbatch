#!/bin/bash
#SBATCH --job-name=noise_v9_functional
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/noise_v9_functional_%j.out
#SBATCH --error=data/validation/noise_v9_functional_%j.err

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
SOURCE_RUN="data/validation/noise_corrected_best_v5_canary_fold_0_run_2332784"
V8_ROOT="data/validation/noise_corrected_v8_query_local_canary_fold_0_seed_20260913_run_2336431"
GRAPH="data/validation/noise_corrected_candidate_graph_v1_20260906"
EVAL_METADATA="data/validation/chemaware_corrected_candidate_manifest_v1"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
OFFICIAL="data/e1/official_embedding_slim.pt"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
INITIAL="data/validation/g8r_noise_final_e8_direct_transfer/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e8_baseline_symmetric_shared/seed_20260830/fold_${FOLD}/final_shared_encoder.pt"
LEDGER="$SOURCE_RUN/ledger"
SOURCE_MANIFEST="tasks/noise_best_v9_source_manifest.sha256"
EXPECTED_SOURCE_MANIFEST_SHA256="b745e64945ede85b06dcd0c27e99a8a491a2059f263c8d4b91328846d9f944fd"
SBATCH_SOURCE="tasks/run_noise_corrected_v9_functional_canary_2gpu.sbatch"
SPEC_SOURCE="docs/NOISE_BEST_ACTION_V9_FUNCTIONAL_DIRECT_SPEC_20260913.md"

FINAL_ROOT="data/validation/noise_corrected_v9_functional_canary_fold_${FOLD}_seed_${SEED}_run_${SLURM_JOB_ID}"
RUN_ROOT="data/validation/.noise_corrected_v9_functional_canary_fold_${FOLD}_seed_${SEED}_run_${SLURM_JOB_ID}.staging"
ARMS="$RUN_ROOT/arms"
SUMMARY="$RUN_ROOT/summary"
SOURCE_SNAPSHOT="$RUN_ROOT/source_snapshot"

for REQUIRED in \
  "$GRAPH/candidate_graph.npz" "$GRAPH/report.json" \
  "$EVAL_METADATA/manifest.npz" "$DATA" "$OFFICIAL" "$ARCHITECTURE" \
  "$INITIAL" "$(dirname "$INITIAL")/decision.json" \
  "$LEDGER/training_actions.csv.gz" "$LEDGER/action_spectra.npz" \
  "$LEDGER/report.json" \
  "$V8_ROOT/arms/query_local_routed/decision.json" \
  "$V8_ROOT/arms/query_local_routed/held_per_query.csv.gz" \
  "$V8_ROOT/arms/query_local_routed/development_per_query.csv.gz" \
  "$V8_ROOT/arms/query_local_shuffled/decision.json" \
  "$V8_ROOT/arms/query_local_shuffled/held_per_query.csv.gz" \
  "$V8_ROOT/arms/query_local_shuffled/development_per_query.csv.gz" \
  "$V8_ROOT/arms/clean_control/decision.json" \
  "$V8_ROOT/arms/clean_control/held_per_query.csv.gz" \
  "$V8_ROOT/arms/clean_control/development_per_query.csv.gz"; do
  [[ -f "$REQUIRED" ]] || { echo "Missing frozen V9 input: $REQUIRED" >&2; exit 4; }
done

printf '%s  %s\n' "$EXPECTED_SOURCE_MANIFEST_SHA256" "$SOURCE_MANIFEST" \
  | sha256sum -c -
sha256sum -c "$SOURCE_MANIFEST"
mapfile -t SOURCE_FILES < <(awk 'NF == 2 {print $2}' "$SOURCE_MANIFEST")
if (( ${#SOURCE_FILES[@]} < 65 )); then
  echo "Incomplete V9 source closure: ${#SOURCE_FILES[@]} files" >&2
  exit 4
fi

[[ ! -e "$RUN_ROOT" && ! -e "$FINAL_ROOT" ]] || {
  echo "Refusing to overwrite unique V9 output" >&2
  exit 5
}
mkdir -p "$ARMS" "$SOURCE_SNAPSHOT"
SNAPSHOT_FILES=(
  "${SOURCE_FILES[@]}"
  "$SOURCE_MANIFEST"
  "$SBATCH_SOURCE"
  "$SPEC_SOURCE"
)
for SOURCE_FILE in "${SNAPSHOT_FILES[@]}"; do
  SNAPSHOT_TARGET="$SOURCE_SNAPSHOT/$SOURCE_FILE"
  mkdir -p "$(dirname "$SNAPSHOT_TARGET")"
  cp "$SOURCE_FILE" "$SNAPSHOT_TARGET"
done
(
  cd "$SOURCE_SNAPSHOT"
  sha256sum "${SNAPSHOT_FILES[@]}" > sha256_manifest.txt
  sha256sum -c "$SOURCE_MANIFEST"
)

SNAPSHOT_ROOT="$PWD/$SOURCE_SNAPSHOT"
export PYTHONPATH="$SNAPSHOT_ROOT:$SNAPSHOT_ROOT/tasks"
PYTHON_SOURCES=()
for SOURCE_FILE in "${SOURCE_FILES[@]}"; do
  [[ "$SOURCE_FILE" == *.py ]] && PYTHON_SOURCES+=("$SNAPSHOT_ROOT/$SOURCE_FILE")
done
python -m py_compile "${PYTHON_SOURCES[@]}"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_corrected_signal_v9.py"
python -u "$SNAPSHOT_ROOT/tasks/test_summarize_noise_corrected_v9_functional_canary.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_corrected_best_action_v6.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_corrected_direct_v3_core.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_corrected_direct_v3_trainer.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_corrected_gradient_locality_v8.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_corrected_update_arbitration_v4.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_corrected_fullgraph_evaluation.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_corrected_v9_functional_sbatch.py"

COMMON=(
  --graph-dir "$GRAPH" --source-manifest-dir "$EVAL_METADATA"
  --routed-ledger-dir "$LEDGER" --data "$DATA"
  --official-checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCHITECTURE"
  --initial-student-checkpoint "$INITIAL" --outer-fold "$FOLD"
  --formula-fold-seed 20260825 --corrective-objective-mode v3_direct
  --direct-contract v3 --corrective-admission strict_top1
  --corrective-gradient-locality query_action_only
  --corrective-margin-floor 0.000005 --no-arm-invariant-targeted-calibration
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
  --optimizer-restoration-scope corrective_only
  --reconcile-restored-adamw-first-moment
  --optimizer-restoration-minimum-risk-component-retention 0.90
  --minimum-optimizer-restoration-target-reached-fraction 0.90
  --continue-after-signal-gate-failure
  --optimizer-attribution-steps-per-epoch 64 --progress-every-steps 25
  --maximum-clip-event-fraction 0.10 --grad-clip 1.0
  --eval-batch-size 256 --maximum-spectra-per-action-forward 512
  --bootstrap-resamples 2000 --device cuda --no-amp --seed "$SEED"
)

run_arm_async() {
  local WORKER_GPU="$1"
  local OUTPUT_NAME="$2"
  local BRANCH_MODE="$3"
  export CUDA_VISIBLE_DEVICES="$WORKER_GPU" OMP_NUM_THREADS=8
  exec python -u "$SNAPSHOT_ROOT/tasks/train_noise_corrected_routed_direct.py" \
    "${COMMON[@]}" --arm routed_direct \
    --corrective-branch-mode "$BRANCH_MODE" \
    --output-dir "$ARMS/$OUTPUT_NAME"
}

worker_is_running() {
  jobs -pr | grep -Fxq -- "$1"
}

wait_pair_fail_fast() {
  local PID_FIRST="$1"
  local PID_SECOND="$2"
  while worker_is_running "$PID_FIRST" && worker_is_running "$PID_SECOND"; do
    sleep 2
  done
  local DONE RUNNING STATUS=0
  if worker_is_running "$PID_FIRST"; then
    DONE="$PID_SECOND"
    RUNNING="$PID_FIRST"
  else
    DONE="$PID_FIRST"
    RUNNING="$PID_SECOND"
  fi
  wait "$DONE" || STATUS=$?
  if (( STATUS != 0 )); then
    kill "$RUNNING" 2>/dev/null || true
    wait "$RUNNING" 2>/dev/null || true
    echo "V9 fail-fast: worker failed with status $STATUS" >&2
    exit 6
  fi
  wait "$RUNNING"
}

echo "=== V9 matched-budget branch decomposition on two GPUs ==="
run_arm_async "$GPU_ZERO" full_action_view full_action_view &
PID_A=$!
run_arm_async "$GPU_ONE" scalar_transfer_only scalar_transfer_only &
PID_B=$!
wait_pair_fail_fast "$PID_A" "$PID_B"

echo "=== V9 exact V8 bridge, formula-cluster CIs, and full metric suite ==="
python -u "$SNAPSHOT_ROOT/tasks/summarize_noise_corrected_v9_functional_canary.py" \
  --full-action-view-dir "$ARMS/full_action_view" \
  --scalar-transfer-only-dir "$ARMS/scalar_transfer_only" \
  --v8-root "$V8_ROOT" --bootstrap-resamples 10000 --seed "$SEED" \
  --output-dir "$SUMMARY"

mv "$RUN_ROOT" "$FINAL_ROOT"
echo "=== V9 bounded result preserved at: $FINAL_ROOT ==="
