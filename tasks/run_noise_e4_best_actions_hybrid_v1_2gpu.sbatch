#!/bin/bash
#SBATCH --job-name=noise_e4_best_hybrid1
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=data/validation/noise_e4_best_hybrid1_%j.out
#SBATCH --error=data/validation/noise_e4_best_hybrid1_%j.err

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
SEED=20260830
SOURCE_RUN="data/validation/noise_corrected_best_v5_canary_fold_0_run_2332784"
LEDGER="$SOURCE_RUN/ledger"
GRAPH_DIR="data/validation/noise_corrected_candidate_graph_v1_20260906"
GRAPH="$GRAPH_DIR/candidate_graph.npz"
EMBEDDING_CACHE="$GRAPH_DIR/official_embeddings.npz"
EVAL_METADATA="data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
OFFICIAL="data/e1/official_embedding_slim.pt"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
HISTORICAL_E4_ROOT="data/validation/g8r_noise_final_e4a_direct/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_highlr_multifold/seed_${SEED}/fold_${FOLD}"
HISTORICAL_E4="$HISTORICAL_E4_ROOT/final_shared_encoder.pt"
HISTORICAL_E4_DECISION="$HISTORICAL_E4_ROOT/decision.json"
SOURCE_MANIFEST="tasks/noise_e4_best_actions_hybrid_v1_source_manifest.sha256"
EXPECTED_SOURCE_MANIFEST_SHA256="27abd3964b4131afb274237a20c078d17b40af480051d0a2233bbc46b8a94508"
SBATCH_SOURCE="tasks/run_noise_e4_best_actions_hybrid_v1_2gpu.sbatch"
SPEC_SOURCE="docs/NOISE_E4_BEST_ACTIONS_HYBRID_V1_SPEC_20260914.md"

FINAL_ROOT="data/validation/noise_e4_best_actions_hybrid_v1_fold_${FOLD}_seed_${SEED}_run_${SLURM_JOB_ID}"
RUN_ROOT="data/validation/.noise_e4_best_actions_hybrid_v1_fold_${FOLD}_seed_${SEED}_run_${SLURM_JOB_ID}.staging"
ARMS="$RUN_ROOT/arms"
SOURCE_SNAPSHOT="$RUN_ROOT/source_snapshot"

for REQUIRED in \
  "$GRAPH" "$GRAPH_DIR/report.json" "$EMBEDDING_CACHE" "$EVAL_METADATA" \
  "$DATA" "$OFFICIAL" "$ARCHITECTURE" \
  "$LEDGER/report.json" "$LEDGER/training_actions.csv.gz" "$LEDGER/action_spectra.npz" \
  "$HISTORICAL_E4" "$HISTORICAL_E4_DECISION" \
  "$SOURCE_MANIFEST" "$SBATCH_SOURCE" "$SPEC_SOURCE"; do
  [[ -f "$REQUIRED" ]] || { echo "Missing frozen hybrid input: $REQUIRED" >&2; exit 4; }
done

printf '%s  %s\n' "$EXPECTED_SOURCE_MANIFEST_SHA256" "$SOURCE_MANIFEST" | sha256sum -c -
sha256sum -c "$SOURCE_MANIFEST"
printf '%s  %s\n' \
  "246e7e871e6669fec9f2c62330a69bf9eb563a1b97b72b89732cecd682844349" \
  "$LEDGER/report.json" | sha256sum -c -
printf '%s  %s\n' \
  "93f0785a69b5e323490a0b543059fa213fabb6ff848697831efba5b3fed667aa" \
  "$LEDGER/training_actions.csv.gz" | sha256sum -c -
printf '%s  %s\n' \
  "6d57615aebbfb7bd6327bb7edb143c6837d2586221c61aa2ff45e833f5e1512a" \
  "$LEDGER/action_spectra.npz" | sha256sum -c -
printf '%s  %s\n' \
  "8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245" \
  "$OFFICIAL" | sha256sum -c -
printf '%s  %s\n' \
  "ccda2c4114d9b21413977df03376ca0fc097956a7fa304b861a3154a2b81e64f" \
  "$DATA" | sha256sum -c -
printf '%s  %s\n' \
  "9884b62ecadf4bd441d22fec79b6787e5ffef168e15e7d8d5804dbdea08b38b2" \
  "$ARCHITECTURE" | sha256sum -c -
printf '%s  %s\n' \
  "8ee5b5aa604bda7f1e3cc99396773cebd10749e8820d02c3de048f2865c2509e" \
  "$HISTORICAL_E4" | sha256sum -c -
printf '%s  %s\n' \
  "1d9144a93ca4e4162854cc67491db33612a8ace0e3457e1777ffb45ae5a125bd" \
  "$HISTORICAL_E4_DECISION" | sha256sum -c -
printf '%s  %s\n' \
  "8a57bb3a9cccdf69a738f2b093bfad8f6fc4393b23f1342c7a035ac54ce58fa1" \
  "$GRAPH" | sha256sum -c -
printf '%s  %s\n' \
  "18d7632adc67dcda5d650a5c0bc344d8db3878f9e1ff3f8d51a6d4918b96bbda" \
  "$EMBEDDING_CACHE" | sha256sum -c -
printf '%s  %s\n' \
  "504ce0a570ac1bc461e76cad81acb3ca3560e58b008ec2d8ff14036e944599f5" \
  "$EVAL_METADATA" | sha256sum -c -

mapfile -t SOURCE_FILES < <(awk 'NF == 2 {print $2}' "$SOURCE_MANIFEST")
if (( ${#SOURCE_FILES[@]} < 90 )); then
  echo "Incomplete best-actions hybrid source closure: ${#SOURCE_FILES[@]} files" >&2
  exit 4
fi
[[ ! -e "$RUN_ROOT" && ! -e "$FINAL_ROOT" ]] || {
  echo "Refusing to overwrite unique hybrid output" >&2
  exit 5
}
mkdir -p "$ARMS" "$SOURCE_SNAPSHOT"

AVAILABLE_KB="$(df -Pk "$PWD" | awk 'NR==2 {print $4}')"
if [[ ! "$AVAILABLE_KB" =~ ^[0-9]+$ ]] || (( AVAILABLE_KB < 6291456 )); then
  echo "Hybrid requires at least 6 GiB filesystem headroom; available_kb=$AVAILABLE_KB" >&2
  exit 5
fi

SNAPSHOT_FILES=("${SOURCE_FILES[@]}" "$SOURCE_MANIFEST" "$SBATCH_SOURCE" "$SPEC_SOURCE")
for SOURCE_FILE in "${SNAPSHOT_FILES[@]}"; do
  SNAPSHOT_TARGET="$SOURCE_SNAPSHOT/$SOURCE_FILE"
  mkdir -p "$(dirname "$SNAPSHOT_TARGET")"
  cp "$SOURCE_FILE" "$SNAPSHOT_TARGET"
done
(
  cd "$SOURCE_SNAPSHOT"
  sha256sum -c "$SOURCE_MANIFEST"
)

SNAPSHOT_ROOT="$PWD/$SOURCE_SNAPSHOT"
# Reuse the historical E4 model implementation without changing the working tree.
cp "$SNAPSHOT_ROOT/dreams/models/dreams/layers_e4_historical.py" \
  "$SNAPSHOT_ROOT/dreams/models/dreams/layers.py"
printf '%s  %s\n' \
  "ab5e44974e7dac2670d687812e2537e8a41c60248512c1e15e72727987536279" \
  "$SNAPSHOT_ROOT/dreams/models/dreams/layers.py" | sha256sum -c -
(
  cd "$SOURCE_SNAPSHOT"
  # Record the files that will actually execute, after the deliberate and
  # hash-checked historical E4 layer substitution.
  sha256sum "${SNAPSHOT_FILES[@]}" > executed_source_manifest.sha256
)

export PYTHONPATH="$SNAPSHOT_ROOT:$SNAPSHOT_ROOT/tasks"
PYTHON_SOURCES=()
for SOURCE_FILE in "${SOURCE_FILES[@]}"; do
  [[ "$SOURCE_FILE" == *.py ]] && PYTHON_SOURCES+=("$SNAPSHOT_ROOT/$SOURCE_FILE")
done
python -m py_compile "${PYTHON_SOURCES[@]}"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_final_e4a_direct_augmentation.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_action_injector_v1.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_e4_action_injector_v1_bridge.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_e4_best_actions_hybrid_v1.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_corrected_fullgraph_evaluation.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_e4_best_actions_hybrid_v1_sbatch.py"

python - "$LEDGER/report.json" "$LEDGER/training_actions.csv.gz" "$SNAPSHOT_ROOT/tasks" <<'PY'
import json
import sys
import pandas as pd
sys.path.insert(0, sys.argv[3])
from noise_final_e4_pmt_core import select_materialized_direct_action_panel

report = json.load(open(sys.argv[1], encoding="utf-8"))
actions = pd.read_csv(sys.argv[2], low_memory=False)
selected, strict = select_materialized_direct_action_panel(
    actions, mode="multi_action_balanced", margin_floor=5e-6,
)
expected_sources = {
    "N_mature", "P_guided_original", "E10B", "E11", "E12B",
    "A4_exact", "V4_gradient_path",
}
assert report["status"] == "noise_corrected_routed_action_ledger_complete"
assert report["contracts"]["outer_held_formula_consumed"] is False
assert report["contracts"]["teacher_embedding_target_used"] is False
assert report["contracts"]["P3_consumed"] is False
assert len(selected) == 32114
assert selected.query_index.nunique() == 3482
assert set(selected.source.astype(str)) == expected_sources
assert not selected.action_id.astype(str).duplicated().any()
assert not selected.formula_fold.astype(int).eq(0).any()
print(json.dumps({
    "status": "noise_e4_best_actions_hybrid_v1_preflight_pass",
    "complete_action_views": len(selected),
    "queries": int(selected.query_index.nunique()),
    "sources": sorted(expected_sources),
    "one_best_compression": False,
    "training_initialization": "official_DreaMS",
    "historical_e4_loss": True,
    "static_preclip_loss_scale": 1.0,
    "satisfied_action_rank_gate": False,
    "source_family_loss_reweighting": False,
    "optimizer_boundary": "ActionInjectorV1",
    "pure_e4_control_retrained": False,
}, indent=2))
PY

COMMON=(
  --graph "$GRAPH" --source-manifest "$EVAL_METADATA"
  --embedding-cache "$EMBEDDING_CACHE" --data "$DATA"
  --official-checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCHITECTURE"
  --action-selection materialized_routed --materialized-action-dir "$LEDGER"
  --materialized-injection-mode complete_panel_historical_e4
  --optimizer-boundary-mode action_injector_v1
  --injector-target-attributable-fraction 0.25
  --injector-minimum-protective-retention 0.90
  --injector-maximum-update-norm-ratio 1.50
  --policy curriculum --action-scope all --outer-fold "$FOLD"
  --formula-fold-seed 20260825 --seed "$SEED"
  --epochs 4 --batch-actions 4 --views-per-identity 4
  --error-views-per-identity 0 --positive-spectra 4 --negative-molecules 8
  --unfreeze-blocks 1 --backbone-lr 2e-6 --head-lr 1e-5
  --weight-decay 1e-4 --rank-margin 0.05 --temperature 0.10
  --lambda-clean-rank 1.0 --lambda-aug-rank 1.0
  --lambda-consistency 0.25 --lambda-margin-floor 2.0
  --lambda-preserve 5.0 --margin-floor-slack 0.005
  --safety-ratio 1.0 --safety-stream-weight 1.0
  --positive-stream-weight 0 --guided-noise-policy none --guided-noise-weight 0
  --direct-transfer-mode symmetric --rank-reference-mode shared
  --pmt-arm none --no-candidate-boundary-loss --no-refresh-hard-negatives
  --grad-clip 1.0 --eval-batch-size 256 --bootstrap-resamples 10000
  --device cuda --no-amp
)

result_dir() {
  local ARM="$1"
  local SUFFIX="e4_best_hybrid1_${ARM}_run${SLURM_JOB_ID}"
  local TAG="curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_as_materialized_routed_marm_${ARM}_minj_complete_panel_historical_e4_ob_action_injector_v1_${SUFFIX}"
  echo "$ARMS/$ARM/$TAG/seed_${SEED}/fold_${FOLD}"
}

run_arm() {
  local WORKER_GPU="$1"
  local ARM="$2"
  local SUFFIX="e4_best_hybrid1_${ARM}_run${SLURM_JOB_ID}"
  export CUDA_VISIBLE_DEVICES="$WORKER_GPU" OMP_NUM_THREADS=8
  python -u "$SNAPSHOT_ROOT/tasks/train_noise_final_e4a_direct_augmentation.py" \
    "${COMMON[@]}" --materialized-action-arm "$ARM" \
    --output-root "$ARMS/$ARM" --run-suffix "$SUFFIX"
  python -u "$SNAPSHOT_ROOT/tasks/validate_noise_final_e4a_direct_augmentation.py" \
    --output-dir "$(result_dir "$ARM")"
}

launch_arm() {
  local WORKER_GPU="$1"
  local ARM="$2"
  local STATUS_FILE="$RUN_ROOT/${ARM}.status"
  (
    set +e
    ( set -e; run_arm "$WORKER_GPU" "$ARM" )
    STATUS=$?
    printf '%s\n' "$STATUS" > "${STATUS_FILE}.tmp"
    mv "${STATUS_FILE}.tmp" "$STATUS_FILE"
    exit "$STATUS"
  ) &
  LAST_PID=$!
}

terminate_worker_tree() {
  local ROOT_PID="$1"
  local CHILD_PID
  while IFS= read -r CHILD_PID; do
    [[ -n "$CHILD_PID" ]] || continue
    terminate_worker_tree "$CHILD_PID"
  done < <(pgrep -P "$ROOT_PID" 2>/dev/null || true)
  kill -TERM "$ROOT_PID" 2>/dev/null || true
}

echo "=== Seven-source best actions + historical E4 loss + Injector V1 ==="
launch_arm "$GPU_ZERO" targeted
PID_TARGETED=$LAST_PID
launch_arm "$GPU_ONE" shuffled
PID_SHUFFLED=$LAST_PID

while [[ ! -f "$RUN_ROOT/targeted.status" || ! -f "$RUN_ROOT/shuffled.status" ]]; do
  if [[ ! -f "$RUN_ROOT/targeted.status" ]] && ! kill -0 "$PID_TARGETED" 2>/dev/null; then
    printf '%s\n' 255 > "$RUN_ROOT/targeted.status.tmp"
    mv "$RUN_ROOT/targeted.status.tmp" "$RUN_ROOT/targeted.status"
  fi
  if [[ ! -f "$RUN_ROOT/shuffled.status" ]] && ! kill -0 "$PID_SHUFFLED" 2>/dev/null; then
    printf '%s\n' 255 > "$RUN_ROOT/shuffled.status.tmp"
    mv "$RUN_ROOT/shuffled.status.tmp" "$RUN_ROOT/shuffled.status"
  fi
  for ARM in targeted shuffled; do
    STATUS_FILE="$RUN_ROOT/${ARM}.status"
    if [[ -f "$STATUS_FILE" ]] && (( $(<"$STATUS_FILE") != 0 )); then
      terminate_worker_tree "$PID_TARGETED"
      terminate_worker_tree "$PID_SHUFFLED"
      wait "$PID_TARGETED" 2>/dev/null || true
      wait "$PID_SHUFFLED" 2>/dev/null || true
      echo "Hybrid $ARM failed with status $(<"$STATUS_FILE")" >&2
      exit "$(<"$STATUS_FILE")"
    fi
  done
  sleep 5
done
wait "$PID_TARGETED"
wait "$PID_SHUFFLED"

export CUDA_VISIBLE_DEVICES="$GPU_ZERO" OMP_NUM_THREADS=8
python -u "$SNAPSHOT_ROOT/tasks/evaluate_noise_e4_best_actions_injector_v1_final.py" \
  --graph "$GRAPH" --source-manifest "$EVAL_METADATA" \
  --embedding-cache "$EMBEDDING_CACHE" --data "$DATA" \
  --official-checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCHITECTURE" \
  --hybrid-checkpoint "$(result_dir targeted)/final_shared_encoder.pt" \
  --hybrid-decision "$(result_dir targeted)/decision.json" \
  --shuffled-checkpoint "$(result_dir shuffled)/final_shared_encoder.pt" \
  --shuffled-decision "$(result_dir shuffled)/decision.json" \
  --historical-e4-checkpoint "$HISTORICAL_E4" \
  --historical-e4-decision "$HISTORICAL_E4_DECISION" \
  --output-dir "$RUN_ROOT/summary" --outer-fold "$FOLD" \
  --formula-fold-seed 20260825 --eval-batch-size 256 \
  --bootstrap-resamples 10000 --device cuda --no-amp

(
  cd "$RUN_ROOT"
  find . -type f \
    ! -name 'artifact_manifest.sha256' \
    ! -name 'artifact_manifest.sha256.tmp' \
    -print0 | sort -z | xargs -0 sha256sum > artifact_manifest.sha256.tmp
  mv artifact_manifest.sha256.tmp artifact_manifest.sha256
)
mv "$RUN_ROOT" "$FINAL_ROOT"
echo "=== Best-actions E4+Injector hybrid result: $FINAL_ROOT ==="
