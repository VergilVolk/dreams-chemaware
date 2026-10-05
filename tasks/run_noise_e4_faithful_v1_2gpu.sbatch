#!/bin/bash
#SBATCH --job-name=noise_e4_faithful_v1
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=data/validation/noise_e4_faithful_v1_%j.out
#SBATCH --error=data/validation/noise_e4_faithful_v1_%j.err

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
PRIMARY_SEED=20260830
REPLICA_SEED=20260829
HISTORICAL_GRAPH="data/validation/g8r_error_atlas_listwise_cache.npz"
HISTORICAL_R0="data/validation/g8r_noise_final_r0_faithful_s3a"
HISTORICAL_CACHE="data/validation/g8r_p2_official_embeddings.npz"
CORRECTED_GRAPH="data/validation/noise_corrected_candidate_graph_v1_20260906/candidate_graph.npz"
CORRECTED_CACHE="data/validation/noise_corrected_candidate_graph_v1_20260906/official_embeddings.npz"
CORRECTED_METADATA="data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
OFFICIAL="data/e1/official_embedding_slim.pt"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
SOURCE_MANIFEST="tasks/noise_e4_faithful_v1_source_manifest.sha256"
EXPECTED_SOURCE_MANIFEST_SHA256="0840f3588b4208c99e291cccce636613fbe77672451ebd356d9b2226f5f3721b"
SBATCH_SOURCE="tasks/run_noise_e4_faithful_v1_2gpu.sbatch"
SPEC_SOURCE="docs/NOISE_E4_FAITHFUL_V1_SPEC_20260914.md"

FINAL_ROOT="data/validation/noise_e4_faithful_v1_fold_${FOLD}_run_${SLURM_JOB_ID}"
RUN_ROOT="data/validation/.noise_e4_faithful_v1_fold_${FOLD}_run_${SLURM_JOB_ID}.staging"
ARMS="$RUN_ROOT/replays"
SOURCE_SNAPSHOT="$RUN_ROOT/source_snapshot"

for REQUIRED in \
  "$HISTORICAL_GRAPH" "$HISTORICAL_R0/report.json" \
  "$HISTORICAL_R0/training_actions.csv.gz" "$HISTORICAL_CACHE" \
  "$CORRECTED_GRAPH" "$CORRECTED_CACHE" "$CORRECTED_METADATA" \
  "$DATA" "$OFFICIAL" "$ARCHITECTURE" "$SOURCE_MANIFEST" \
  "$SBATCH_SOURCE" "$SPEC_SOURCE"; do
  [[ -f "$REQUIRED" ]] || { echo "Missing frozen E4 input: $REQUIRED" >&2; exit 4; }
done

printf '%s  %s\n' "$EXPECTED_SOURCE_MANIFEST_SHA256" "$SOURCE_MANIFEST" \
  | sha256sum -c -
sha256sum -c "$SOURCE_MANIFEST"
printf '%s  %s\n' \
  "5f2340751c7521c5a93114e2b134d5796f157148736ad9162d545b84c11d9f71" \
  "$HISTORICAL_GRAPH" | sha256sum -c -
printf '%s  %s\n' \
  "4f68bfd950d44d02664f67ae9d7ac6700fd3ea33ae0f914f6e623146c57749b6" \
  "$HISTORICAL_R0/report.json" | sha256sum -c -
printf '%s  %s\n' \
  "35d52f13e4441141d622c3d60208e3ae7f11dc554bfc129fef2baa4a1cd27843" \
  "$HISTORICAL_R0/training_actions.csv.gz" | sha256sum -c -
printf '%s  %s\n' \
  "86b7e60194b059b839d266e9e6b52a84b185e5b7160aa934531d4c1c026a5486" \
  "$HISTORICAL_CACHE" | sha256sum -c -
printf '%s  %s\n' \
  "ccda2c4114d9b21413977df03376ca0fc097956a7fa304b861a3154a2b81e64f" \
  "$DATA" | sha256sum -c -
printf '%s  %s\n' \
  "8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245" \
  "$OFFICIAL" | sha256sum -c -
printf '%s  %s\n' \
  "9884b62ecadf4bd441d22fec79b6787e5ffef168e15e7d8d5804dbdea08b38b2" \
  "$ARCHITECTURE" | sha256sum -c -
printf '%s  %s\n' \
  "8a57bb3a9cccdf69a738f2b093bfad8f6fc4393b23f1342c7a035ac54ce58fa1" \
  "$CORRECTED_GRAPH" | sha256sum -c -
printf '%s  %s\n' \
  "18d7632adc67dcda5d650a5c0bc344d8db3878f9e1ff3f8d51a6d4918b96bbda" \
  "$CORRECTED_CACHE" | sha256sum -c -
printf '%s  %s\n' \
  "504ce0a570ac1bc461e76cad81acb3ca3560e58b008ec2d8ff14036e944599f5" \
  "$CORRECTED_METADATA" | sha256sum -c -

mapfile -t SOURCE_FILES < <(awk 'NF == 2 {print $2}' "$SOURCE_MANIFEST")
if (( ${#SOURCE_FILES[@]} < 35 )); then
  echo "Incomplete historical E4 source closure: ${#SOURCE_FILES[@]} files" >&2
  exit 4
fi
[[ ! -e "$RUN_ROOT" && ! -e "$FINAL_ROOT" ]] || {
  echo "Refusing to overwrite unique historical E4 output" >&2
  exit 5
}
mkdir -p "$ARMS" "$SOURCE_SNAPSHOT"

AVAILABLE_KB="$(df -Pk "$PWD" | awk 'NR==2 {print $4}')"
if [[ ! "$AVAILABLE_KB" =~ ^[0-9]+$ ]] || (( AVAILABLE_KB < 6291456 )); then
  echo "Historical E4 requires at least 6 GiB filesystem headroom; available_kb=$AVAILABLE_KB" >&2
  exit 5
fi

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

SNAPSHOT_ROOT="$PWD/$SOURCE_SNAPSHOT"
# Restore the exact historical import names without touching the working-tree
# files that newer experiments still use.
cp "$SNAPSHOT_ROOT/tasks/train_noise_e4_faithful_v1.py" \
  "$SNAPSHOT_ROOT/tasks/train_noise_final_e4a_direct_augmentation.py"
cp "$SNAPSHOT_ROOT/tasks/validate_noise_e4_faithful_v1.py" \
  "$SNAPSHOT_ROOT/tasks/validate_noise_final_e4a_direct_augmentation.py"
cp "$SNAPSHOT_ROOT/tasks/test_noise_e4_faithful_v1.py" \
  "$SNAPSHOT_ROOT/tasks/test_noise_final_e4a_direct_augmentation.py"
cp "$SNAPSHOT_ROOT/tasks/noise_e4_faithful_v1_noise_v3_core.py" \
  "$SNAPSHOT_ROOT/tasks/noise_v3_core.py"
cp "$SNAPSHOT_ROOT/dreams/models/dreams/layers_e4_historical.py" \
  "$SNAPSHOT_ROOT/dreams/models/dreams/layers.py"

(
  cd "$SOURCE_SNAPSHOT"
  sha256sum "${SNAPSHOT_FILES[@]}" > sha256_manifest.txt
  sha256sum -c "$SOURCE_MANIFEST"
)
printf '%s  %s\n' \
  "28c3b375d270fc2030783938d9390710c2c5ab8f0926a0b4ff507d62afa26885" \
  "$SNAPSHOT_ROOT/tasks/train_noise_final_e4a_direct_augmentation.py" | sha256sum -c -
printf '%s  %s\n' \
  "95b0e6a5bcdd0a473dacc63b8b15e8f63d88b0a90ae6612acae77931330973cc" \
  "$SNAPSHOT_ROOT/tasks/validate_noise_final_e4a_direct_augmentation.py" | sha256sum -c -
printf '%s  %s\n' \
  "c20e5a9e9ec2782fa9e94f62d2aa8481c844f0af5d3a2e91975466f1f7a99ab3" \
  "$SNAPSHOT_ROOT/tasks/test_noise_final_e4a_direct_augmentation.py" | sha256sum -c -
printf '%s  %s\n' \
  "a7dd0b07ed2093f0e38b6e36d2645879853cbed9b3d1e0a51c1c7129d43b7997" \
  "$SNAPSHOT_ROOT/tasks/noise_v3_core.py" | sha256sum -c -
printf '%s  %s\n' \
  "ab5e44974e7dac2670d687812e2537e8a41c60248512c1e15e72727987536279" \
  "$SNAPSHOT_ROOT/dreams/models/dreams/layers.py" | sha256sum -c -

export PYTHONPATH="$SNAPSHOT_ROOT:$SNAPSHOT_ROOT/tasks"
PYTHON_SOURCES=()
for SOURCE_FILE in "${SOURCE_FILES[@]}"; do
  [[ "$SOURCE_FILE" == *.py ]] && PYTHON_SOURCES+=("$SNAPSHOT_ROOT/$SOURCE_FILE")
done
python -m py_compile "${PYTHON_SOURCES[@]}" \
  "$SNAPSHOT_ROOT/tasks/train_noise_final_e4a_direct_augmentation.py" \
  "$SNAPSHOT_ROOT/tasks/validate_noise_final_e4a_direct_augmentation.py" \
  "$SNAPSHOT_ROOT/tasks/test_noise_final_e4a_direct_augmentation.py" \
  "$SNAPSHOT_ROOT/tasks/noise_v3_core.py" \
  "$SNAPSHOT_ROOT/dreams/models/dreams/layers.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_final_e4a_direct_augmentation.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_corrected_fullgraph_evaluation.py"
python -u "$SNAPSHOT_ROOT/tasks/test_noise_e4_faithful_v1_sbatch.py"

python -u "$SNAPSHOT_ROOT/tasks/audit_noise_e4_faithful_v1.py" \
  --trainer "$SNAPSHOT_ROOT/tasks/train_noise_final_e4a_direct_augmentation.py" \
  --validator "$SNAPSHOT_ROOT/tasks/validate_noise_final_e4a_direct_augmentation.py" \
  --historical-test "$SNAPSHOT_ROOT/tasks/test_noise_final_e4a_direct_augmentation.py" \
  --noise-v3-core "$SNAPSHOT_ROOT/tasks/noise_v3_core.py" \
  --dreams-layers "$SNAPSHOT_ROOT/dreams/models/dreams/layers.py" \
  --sbatch "$SNAPSHOT_ROOT/$SBATCH_SOURCE" \
  --source-manifest "$SNAPSHOT_ROOT/$SOURCE_MANIFEST" \
  --r0-dir "$HISTORICAL_R0" --graph "$HISTORICAL_GRAPH" \
  --historical-cache "$HISTORICAL_CACHE" --data "$DATA" \
  --official-checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCHITECTURE" \
  --corrected-graph "$CORRECTED_GRAPH" --corrected-cache "$CORRECTED_CACHE" \
  --corrected-metadata "$CORRECTED_METADATA" \
  --runtime-report "$RUN_ROOT/preflight/report.json"

COMMON=(
  --graph "$HISTORICAL_GRAPH"
  --r0-dir "$HISTORICAL_R0"
  --embedding-cache "$HISTORICAL_CACHE"
  --data "$DATA"
  --official-checkpoint "$OFFICIAL"
  --architecture-checkpoint "$ARCHITECTURE"
  --policy curriculum
  --action-scope all
  --outer-fold 0
  --formula-fold-seed 20260825
  --epochs 4
  --batch-actions 4
  --views-per-identity 4
  --positive-spectra 4
  --negative-molecules 8
  --unfreeze-blocks 1
  --backbone-lr 2e-6
  --head-lr 1e-5
  --weight-decay 1e-4
  --rank-margin 0.05
  --temperature 0.10
  --lambda-clean-rank 1.0
  --lambda-aug-rank 1.0
  --lambda-consistency 0.25
  --lambda-margin-floor 2.0
  --lambda-preserve 5.0
  --margin-floor-slack 0.005
  --safety-ratio 1.0
  --grad-clip 1.0
  --run-suffix highlr_multifold
  --device cuda
  --no-amp
)

run_replay() {
  local WORKER_GPU="$1"
  local ROLE="$2"
  local SEED="$3"
  local ROLE_ROOT="$ARMS/$ROLE"
  local RAW_ROOT="$ROLE_ROOT/raw"
  local TRAIN_RESULT="$RAW_ROOT/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_highlr_multifold/seed_${SEED}/fold_${FOLD}"
  export CUDA_VISIBLE_DEVICES="$WORKER_GPU" OMP_NUM_THREADS=8
  mkdir -p "$ROLE_ROOT"
  python -u "$SNAPSHOT_ROOT/tasks/train_noise_final_e4a_direct_augmentation.py" \
    "${COMMON[@]}" --seed "$SEED" --output-root "$RAW_ROOT"
  python -u "$SNAPSHOT_ROOT/tasks/validate_noise_final_e4a_direct_augmentation.py" \
    --output-dir "$TRAIN_RESULT"
  [[ ! -e "$ROLE_ROOT/training" ]] || {
    echo "Refusing to overwrite normalized training output for $ROLE" >&2
    return 9
  }
  mv "$TRAIN_RESULT" "$ROLE_ROOT/training"
  python -u "$SNAPSHOT_ROOT/tasks/evaluate_noise_e4_faithful_v1.py" \
    --graph "$CORRECTED_GRAPH" --source-manifest "$CORRECTED_METADATA" \
    --embedding-cache "$CORRECTED_CACHE" --data "$DATA" \
    --official-checkpoint "$OFFICIAL" --architecture-checkpoint "$ARCHITECTURE" \
    --student-checkpoint "$ROLE_ROOT/training/final_shared_encoder.pt" \
    --training-decision "$ROLE_ROOT/training/decision.json" \
    --output-dir "$ROLE_ROOT/corrected_evaluation" \
    --outer-fold "$FOLD" --formula-fold-seed 20260825 \
    --eval-batch-size 256 --bootstrap-resamples 10000 --device cuda --no-amp
}

launch_replay() {
  local WORKER_GPU="$1"
  local ROLE="$2"
  local SEED="$3"
  local STATUS_FILE="$RUN_ROOT/${ROLE}.status"
  (
    set +e
    (
      set -e
      run_replay "$WORKER_GPU" "$ROLE" "$SEED"
    )
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
  local -a CHILDREN=()
  mapfile -t CHILDREN < <(pgrep -P "$ROOT_PID" 2>/dev/null || true)
  for CHILD_PID in "${CHILDREN[@]}"; do
    terminate_worker_tree "$CHILD_PID"
  done
  kill -TERM "$ROOT_PID" 2>/dev/null || true
}

echo "=== Byte-exact E4: primary seed 20260830 and independent seed 20260829 ==="
launch_replay "$GPU_ZERO" primary 20260830
PID_PRIMARY=$LAST_PID
launch_replay "$GPU_ONE" replica 20260829
PID_REPLICA=$LAST_PID

while [[ ! -f "$RUN_ROOT/primary.status" || ! -f "$RUN_ROOT/replica.status" ]]; do
  if [[ ! -f "$RUN_ROOT/primary.status" ]] && ! kill -0 "$PID_PRIMARY" 2>/dev/null; then
    printf '%s\n' 255 > "$RUN_ROOT/primary.status.tmp"
    mv "$RUN_ROOT/primary.status.tmp" "$RUN_ROOT/primary.status"
  fi
  if [[ ! -f "$RUN_ROOT/replica.status" ]] && ! kill -0 "$PID_REPLICA" 2>/dev/null; then
    printf '%s\n' 255 > "$RUN_ROOT/replica.status.tmp"
    mv "$RUN_ROOT/replica.status.tmp" "$RUN_ROOT/replica.status"
  fi
  for ROLE in primary replica; do
    STATUS_FILE="$RUN_ROOT/${ROLE}.status"
    if [[ -f "$STATUS_FILE" ]] && (( $(<"$STATUS_FILE") != 0 )); then
      terminate_worker_tree "$PID_PRIMARY"
      terminate_worker_tree "$PID_REPLICA"
      wait "$PID_PRIMARY" 2>/dev/null || true
      wait "$PID_REPLICA" 2>/dev/null || true
      echo "Historical E4 $ROLE replay failed with status $(<"$STATUS_FILE")" >&2
      exit "$(<"$STATUS_FILE")"
    fi
  done
  sleep 5
done
wait "$PID_PRIMARY"
wait "$PID_REPLICA"

python -u "$SNAPSHOT_ROOT/tasks/summarize_noise_e4_faithful_v1.py" \
  --primary "$ARMS/primary" --replica "$ARMS/replica" \
  --output-dir "$RUN_ROOT/summary" \
  --bootstrap-resamples 10000 --seed "$PRIMARY_SEED"

(
  cd "$RUN_ROOT"
  find . -type f \
    ! -name 'artifact_manifest.sha256' \
    ! -name 'artifact_manifest.sha256.tmp' \
    -print0 | sort -z | xargs -0 sha256sum > artifact_manifest.sha256.tmp
  mv artifact_manifest.sha256.tmp artifact_manifest.sha256
)
mv "$RUN_ROOT" "$FINAL_ROOT"
echo "=== Byte-exact historical E4 result: $FINAL_ROOT ==="
