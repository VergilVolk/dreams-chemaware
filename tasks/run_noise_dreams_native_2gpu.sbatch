#!/bin/bash
#SBATCH --job-name=noise_dreams_native
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=168:00:00
#SBATCH --output=data/validation/noise_dreams_native_%j.out
#SBATCH --error=data/validation/noise_dreams_native_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

ALLOCATED_GPU_LIST="${CUDA_VISIBLE_DEVICES:-}"
IFS=',' read -r -a ALLOCATED_GPUS <<< "$ALLOCATED_GPU_LIST"
if (( ${#ALLOCATED_GPUS[@]} != 2 )); then
  echo "Expected exactly two allocated GPUs, got CUDA_VISIBLE_DEVICES=$ALLOCATED_GPU_LIST" >&2
  exit 3
fi

FOLD=0
LEDGER="data/validation/noise_corrected_best_v5_canary_fold_0_run_2332784/ledger"
GRAPH_DIR="data/validation/noise_corrected_candidate_graph_v1_20260906"
GRAPH="$GRAPH_DIR/candidate_graph.npz"
EMBEDDING_CACHE="$GRAPH_DIR/official_embeddings.npz"
SOURCE_MANIFEST="data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
OFFICIAL_SLIM="data/e1/official_embedding_slim.pt"
E8_ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
MATURE_E8="data/validation/g8r_noise_final_e8_direct_transfer/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e8_baseline_symmetric_shared/seed_20260830/fold_0/final_shared_encoder.pt"
RESUME_ROOT="${RESUME_ROOT:-}"
if [[ -n "$RESUME_ROOT" ]]; then
  echo "Noise reference-aligned native v3 requires a fresh run; unset RESUME_ROOT" >&2
  exit 4
fi
RUN_ROOT="data/validation/noise_dreams_native_fold_${FOLD}_run_${SLURM_JOB_ID}"
TRIPLETS="$RUN_ROOT/triplets"
LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
LOCAL_ROOT="$LOCAL_BASE/noise_dreams_native_${SLURM_JOB_ID}"
ARMS="$LOCAL_ROOT/arms"
EVALUATION="$RUN_ROOT/evaluation"

[[ ! -e "$RUN_ROOT" ]] || { echo "Output collision: $RUN_ROOT" >&2; exit 4; }
mkdir -p "$ARMS"
cleanup_local() {
  rm -rf -- "$LOCAL_ROOT"
}
trap cleanup_local EXIT
LOCAL_FREE_KB=$(df -Pk "$LOCAL_BASE" | awk 'NR==2 {print $4}')
if [[ -z "$LOCAL_FREE_KB" ]] || (( LOCAL_FREE_KB < 3145728 )); then
  echo "Node-local checkpoint space below 3 GiB: base=$LOCAL_BASE free_kb=${LOCAL_FREE_KB:-unknown}" >&2
  exit 4
fi
echo "Node-local checkpoints: $ARMS free_kb=$LOCAL_FREE_KB"

required_files=(
  "$LEDGER/report.json"
  "$LEDGER/training_actions.csv.gz"
  "$LEDGER/action_spectra.npz"
  "$GRAPH"
  "$EMBEDDING_CACHE"
  "$SOURCE_MANIFEST"
  "$DATA"
  "$OFFICIAL_SLIM"
  "$E8_ARCHITECTURE"
  "$MATURE_E8"
  tasks/build_noise_reference_aligned_native_triplets.py
  tasks/build_noise_dreams_native_triplets.py
  tasks/noise_corrected_shuffled_control_v3.py
  tasks/noise_dreams_native_spectrum.py
  tasks/train_noise_reference_native.py
  tasks/evaluate_noise_dreams_native.py
  tasks/summarize_noise_dreams_native_replicates.py
  tasks/noise_corrected_fullgraph_evaluation.py
  tasks/test_noise_reference_aligned_native_triplets.py
  tasks/test_noise_dreams_native.py
  tasks/test_noise_reference_native_runtime.py
)
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || {
    echo "Missing native Noise input: $required_file" >&2
    exit 5
  }
done

grep -Fq 'VERSION = "noise_reference_aligned_native_v3"' \
  tasks/build_noise_reference_aligned_native_triplets.py || {
  echo "Stale Noise reference-aligned native triplet adapter" >&2
  exit 5
}
grep -Fq 'NATIVE_EVALUATION_BASELINE_VERSION = "official_slim_reencoded_v3"' \
  tasks/evaluate_noise_dreams_native.py || {
  echo "Stale native evaluator: official slim baseline is not re-encoded" >&2
  exit 5
}
# All tests execute inside the Slurm allocation, never on a login node.
python -m py_compile \
  tasks/build_noise_reference_aligned_native_triplets.py \
  tasks/build_noise_dreams_native_triplets.py \
  tasks/noise_corrected_shuffled_control_v3.py \
  tasks/noise_dreams_native_spectrum.py \
  tasks/train_noise_reference_native.py \
  tasks/evaluate_noise_dreams_native.py \
  tasks/summarize_noise_dreams_native_replicates.py \
  tasks/test_noise_reference_aligned_native_triplets.py \
  tasks/test_noise_dreams_native.py \
  tasks/test_noise_reference_native_runtime.py
python -u tasks/test_noise_reference_aligned_native_triplets.py
python -u tasks/test_noise_reference_native_runtime.py

# Noise only selects identity-valid false candidate molecules. Pool arrays use
# the native DreaMS five-field contrastive interface; action tensors never
# enter the model or loss.
python -u tasks/build_noise_reference_aligned_native_triplets.py \
  --ledger-dir "$LEDGER" \
  --graph "$GRAPH" \
  --embedding-cache "$EMBEDDING_CACHE" \
  --data "$DATA" \
  --output "$TRIPLETS" \
  --outer-fold "$FOLD" \
  --formula-fold-seed 20260825 \
  --validation-folds 10 --validation-fold 0 --validation-seed 20260920 \
  --margin 0.1 \
  --expected-actions 32114 \
  --base-queries-per-formula 1 \
  --negative-references-per-action-candidate 2 \
  --positive-references-per-negative 2 \
  --minimum-train-formulas 800 \
  --minimum-triplet-events 6500 \
  --minimum-active-events 4000 \
  --minimum-exact-action-events 1000 \
  --minimum-active-action-fraction 0.50

TRAIN_EVENTS=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["targeted_curriculum"]["triplet_events"])' "$TRIPLETS/report.json")
TRAIN_BATCHES=$(( (TRAIN_EVENTS + 3) / 4 ))
TRAIN_STEPS=3000
CHECKPOINT_INTERVAL="$TRAIN_STEPS"
TRAIN_CHECKPOINT="step-003000.ckpt"
if (( TRAIN_EVENTS < 6500 || TRAIN_EVENTS > 7500 )); then
  echo "Native query-pool curriculum outside frozen dose window: events=$TRAIN_EVENTS expected=6500..7500" >&2
  exit 5
fi
echo "Native relation curriculum: events=$TRAIN_EVENTS batches/pass=$TRAIN_BATCHES fixed_steps=$TRAIN_STEPS interval=$CHECKPOINT_INTERVAL checkpoint=$TRAIN_CHECKPOINT"

run_arm() {
  local gpu="$1"
  local arm="$2"
  local seed="$3"
  CUDA_VISIBLE_DEVICES="$gpu" python -u tasks/train_noise_reference_native.py \
    --data "$DATA" \
    --train-pool "$TRIPLETS/train_pool_targeted.npz" \
    --val-pool "$TRIPLETS/val_pool.npz" \
    --official-checkpoint "$OFFICIAL_SLIM" \
    --architecture-checkpoint "$E8_ARCHITECTURE" \
    --output "$ARMS/$arm" \
    --seed "$seed" \
    --lr 5e-6 --weight-decay 0 \
    --triplet-loss-margin 0.1 \
    --batch-size 4 --num-workers 0 \
    --max-epochs 2 --max-steps "$TRAIN_STEPS" \
    --checkpoint-mode fixed_steps --save-every-n-steps "$CHECKPOINT_INTERVAL" \
    --checkpoint-save-weights-only --no-save-last-checkpoint \
    --keep-final-partial-batch \
    --n-highest-peaks 100 --device cuda \
    >"$RUN_ROOT/${arm}.out" 2>"$RUN_ROOT/${arm}.err"
}

run_arm "${ALLOCATED_GPUS[0]}" primary 3407 &
PID_PRIMARY=$!
run_arm "${ALLOCATED_GPUS[1]}" replicate 3408 &
PID_REPLICATE=$!

echo "=== Noise-owned DreaMS native runtime; fixed 3,000-step budget ==="
arm_heartbeat() {
  while kill -0 "$PID_PRIMARY" 2>/dev/null || kill -0 "$PID_REPLICATE" 2>/dev/null; do
    local primary_alive=0
    local replicate_alive=0
    kill -0 "$PID_PRIMARY" 2>/dev/null && primary_alive=1
    kill -0 "$PID_REPLICATE" 2>/dev/null && replicate_alive=1
    echo "[native-train-heartbeat] $(date) primary_alive=$primary_alive replicate_alive=$replicate_alive"
    sleep 120
  done
}
arm_heartbeat &
PID_HEARTBEAT=$!

set +e
wait "$PID_PRIMARY"
STATUS_PRIMARY=$?
wait "$PID_REPLICATE"
STATUS_REPLICATE=$?
kill "$PID_HEARTBEAT" 2>/dev/null
wait "$PID_HEARTBEAT" 2>/dev/null
set -e

if (( STATUS_PRIMARY != 0 || STATUS_REPLICATE != 0 )); then
  echo "=== primary.err tail ===" >&2
  tail -n 120 "$RUN_ROOT/primary.err" >&2 || true
  echo "=== replicate.err tail ===" >&2
  tail -n 120 "$RUN_ROOT/replicate.err" >&2 || true
  echo "Native DreaMS seed failure: primary=$STATUS_PRIMARY replicate=$STATUS_REPLICATE" >&2
  exit 6
fi

for arm in primary replicate; do
  [[ -f "$ARMS/$arm/$TRAIN_CHECKPOINT" ]] || {
    echo "Native arm $arm produced no fixed $TRAIN_CHECKPOINT" >&2
    exit 7
  }
  [[ -f "$ARMS/$arm/report.json" ]] || {
    echo "Native arm $arm produced no report.json" >&2
    exit 7
  }
done

evaluate_arm() {
  local gpu="$1"
  local arm="$2"
  CUDA_VISIBLE_DEVICES="$gpu" python -u tasks/evaluate_noise_dreams_native.py \
    --graph "$GRAPH" \
    --embedding-cache "$EMBEDDING_CACHE" \
    --official-slim-checkpoint "$OFFICIAL_SLIM" \
    --source-manifest "$SOURCE_MANIFEST" \
    --data "$DATA" \
    --checkpoint "$ARMS/$arm/$TRAIN_CHECKPOINT" \
    --architecture-checkpoint "$E8_ARCHITECTURE" \
    --output "$EVALUATION/$arm" \
    --arm "$arm" \
    --outer-fold "$FOLD" --formula-fold-seed 20260825 \
    --batch-size 64 --bootstrap-resamples 10000 --bootstrap-seed 20260920 \
    --official-finetuned-checkpoint "$OFFICIAL_SLIM" \
    --mature-e8-checkpoint "$MATURE_E8" \
    --e8-architecture-checkpoint "$E8_ARCHITECTURE" \
    >"$RUN_ROOT/evaluate_${arm}.out" 2>"$RUN_ROOT/evaluate_${arm}.err"
}

evaluate_arm "${ALLOCATED_GPUS[0]}" primary &
PID_EVAL_PRIMARY=$!
evaluate_arm "${ALLOCATED_GPUS[1]}" replicate &
PID_EVAL_REPLICATE=$!

set +e
wait "$PID_EVAL_PRIMARY"
STATUS_EVAL_PRIMARY=$?
wait "$PID_EVAL_REPLICATE"
STATUS_EVAL_REPLICATE=$?
set -e
if (( STATUS_EVAL_PRIMARY != 0 || STATUS_EVAL_REPLICATE != 0 )); then
  echo "Native held evaluation failure: primary=$STATUS_EVAL_PRIMARY replicate=$STATUS_EVAL_REPLICATE" >&2
  exit 8
fi

python -u tasks/summarize_noise_dreams_native_replicates.py \
  --primary "$EVALUATION/primary" \
  --replicate "$EVALUATION/replicate" \
  --output "$RUN_ROOT/summary" \
  --minimum-delta-recall1-pp 2.0 \
  --bootstrap-resamples 10000 --seed 20260920

# Checkpoints intentionally remain on node-local storage. The persistent run
# contains the full paired held evaluation and promotion decision without
# consuming project quota with two 468 MB transient model copies.
mkdir -p "$RUN_ROOT/training_reports"
cp "$ARMS/primary/report.json" "$RUN_ROOT/training_reports/primary.json"
cp "$ARMS/replicate/report.json" "$RUN_ROOT/training_reports/replicate.json"

echo "=== Native DreaMS Noise training complete: $RUN_ROOT ==="
