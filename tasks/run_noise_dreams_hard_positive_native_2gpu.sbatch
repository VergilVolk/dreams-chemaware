#!/bin/bash
#SBATCH --job-name=noise_hardpos_native
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=168:00:00
#SBATCH --output=data/validation/noise_hardpos_native_%j.out
#SBATCH --error=data/validation/noise_hardpos_native_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

IFS=',' read -r -a ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES:-}"
if (( ${#ALLOCATED_GPUS[@]} != 2 )); then
  echo "Expected exactly two allocated GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" >&2
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
RUN_ROOT="data/validation/noise_dreams_hard_positive_native_fold_${FOLD}_run_${SLURM_JOB_ID}"
TRIPLETS="$RUN_ROOT/triplets"
EVALUATION="$RUN_ROOT/evaluation"
LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
LOCAL_ROOT="$LOCAL_BASE/noise_hardpos_native_${SLURM_JOB_ID}"
ARMS="$LOCAL_ROOT/arms"

[[ ! -e "$RUN_ROOT" ]] || { echo "Output collision: $RUN_ROOT" >&2; exit 4; }
mkdir -p "$RUN_ROOT" "$EVALUATION" "$ARMS"
cleanup_local() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup_local EXIT
LOCAL_FREE_KB=$(df -Pk "$LOCAL_BASE" | awk 'NR==2 {print $4}')
if [[ -z "$LOCAL_FREE_KB" ]] || (( LOCAL_FREE_KB < 4194304 )); then
  echo "Node-local checkpoint space below 4 GiB: base=$LOCAL_BASE free_kb=${LOCAL_FREE_KB:-unknown}" >&2
  exit 4
fi

required_files=(
  "$LEDGER/report.json"
  "$LEDGER/training_actions.csv.gz"
  "$LEDGER/action_spectra.npz"
  "$GRAPH" "$EMBEDDING_CACHE" "$SOURCE_MANIFEST" "$DATA"
  "$OFFICIAL_SLIM" "$E8_ARCHITECTURE" "$MATURE_E8"
  tasks/build_noise_dreams_native_triplets.py
  tasks/noise_dreams_native_spectrum.py
  tasks/audit_noise_dreams_native_official_replay.py
  tasks/train_noise_dreams_native.py
  tasks/train_e1_identity.py
  tasks/e1_checkpoint_io.py
  tasks/evaluate_noise_dreams_native.py
  tasks/noise_corrected_fullgraph_evaluation.py
  tasks/summarize_noise_dreams_hard_positive_arms.py
  tasks/summarize_noise_dreams_native_replicates.py
  tasks/test_noise_dreams_hard_positive_bridge.py
  tasks/test_noise_dreams_native.py
)
for path in "${required_files[@]}"; do
  [[ -f "$path" ]] || { echo "Missing hard-positive input: $path" >&2; exit 5; }
done

python -m py_compile \
  tasks/build_noise_dreams_native_triplets.py \
  tasks/noise_dreams_native_spectrum.py \
  tasks/audit_noise_dreams_native_official_replay.py \
  tasks/train_noise_dreams_native.py \
  tasks/train_e1_identity.py \
  tasks/e1_checkpoint_io.py \
  tasks/evaluate_noise_dreams_native.py \
  tasks/noise_corrected_fullgraph_evaluation.py \
  tasks/summarize_noise_dreams_hard_positive_arms.py \
  tasks/summarize_noise_dreams_native_replicates.py \
  tasks/test_noise_dreams_hard_positive_bridge.py
python -u tasks/test_noise_dreams_hard_positive_bridge.py

python -u tasks/build_noise_dreams_native_triplets.py \
  --ledger-dir "$LEDGER" \
  --graph "$GRAPH" \
  --embedding-cache "$EMBEDDING_CACHE" \
  --data "$DATA" \
  --output "$TRIPLETS" \
  --outer-fold "$FOLD" --formula-fold-seed 20260825 \
  --validation-folds 10 --validation-fold 0 --validation-seed 20260920 \
  --margin-floor 5e-6 --hard-negative-molecules 5 \
  --max-positive-pool 32 --max-negative-pool 32 \
  --expected-actions 32114 \
  --clean-anchor-scope action_queries_plus_formula \
  --minimum-train-clean-anchors 4000 \
  --minimum-validation-clean-anchors 1000 \
  --minimum-train-formulas 4000 \
  --epochs 1

CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[0]}" python -u \
  tasks/audit_noise_dreams_native_official_replay.py \
  --triplet-dir "$TRIPLETS" \
  --embedding-cache "$EMBEDDING_CACHE" \
  --data "$DATA" \
  --official-finetuned-checkpoint "$OFFICIAL_SLIM" \
  --architecture-checkpoint "$E8_ARCHITECTURE" \
  --native-embedding-cache-output "$TRIPLETS/native_official_embeddings.npz" \
  --output "$TRIPLETS/official_replay.json" \
  --batch-size 64

run_arm() {
  local gpu="$1"
  local arm="$2"
  CUDA_VISIBLE_DEVICES="$gpu" python -u tasks/train_noise_dreams_native.py \
    --data "$DATA" \
    --triplet-dir "$TRIPLETS" \
    --official-finetuned-checkpoint "$OFFICIAL_SLIM" \
    --architecture-checkpoint "$E8_ARCHITECTURE" \
    --output "$ARMS/$arm" \
    --arm "$arm" --seed 3407 \
    --lr 5e-6 --weight-decay 0 --triplet-loss-margin 0.1 \
    --batch-size 4 --num-workers 4 --max-epochs 1 \
    --n-highest-peaks 100 \
    >"$RUN_ROOT/${arm}.out" 2>"$RUN_ROOT/${arm}.err"
}

run_arm "${ALLOCATED_GPUS[0]}" targeted &
PID_TARGETED=$!
run_arm "${ALLOCATED_GPUS[1]}" control &
PID_CONTROL=$!

heartbeat() {
  while kill -0 "$PID_TARGETED" 2>/dev/null || kill -0 "$PID_CONTROL" 2>/dev/null; do
    local targeted_alive=0 control_alive=0
    kill -0 "$PID_TARGETED" 2>/dev/null && targeted_alive=1
    kill -0 "$PID_CONTROL" 2>/dev/null && control_alive=1
    echo "[hard-positive-heartbeat] $(date) targeted_alive=$targeted_alive control_alive=$control_alive"
    sleep 120
  done
}
heartbeat &
PID_HEARTBEAT=$!

set +e
wait "$PID_TARGETED"; STATUS_TARGETED=$?
wait "$PID_CONTROL"; STATUS_CONTROL=$?
kill "$PID_HEARTBEAT" 2>/dev/null
wait "$PID_HEARTBEAT" 2>/dev/null
set -e
if (( STATUS_TARGETED != 0 || STATUS_CONTROL != 0 )); then
  tail -n 120 "$RUN_ROOT/targeted.err" >&2 || true
  tail -n 120 "$RUN_ROOT/control.err" >&2 || true
  echo "Hard-positive native training failure: targeted=$STATUS_TARGETED control=$STATUS_CONTROL" >&2
  exit 6
fi

evaluate_arm() {
  local gpu="$1"
  local arm="$2"
  CUDA_VISIBLE_DEVICES="$gpu" python -u tasks/evaluate_noise_dreams_native.py \
    --graph "$GRAPH" --embedding-cache "$EMBEDDING_CACHE" \
    --official-slim-checkpoint "$OFFICIAL_SLIM" \
    --source-manifest "$SOURCE_MANIFEST" --data "$DATA" \
    --checkpoint "$ARMS/$arm/final.ckpt" \
    --architecture-checkpoint "$E8_ARCHITECTURE" \
    --output "$EVALUATION/$arm" --arm "$arm" \
    --outer-fold "$FOLD" --formula-fold-seed 20260825 \
    --batch-size 64 --bootstrap-resamples 10000 --bootstrap-seed 20260925 \
    --official-finetuned-checkpoint "$OFFICIAL_SLIM" \
    --mature-e8-checkpoint "$MATURE_E8" \
    --e8-architecture-checkpoint "$E8_ARCHITECTURE" \
    >"$RUN_ROOT/evaluate_${arm}.out" 2>"$RUN_ROOT/evaluate_${arm}.err"
}

evaluate_arm "${ALLOCATED_GPUS[0]}" targeted &
PID_EVAL_TARGETED=$!
evaluate_arm "${ALLOCATED_GPUS[1]}" control &
PID_EVAL_CONTROL=$!
set +e
wait "$PID_EVAL_TARGETED"; STATUS_EVAL_TARGETED=$?
wait "$PID_EVAL_CONTROL"; STATUS_EVAL_CONTROL=$?
set -e
if (( STATUS_EVAL_TARGETED != 0 || STATUS_EVAL_CONTROL != 0 )); then
  echo "Hard-positive held evaluation failure: targeted=$STATUS_EVAL_TARGETED control=$STATUS_EVAL_CONTROL" >&2
  exit 7
fi

python -u tasks/summarize_noise_dreams_hard_positive_arms.py \
  --targeted "$EVALUATION/targeted" \
  --control "$EVALUATION/control" \
  --output "$RUN_ROOT/summary" \
  --minimum-delta-recall1-pp 2.0 \
  --bootstrap-resamples 10000 --seed 20260925

mkdir -p "$RUN_ROOT/checkpoint" "$RUN_ROOT/training_reports"
cp "$ARMS/targeted/final.ckpt" "$RUN_ROOT/checkpoint/targeted_final.ckpt"
cp "$ARMS/targeted/report.json" "$RUN_ROOT/training_reports/targeted.json"
cp "$ARMS/control/report.json" "$RUN_ROOT/training_reports/control.json"
echo "=== Native hard-positive result: $RUN_ROOT ==="
