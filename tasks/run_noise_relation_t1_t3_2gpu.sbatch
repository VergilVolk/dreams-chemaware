#!/bin/bash
#SBATCH --job-name=noise_t1_t3_relation
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=96:00:00
#SBATCH --output=data/validation/noise_t1_t3_%j.out
#SBATCH --error=data/validation/noise_t1_t3_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

IFS=',' read -r -a ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES:-}"
if (( ${#ALLOCATED_GPUS[@]} != 2 )); then
  echo "Expected exactly two GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}" >&2
  exit 3
fi

GRAPH="data/validation/noise_corrected_candidate_graph_v1_20260906/candidate_graph.npz"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
STAGE1="data/validation/noise_dreams_hard_positive_native_fold_0_run_2344820"
STAGE1_CHECKPOINT="$STAGE1/checkpoint/targeted_final.ckpt"
STAGE1_TRIPLETS="$STAGE1/triplets"
STAGE1_EVALUATION="$STAGE1/evaluation/targeted"
RUN_ROOT="data/validation/noise_relation_t1_t3_run_${SLURM_JOB_ID}"
CORPUS="$RUN_ROOT/corpus"
EVALUATION="$RUN_ROOT/evaluation"
LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
LOCAL_ROOT="$LOCAL_BASE/noise_relation_t1_t3_${SLURM_JOB_ID}"
STAGE1_EMBEDDINGS="$LOCAL_ROOT/stage1_embeddings.npz"
PRIMARY="$LOCAL_ROOT/seed_3407"
REPLICATE="$LOCAL_ROOT/seed_3408"

[[ ! -e "$RUN_ROOT" ]] || { echo "Output collision: $RUN_ROOT" >&2; exit 4; }
mkdir -p "$RUN_ROOT" "$EVALUATION" "$LOCAL_ROOT"
cleanup_local() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup_local EXIT

required_files=(
  "$GRAPH" "$DATA" "$ARCHITECTURE" "$STAGE1_CHECKPOINT"
  "$STAGE1_TRIPLETS/train_pool.npz" "$STAGE1_TRIPLETS/action_spectra.npz"
  "$STAGE1_EVALUATION/report.json" "$STAGE1_EVALUATION/held_per_query.csv.gz"
  tasks/noise_relation_t1_t3_core.py
  tasks/build_noise_relation_t1_t3_corpus.py
  tasks/train_noise_relation_t1_t3.py
  tasks/summarize_noise_relation_t1_t3.py
  tasks/test_noise_relation_t1_t3_core.py
  tasks/test_noise_relation_t1_t3_sbatch.py
  tasks/encode_noise_corrected_graph_checkpoint.py
  tasks/evaluate_noise_dreams_native.py
)
for path in "${required_files[@]}"; do
  [[ -f "$path" ]] || { echo "Missing T1/T3 input: $path" >&2; exit 5; }
done

python -m py_compile \
  tasks/noise_relation_t1_t3_core.py \
  tasks/build_noise_relation_t1_t3_corpus.py \
  tasks/train_noise_relation_t1_t3.py \
  tasks/summarize_noise_relation_t1_t3.py \
  tasks/test_noise_relation_t1_t3_core.py
python -u tasks/test_noise_relation_t1_t3_core.py
python -u tasks/test_noise_relation_t1_t3_sbatch.py

# Stage-1 geometry is used only to choose live hard relations from outer-train formulas.
CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[0]}" python -u \
  tasks/encode_noise_corrected_graph_checkpoint.py \
  --graph "$GRAPH" --data "$DATA" \
  --checkpoint "$STAGE1_CHECKPOINT" --architecture-checkpoint "$ARCHITECTURE" \
  --output "$STAGE1_EMBEDDINGS" --batch-size 64

python -u tasks/build_noise_relation_t1_t3_corpus.py \
  --graph "$GRAPH" --stage1-embeddings "$STAGE1_EMBEDDINGS" \
  --stage1-triplets "$STAGE1_TRIPLETS" --output "$CORPUS" \
  --outer-fold 0 --validation-fold 1 --fold-seed 20260825 \
  --max-positive-refs 4 --max-negative-molecules 4 \
  --additional-boundary-molecules 2 --max-negative-refs 3 \
  --max-actions-per-query 40

train_seed() {
  local gpu="$1"
  local seed="$2"
  local output="$3"
  CUDA_VISIBLE_DEVICES="$gpu" python -u tasks/train_noise_relation_t1_t3.py \
    --data "$DATA" --corpus "$CORPUS" \
    --action-bank "$STAGE1_TRIPLETS/action_spectra.npz" \
    --warm-start-checkpoint "$STAGE1_CHECKPOINT" \
    --architecture-checkpoint "$ARCHITECTURE" \
    --output "$output" --seed "$seed" \
    --lr 5e-6 --weight-decay 0 --triplet-loss-margin 0.1 \
    --listwise-temperature 0.07 --n-highest-peaks 100 \
    --maximum-spectra-per-step 64 --maximum-queries-per-step 4 \
    >"$RUN_ROOT/train_${seed}.out" 2>"$RUN_ROOT/train_${seed}.err"
}

train_seed "${ALLOCATED_GPUS[0]}" 3407 "$PRIMARY" & PID_PRIMARY=$!
train_seed "${ALLOCATED_GPUS[1]}" 3408 "$REPLICATE" & PID_REPLICATE=$!
set +e
wait "$PID_PRIMARY"; STATUS_PRIMARY=$?
wait "$PID_REPLICATE"; STATUS_REPLICATE=$?
set -e
if (( STATUS_PRIMARY != 0 || STATUS_REPLICATE != 0 )); then
  tail -n 160 "$RUN_ROOT/train_3407.err" >&2 || true
  tail -n 160 "$RUN_ROOT/train_3408.err" >&2 || true
  echo "T1/T3 training failure: primary=$STATUS_PRIMARY replicate=$STATUS_REPLICATE" >&2
  exit 6
fi

evaluate_seed() {
  local gpu="$1"
  local seed="$2"
  local checkpoint="$3"
  CUDA_VISIBLE_DEVICES="$gpu" python -u tasks/evaluate_noise_dreams_native.py \
    --graph "$GRAPH" \
    --embedding-cache "data/validation/noise_corrected_candidate_graph_v1_20260906/official_embeddings.npz" \
    --official-slim-checkpoint "data/e1/official_embedding_slim.pt" \
    --source-manifest "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz" \
    --data "$DATA" --checkpoint "$checkpoint" \
    --architecture-checkpoint "$ARCHITECTURE" \
    --output "$EVALUATION/seed_${seed}" --arm targeted \
    --outer-fold 0 --formula-fold-seed 20260825 --batch-size 64 \
    --bootstrap-resamples 10000 --bootstrap-seed $((20260930 + seed)) \
    --no-write-held-metric-evidence \
    >"$RUN_ROOT/evaluate_${seed}.out" 2>"$RUN_ROOT/evaluate_${seed}.err"
}

evaluate_seed "${ALLOCATED_GPUS[0]}" 3407 "$PRIMARY/final_slim.pt" & PID_EVAL_PRIMARY=$!
evaluate_seed "${ALLOCATED_GPUS[1]}" 3408 "$REPLICATE/final_slim.pt" & PID_EVAL_REPLICATE=$!
set +e
wait "$PID_EVAL_PRIMARY"; STATUS_EVAL_PRIMARY=$?
wait "$PID_EVAL_REPLICATE"; STATUS_EVAL_REPLICATE=$?
set -e
if (( STATUS_EVAL_PRIMARY != 0 || STATUS_EVAL_REPLICATE != 0 )); then
  tail -n 160 "$RUN_ROOT/evaluate_3407.err" >&2 || true
  tail -n 160 "$RUN_ROOT/evaluate_3408.err" >&2 || true
  echo "T1/T3 evaluation failure: primary=$STATUS_EVAL_PRIMARY replicate=$STATUS_EVAL_REPLICATE" >&2
  exit 7
fi

python -u tasks/summarize_noise_relation_t1_t3.py \
  --stage1 "$STAGE1_EVALUATION" \
  --primary "$EVALUATION/seed_3407" \
  --replicate "$EVALUATION/seed_3408" \
  --output "$RUN_ROOT/summary/report.json" --bootstrap-resamples 10000

mkdir -p "$RUN_ROOT/checkpoint" "$RUN_ROOT/training_reports"
cp "$PRIMARY/final_slim.pt" "$RUN_ROOT/checkpoint/primary_seed_3407_slim.pt"
cp "$REPLICATE/final_slim.pt" "$RUN_ROOT/checkpoint/replicate_seed_3408_slim.pt"
cp "$PRIMARY/report.json" "$RUN_ROOT/training_reports/primary_seed_3407.json"
cp "$REPLICATE/report.json" "$RUN_ROOT/training_reports/replicate_seed_3408.json"
echo "=== Noise relation-complete T1/T3 result: $RUN_ROOT ==="
