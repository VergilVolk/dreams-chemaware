#!/bin/bash
#SBATCH --job-name=noise_t1t3_v2
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=96:00:00
#SBATCH --output=data/validation/noise_t1t3_v2_%j.out
#SBATCH --error=data/validation/noise_t1t3_v2_%j.err

# v2 protocol (predeclared, two stages):
#   Stage A (train-side): build the v2 corpus, train clean-only and
#     action-rotation arms (seed 3407, one per GPU), score the averaged arm
#     by reusing run 2347055's checkpoint, and select the winner on fold-1
#     strict recall@1 delta over the Stage-1 warm start (ties: mean-margin
#     delta).  No-go (winner delta <= 0) exits without touching held.
#   Stage B (only on go, winner earns it): replicate the winner arm with
#     seed 3408, evaluate both seeds on the outer held fold exactly as the
#     v1 run did, summarize against Stage-1, and run the independent GNPS
#     gold/silver panel paired against Stage-1.  If the averaged arm wins,
#     its held evidence already exists in run 2347055 and only the GNPS
#     panel is added.
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
AB_V1_RUN="${AB_V1_RUN:-data/validation/noise_relation_t1_t3_run_2347055}"
AB_SLIM="$AB_V1_RUN/checkpoint/primary_seed_3407_slim.pt"
GNPS_BENCHMARK="data/validation/gnps_gold_silver_10ppm_benchmark_v1"
RUN_ROOT="data/validation/noise_relation_t1_t3_v2_run_${SLURM_JOB_ID}"
CORPUS="$RUN_ROOT/corpus"
EVALUATION="$RUN_ROOT/evaluation"
FOLD1="$RUN_ROOT/fold1"
GNPS="$RUN_ROOT/gnps"
LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
LOCAL_ROOT="$LOCAL_BASE/noise_relation_t1_t3_v2_${SLURM_JOB_ID}"
STAGE1_EMBEDDINGS="$LOCAL_ROOT/stage1_embeddings.npz"
STAGE1_SLIM="$LOCAL_ROOT/stage1_targeted_slim.pt"
CLEAN_3407="$LOCAL_ROOT/clean_only_seed_3407"
ROTATION_3407="$LOCAL_ROOT/action_rotation_seed_3407"
WINNER_3408="$LOCAL_ROOT/winner_seed_3408"

[[ ! -e "$RUN_ROOT" ]] || { echo "Output collision: $RUN_ROOT" >&2; exit 4; }
mkdir -p "$RUN_ROOT" "$EVALUATION" "$FOLD1" "$GNPS" "$LOCAL_ROOT"
cleanup_local() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup_local EXIT

required_files=(
  "$GRAPH" "$DATA" "$ARCHITECTURE" "$STAGE1_CHECKPOINT" "$AB_SLIM"
  "$STAGE1_TRIPLETS/train_pool.npz" "$STAGE1_TRIPLETS/action_spectra.npz"
  "$STAGE1_EVALUATION/report.json" "$STAGE1_EVALUATION/held_per_query.csv.gz"
  tasks/noise_relation_t1_t3_core.py
  tasks/build_noise_relation_t1_t3_corpus.py
  tasks/train_noise_relation_t1_t3.py
  tasks/build_noise_relation_t1_t3_corpus_v2.py
  tasks/train_noise_relation_t1_t3_v2.py
  tasks/score_fold1_noise_relation_checkpoint.py
  tasks/summarize_noise_relation_t1_t3_v2_fold1.py
  tasks/summarize_noise_relation_t1_t3.py
  tasks/test_noise_relation_t1_t3_core.py
  tasks/test_noise_relation_t1_t3_v2.py
  tasks/test_noise_relation_t1_t3_v2_sbatch.py
  tasks/encode_noise_corrected_graph_checkpoint.py
  tasks/evaluate_noise_dreams_native.py
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py
  tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py
  tasks/prepare_official_embedding_checkpoint.py
)
for path in "${required_files[@]}"; do
  [[ -f "$path" ]] || { echo "Missing T1/T3 v2 input: $path" >&2; exit 5; }
done

python -m py_compile \
  tasks/noise_relation_t1_t3_core.py \
  tasks/build_noise_relation_t1_t3_corpus.py \
  tasks/train_noise_relation_t1_t3.py \
  tasks/build_noise_relation_t1_t3_corpus_v2.py \
  tasks/train_noise_relation_t1_t3_v2.py \
  tasks/score_fold1_noise_relation_checkpoint.py \
  tasks/summarize_noise_relation_t1_t3_v2_fold1.py \
  tasks/summarize_noise_relation_t1_t3.py
python -u tasks/test_noise_relation_t1_t3_core.py
python -u tasks/test_noise_relation_t1_t3_v2.py
python -u tasks/test_noise_relation_t1_t3_v2_sbatch.py

# Stage A.1: Stage-1 geometry for live hard relation selection (train side only).
CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[0]}" python -u \
  tasks/encode_noise_corrected_graph_checkpoint.py \
  --graph "$GRAPH" --data "$DATA" \
  --checkpoint "$STAGE1_CHECKPOINT" --architecture-checkpoint "$ARCHITECTURE" \
  --output "$STAGE1_EMBEDDINGS" --batch-size 64

# Stage A.2: v2 corpus.  Clean relations carry no action anchors; every exact
# Stage-1 hard-positive bridge lands in action_events.npz (train folds; fold-1
# events are kept for diagnostics only).
python -u tasks/build_noise_relation_t1_t3_corpus_v2.py \
  --graph "$GRAPH" --stage1-embeddings "$STAGE1_EMBEDDINGS" \
  --stage1-triplets "$STAGE1_TRIPLETS" --output "$CORPUS" \
  --outer-fold 0 --validation-fold 1 --fold-seed 20260825 \
  --max-positive-refs 4 --max-negative-molecules 4 \
  --additional-boundary-molecules 2 --max-negative-refs 3

# Stage A.3: the two new arms, one per GPU, seed 3407.  The averaged arm is
# NOT retrained: run 2347055 already trained and evaluated it on this exact
# corpus definition with the v1 trainer.
train_v2_seed() {
  local gpu="$1"
  local arm="$2"
  local seed="$3"
  local output="$4"
  CUDA_VISIBLE_DEVICES="$gpu" python -u tasks/train_noise_relation_t1_t3_v2.py \
    --data "$DATA" --corpus "$CORPUS" \
    --action-bank "$STAGE1_TRIPLETS/action_spectra.npz" \
    --warm-start-checkpoint "$STAGE1_CHECKPOINT" \
    --architecture-checkpoint "$ARCHITECTURE" \
    --output "$output" --seed "$seed" --arm "$arm" \
    --lr 5e-6 --weight-decay 0 --triplet-loss-margin 0.1 \
    --listwise-temperature 0.07 --n-highest-peaks 100 \
    --maximum-spectra-per-step 64 --maximum-queries-per-step 4 \
    >"$RUN_ROOT/train_${arm}_seed_${seed}.out" 2>"$RUN_ROOT/train_${arm}_seed_${seed}.err"
}

train_v2_seed "${ALLOCATED_GPUS[0]}" clean-only 3407 "$CLEAN_3407" & PID_CLEAN=$!
train_v2_seed "${ALLOCATED_GPUS[1]}" action-rotation 3407 "$ROTATION_3407" & PID_ROTATION=$!
set +e
wait "$PID_CLEAN"; STATUS_CLEAN=$?
wait "$PID_ROTATION"; STATUS_ROTATION=$?
set -e
if (( STATUS_CLEAN != 0 || STATUS_ROTATION != 0 )); then
  tail -n 160 "$RUN_ROOT/train_clean-only_seed_3407.err" >&2 || true
  tail -n 160 "$RUN_ROOT/train_action-rotation_seed_3407.err" >&2 || true
  echo "T1/T3 v2 training failure: clean=$STATUS_CLEAN rotation=$STATUS_ROTATION" >&2
  exit 6
fi

# Stage A.4: fold-1 standalone score for the averaged arm's checkpoint.
CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[0]}" python -u \
  tasks/score_fold1_noise_relation_checkpoint.py \
  --data "$DATA" --corpus "$CORPUS" \
  --warm-start-checkpoint "$STAGE1_CHECKPOINT" \
  --architecture-checkpoint "$ARCHITECTURE" \
  --candidate-checkpoint "$AB_SLIM" \
  --output "$FOLD1/averaged_fold1.json" \
  >"$RUN_ROOT/score_averaged_fold1.out" 2>"$RUN_ROOT/score_averaged_fold1.err"

# Stage A.5: predeclared selection on train-side material only.
python -u tasks/summarize_noise_relation_t1_t3_v2_fold1.py \
  --clean-only-report "$CLEAN_3407/report.json" \
  --action-rotation-report "$ROTATION_3407/report.json" \
  --averaged-scorer-json "$FOLD1/averaged_fold1.json" \
  --output "$FOLD1/selection.json"

mkdir -p "$RUN_ROOT/checkpoint" "$RUN_ROOT/training_reports"
cp "$CLEAN_3407/report.json" "$RUN_ROOT/training_reports/clean_only_seed_3407.json"
cp "$ROTATION_3407/report.json" "$RUN_ROOT/training_reports/action_rotation_seed_3407.json"
cp "$CLEAN_3407/final_slim.pt" "$RUN_ROOT/checkpoint/clean_only_seed_3407_slim.pt"
cp "$ROTATION_3407/final_slim.pt" "$RUN_ROOT/checkpoint/action_rotation_seed_3407_slim.pt"
cp "$ROTATION_3407/final_optimizer.pt" "$RUN_ROOT/checkpoint/action_rotation_seed_3407_optimizer.pt"
cp "$CLEAN_3407/final_optimizer.pt" "$RUN_ROOT/checkpoint/clean_only_seed_3407_optimizer.pt"

SELECTED_ARM=$(python -c "import json;print(json.load(open('$FOLD1/selection.json'))['selected_arm'] or '')")
PROCEED=$(python -c "import json;print('yes' if json.load(open('$FOLD1/selection.json'))['proceed_to_held'] else 'no')")
echo "=== fold-1 selection: selected_arm='${SELECTED_ARM}' proceed='${PROCEED}' ==="

if [[ "$PROCEED" != "yes" ]]; then
  echo "=== NO-GO: no arm improved fold-1 strict recall over Stage-1; held fold untouched ==="
  echo "=== T1/T3 v2 fold-1 evidence: $RUN_ROOT ==="
  exit 0
fi

[[ -d "$GNPS_BENCHMARK" ]] || { echo "Missing GNPS benchmark: $GNPS_BENCHMARK" >&2; exit 8; }

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

# Stage B: GNPS Stage-1 baseline embeddings are winner-independent.  The
# 1.2 GB Lightning checkpoint is converted to a slim once (format-only, no
# optimization), matching the proven Stage-1 GNPS pipeline pattern.
python -u tasks/prepare_official_embedding_checkpoint.py \
  --source "$STAGE1_CHECKPOINT" --output "$STAGE1_SLIM"
CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[1]}" python -u \
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  --benchmark "$GNPS_BENCHMARK" \
  --checkpoint "$STAGE1_SLIM" \
  --architecture-checkpoint "$ARCHITECTURE" \
  --output "$GNPS/stage1_embeddings.npz" --batch-size 64 --label stage1 \
  >"$RUN_ROOT/gnps_encode_stage1.out" 2>"$RUN_ROOT/gnps_encode_stage1.err" & PID_GNPS_STAGE1=$!

if [[ "$SELECTED_ARM" == "averaged" ]]; then
  # The averaged arm's held evidence already exists in run 2347055
  # (both seeds, summarized against Stage-1).  Only the GNPS panel is new.
  wait "$PID_GNPS_STAGE1" || { tail -n 160 "$RUN_ROOT/gnps_encode_stage1.err" >&2; exit 9; }
  CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[0]}" python -u \
    tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
    --benchmark "$GNPS_BENCHMARK" \
    --checkpoint "$AB_SLIM" \
    --architecture-checkpoint "$ARCHITECTURE" \
    --output "$GNPS/averaged_embeddings.npz" --batch-size 64 --label averaged \
    >"$RUN_ROOT/gnps_encode_averaged.out" 2>"$RUN_ROOT/gnps_encode_averaged.err"
  python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
    --baseline-embeddings "$GNPS/stage1_embeddings.npz" \
    --candidate-embeddings "$GNPS/averaged_embeddings.npz" \
    --output "$GNPS/paired_stage1_vs_averaged" \
    --bootstrap-resamples 10000 --bootstrap-seed 20261001
  echo "=== averaged arm selected; held evidence: $AB_V1_RUN/summary ==="
  echo "=== T1/T3 v2 result: $RUN_ROOT ==="
  exit 0
fi

# Winner is a v2 arm: replicate it with seed 3408 before spending held.
train_v2_seed "${ALLOCATED_GPUS[0]}" "$SELECTED_ARM" 3408 "$WINNER_3408" & PID_WINNER=$!
wait "$PID_WINNER" || { tail -n 200 "$RUN_ROOT/train_${SELECTED_ARM}_seed_3408.err" >&2; exit 7; }
wait "$PID_GNPS_STAGE1" || { tail -n 160 "$RUN_ROOT/gnps_encode_stage1.err" >&2; exit 9; }
cp "$WINNER_3408/report.json" "$RUN_ROOT/training_reports/${SELECTED_ARM}_seed_3408.json"
cp "$WINNER_3408/final_slim.pt" "$RUN_ROOT/checkpoint/${SELECTED_ARM}_seed_3408_slim.pt"
cp "$WINNER_3408/final_optimizer.pt" "$RUN_ROOT/checkpoint/${SELECTED_ARM}_seed_3408_optimizer.pt"
if [[ "$SELECTED_ARM" == "clean-only" ]]; then
  PRIMARY="$CLEAN_3407"
else
  PRIMARY="$ROTATION_3407"
fi

evaluate_seed "${ALLOCATED_GPUS[0]}" 3407 "$PRIMARY/final_slim.pt" & PID_EVAL_PRIMARY=$!
evaluate_seed "${ALLOCATED_GPUS[1]}" 3408 "$WINNER_3408/final_slim.pt" & PID_EVAL_REPLICATE=$!
set +e
wait "$PID_EVAL_PRIMARY"; STATUS_EVAL_PRIMARY=$?
wait "$PID_EVAL_REPLICATE"; STATUS_EVAL_REPLICATE=$?
set -e
if (( STATUS_EVAL_PRIMARY != 0 || STATUS_EVAL_REPLICATE != 0 )); then
  tail -n 160 "$RUN_ROOT/evaluate_3407.err" >&2 || true
  tail -n 160 "$RUN_ROOT/evaluate_3408.err" >&2 || true
  echo "T1/T3 v2 evaluation failure: primary=$STATUS_EVAL_PRIMARY replicate=$STATUS_EVAL_REPLICATE" >&2
  exit 7
fi

python -u tasks/summarize_noise_relation_t1_t3.py \
  --stage1 "$STAGE1_EVALUATION" \
  --primary "$EVALUATION/seed_3407" \
  --replicate "$EVALUATION/seed_3408" \
  --output "$RUN_ROOT/summary/report.json" --bootstrap-resamples 10000

CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[0]}" python -u \
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  --benchmark "$GNPS_BENCHMARK" \
  --checkpoint "$PRIMARY/final_slim.pt" \
  --architecture-checkpoint "$ARCHITECTURE" \
  --output "$GNPS/${SELECTED_ARM}_embeddings.npz" --batch-size 64 --label "$SELECTED_ARM" \
  >"$RUN_ROOT/gnps_encode_${SELECTED_ARM}.out" 2>"$RUN_ROOT/gnps_encode_${SELECTED_ARM}.err"
python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
  --baseline-embeddings "$GNPS/stage1_embeddings.npz" \
  --candidate-embeddings "$GNPS/${SELECTED_ARM}_embeddings.npz" \
  --output "$GNPS/paired_stage1_vs_${SELECTED_ARM}" \
  --bootstrap-resamples 10000 --bootstrap-seed 20261001

echo "=== ${SELECTED_ARM} arm won fold-1; held+GNPS evidence: $RUN_ROOT ==="
