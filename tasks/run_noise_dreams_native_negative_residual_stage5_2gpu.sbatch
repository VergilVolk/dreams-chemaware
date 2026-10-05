#!/bin/bash
#SBATCH --job-name=noise_native_s5
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH --output=data/validation/noise_native_s5_%j.out
#SBATCH --error=data/validation/noise_native_s5_%j.err

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
STAGE1="data/validation/noise_dreams_hard_positive_native_fold_0_run_2344820"
GNPS_BENCHMARK="data/validation/gnps_gold_silver_10ppm_benchmark_v1"
GRAPH_DIR="data/validation/noise_corrected_candidate_graph_v1_20260906"
GRAPH="$GRAPH_DIR/candidate_graph.npz"
EMBEDDING_CACHE="$GRAPH_DIR/official_embeddings.npz"
SOURCE_MANIFEST="data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
OFFICIAL_SLIM="data/e1/official_embedding_slim.pt"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
MATURE_E8="data/validation/g8r_noise_final_e8_direct_transfer/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e8_baseline_symmetric_shared/seed_20260830/fold_0/final_shared_encoder.pt"
WARM_START="$STAGE1/checkpoint/targeted_final.ckpt"
RUN_ROOT="data/validation/noise_dreams_native_negative_residual_stage5_fold_${FOLD}_run_${SLURM_JOB_ID}"
TRIPLETS="$RUN_ROOT/triplets"
EVALUATION="$RUN_ROOT/evaluation"
GNPS_EVALUATION="$RUN_ROOT/gnps_evaluation"
LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
LOCAL_ROOT="$LOCAL_BASE/noise_native_s5_${SLURM_JOB_ID}"
ARMS="$LOCAL_ROOT/arms"
GNPS_CACHE="$LOCAL_ROOT/gnps_cache"
SOURCE_SNAPSHOT="$LOCAL_ROOT/source_snapshot"

[[ ! -e "$RUN_ROOT" ]] || { echo "Output collision: $RUN_ROOT" >&2; exit 4; }
mkdir -p "$RUN_ROOT" "$EVALUATION" "$GNPS_EVALUATION" "$ARMS" "$GNPS_CACHE"
cleanup_local() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup_local EXIT
LOCAL_FREE_KB=$(df -Pk "$LOCAL_BASE" | awk 'NR==2 {print $4}')
if [[ -z "$LOCAL_FREE_KB" ]] || (( LOCAL_FREE_KB < 4194304 )); then
  echo "Node-local checkpoint space below 4 GiB: base=$LOCAL_BASE free_kb=${LOCAL_FREE_KB:-unknown}" >&2
  exit 4
fi

required_files=(
  "$STAGE1/summary/report.json"
  "$STAGE1/evaluation/targeted/report.json"
  "$STAGE1/triplets/report.json"
  "$STAGE1/triplets/train_pool.npz"
  "$STAGE1/triplets/validation_pool.npz"
  "$STAGE1/triplets/action_spectra.npz"
  "$STAGE1/triplets/selected_actions.csv.gz"
  "$GNPS_BENCHMARK/checksums.sha256"
  "$GNPS_BENCHMARK/spectra.mgf"
  "$GNPS_BENCHMARK/manifest.csv.gz"
  "$GNPS_BENCHMARK/panel_identity_disjoint.npz"
  "$GNPS_BENCHMARK/panel_formula_disjoint.npz"
  "$WARM_START" "$GRAPH" "$EMBEDDING_CACHE" "$SOURCE_MANIFEST"
  "$DATA" "$OFFICIAL_SLIM" "$ARCHITECTURE" "$MATURE_E8"
  tasks/build_noise_dreams_native_negative_residual_stage5.py
  tasks/train_noise_dreams_native_residual_stage2.py
  tasks/prepare_official_embedding_checkpoint.py
  tasks/evaluate_noise_dreams_native.py
  tasks/summarize_noise_dreams_native_residual_stage2.py
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py
  tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py
  tasks/summarize_noise_dreams_native_stage4_gnps.py
  tasks/test_noise_dreams_native_negative_residual_stage5.py
)
for path in "${required_files[@]}"; do
  [[ -f "$path" ]] || { echo "Missing Stage-5 input: $path" >&2; exit 5; }
done

# Freeze source before GPU work because other project tasks share this checkout.
mkdir -p "$SOURCE_SNAPSHOT"
cp -a dreams "$SOURCE_SNAPSHOT/dreams"
cp -a tasks "$SOURCE_SNAPSHOT/tasks"
export PYTHONPATH="$SOURCE_SNAPSHOT:$SOURCE_SNAPSHOT/tasks:${PYTHONPATH:-}"
S5_TASKS="$SOURCE_SNAPSHOT/tasks"

python -m py_compile \
  "$S5_TASKS/build_noise_dreams_native_negative_residual_stage5.py" \
  "$S5_TASKS/train_noise_dreams_native_residual_stage2.py" \
  "$S5_TASKS/evaluate_noise_dreams_native.py" \
  "$S5_TASKS/summarize_noise_dreams_native_residual_stage2.py" \
  "$S5_TASKS/encode_gnps_gold_silver_10ppm_checkpoint.py" \
  "$S5_TASKS/evaluate_gnps_gold_silver_10ppm_embeddings.py" \
  "$S5_TASKS/summarize_noise_dreams_native_stage4_gnps.py" \
  "$S5_TASKS/test_noise_dreams_native_negative_residual_stage5.py"
python -u "$S5_TASKS/test_noise_dreams_native_negative_residual_stage5.py"

# Re-evaluate the real Stage-1 checkpoint with the same current evaluator while
# GPU 0 builds the paired negative ledgers.  This is numerical AUC/AUPRC, not a
# historical boolean proxy.
python -u "$S5_TASKS/prepare_official_embedding_checkpoint.py" \
  --source "$WARM_START" --output "$ARMS/stage1_slim.pt"
CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[1]}" python -u \
  "$S5_TASKS/evaluate_noise_dreams_native.py" \
  --graph "$GRAPH" --embedding-cache "$EMBEDDING_CACHE" \
  --official-slim-checkpoint "$OFFICIAL_SLIM" \
  --source-manifest "$SOURCE_MANIFEST" --data "$DATA" \
  --checkpoint "$ARMS/stage1_slim.pt" \
  --architecture-checkpoint "$ARCHITECTURE" \
  --output "$EVALUATION/stage1" --arm targeted \
  --outer-fold "$FOLD" --formula-fold-seed 20260825 \
  --batch-size 64 --bootstrap-resamples 10000 --bootstrap-seed 20260929 \
  --official-finetuned-checkpoint "$OFFICIAL_SLIM" \
  --mature-e8-checkpoint "$MATURE_E8" \
  --e8-architecture-checkpoint "$ARCHITECTURE" \
  --no-write-held-metric-evidence \
  >"$RUN_ROOT/evaluate_stage1.out" 2>"$RUN_ROOT/evaluate_stage1.err" &
PID_EVAL_STAGE1=$!

set +e
CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[0]}" python -u \
  "$S5_TASKS/build_noise_dreams_native_negative_residual_stage5.py" \
  --stage1-run "$STAGE1" --graph "$GRAPH" \
  --embedding-cache "$EMBEDDING_CACHE" --data "$DATA" \
  --architecture-checkpoint "$ARCHITECTURE" --output "$TRIPLETS" \
  --negative-molecules-per-query 3 --margin 0.1 --batch-size 64
STATUS_TRIPLETS=$?
wait "$PID_EVAL_STAGE1"; STATUS_EVAL_STAGE1=$?
set -e
if (( STATUS_TRIPLETS != 0 || STATUS_EVAL_STAGE1 != 0 )); then
  tail -n 120 "$RUN_ROOT/evaluate_stage1.err" >&2 || true
  echo "Stage-5 pretraining failure: triplets=$STATUS_TRIPLETS stage1_auc=$STATUS_EVAL_STAGE1" >&2
  exit 5
fi

run_arm() {
  local gpu="$1"
  local arm="$2"
  CUDA_VISIBLE_DEVICES="$gpu" python -u \
    "$S5_TASKS/train_noise_dreams_native_residual_stage2.py" \
    --curriculum negative_residual_stage5 \
    --data "$DATA" --triplet-dir "$TRIPLETS/$arm" \
    --warm-start-checkpoint "$WARM_START" \
    --architecture-checkpoint "$ARCHITECTURE" \
    --output "$ARMS/$arm" --arm "$arm" --seed 3407 \
    --lr 5e-6 --weight-decay 0 --triplet-loss-margin 0.1 \
    --batch-size 4 --num-workers 4 --max-epochs 1 \
    --n-highest-peaks 100 \
    >"$RUN_ROOT/${arm}.out" 2>"$RUN_ROOT/${arm}.err"
}

run_arm "${ALLOCATED_GPUS[0]}" targeted &
PID_TARGETED=$!
run_arm "${ALLOCATED_GPUS[1]}" control &
PID_CONTROL=$!
set +e
wait "$PID_TARGETED"; STATUS_TARGETED=$?
wait "$PID_CONTROL"; STATUS_CONTROL=$?
set -e
if (( STATUS_TARGETED != 0 || STATUS_CONTROL != 0 )); then
  tail -n 120 "$RUN_ROOT/targeted.err" >&2 || true
  tail -n 120 "$RUN_ROOT/control.err" >&2 || true
  echo "Stage-5 native training failure: targeted=$STATUS_TARGETED control=$STATUS_CONTROL" >&2
  exit 6
fi

mkdir -p "$RUN_ROOT/training_reports"
cp "$ARMS/targeted/report.json" "$RUN_ROOT/training_reports/targeted.json"
cp "$ARMS/control/report.json" "$RUN_ROOT/training_reports/control.json"
python -u "$S5_TASKS/prepare_official_embedding_checkpoint.py" \
  --source "$ARMS/targeted/final.ckpt" --output "$ARMS/targeted_slim.pt"
rm -f -- "$ARMS/targeted/final.ckpt"
python -u "$S5_TASKS/prepare_official_embedding_checkpoint.py" \
  --source "$ARMS/control/final.ckpt" --output "$ARMS/control_slim.pt"
rm -f -- "$ARMS/control/final.ckpt"

evaluate_arm() {
  local gpu="$1"
  local arm="$2"
  CUDA_VISIBLE_DEVICES="$gpu" python -u "$S5_TASKS/evaluate_noise_dreams_native.py" \
    --graph "$GRAPH" --embedding-cache "$EMBEDDING_CACHE" \
    --official-slim-checkpoint "$OFFICIAL_SLIM" \
    --source-manifest "$SOURCE_MANIFEST" --data "$DATA" \
    --checkpoint "$ARMS/${arm}_slim.pt" \
    --architecture-checkpoint "$ARCHITECTURE" \
    --output "$EVALUATION/$arm" --arm "$arm" \
    --outer-fold "$FOLD" --formula-fold-seed 20260825 \
    --batch-size 64 --bootstrap-resamples 10000 --bootstrap-seed 20260929 \
    --official-finetuned-checkpoint "$OFFICIAL_SLIM" \
    --mature-e8-checkpoint "$MATURE_E8" \
    --e8-architecture-checkpoint "$ARCHITECTURE" \
    --no-write-held-metric-evidence \
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
  tail -n 120 "$RUN_ROOT/evaluate_targeted.err" >&2 || true
  tail -n 120 "$RUN_ROOT/evaluate_control.err" >&2 || true
  echo "Stage-5 internal evaluation failure: targeted=$STATUS_EVAL_TARGETED control=$STATUS_EVAL_CONTROL" >&2
  exit 7
fi

python -u "$S5_TASKS/summarize_noise_dreams_native_residual_stage2.py" \
  --stage1 "$EVALUATION/stage1" \
  --targeted "$EVALUATION/targeted" --control "$EVALUATION/control" \
  --output "$RUN_ROOT/summary" \
  --result-prefix NOISE_DREAMS_NATIVE_NEGATIVE_RESIDUAL_STAGE5 \
  --absolute-target-pp 5.0 --bootstrap-resamples 10000 --seed 20260929

encode_gnps() {
  local gpu="$1" checkpoint="$2" label="$3" output="$4"
  CUDA_VISIBLE_DEVICES="$gpu" python -u \
    "$S5_TASKS/encode_gnps_gold_silver_10ppm_checkpoint.py" \
    --benchmark "$GNPS_BENCHMARK" --checkpoint "$checkpoint" \
    --architecture-checkpoint "$ARCHITECTURE" --output "$output" \
    --batch-size 64 --label "$label" \
    >"$RUN_ROOT/encode_gnps_${label}.out" 2>"$RUN_ROOT/encode_gnps_${label}.err"
}

encode_gnps "${ALLOCATED_GPUS[0]}" "$OFFICIAL_SLIM" official "$GNPS_CACHE/official.npz" &
P1=$!
encode_gnps "${ALLOCATED_GPUS[1]}" "$ARMS/stage1_slim.pt" stage1 "$GNPS_CACHE/stage1.npz" &
P2=$!
set +e; wait "$P1"; S1=$?; wait "$P2"; S2=$?; set -e
(( S1 == 0 && S2 == 0 )) || { echo "Stage-5 GNPS reference encoding failed: $S1/$S2" >&2; exit 8; }

encode_gnps "${ALLOCATED_GPUS[0]}" "$ARMS/targeted_slim.pt" targeted "$GNPS_CACHE/targeted.npz" &
P1=$!
encode_gnps "${ALLOCATED_GPUS[1]}" "$ARMS/control_slim.pt" control "$GNPS_CACHE/control.npz" &
P2=$!
set +e; wait "$P1"; S1=$?; wait "$P2"; S2=$?; set -e
(( S1 == 0 && S2 == 0 )) || { echo "Stage-5 GNPS candidate encoding failed: $S1/$S2" >&2; exit 8; }

evaluate_gnps() {
  local name="$1" baseline="$2" candidate="$3"
  python -u "$S5_TASKS/evaluate_gnps_gold_silver_10ppm_embeddings.py" \
    --benchmark "$GNPS_BENCHMARK" --baseline-embeddings "$baseline" \
    --candidate-embeddings "$candidate" --output "$GNPS_EVALUATION/$name" \
    --bootstrap-resamples 10000 --bootstrap-seed 20260929 \
    >"$RUN_ROOT/evaluate_gnps_${name}.out" 2>"$RUN_ROOT/evaluate_gnps_${name}.err"
}

evaluate_gnps stage1_vs_official "$GNPS_CACHE/official.npz" "$GNPS_CACHE/stage1.npz" & P1=$!
evaluate_gnps targeted_vs_official "$GNPS_CACHE/official.npz" "$GNPS_CACHE/targeted.npz" & P2=$!
evaluate_gnps targeted_vs_stage1 "$GNPS_CACHE/stage1.npz" "$GNPS_CACHE/targeted.npz" & P3=$!
evaluate_gnps targeted_vs_control "$GNPS_CACHE/control.npz" "$GNPS_CACHE/targeted.npz" & P4=$!
set +e
wait "$P1"; S1=$?; wait "$P2"; S2=$?; wait "$P3"; S3=$?; wait "$P4"; S4=$?
set -e
(( S1 == 0 && S2 == 0 && S3 == 0 && S4 == 0 )) || {
  echo "Stage-5 GNPS comparisons failed: $S1/$S2/$S3/$S4" >&2
  exit 9
}

python -u "$S5_TASKS/summarize_noise_dreams_native_stage4_gnps.py" \
  --internal-summary "$RUN_ROOT/summary/report.json" \
  --gnps-root "$GNPS_EVALUATION" --output "$RUN_ROOT/combined_summary" \
  --result-prefix NOISE_DREAMS_NATIVE_STAGE5 --absolute-target-pp 5.0

PROMOTION_STATUS=$(python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
  "$RUN_ROOT/combined_summary/report.json")
if [[ "$PROMOTION_STATUS" == "NOISE_DREAMS_NATIVE_STAGE5_INTERNAL_AND_GNPS_PROMOTION_PASS" ]]; then
  mkdir -p "$RUN_ROOT/checkpoint"
  cp "$ARMS/targeted_slim.pt" "$RUN_ROOT/checkpoint/targeted_slim.pt"
fi

echo "=== Noise native hard-negative Stage-5 + GNPS result: $RUN_ROOT ==="
