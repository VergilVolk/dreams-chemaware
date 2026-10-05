#!/bin/bash
#SBATCH --job-name=noise_s3_repair
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=data/validation/noise_s3_repair_%j.out
#SBATCH --error=data/validation/noise_s3_repair_%j.err

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
STAGE3_RUN="${NOISE_STAGE3_RUN:-data/validation/noise_dreams_native_multidifficulty_stage3_fold_0_run_2345317}"
STAGE3_TRIPLETS="$STAGE3_RUN/triplets"
ZERO_AUDIT="${NOISE_STAGE3_ZERO_AUDIT:-data/validation/noise_dreams_native_stage3_zero_update_run_2345440}"
GRAPH_DIR="data/validation/noise_corrected_candidate_graph_v1_20260906"
GRAPH="$GRAPH_DIR/candidate_graph.npz"
EMBEDDING_CACHE="$GRAPH_DIR/official_embeddings.npz"
SOURCE_MANIFEST="data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
OFFICIAL_SLIM="data/e1/official_embedding_slim.pt"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
MATURE_E8="data/validation/g8r_noise_final_e8_direct_transfer/curriculum_all_views4_blocks1_blr_2e-06_hlr_1e-05_e8_baseline_symmetric_shared/seed_20260830/fold_0/final_shared_encoder.pt"
WARM_START="$STAGE1/checkpoint/targeted_final.ckpt"
RUN_ROOT="data/validation/noise_dreams_native_stage3_calibrated_repair_fold_${FOLD}_run_${SLURM_JOB_ID}"
TRIPLETS="$RUN_ROOT/triplets"
EVALUATION="$RUN_ROOT/evaluation"
LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
LOCAL_ROOT="$LOCAL_BASE/noise_native_s3_repair_${SLURM_JOB_ID}"
ARMS="$LOCAL_ROOT/arms"
SOURCE_SNAPSHOT="$LOCAL_ROOT/source_snapshot"

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
  "$STAGE1/evaluation/targeted/report.json"
  "$STAGE1/evaluation/targeted/held_per_query.csv.gz"
  "$WARM_START"
  "$STAGE3_TRIPLETS/report.json"
  "$STAGE3_TRIPLETS/train_pool.npz"
  "$STAGE3_TRIPLETS/validation_pool.npz"
  "$STAGE3_TRIPLETS/action_spectra.npz"
  "$STAGE3_TRIPLETS/selected_actions.csv.gz"
  "$ZERO_AUDIT/report.json"
  "$ZERO_AUDIT/action_geometry.csv.gz"
  "$GRAPH" "$EMBEDDING_CACHE" "$SOURCE_MANIFEST"
  "$DATA" "$OFFICIAL_SLIM" "$ARCHITECTURE" "$MATURE_E8"
  tasks/repair_noise_dreams_native_multidifficulty_stage3.py
  tasks/train_noise_dreams_native_residual_stage2.py
  tasks/prepare_official_embedding_checkpoint.py
  tasks/evaluate_noise_dreams_native.py
  tasks/summarize_noise_dreams_native_residual_stage2.py
  tasks/test_noise_dreams_native_stage3_calibrated_repair.py
)
for path in "${required_files[@]}"; do
  [[ -f "$path" ]] || { echo "Missing calibrated Stage-3 repair input: $path" >&2; exit 5; }
done

# Freeze the exact source used by both arms.  The repair selector consumes the
# already-built Stage-3 relations and zero-update geometry; it never re-mines.
mkdir -p "$SOURCE_SNAPSHOT"
cp -a dreams "$SOURCE_SNAPSHOT/dreams"
cp -a tasks "$SOURCE_SNAPSHOT/tasks"
export PYTHONPATH="$SOURCE_SNAPSHOT:$SOURCE_SNAPSHOT/tasks:${PYTHONPATH:-}"
S3_TASKS="$SOURCE_SNAPSHOT/tasks"

python -m py_compile \
  "$S3_TASKS/repair_noise_dreams_native_multidifficulty_stage3.py" \
  "$S3_TASKS/train_noise_dreams_native_residual_stage2.py" \
  "$S3_TASKS/evaluate_noise_dreams_native.py" \
  "$S3_TASKS/summarize_noise_dreams_native_residual_stage2.py" \
  "$S3_TASKS/test_noise_dreams_native_stage3_calibrated_repair.py"
python -u "$S3_TASKS/test_noise_dreams_native_stage3_calibrated_repair.py"

CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[0]}" python -u \
  "$S3_TASKS/repair_noise_dreams_native_multidifficulty_stage3.py" \
  --stage3-triplets "$STAGE3_TRIPLETS" \
  --zero-update-audit "$ZERO_AUDIT" \
  --data "$DATA" --warm-start-checkpoint "$WARM_START" \
  --architecture-checkpoint "$ARCHITECTURE" \
  --output "$TRIPLETS" \
  --minimum-action-events 1500 --protection-events 256 --batch-size 64

run_arm() {
  local gpu="$1"
  local arm="$2"
  CUDA_VISIBLE_DEVICES="$gpu" python -u \
    "$S3_TASKS/train_noise_dreams_native_residual_stage2.py" \
    --curriculum multidifficulty_stage3_repair \
    --data "$DATA" --triplet-dir "$TRIPLETS" \
    --warm-start-checkpoint "$WARM_START" \
    --architecture-checkpoint "$ARCHITECTURE" \
    --output "$ARMS/$arm" --arm "$arm" --seed 3407 \
    --lr 1e-6 --weight-decay 0 --triplet-loss-margin 0.1 \
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
  echo "Calibrated Stage-3 training failure: targeted=$STATUS_TARGETED control=$STATUS_CONTROL" >&2
  exit 6
fi

mkdir -p "$RUN_ROOT/training_reports"
cp "$ARMS/targeted/report.json" "$RUN_ROOT/training_reports/targeted.json"
cp "$ARMS/control/report.json" "$RUN_ROOT/training_reports/control.json"

python -u "$S3_TASKS/prepare_official_embedding_checkpoint.py" \
  --source "$ARMS/targeted/final.ckpt" --output "$ARMS/targeted_slim.pt"
rm -f -- "$ARMS/targeted/final.ckpt"
python -u "$S3_TASKS/prepare_official_embedding_checkpoint.py" \
  --source "$ARMS/control/final.ckpt" --output "$ARMS/control_slim.pt"
rm -f -- "$ARMS/control/final.ckpt"

evaluate_arm() {
  local gpu="$1"
  local arm="$2"
  CUDA_VISIBLE_DEVICES="$gpu" python -u "$S3_TASKS/evaluate_noise_dreams_native.py" \
    --graph "$GRAPH" --embedding-cache "$EMBEDDING_CACHE" \
    --official-slim-checkpoint "$OFFICIAL_SLIM" \
    --source-manifest "$SOURCE_MANIFEST" --data "$DATA" \
    --checkpoint "$ARMS/${arm}_slim.pt" \
    --architecture-checkpoint "$ARCHITECTURE" \
    --output "$EVALUATION/$arm" --arm "$arm" \
    --outer-fold "$FOLD" --formula-fold-seed 20260825 \
    --batch-size 64 --bootstrap-resamples 10000 --bootstrap-seed 20260927 \
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
  echo "Calibrated Stage-3 held evaluation failure: targeted=$STATUS_EVAL_TARGETED control=$STATUS_EVAL_CONTROL" >&2
  exit 7
fi

python -u "$S3_TASKS/summarize_noise_dreams_native_residual_stage2.py" \
  --stage1 "$STAGE1/evaluation/targeted" \
  --targeted "$EVALUATION/targeted" --control "$EVALUATION/control" \
  --output "$RUN_ROOT/summary" \
  --result-prefix NOISE_DREAMS_NATIVE_MULTIDIFFICULTY_STAGE3_REPAIR \
  --absolute-target-pp 5.0 --bootstrap-resamples 10000 --seed 20260927

PROMOTION_STATUS=$(python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
  "$RUN_ROOT/summary/report.json")
if [[ "$PROMOTION_STATUS" == "NOISE_DREAMS_NATIVE_MULTIDIFFICULTY_STAGE3_REPAIR_PROMOTION_PASS" ]]; then
  mkdir -p "$RUN_ROOT/checkpoint"
  cp "$ARMS/targeted_slim.pt" "$RUN_ROOT/checkpoint/targeted_slim.pt"
fi

echo "=== Noise native calibrated Stage-3 repair result: $RUN_ROOT ==="
