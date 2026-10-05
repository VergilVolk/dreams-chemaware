#!/bin/bash
#SBATCH --job-name=noise_s1_rrf_confirm
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=36:00:00
#SBATCH --output=data/validation/noise_s1_rrf_confirm_%j.out
#SBATCH --error=data/validation/noise_s1_rrf_confirm_%j.err

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

GRAPH="data/validation/noise_corrected_candidate_graph_v1_20260906/candidate_graph.npz"
DATA="data/models/MassSpecGym_MurckoHist_split.hdf5"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
STAGE1="data/validation/noise_dreams_hard_positive_native_fold_0_run_2344820"
STAGE1_NATIVE="$STAGE1/checkpoint/targeted_final.ckpt"
STAGE1_REPORT="$STAGE1/evaluation/targeted/report.json"
GNPS="data/validation/gnps_gold_silver_10ppm_benchmark_v1"
RUN_ROOT="data/validation/noise_stage1_rrf_confirmation_run_${SLURM_JOB_ID}"
LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
LOCAL_ROOT="$LOCAL_BASE/noise_stage1_rrf_confirmation_${SLURM_JOB_ID}"
STAGE1_SLIM="$LOCAL_ROOT/stage1_targeted_slim.pt"
INTERNAL_EMBEDDINGS="$LOCAL_ROOT/stage1_internal_embeddings.npz"
GNPS_EMBEDDINGS="$LOCAL_ROOT/stage1_gnps_embeddings.npz"

[[ ! -e "$RUN_ROOT" ]] || { echo "Output collision: $RUN_ROOT" >&2; exit 4; }
mkdir -p "$RUN_ROOT" "$LOCAL_ROOT"
cleanup_local() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup_local EXIT

required_files=(
  "$GRAPH" "$DATA" "$ARCHITECTURE"
  "$STAGE1_NATIVE" "$STAGE1_REPORT"
  "$GNPS/spectra.mgf" "$GNPS/manifest.csv.gz"
  "$GNPS/panel_identity_disjoint.npz" "$GNPS/panel_formula_disjoint.npz"
  tasks/prepare_official_embedding_checkpoint.py
  tasks/encode_noise_corrected_graph_checkpoint.py
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py
  tasks/noise_stage1_rrf_core.py
  tasks/evaluate_noise_stage1_rrf_confirmation.py
  tasks/evaluate_gnps_stage1_rrf_confirmation.py
  tasks/summarize_noise_stage1_rrf_confirmation.py
  tasks/test_noise_stage1_rrf_confirmation.py
  tasks/test_noise_stage1_rrf_confirmation_sbatch.py
)
for path in "${required_files[@]}"; do
  [[ -f "$path" ]] || { echo "Missing Stage-1 RRF confirmation input: $path" >&2; exit 5; }
done

python -m py_compile \
  tasks/encode_noise_corrected_graph_checkpoint.py \
  tasks/noise_stage1_rrf_core.py \
  tasks/evaluate_noise_stage1_rrf_confirmation.py \
  tasks/evaluate_gnps_stage1_rrf_confirmation.py \
  tasks/summarize_noise_stage1_rrf_confirmation.py \
  tasks/test_noise_stage1_rrf_confirmation.py
python -u tasks/test_noise_stage1_rrf_confirmation_sbatch.py
python -u tasks/test_noise_stage1_rrf_confirmation.py

# Format conversion only: no optimization and no weight change.
python -u tasks/prepare_official_embedding_checkpoint.py \
  --source "$STAGE1_NATIVE" --output "$STAGE1_SLIM"

CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[0]}" python -u \
  tasks/encode_noise_corrected_graph_checkpoint.py \
  --graph "$GRAPH" --data "$DATA" \
  --checkpoint "$STAGE1_SLIM" --architecture-checkpoint "$ARCHITECTURE" \
  --output "$INTERNAL_EMBEDDINGS" --batch-size 64 \
  >"$RUN_ROOT/encode_internal.out" 2>"$RUN_ROOT/encode_internal.err" &
PID_INTERNAL=$!

CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[1]}" python -u \
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  --benchmark "$GNPS" --checkpoint "$STAGE1_SLIM" \
  --architecture-checkpoint "$ARCHITECTURE" \
  --output "$GNPS_EMBEDDINGS" --batch-size 64 --label noise-stage1-targeted \
  >"$RUN_ROOT/encode_gnps.out" 2>"$RUN_ROOT/encode_gnps.err" &
PID_GNPS=$!

set +e
wait "$PID_INTERNAL"; STATUS_INTERNAL=$?
wait "$PID_GNPS"; STATUS_GNPS=$?
set -e
if (( STATUS_INTERNAL != 0 || STATUS_GNPS != 0 )); then
  tail -n 120 "$RUN_ROOT/encode_internal.err" >&2 || true
  tail -n 120 "$RUN_ROOT/encode_gnps.err" >&2 || true
  echo "Stage-1 encoding failed: internal=$STATUS_INTERNAL gnps=$STATUS_GNPS" >&2
  exit 6
fi
cp "${INTERNAL_EMBEDDINGS%.npz}.json" "$RUN_ROOT/internal_encoding.json"
cp "${GNPS_EMBEDDINGS%.npz}.json" "$RUN_ROOT/gnps_encoding.json"

python -u tasks/evaluate_noise_stage1_rrf_confirmation.py \
  --graph "$GRAPH" --embeddings "$INTERNAL_EMBEDDINGS" \
  --data "$DATA" \
  --stage1-report "$STAGE1_REPORT" --output "$RUN_ROOT/internal" \
  --outer-fold 0 --formula-fold-seed 20260825 \
  --bootstrap-resamples 10000 --bootstrap-seed 20260929 \
  >"$RUN_ROOT/evaluate_internal.out" 2>"$RUN_ROOT/evaluate_internal.err" &
PID_EVAL_INTERNAL=$!

python -u tasks/evaluate_gnps_stage1_rrf_confirmation.py \
  --benchmark "$GNPS" --stage1-embeddings "$GNPS_EMBEDDINGS" \
  --output "$RUN_ROOT/gnps" \
  --bootstrap-resamples 10000 --bootstrap-seed 20260929 \
  >"$RUN_ROOT/evaluate_gnps.out" 2>"$RUN_ROOT/evaluate_gnps.err" &
PID_EVAL_GNPS=$!

set +e
wait "$PID_EVAL_INTERNAL"; STATUS_EVAL_INTERNAL=$?
wait "$PID_EVAL_GNPS"; STATUS_EVAL_GNPS=$?
set -e
if (( STATUS_EVAL_INTERNAL != 0 || STATUS_EVAL_GNPS != 0 )); then
  tail -n 160 "$RUN_ROOT/evaluate_internal.err" >&2 || true
  tail -n 160 "$RUN_ROOT/evaluate_gnps.err" >&2 || true
  echo "Stage-1 fixed-RRF evaluation failed: internal=$STATUS_EVAL_INTERNAL gnps=$STATUS_EVAL_GNPS" >&2
  exit 7
fi

python -u tasks/summarize_noise_stage1_rrf_confirmation.py \
  --internal "$RUN_ROOT/internal/report.json" \
  --gnps "$RUN_ROOT/gnps/report.json" \
  --output "$RUN_ROOT/summary/report.json"

echo "=== Stage-1 fixed-RRF internal + independent GNPS result: $RUN_ROOT ==="
