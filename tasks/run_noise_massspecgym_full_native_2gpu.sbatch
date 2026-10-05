#!/bin/bash
#SBATCH --job-name=noise_msg_full
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=96:00:00
#SBATCH --output=data/validation/noise_msg_full_%j.out
#SBATCH --error=data/validation/noise_msg_full_%j.err

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
STAGE1_TRIPLETS="data/validation/noise_dreams_hard_positive_native_fold_0_run_2344820/triplets"
ACTION_BANK="$STAGE1_TRIPLETS/action_spectra.npz"
V1="data/validation/noise_relation_t1_t3_run_2347055/checkpoint/primary_seed_3407_slim.pt"
GNPS_BENCHMARK="data/validation/gnps_gold_silver_10ppm_benchmark_v1"
GNPS_BASELINE_RUN="${GNPS_BASELINE_RUN:-data/validation/gnps_absolute_baselines_run_2347472}"
OFFICIAL_GNPS="$GNPS_BASELINE_RUN/gnps/official_embeddings.npz"
V1_GNPS="$GNPS_BASELINE_RUN/gnps/v1_embeddings.npz"
RUN_ROOT="data/validation/noise_massspecgym_full_native_run_${SLURM_JOB_ID}"
LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
LOCAL_ROOT="$LOCAL_BASE/noise_massspecgym_full_native_${SLURM_JOB_ID}"
EMBEDDINGS="$LOCAL_ROOT/embeddings"
TRIPLETS="$LOCAL_ROOT/triplets"
TRAINING="$LOCAL_ROOT/training"
GNPS="$LOCAL_ROOT/gnps"

[[ ! -e "$RUN_ROOT" ]] || { echo "Output collision: $RUN_ROOT" >&2; exit 4; }
mkdir -p "$RUN_ROOT" "$LOCAL_ROOT" "$EMBEDDINGS" "$GNPS"
cleanup_local() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup_local EXIT

required=(
  "$GRAPH" "$DATA" "$ARCHITECTURE" "$ACTION_BANK" "$V1"
  "$STAGE1_TRIPLETS/train_pool.npz"
  "$GNPS_BENCHMARK/spectra.mgf" "$GNPS_BENCHMARK/manifest.csv.gz"
  "$GNPS_BENCHMARK/panel_identity_disjoint.npz"
  "$GNPS_BENCHMARK/panel_formula_disjoint.npz"
  "$OFFICIAL_GNPS" "$V1_GNPS"
  tasks/noise_massspecgym_full_triplet_core.py
  tasks/encode_noise_massspecgym_checkpoint.py
  tasks/build_noise_massspecgym_full_triplets.py
  tasks/train_noise_massspecgym_full_native.py
  tasks/summarize_noise_massspecgym_full_gnps.py
  tasks/test_noise_massspecgym_full_native.py
  tasks/test_noise_massspecgym_full_native_sbatch.py
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py
  tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "Missing full-MassSpecGym input: $path" >&2; exit 5; }
done

python -m py_compile \
  tasks/noise_massspecgym_full_triplet_core.py \
  tasks/encode_noise_massspecgym_checkpoint.py \
  tasks/build_noise_massspecgym_full_triplets.py \
  tasks/train_noise_massspecgym_full_native.py \
  tasks/summarize_noise_massspecgym_full_gnps.py \
  tasks/test_noise_massspecgym_full_native.py \
  tasks/test_noise_massspecgym_full_native_sbatch.py
python -u tasks/test_noise_massspecgym_full_native.py
python -u tasks/test_noise_massspecgym_full_native_sbatch.py

encode_shard() {
  local gpu="$1"
  local shard="$2"
  CUDA_VISIBLE_DEVICES="$gpu" python -u tasks/encode_noise_massspecgym_checkpoint.py \
    --data "$DATA" --checkpoint "$V1" --architecture-checkpoint "$ARCHITECTURE" \
    --output "$EMBEDDINGS/shard_${shard}.npz" \
    --shard-index "$shard" --shard-count 2 --batch-size 64 \
    >"$RUN_ROOT/encode_shard_${shard}.out" 2>"$RUN_ROOT/encode_shard_${shard}.err"
}
encode_shard "${ALLOCATED_GPUS[0]}" 0 & PID0=$!
encode_shard "${ALLOCATED_GPUS[1]}" 1 & PID1=$!
set +e
wait "$PID0"; STATUS0=$?
wait "$PID1"; STATUS1=$?
set -e
if (( STATUS0 != 0 || STATUS1 != 0 )); then
  tail -n 160 "$RUN_ROOT/encode_shard_0.err" >&2 || true
  tail -n 160 "$RUN_ROOT/encode_shard_1.err" >&2 || true
  echo "Full MassSpecGym V1 encoding failed: $STATUS0/$STATUS1" >&2
  exit 6
fi

python -u tasks/build_noise_massspecgym_full_triplets.py \
  --graph "$GRAPH" --data "$DATA" \
  --embedding-shard "$EMBEDDINGS/shard_0.npz" \
  --embedding-shard "$EMBEDDINGS/shard_1.npz" \
  --stage1-triplets "$STAGE1_TRIPLETS" --output "$TRIPLETS" \
  --seed 20261001 --ppm 10 --max-measured-positives 6 \
  --max-negatives 8 --fallback-neighbours 64 \
  >"$RUN_ROOT/build_triplets.out" 2>"$RUN_ROOT/build_triplets.err"

CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[0]}" python -u \
  tasks/train_noise_massspecgym_full_native.py \
  --data "$DATA" --triplet-dir "$TRIPLETS" \
  --stage1-action-bank "$ACTION_BANK" \
  --warm-start-checkpoint "$V1" --architecture-checkpoint "$ARCHITECTURE" \
  --output "$TRAINING" --seed 3407 --lr 5e-6 --weight-decay 0 \
  --triplet-loss-margin 0.1 --batch-size 4 --max-epochs 1 \
  --n-highest-peaks 100 \
  >"$RUN_ROOT/train.out" 2>"$RUN_ROOT/train.err"

CUDA_VISIBLE_DEVICES="${ALLOCATED_GPUS[1]}" python -u \
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  --benchmark "$GNPS_BENCHMARK" --checkpoint "$TRAINING/final_slim.pt" \
  --architecture-checkpoint "$ARCHITECTURE" \
  --output "$GNPS/new_embeddings.npz" --batch-size 64 \
  --label noise_massspecgym_full_native \
  >"$RUN_ROOT/gnps_encode.out" 2>"$RUN_ROOT/gnps_encode.err"

python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
  --benchmark "$GNPS_BENCHMARK" \
  --baseline-embeddings "$OFFICIAL_GNPS" --candidate-embeddings "$GNPS/new_embeddings.npz" \
  --output "$GNPS/official_vs_new" --bootstrap-resamples 10000 --bootstrap-seed 20261001
python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
  --benchmark "$GNPS_BENCHMARK" \
  --baseline-embeddings "$V1_GNPS" --candidate-embeddings "$GNPS/new_embeddings.npz" \
  --output "$GNPS/v1_vs_new" --bootstrap-resamples 10000 --bootstrap-seed 20261001

python -u tasks/summarize_noise_massspecgym_full_gnps.py \
  --official-vs-new "$GNPS/official_vs_new/report.json" \
  --v1-vs-new "$GNPS/v1_vs_new/report.json" \
  --training-report "$TRAINING/report.json" \
  --triplet-report "$TRIPLETS/report.json" \
  --output "$RUN_ROOT/summary/report.json"

mkdir -p "$RUN_ROOT/checkpoint" "$RUN_ROOT/triplets" "$RUN_ROOT/training" "$RUN_ROOT/gnps"
cp "$TRAINING/final_slim.pt" "$RUN_ROOT/checkpoint/final_slim.pt"
cp "$TRAINING/report.json" "$RUN_ROOT/training/report.json"
cp "$TRIPLETS/report.json" "$RUN_ROOT/triplets/report.json"
cp "$TRIPLETS/train_pool.npz" "$RUN_ROOT/triplets/train_pool.npz"
cp "$TRIPLETS/noise_spectra.npz" "$RUN_ROOT/triplets/noise_spectra.npz"
cp -r "$GNPS/official_vs_new" "$RUN_ROOT/gnps/official_vs_new"
cp -r "$GNPS/v1_vs_new" "$RUN_ROOT/gnps/v1_vs_new"

echo "=== Full MassSpecGym native Noise result: $RUN_ROOT ==="
