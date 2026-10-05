#!/bin/bash
#SBATCH --job-name=noise_wse_native
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/noise_wse_native_%j.out
#SBATCH --error=data/validation/noise_wse_native_%j.err

# WSE-discordant native DreaMS continuation (pre-registered design).
#
# Weighted spectral entropy is a label-blind spectrum score used ONLY to pick
# training relations on MassSpecGym: measured hard positives that WSE still
# ranks above the candidate-graph hard negatives while the frozen V1 encoder
# still has an active native margin.  WSE never enters the loss; the model,
# dataset, cosine triplet-margin loss and Adam are unmodified DreaMS.  Every
# corrected-graph query contributes exactly one triplet (WSE relation, else
# one exact Stage-1 action replay, else a measured V1-preservation relation,
# else a light singleton noise-to-clean relation).  Warm start is the V1
# champion; both registered seeds run once.  GNPS never enters mining or
# training and is the only external judge.
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

LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
LOCAL_DEPS="$LOCAL_BASE/noise_wse_native_deps_${SLURM_JOB_ID}"
mkdir -p "$LOCAL_DEPS"
python - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit(
        f"pinned ms_entropy wheel requires Python 3.11, got {sys.version.split()[0]}"
    )
PY
python -m pip install --no-deps --target "$LOCAL_DEPS" \
  third_party/noise_article_wheels/ms_entropy-1.5.2-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl
export PYTHONPATH="$LOCAL_DEPS:$PYTHONPATH"

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
RUN_ROOT="data/validation/noise_wse_discordant_native_run_${SLURM_JOB_ID}"
LOCAL_ROOT="$LOCAL_BASE/noise_wse_discordant_native_${SLURM_JOB_ID}"
EMBEDDINGS="$LOCAL_ROOT/embeddings"
TRIPLETS="$LOCAL_ROOT/triplets"

[[ ! -e "$RUN_ROOT" ]] || { echo "Output collision: $RUN_ROOT" >&2; exit 4; }
mkdir -p "$RUN_ROOT" "$LOCAL_ROOT" "$EMBEDDINGS"
cleanup_local() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup_local EXIT

required=(
  "$GRAPH" "$DATA" "$ARCHITECTURE" "$V1" "$ACTION_BANK"
  "$STAGE1_TRIPLETS/train_pool.npz"
  "$GNPS_BENCHMARK/spectra.mgf" "$GNPS_BENCHMARK/manifest.csv.gz"
  "$GNPS_BENCHMARK/panel_identity_disjoint.npz"
  "$GNPS_BENCHMARK/panel_formula_disjoint.npz"
  "$OFFICIAL_GNPS" "$V1_GNPS"
  tasks/noise_wse_discordant_triplet_core.py
  tasks/encode_noise_massspecgym_checkpoint.py
  tasks/build_noise_wse_discordant_native_triplets.py
  tasks/train_noise_wse_discordant_native.py
  tasks/summarize_noise_wse_discordant_gnps.py
  tasks/test_noise_wse_discordant_native.py
  tasks/test_noise_wse_discordant_native_sbatch.py
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py
  tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py
)
for path in "${required[@]}"; do
  [[ -f "$path" ]] || { echo "Missing WSE-native input: $path" >&2; exit 5; }
done

python -u tasks/test_noise_wse_discordant_native.py
python -u tasks/test_noise_wse_discordant_native_sbatch.py

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
  echo "V1 MassSpecGym embedding shards failed: $STATUS0/$STATUS1" >&2
  exit 6
fi

python -u tasks/build_noise_wse_discordant_native_triplets.py \
  --graph "$GRAPH" --data "$DATA" \
  --embedding-shard "$EMBEDDINGS/shard_0.npz" \
  --embedding-shard "$EMBEDDINGS/shard_1.npz" \
  --stage1-triplets "$STAGE1_TRIPLETS" \
  --output "$TRIPLETS" \
  --seed 20261003 --native-margin 0.1 --fragment-tolerance 0.02 \
  --max-positive-candidates 6 --max-negative-candidates 8 \
  >"$RUN_ROOT/build_triplets.out" 2>"$RUN_ROOT/build_triplets.err"
cp "$TRIPLETS/report.json" "$RUN_ROOT/triplet_report.json"
cp "$TRIPLETS/selection_ledger.npz" "$RUN_ROOT/selection_ledger.npz"

train_seed() {
  local gpu="$1"
  local seed="$2"
  CUDA_VISIBLE_DEVICES="$gpu" python -u tasks/train_noise_wse_discordant_native.py \
    --data "$DATA" --triplet-dir "$TRIPLETS" \
    --stage1-action-bank "$ACTION_BANK" \
    --warm-start-checkpoint "$V1" \
    --architecture-checkpoint "$ARCHITECTURE" \
    --output "$LOCAL_ROOT/seed_${seed}" --seed "$seed" \
    >"$RUN_ROOT/train_seed_${seed}.out" 2>"$RUN_ROOT/train_seed_${seed}.err"
}
train_seed "${ALLOCATED_GPUS[0]}" 3407 & PID3407=$!
train_seed "${ALLOCATED_GPUS[1]}" 3408 & PID3408=$!
set +e
wait "$PID3407"; STATUS3407=$?
wait "$PID3408"; STATUS3408=$?
set -e
if (( STATUS3407 != 0 || STATUS3408 != 0 )); then
  echo "WSE-native training failed: $STATUS3407/$STATUS3408" >&2
  exit 7
fi

encode_gnps() {
  local gpu="$1"
  local seed="$2"
  CUDA_VISIBLE_DEVICES="$gpu" python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
    --benchmark "$GNPS_BENCHMARK" \
    --checkpoint "$LOCAL_ROOT/seed_${seed}/final_slim.pt" \
    --architecture-checkpoint "$ARCHITECTURE" \
    --output "$RUN_ROOT/gnps_seed_${seed}_embeddings.npz" --batch-size 64 \
    --label "wse_native_seed_${seed}" \
    >"$RUN_ROOT/gnps_encode_seed_${seed}.out" 2>"$RUN_ROOT/gnps_encode_seed_${seed}.err"
}
encode_gnps "${ALLOCATED_GPUS[0]}" 3407 & PIDG0=$!
encode_gnps "${ALLOCATED_GPUS[1]}" 3408 & PIDG1=$!
set +e
wait "$PIDG0"; STATUSG0=$?
wait "$PIDG1"; STATUSG1=$?
set -e
if (( STATUSG0 != 0 || STATUSG1 != 0 )); then
  echo "GNPS encoding failed: $STATUSG0/$STATUSG1" >&2
  exit 8
fi

for seed in 3407 3408; do
  python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
    --benchmark "$GNPS_BENCHMARK" \
    --baseline-embeddings "$OFFICIAL_GNPS" \
    --candidate-embeddings "$RUN_ROOT/gnps_seed_${seed}_embeddings.npz" \
    --output "$RUN_ROOT/gnps_official_vs_seed_${seed}" \
    --bootstrap-resamples 10000 --bootstrap-seed 20261003 \
    >"$RUN_ROOT/eval_official_seed_${seed}.out" 2>&1
  python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
    --benchmark "$GNPS_BENCHMARK" \
    --baseline-embeddings "$V1_GNPS" \
    --candidate-embeddings "$RUN_ROOT/gnps_seed_${seed}_embeddings.npz" \
    --output "$RUN_ROOT/gnps_v1_vs_seed_${seed}" \
    --bootstrap-resamples 10000 --bootstrap-seed 20261003 \
    >"$RUN_ROOT/eval_v1_seed_${seed}.out" 2>&1
done

python -u tasks/summarize_noise_wse_discordant_gnps.py \
  --triplet-report "$RUN_ROOT/triplet_report.json" \
  --training-report "$LOCAL_ROOT/seed_3407/report.json" \
  --training-report "$LOCAL_ROOT/seed_3408/report.json" \
  --vs-official "$RUN_ROOT/gnps_official_vs_seed_3407/report.json" \
  --vs-official "$RUN_ROOT/gnps_official_vs_seed_3408/report.json" \
  --vs-v1 "$RUN_ROOT/gnps_v1_vs_seed_3407/report.json" \
  --vs-v1 "$RUN_ROOT/gnps_v1_vs_seed_3408/report.json" \
  --output "$RUN_ROOT/summary/report.json"

mkdir -p "$RUN_ROOT/seed_3407" "$RUN_ROOT/seed_3408"
cp "$LOCAL_ROOT/seed_3407/final_slim.pt" "$LOCAL_ROOT/seed_3407/report.json" "$RUN_ROOT/seed_3407/"
cp "$LOCAL_ROOT/seed_3408/final_slim.pt" "$LOCAL_ROOT/seed_3408/report.json" "$RUN_ROOT/seed_3408/"

echo "=== WSE-discordant native continuation complete: $RUN_ROOT (summary/report.json) ==="
