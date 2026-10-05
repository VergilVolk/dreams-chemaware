#!/bin/bash
#SBATCH --job-name=gnps_noise_s1_eval
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/gnps_noise_s1_eval_%j.out
#SBATCH --error=data/validation/gnps_noise_s1_eval_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

BENCHMARK="data/validation/gnps_gold_silver_10ppm_benchmark_v1"
OFFICIAL="data/e1/official_embedding_slim.pt"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
STAGE1="data/validation/noise_dreams_hard_positive_native_fold_0_run_2344820"
STAGE1_NATIVE="$STAGE1/checkpoint/targeted_final.ckpt"
RUN_ROOT="data/validation/gnps_gold_silver_noise_stage1_run_${SLURM_JOB_ID}"
LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
LOCAL_ROOT="$LOCAL_BASE/gnps_noise_s1_eval_${SLURM_JOB_ID}"
STAGE1_SLIM="$LOCAL_ROOT/stage1_targeted_slim.pt"
OFFICIAL_EMBEDDINGS="$LOCAL_ROOT/official_embeddings.npz"
STAGE1_EMBEDDINGS="$LOCAL_ROOT/stage1_embeddings.npz"
OFFICIAL_ENCODING_REPORT="$LOCAL_ROOT/official_embeddings.json"
STAGE1_ENCODING_REPORT="$LOCAL_ROOT/stage1_embeddings.json"

[[ ! -e "$RUN_ROOT" ]] || { echo "Output collision: $RUN_ROOT" >&2; exit 4; }
mkdir -p "$RUN_ROOT" "$LOCAL_ROOT"
cleanup_local() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup_local EXIT

required_files=(
  "$BENCHMARK/checksums.sha256"
  "$BENCHMARK/spectra.mgf"
  "$BENCHMARK/manifest.csv.gz"
  "$BENCHMARK/panel_identity_disjoint.npz"
  "$BENCHMARK/panel_formula_disjoint.npz"
  "$OFFICIAL" "$ARCHITECTURE" "$STAGE1_NATIVE"
  tasks/prepare_official_embedding_checkpoint.py
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py
  tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py
)
for path in "${required_files[@]}"; do
  [[ -f "$path" ]] || { echo "Missing GNPS Stage-1 evaluation input: $path" >&2; exit 5; }
done

python -m py_compile \
  tasks/prepare_official_embedding_checkpoint.py \
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py

# This is a checkpoint-format conversion only. It performs no optimization.
python -u tasks/prepare_official_embedding_checkpoint.py \
  --source "$STAGE1_NATIVE" --output "$STAGE1_SLIM"

python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  --benchmark "$BENCHMARK" \
  --checkpoint "$OFFICIAL" \
  --architecture-checkpoint "$ARCHITECTURE" \
  --output "$OFFICIAL_EMBEDDINGS" \
  --batch-size 64 --label official

python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  --benchmark "$BENCHMARK" \
  --checkpoint "$STAGE1_SLIM" \
  --architecture-checkpoint "$ARCHITECTURE" \
  --output "$STAGE1_EMBEDDINGS" \
  --batch-size 64 --label noise-stage1-targeted

python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
  --benchmark "$BENCHMARK" \
  --baseline-embeddings "$OFFICIAL_EMBEDDINGS" \
  --candidate-embeddings "$STAGE1_EMBEDDINGS" \
  --output "$RUN_ROOT/summary" \
  --bootstrap-resamples 10000 --bootstrap-seed 20260928 \
  | tee "$RUN_ROOT/evaluation.json.stdout"

[[ -f "$OFFICIAL_ENCODING_REPORT" ]] || {
  echo "Missing official encoding report: $OFFICIAL_ENCODING_REPORT" >&2
  exit 8
}
[[ -f "$STAGE1_ENCODING_REPORT" ]] || {
  echo "Missing Stage-1 encoding report: $STAGE1_ENCODING_REPORT" >&2
  exit 8
}
cp "$OFFICIAL_ENCODING_REPORT" "$RUN_ROOT/official_encoding.json"
cp "$STAGE1_ENCODING_REPORT" "$RUN_ROOT/stage1_encoding.json"
echo "=== GNPS Gold/Silver official vs Noise Stage-1: $RUN_ROOT ==="
