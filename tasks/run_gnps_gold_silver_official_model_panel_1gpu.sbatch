#!/bin/bash
#SBATCH --job-name=gnps_official_panel
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/gnps_official_panel_%j.out
#SBATCH --error=data/validation/gnps_official_panel_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

BENCHMARK="data/validation/gnps_gold_silver_10ppm_benchmark_v1"
GENERAL_OFFICIAL="data/e1/official_embedding_slim.pt"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
STAGE1_NATIVE="data/validation/noise_dreams_hard_positive_native_fold_0_run_2344820/checkpoint/targeted_final.ckpt"
MASSPEC_DREAMS_URL="https://huggingface.co/roman-bushuiev/DreaMS/resolve/c81a62766b10dd1d39fcda3edec5ef88623e5f6b/embedding_model_massspecgym.ckpt"
MASSPEC_DREAMS_SHA256="d9b4e8b920ff367370c5a3fa154eaa951f90bafa58d5224a594424d9c54b1832"
RUN_ROOT="data/validation/gnps_gold_silver_official_model_panel_run_${SLURM_JOB_ID}"
LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
LOCAL_ROOT="$LOCAL_BASE/gnps_official_panel_${SLURM_JOB_ID}"

MASSPEC_NATIVE="$LOCAL_ROOT/embedding_model_massspecgym.ckpt"
MASSPEC_SLIM="$LOCAL_ROOT/embedding_model_massspecgym_slim.pt"
STAGE1_SLIM="$LOCAL_ROOT/noise_stage1_slim.pt"
GENERAL_EMB="$LOCAL_ROOT/general_official_embeddings.npz"
MASSPEC_EMB="$LOCAL_ROOT/massspecgym_official_embeddings.npz"
STAGE1_EMB="$LOCAL_ROOT/noise_stage1_embeddings.npz"

[[ ! -e "$RUN_ROOT" ]] || { echo "Output collision: $RUN_ROOT" >&2; exit 4; }
mkdir -p "$RUN_ROOT" "$LOCAL_ROOT"
cleanup_local() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup_local EXIT

required_files=(
  "$BENCHMARK/spectra.mgf"
  "$BENCHMARK/manifest.csv.gz"
  "$BENCHMARK/panel_identity_disjoint.npz"
  "$BENCHMARK/panel_formula_disjoint.npz"
  "$GENERAL_OFFICIAL" "$ARCHITECTURE" "$STAGE1_NATIVE"
  tasks/prepare_official_embedding_checkpoint.py
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py
  tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py
)
for path in "${required_files[@]}"; do
  [[ -f "$path" ]] || { echo "Missing official-model-panel input: $path" >&2; exit 5; }
done

LOCAL_FREE_KB=$(df -Pk "$LOCAL_BASE" | awk 'NR==2 {print $4}')
if [[ -z "$LOCAL_FREE_KB" ]] || (( LOCAL_FREE_KB < 6291456 )); then
  echo "Node-local space below 6 GiB: base=$LOCAL_BASE free_kb=${LOCAL_FREE_KB:-unknown}" >&2
  exit 5
fi

python -m py_compile \
  tasks/prepare_official_embedding_checkpoint.py \
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py

# Download only into node-local scratch; this consumes no shared /data02 quota.
curl --fail --location --retry 3 --retry-delay 5 \
  "$MASSPEC_DREAMS_URL" --output "$MASSPEC_NATIVE"
echo "$MASSPEC_DREAMS_SHA256  $MASSPEC_NATIVE" | sha256sum --check --strict

# Format conversions only; neither command performs optimization.
python -u tasks/prepare_official_embedding_checkpoint.py \
  --source "$MASSPEC_NATIVE" --output "$MASSPEC_SLIM"
python -u tasks/prepare_official_embedding_checkpoint.py \
  --source "$STAGE1_NATIVE" --output "$STAGE1_SLIM"
rm -f -- "$MASSPEC_NATIVE"

python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  --benchmark "$BENCHMARK" --checkpoint "$GENERAL_OFFICIAL" \
  --architecture-checkpoint "$ARCHITECTURE" --output "$GENERAL_EMB" \
  --batch-size 64 --label general-official
python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  --benchmark "$BENCHMARK" --checkpoint "$MASSPEC_SLIM" \
  --architecture-checkpoint "$ARCHITECTURE" --output "$MASSPEC_EMB" \
  --batch-size 64 --label massspecgym-official
python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  --benchmark "$BENCHMARK" --checkpoint "$STAGE1_SLIM" \
  --architecture-checkpoint "$ARCHITECTURE" --output "$STAGE1_EMB" \
  --batch-size 64 --label noise-stage1

python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
  --benchmark "$BENCHMARK" --baseline-embeddings "$GENERAL_EMB" \
  --candidate-embeddings "$MASSPEC_EMB" \
  --output "$RUN_ROOT/massspecgym_official_vs_general_official" \
  --bootstrap-resamples 10000 --bootstrap-seed 20260929

python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
  --benchmark "$BENCHMARK" --baseline-embeddings "$MASSPEC_EMB" \
  --candidate-embeddings "$STAGE1_EMB" \
  --output "$RUN_ROOT/noise_stage1_vs_massspecgym_official" \
  --bootstrap-resamples 10000 --bootstrap-seed 20260929

cp "${GENERAL_EMB%.npz}.json" "$RUN_ROOT/general_official_encoding.json"
cp "${MASSPEC_EMB%.npz}.json" "$RUN_ROOT/massspecgym_official_encoding.json"
cp "${STAGE1_EMB%.npz}.json" "$RUN_ROOT/noise_stage1_encoding.json"

cat > "$RUN_ROOT/provenance.txt" <<EOF
general_official_sha256=8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245
massspecgym_official_sha256=$MASSPEC_DREAMS_SHA256
massspecgym_official_huggingface_commit=c81a62766b10dd1d39fcda3edec5ef88623e5f6b
massspecgym_checkpoint_role=official_specialized_comparator_not_universal_replacement
noise_stage1_role=project_candidate
EOF

echo "=== GNPS official-model panel complete: $RUN_ROOT ==="
