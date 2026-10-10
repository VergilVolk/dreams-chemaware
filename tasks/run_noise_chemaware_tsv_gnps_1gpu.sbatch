#!/bin/bash
#SBATCH --job-name=noise_chem_tsv
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/noise_chem_tsv_%j.out
#SBATCH --error=data/validation/noise_chem_tsv_%j.err

# Fixed 24-hour Noise V1 x ChemAware Phase-A merge decision.
# Exactly two constructed arms: linear task-vector mean and official TSV-Merge.
# MassSpecGym is not read. Enveda-180 remains unopened.
set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

BASE="data/e1/official_embedding_slim.pt"
NOISE="data/validation/noise_relation_t1_t3_run_2347055/checkpoint/primary_seed_3407_slim.pt"
CHEM_NATIVE="data/validation/chemaware_online_role_budget_native/run_2345481/phasea_reproduction_gate/chemaware_phasea_max_boundary_step2000.ckpt"
BENCHMARK="data/validation/gnps_gold_silver_10ppm_benchmark_v1"
ARCHITECTURE="dreams/models/pretrained/ssl_model_server.pt"
RUN_ROOT="data/validation/noise_chemaware_tsv_gnps_run_${SLURM_JOB_ID}"
LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
LOCAL_ROOT="$LOCAL_BASE/noise_chem_tsv_${SLURM_JOB_ID}"
CHEM_SLIM="$LOCAL_ROOT/chemaware_phasea_slim.pt"
LINEAR="$RUN_ROOT/checkpoints/linear_task_vector_mean.pt"
TSV="$RUN_ROOT/checkpoints/tsv_merge_scale1.pt"

EXPECTED_BASE_SHA="8928f908606c0bd652c5a4107d3c35102f660622958c225a1f625abe4b1ba245"
EXPECTED_NOISE_SHA="01ac8c7bfa75d89862e0b1eb2b5ffb297955cffae471fd483ad6c38d13f33beb"
EXPECTED_CHEM_SHA="a8428329ca1d12bfe735f3f8b848ed020a8db18d0a654d4f62cdbeb006e61135"

[[ ! -e "$RUN_ROOT" ]] || { echo "Output collision: $RUN_ROOT" >&2; exit 3; }
mkdir -p "$RUN_ROOT/checkpoints" "$RUN_ROOT/embeddings" "$RUN_ROOT/evaluation" "$LOCAL_ROOT"
cleanup_local() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup_local EXIT

required_files=(
  "$BASE" "$NOISE" "$CHEM_NATIVE" "$ARCHITECTURE"
  "$BENCHMARK/spectra.mgf" "$BENCHMARK/manifest.csv.gz"
  "$BENCHMARK/panel_identity_disjoint.npz" "$BENCHMARK/panel_formula_disjoint.npz"
  tasks/normalize_dreams_slim_checkpoint.py
  tasks/merge_dreams_tsv.py
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py
  tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py
  tasks/select_dreams_tsv_gnps.py
)
for path in "${required_files[@]}"; do
  [[ -f "$path" ]] || { echo "Missing TSV input: $path" >&2; exit 4; }
done

check_sha() {
  local path="$1" expected="$2" actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$actual" == "$expected" ]] || {
    echo "Checkpoint SHA256 mismatch: $path ($actual != $expected)" >&2
    exit 5
  }
}
check_sha "$BASE" "$EXPECTED_BASE_SHA"
check_sha "$NOISE" "$EXPECTED_NOISE_SHA"
check_sha "$CHEM_NATIVE" "$EXPECTED_CHEM_SHA"

python -m py_compile \
  tasks/normalize_dreams_slim_checkpoint.py \
  tasks/merge_dreams_tsv.py \
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
  tasks/select_dreams_tsv_gnps.py

# Format conversion only; no training or data access.
python -u tasks/normalize_dreams_slim_checkpoint.py \
  --input "$CHEM_NATIVE" --output "$CHEM_SLIM"

# One construction call, two fixed arms, no coefficient/rank search.
python -u tasks/merge_dreams_tsv.py \
  --base "$BASE" --noise "$NOISE" --chem "$CHEM_SLIM" \
  --linear-output "$LINEAR" --tsv-output "$TSV" --device cuda \
  | tee "$RUN_ROOT/merge.stdout"

encode() {
  local label="$1" checkpoint="$2" output="$3"
  python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
    --benchmark "$BENCHMARK" --checkpoint "$checkpoint" \
    --architecture-checkpoint "$ARCHITECTURE" --output "$output" \
    --batch-size 64 --label "$label"
}
encode official "$BASE" "$RUN_ROOT/embeddings/official.npz"
encode noise-v1 "$NOISE" "$RUN_ROOT/embeddings/noise_v1.npz"
encode linear-mean "$LINEAR" "$RUN_ROOT/embeddings/linear.npz"
encode tsv-scale1 "$TSV" "$RUN_ROOT/embeddings/tsv.npz"

evaluate() {
  local baseline="$1" candidate="$2" output="$3"
  python -u tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py \
    --benchmark "$BENCHMARK" --baseline-embeddings "$baseline" \
    --candidate-embeddings "$candidate" --output "$output" \
    --bootstrap-resamples 10000 --bootstrap-seed 20261010
}

# Absolute improvement vs official for both fixed arms.
evaluate "$RUN_ROOT/embeddings/official.npz" "$RUN_ROOT/embeddings/linear.npz" \
  "$RUN_ROOT/evaluation/linear_vs_official"
evaluate "$RUN_ROOT/embeddings/official.npz" "$RUN_ROOT/embeddings/tsv.npz" \
  "$RUN_ROOT/evaluation/tsv_vs_official"
# Direct tests of the intended advance over the current Noise encoder and the
# required TSV-vs-linear method comparison.
evaluate "$RUN_ROOT/embeddings/noise_v1.npz" "$RUN_ROOT/embeddings/tsv.npz" \
  "$RUN_ROOT/evaluation/tsv_vs_noise"
evaluate "$RUN_ROOT/embeddings/linear.npz" "$RUN_ROOT/embeddings/tsv.npz" \
  "$RUN_ROOT/evaluation/tsv_vs_linear"

python -u tasks/select_dreams_tsv_gnps.py \
  --linear-report "$RUN_ROOT/evaluation/linear_vs_official/report.json" \
  --tsv-report "$RUN_ROOT/evaluation/tsv_vs_official/report.json" \
  --linear-checkpoint "$LINEAR" --tsv-checkpoint "$TSV" \
  --output "$RUN_ROOT/selection.json" | tee "$RUN_ROOT/selection.stdout"

echo "COMPLETE: fixed Noise x ChemAware linear/TSV GNPS development decision -> $RUN_ROOT"
