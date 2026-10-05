#!/bin/bash
#SBATCH --job-name=bioaware_ub_gnps
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/bioaware_unified_benchmark_gnps_%j.out
#SBATCH --error=data/validation/bioaware_unified_benchmark_gnps_%j.err

# BioAware unified benchmark, Track A, GNPS Gold/Silver sealed panels.
# Encodes the frozen benchmark with the official DreaMS checkpoint, builds
# official molecule-max predictions plus frozen classical baselines
# (modified cosine, spectral entropy), and evaluates every method with the
# single unified evaluator.  No B47 truth is touched; no model is trained.

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
RUN_ROOT="data/validation/bioaware_unified_benchmark_gnps_run_${SLURM_JOB_ID}"
LOCAL_BASE="${SLURM_TMPDIR:-/tmp}"
LOCAL_ROOT="$LOCAL_BASE/bioaware_ub_gnps_${SLURM_JOB_ID}"
OFFICIAL_EMBEDDINGS="$LOCAL_ROOT/official_embeddings.npz"
OFFICIAL_ENCODING_REPORT="$LOCAL_ROOT/official_embeddings.json"

PANEL_IDENTITY="$BENCHMARK/panel_identity_disjoint.npz"
PANEL_FORMULA="$BENCHMARK/panel_formula_disjoint.npz"

[[ ! -e "$RUN_ROOT" ]] || { echo "Output collision: $RUN_ROOT" >&2; exit 4; }
mkdir -p "$RUN_ROOT" "$LOCAL_ROOT"
# The four builders below own their output directories and deliberately
# fail closed when one already exists.  Do not pre-create those directories:
# doing so makes a fresh run look like an overwrite before any baseline is
# computed.
cleanup_local() { rm -rf -- "$LOCAL_ROOT"; }
trap cleanup_local EXIT

required_files=(
  "$BENCHMARK/checksums.sha256"
  "$BENCHMARK/spectra.mgf"
  "$BENCHMARK/manifest.csv.gz"
  "$PANEL_IDENTITY"
  "$PANEL_FORMULA"
  "$OFFICIAL" "$ARCHITECTURE"
  tasks/encode_gnps_gold_silver_10ppm_checkpoint.py
  tasks/evaluate_gnps_gold_silver_10ppm_embeddings.py
  tasks/bioaware_unified_benchmark_core.py
  tasks/evaluate_bioaware_unified_benchmark.py
  tasks/build_bioaware_tracka_gnps_classical_baselines.py
  tasks/build_bioaware_tracka_gnps_official_predictions.py
  tasks/test_bioaware_unified_benchmark_core.py
  tasks/test_evaluate_bioaware_unified_benchmark.py
  tasks/test_bioaware_tracka_gnps_official_predictions.py
)
for path in "${required_files[@]}"; do
  [[ -f "$path" ]] || { echo "Missing BioAware benchmark input: $path" >&2; exit 5; }
done

python -m py_compile \
  tasks/bioaware_unified_benchmark_core.py \
  tasks/evaluate_bioaware_unified_benchmark.py \
  tasks/build_bioaware_tracka_gnps_classical_baselines.py \
  tasks/build_bioaware_tracka_gnps_official_predictions.py

# Contract tests before any compute leaves the login-free path.
python -u tasks/test_bioaware_unified_benchmark_core.py
python -u tasks/test_evaluate_bioaware_unified_benchmark.py
python -u tasks/test_bioaware_tracka_gnps_official_predictions.py

# 1. Official DreaMS encoding of exactly the panel-required spectra.
python -u tasks/encode_gnps_gold_silver_10ppm_checkpoint.py \
  --benchmark "$BENCHMARK" \
  --checkpoint "$OFFICIAL" \
  --architecture-checkpoint "$ARCHITECTURE" \
  --output "$OFFICIAL_EMBEDDINGS" \
  --batch-size 64 --label official

# 2. Frozen classical baselines on both sealed panels (CPU, deterministic).
python -u tasks/build_bioaware_tracka_gnps_classical_baselines.py \
  --panel-dir "$BENCHMARK" \
  --panel-file "$PANEL_FORMULA" \
  --spectra-mgf "$BENCHMARK/spectra.mgf" \
  --source-label gnps_gold_silver_formula_disjoint \
  --output-dir "$LOCAL_ROOT/classical_formula"

python -u tasks/build_bioaware_tracka_gnps_classical_baselines.py \
  --panel-dir "$BENCHMARK" \
  --panel-file "$PANEL_IDENTITY" \
  --spectra-mgf "$BENCHMARK/spectra.mgf" \
  --source-label gnps_gold_silver_identity_disjoint \
  --output-dir "$LOCAL_ROOT/classical_identity"

# 3. Official molecule-max predictions on both sealed panels.
python -u tasks/build_bioaware_tracka_gnps_official_predictions.py \
  --panel-file "$PANEL_FORMULA" \
  --embeddings "$OFFICIAL_EMBEDDINGS" \
  --output-dir "$LOCAL_ROOT/official_formula" \
  --source-label gnps_gold_silver_formula_disjoint \
  --expected-queries 5261 --expected-candidate-rows 24401

python -u tasks/build_bioaware_tracka_gnps_official_predictions.py \
  --panel-file "$PANEL_IDENTITY" \
  --embeddings "$OFFICIAL_EMBEDDINGS" \
  --output-dir "$LOCAL_ROOT/official_identity" \
  --source-label gnps_gold_silver_identity_disjoint \
  --expected-queries 10995 --expected-candidate-rows 87518

# 4. Unified evaluation, sealed-external scope, official DreaMS baseline.
python -u tasks/evaluate_bioaware_unified_benchmark.py \
  --candidate-manifest "$LOCAL_ROOT/classical_formula/candidate_manifest.csv.gz" \
  --method official_dreams="$LOCAL_ROOT/official_formula/predictions_official_dreams.csv.gz" \
  --method modified_cosine="$LOCAL_ROOT/classical_formula/predictions_modified_cosine.csv.gz" \
  --method spectral_entropy="$LOCAL_ROOT/classical_formula/predictions_spectral_entropy.csv.gz" \
  --baseline-method official_dreams \
  --benchmark-track spectrum_reference \
  --protocol-scope sealed_external \
  --output-dir "$RUN_ROOT/formula_disjoint_summary" \
  --expected-queries 5261 --expected-candidate-rows 24401 \
  | tee "$RUN_ROOT/formula_disjoint_summary.stdout"

python -u tasks/evaluate_bioaware_unified_benchmark.py \
  --candidate-manifest "$LOCAL_ROOT/classical_identity/candidate_manifest.csv.gz" \
  --method official_dreams="$LOCAL_ROOT/official_identity/predictions_official_dreams.csv.gz" \
  --method modified_cosine="$LOCAL_ROOT/classical_identity/predictions_modified_cosine.csv.gz" \
  --method spectral_entropy="$LOCAL_ROOT/classical_identity/predictions_spectral_entropy.csv.gz" \
  --baseline-method official_dreams \
  --benchmark-track spectrum_reference \
  --protocol-scope sealed_external \
  --output-dir "$RUN_ROOT/identity_disjoint_summary" \
  --expected-queries 10995 --expected-candidate-rows 87518 \
  | tee "$RUN_ROOT/identity_disjoint_summary.stdout"

# 5. Archive the encoding report and provenance next to the frozen summaries.
[[ -f "$OFFICIAL_ENCODING_REPORT" ]] || {
  echo "Missing official encoding report: $OFFICIAL_ENCODING_REPORT" >&2
  exit 8
}
cp "$OFFICIAL_ENCODING_REPORT" "$RUN_ROOT/official_encoding.json"
sha256sum "$OFFICIAL" "$PANEL_IDENTITY" "$PANEL_FORMULA" \
  > "$RUN_ROOT/input_checksums.sha256"

echo "=== BioAware unified benchmark GNPS run: $RUN_ROOT ==="
