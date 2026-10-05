#!/bin/bash
#SBATCH --job-name=bioaware_b44_portable
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=bioaware_b44_portable_%j.out
#SBATCH --error=bioaware_b44_portable_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
BUNDLE_DIR="data/validation/bioaware_b44_portable_bundle_20260913_v3"
RESULT_DIR="data/validation/bioaware_b44_massbank_once_${SLURM_JOB_ID}"

echo "=== BioAware B44 portable external evaluation started: $(date) ==="
for path in \
  "${BUNDLE_DIR}/report.json" \
  "${BUNDLE_DIR}/preprocessed_spectra.npz" \
  "${BUNDLE_DIR}/panel/report.json" \
  "${BUNDLE_DIR}/panel/queries.csv.gz" \
  "${BUNDLE_DIR}/panel/candidate_references.csv.gz" \
  "${BUNDLE_DIR}/artifact/report.json" \
  "${BUNDLE_DIR}/artifact/portable_model.npz" \
  "${BUNDLE_DIR}/artifact/catalogue_lookup.csv.gz" \
  data/e1/official_embedding_slim.pt \
  dreams/models/pretrained/ssl_model_server.pt; do
  test -s "${path}" || { echo "Missing dependency: ${path}" >&2; exit 2; }
done

"${PYTHON_BIN}" tasks/validate_bioaware_b44_portable_bundle.py --input-dir "${BUNDLE_DIR}"
"${PYTHON_BIN}" tasks/test_bioaware_b44_massbank_once.py
"${PYTHON_BIN}" tasks/evaluate_bioaware_b44_massbank_once.py \
  --panel-dir "${BUNDLE_DIR}/panel" \
  --artifact-dir "${BUNDLE_DIR}/artifact" \
  --spectrum-cache "${BUNDLE_DIR}/preprocessed_spectra.npz" \
  --portable-model "${BUNDLE_DIR}/artifact/portable_model.npz" \
  --output-dir "${RESULT_DIR}" \
  --device cuda
"${PYTHON_BIN}" tasks/validate_bioaware_b44_massbank_once.py --input-dir "${RESULT_DIR}"

echo "=== BioAware B44 portable external evaluation completed: $(date) ==="
echo "Result: ${RESULT_DIR}"
