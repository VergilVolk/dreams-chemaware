#!/bin/bash
#SBATCH --job-name=bioaware_b44_once
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=bioaware_b44_once_%j.out
#SBATCH --error=bioaware_b44_once_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
PANEL_DIR="${PANEL_DIR:?Set PANEL_DIR to the sealed B44 MassBank panel}"
ARTIFACT_DIR="${ARTIFACT_DIR:?Set ARTIFACT_DIR to the frozen BioAware catalogue artefact}"
OUTPUT_DIR="data/validation/bioaware_b44_massbank_once_${SLURM_JOB_ID}"

echo "=== BioAware B44 one-time MassBank evaluation started: $(date) ==="
for path in \
  data/massbank/massbank_full.hdf5 \
  data/e1/official_embedding_slim.pt \
  dreams/models/pretrained/ssl_model_server.pt \
  "${PANEL_DIR}/report.json" \
  "${ARTIFACT_DIR}/report.json" \
  "${ARTIFACT_DIR}/model.joblib"; do
  test -s "${path}" || { echo "Missing dependency: ${path}" >&2; exit 2; }
done
"${PYTHON_BIN}" tasks/test_bioaware_b44_massbank_once.py
"${PYTHON_BIN}" tasks/validate_bioaware_b44_massbank_blind_panel.py --input-dir "${PANEL_DIR}"
"${PYTHON_BIN}" tasks/validate_bioaware_b44_catalogue_v1.py --input-dir "${ARTIFACT_DIR}"
"${PYTHON_BIN}" tasks/evaluate_bioaware_b44_massbank_once.py \
  --panel-dir "${PANEL_DIR}" \
  --artifact-dir "${ARTIFACT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --device cuda
"${PYTHON_BIN}" tasks/validate_bioaware_b44_massbank_once.py --input-dir "${OUTPUT_DIR}"
echo "=== BioAware B44 one-time MassBank evaluation completed: $(date) ==="
echo "Result: ${OUTPUT_DIR}"
