#!/bin/bash
#SBATCH --job-name=bioaware_b43
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=bioaware_b43_%j.out
#SBATCH --error=bioaware_b43_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
B42_DIR="${B42_DIR:?Set B42_DIR to a validated BioAware B42 output directory}"
OUTPUT_DIR="data/validation/bioaware_b43_catalog_gain_mechanism_${SLURM_JOB_ID}"

echo "=== BioAware B43 catalogue-gain mechanism audit started: $(date) ==="
"${PYTHON_BIN}" tasks/test_bioaware_b43_catalog_gain_mechanism.py
"${PYTHON_BIN}" tasks/validate_bioaware_b42_independent_catalog_topology.py --input-dir "${B42_DIR}"
"${PYTHON_BIN}" tasks/audit_bioaware_b43_catalog_gain_mechanism.py \
  --b42-dir "${B42_DIR}" \
  --output-dir "${OUTPUT_DIR}"
"${PYTHON_BIN}" tasks/validate_bioaware_b43_catalog_gain_mechanism.py --input-dir "${OUTPUT_DIR}"
echo "=== BioAware B43 catalogue-gain mechanism audit completed: $(date) ==="
echo "Result: ${OUTPUT_DIR}"
