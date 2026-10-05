#!/bin/bash
#SBATCH --job-name=bioaware_b44_lock
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=bioaware_b44_lock_%j.out
#SBATCH --error=bioaware_b44_lock_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
B42_DIR="${B42_DIR:?Set B42_DIR to the validated B42 directory}"
OUTPUT_DIR="data/validation/bioaware_b44_massbank_panel_${SLURM_JOB_ID}"

echo "=== BioAware B44 MassBank panel lock started: $(date) ==="
"${PYTHON_BIN}" tasks/test_bioaware_b44_massbank_blind_panel.py
"${PYTHON_BIN}" tasks/validate_bioaware_b42_independent_catalog_topology.py --input-dir "${B42_DIR}"
"${PYTHON_BIN}" tasks/build_bioaware_b44_massbank_blind_panel.py \
  --b42-dir "${B42_DIR}" \
  --output-dir "${OUTPUT_DIR}"
"${PYTHON_BIN}" tasks/validate_bioaware_b44_massbank_blind_panel.py --input-dir "${OUTPUT_DIR}"
echo "=== BioAware B44 MassBank panel lock completed: $(date) ==="
echo "Result: ${OUTPUT_DIR}"
