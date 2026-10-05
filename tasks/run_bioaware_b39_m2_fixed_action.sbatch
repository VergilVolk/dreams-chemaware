#!/bin/bash
#SBATCH --job-name=bioaware_b39_m2
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=bioaware_b39_m2_%j.out
#SBATCH --error=bioaware_b39_m2_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
: "${B39_M0_DIR:?Set B39_M0_DIR to the completed B39-M0 directory}"
: "${B39_M1_DIR:?Set B39_M1_DIR to the completed B39-M1 directory}"
CANDIDATES="${B39_CANDIDATES:-data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz}"
B37_DIR="${B39_B37_DIR:-data/validation/bioaware_b37_local_fullcheck_v2_20260913}"
MANIFEST_DIR="data/validation/bioaware_b39_m2_manifest_${SLURM_JOB_ID}"
RESULT_DIR="data/validation/bioaware_b39_m2_fixed_action_${SLURM_JOB_ID}"

echo "=== BioAware B39-M2 fixed action started: $(date) ==="
"${PYTHON_BIN}" tasks/test_bioaware_b39_m2_fixed_action.py
"${PYTHON_BIN}" tasks/freeze_bioaware_b39_m2_fixed_action_manifest.py \
  --output-dir "${MANIFEST_DIR}"
"${PYTHON_BIN}" tasks/evaluate_bioaware_b39_m2_fixed_action.py \
  --manifest-dir "${MANIFEST_DIR}" \
  --m0-dir "${B39_M0_DIR}" \
  --m1-dir "${B39_M1_DIR}" \
  --candidate-features "${CANDIDATES}" \
  --b37-dir "${B37_DIR}" \
  --output-dir "${RESULT_DIR}"
"${PYTHON_BIN}" tasks/validate_bioaware_b39_m2_fixed_action.py \
  --input-dir "${RESULT_DIR}"
"${PYTHON_BIN}" tasks/audit_bioaware_b39_m2_coverage_funnel.py \
  --m1-dir "${B39_M1_DIR}" \
  --m2-dir "${RESULT_DIR}" \
  --candidate-features "${CANDIDATES}" \
  --b37-dir "${B37_DIR}" \
  --output "${RESULT_DIR}/coverage_funnel.json"
echo "=== BioAware B39-M2 fixed action completed: $(date) ==="
echo "Manifest: ${MANIFEST_DIR}"
echo "Result: ${RESULT_DIR}"
