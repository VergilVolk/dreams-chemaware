#!/bin/bash
#SBATCH --job-name=bioaware_b39_m1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=bioaware_b39_m1_%j.out
#SBATCH --error=bioaware_b39_m1_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
: "${B39_M0_DIR:?Set B39_M0_DIR to the completed BioAware B39-M0 directory}"
OUTPUT_DIR="data/validation/bioaware_b39_m1_internal_edge_evidence_${SLURM_JOB_ID}"

echo "=== BioAware B39-M1 internal edge evidence started: $(date) ==="
"${PYTHON_BIN}" - <<'PY'
import importlib.util
for module in ("numpy", "pandas", "sklearn", "scipy"):
    if importlib.util.find_spec(module) is None:
        raise SystemExit(f"missing Python dependency: {module}")
print("[BioAware B39-M1 dependencies] PASS", flush=True)
PY

"${PYTHON_BIN}" tasks/test_bioaware_b39_m1_internal_edge_evidence.py
"${PYTHON_BIN}" tasks/build_bioaware_b39_m1_internal_edge_evidence.py \
  --b39-m0-dir "${B39_M0_DIR}" \
  --output-dir "${OUTPUT_DIR}"
"${PYTHON_BIN}" tasks/validate_bioaware_b39_m1_internal_edge_evidence.py \
  --input-dir "${OUTPUT_DIR}"

echo "=== BioAware B39-M1 completed: $(date) ==="
echo "Result: ${OUTPUT_DIR}"
