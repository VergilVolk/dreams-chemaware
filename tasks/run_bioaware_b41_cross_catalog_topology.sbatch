#!/bin/bash
#SBATCH --job-name=bioaware_b41
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=bioaware_b41_%j.out
#SBATCH --error=bioaware_b41_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="data/validation/bioaware_b41_cross_catalog_topology_${SLURM_JOB_ID}"

echo "=== BioAware B41 cross-catalog topology audit started: $(date) ==="
"${PYTHON_BIN}" - <<'PY'
import importlib.util
for module in ("numpy", "pandas", "scipy", "sklearn"):
    if importlib.util.find_spec(module) is None:
        raise SystemExit(f"missing Python dependency: {module}")
print("[BioAware B41 dependencies] PASS", flush=True)
PY
"${PYTHON_BIN}" tasks/test_bioaware_b41_cross_catalog_topology.py
"${PYTHON_BIN}" tasks/audit_bioaware_b41_cross_catalog_topology.py \
  --output-dir "${OUTPUT_DIR}"
"${PYTHON_BIN}" tasks/validate_bioaware_b41_cross_catalog_topology.py \
  --input-dir "${OUTPUT_DIR}"
echo "=== BioAware B41 cross-catalog topology audit completed: $(date) ==="
echo "Result: ${OUTPUT_DIR}"
