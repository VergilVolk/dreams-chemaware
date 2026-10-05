#!/bin/bash
#SBATCH --job-name=bioaware_b38_m0
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=bioaware_b38_m0_%j.out
#SBATCH --error=bioaware_b38_m0_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="data/validation/bioaware_b38_m0_explicit_path_${SLURM_JOB_ID}"

echo "=== BioAware B38-M0 explicit path coverage started: $(date) ==="
"${PYTHON_BIN}" - <<'PY'
import importlib.util
for module in ("numpy", "pandas", "sklearn", "scipy"):
    if importlib.util.find_spec(module) is None:
        raise SystemExit(f"missing Python dependency: {module}")
print("[BioAware B38-M0 dependencies] PASS", flush=True)
PY

"${PYTHON_BIN}" tasks/test_bioaware_b38_m0_explicit_path_coverage.py
"${PYTHON_BIN}" tasks/audit_bioaware_b38_m0_explicit_path_coverage.py \
  --output-dir "${OUTPUT_DIR}" \
  --decoy-repeats 20
"${PYTHON_BIN}" tasks/validate_bioaware_b38_m0_explicit_path_coverage.py \
  --input-dir "${OUTPUT_DIR}"

echo "=== BioAware B38-M0 completed: $(date) ==="
echo "Result: ${OUTPUT_DIR}"
