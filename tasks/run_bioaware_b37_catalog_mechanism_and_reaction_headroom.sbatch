#!/bin/bash
#SBATCH --job-name=bioaware_b37
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=bioaware_b37_%j.out
#SBATCH --error=bioaware_b37_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
B36_DIR="${B36_DIR:-data/validation/bioaware_b36_reaction_specificity_2336381}"
OUTPUT_DIR="data/validation/bioaware_b37_catalog_mechanism_${SLURM_JOB_ID}"

echo "=== BioAware B37 catalogue-mechanism audit started: $(date) ==="
"${PYTHON_BIN}" - <<'PY'
import importlib.util
for module in ("numpy", "pandas", "sklearn", "scipy"):
    if importlib.util.find_spec(module) is None:
        raise SystemExit(f"missing Python dependency: {module}")
print("[BioAware B37 dependencies] PASS", flush=True)
PY

"${PYTHON_BIN}" tasks/test_bioaware_b37_catalog_mechanism_and_reaction_headroom.py
"${PYTHON_BIN}" tasks/audit_bioaware_b37_catalog_mechanism_and_reaction_headroom.py \
  --b36-dir "${B36_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --bootstrap-resamples 10000 \
  --seed 20260907
"${PYTHON_BIN}" tasks/validate_bioaware_b37_catalog_mechanism_and_reaction_headroom.py \
  --input-dir "${OUTPUT_DIR}"

echo "=== BioAware B37 completed: $(date) ==="
echo "Result: ${OUTPUT_DIR}"
