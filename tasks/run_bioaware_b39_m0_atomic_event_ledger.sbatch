#!/bin/bash
#SBATCH --job-name=bioaware_b39_m0
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=bioaware_b39_m0_%j.out
#SBATCH --error=bioaware_b39_m0_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
: "${B38_M0_DIR:?Set B38_M0_DIR to the completed BioAware B38-M0 directory}"
OUTPUT_DIR="data/validation/bioaware_b39_m0_atomic_event_ledger_${SLURM_JOB_ID}"

echo "=== BioAware B39-M0 atomic event ledger started: $(date) ==="
"${PYTHON_BIN}" - <<'PY'
import importlib.util
for module in ("numpy", "pandas", "sklearn", "scipy"):
    if importlib.util.find_spec(module) is None:
        raise SystemExit(f"missing Python dependency: {module}")
print("[BioAware B39-M0 dependencies] PASS", flush=True)
PY

"${PYTHON_BIN}" tasks/test_bioaware_b39_atomic_event_ledger.py
"${PYTHON_BIN}" tasks/build_bioaware_b39_atomic_event_ledger.py \
  --b38-m0-dir "${B38_M0_DIR}" \
  --output-dir "${OUTPUT_DIR}"
"${PYTHON_BIN}" tasks/validate_bioaware_b39_atomic_event_ledger.py \
  --input-dir "${OUTPUT_DIR}"

echo "=== BioAware B39-M0 completed: $(date) ==="
echo "Result: ${OUTPUT_DIR}"
