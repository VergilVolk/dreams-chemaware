#!/bin/bash
#SBATCH --job-name=bioaware_b39_m01
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=05:00:00
#SBATCH --output=bioaware_b39_m01_%j.out
#SBATCH --error=bioaware_b39_m01_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
: "${B38_M0_DIR:?Set B38_M0_DIR to the completed BioAware B38-M0 directory}"
M0_DIR="data/validation/bioaware_b39_m0_atomic_event_ledger_${SLURM_JOB_ID}"
M1_DIR="data/validation/bioaware_b39_m1_internal_edge_evidence_${SLURM_JOB_ID}"

echo "=== BioAware B39-M0/M1 started: $(date) ==="
"${PYTHON_BIN}" - <<'PY'
import importlib.util
for module in ("numpy", "pandas", "sklearn", "scipy"):
    if importlib.util.find_spec(module) is None:
        raise SystemExit(f"missing Python dependency: {module}")
print("[BioAware B39 dependencies] PASS", flush=True)
PY

"${PYTHON_BIN}" tasks/test_bioaware_b39_atomic_event_ledger.py
"${PYTHON_BIN}" tasks/test_bioaware_b39_m1_internal_edge_evidence.py

"${PYTHON_BIN}" tasks/build_bioaware_b39_atomic_event_ledger.py \
  --b38-m0-dir "${B38_M0_DIR}" \
  --output-dir "${M0_DIR}"
"${PYTHON_BIN}" tasks/validate_bioaware_b39_atomic_event_ledger.py \
  --input-dir "${M0_DIR}"

"${PYTHON_BIN}" tasks/build_bioaware_b39_m1_internal_edge_evidence.py \
  --b39-m0-dir "${M0_DIR}" \
  --output-dir "${M1_DIR}"
"${PYTHON_BIN}" tasks/validate_bioaware_b39_m1_internal_edge_evidence.py \
  --input-dir "${M1_DIR}"

echo "=== BioAware B39-M0/M1 completed: $(date) ==="
echo "M0 result: ${M0_DIR}"
echo "M1 result: ${M1_DIR}"
