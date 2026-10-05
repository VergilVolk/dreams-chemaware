#!/bin/bash
#SBATCH --job-name=bioaware_b40_m1m2
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=bioaware_b40_m1m2_%j.out
#SBATCH --error=bioaware_b40_m1m2_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
: "${B40_M0_DIR:?Set B40_M0_DIR to a validated B40-M0 directory}"
B37_DIR="${B37_DIR:-data/validation/bioaware_b37_catalog_mechanism_2336398}"
MANIFEST_DIR="data/validation/bioaware_b40_m1_manifest_${SLURM_JOB_ID}"
M1_DIR="data/validation/bioaware_b40_m1_fixed_action_${SLURM_JOB_ID}"
M2_DIR="data/validation/bioaware_b40_m2_context_strata_${SLURM_JOB_ID}"

echo "=== BioAware B40-M1/M2 frozen graph-completion audit started: $(date) ==="
"${PYTHON_BIN}" - <<'PY'
import importlib.util
for module in ("numpy", "pandas", "scipy", "sklearn"):
    if importlib.util.find_spec(module) is None:
        raise SystemExit(f"missing Python dependency: {module}")
print("[BioAware B40-M1/M2 dependencies] PASS", flush=True)
PY
"${PYTHON_BIN}" tasks/test_bioaware_b40_m1_graph_completion.py
"${PYTHON_BIN}" tasks/test_bioaware_b40_m2_context_strata.py
"${PYTHON_BIN}" tasks/freeze_bioaware_b40_m1_graph_completion_manifest.py \
  --output-dir "${MANIFEST_DIR}"
"${PYTHON_BIN}" tasks/evaluate_bioaware_b40_m1_graph_completion.py \
  --manifest-dir "${MANIFEST_DIR}" \
  --b40-m0-dir "${B40_M0_DIR}" \
  --b37-dir "${B37_DIR}" \
  --output-dir "${M1_DIR}"
"${PYTHON_BIN}" tasks/validate_bioaware_b40_m1_graph_completion.py \
  --input-dir "${M1_DIR}"
"${PYTHON_BIN}" tasks/evaluate_bioaware_b40_m2_context_strata.py \
  --manifest-dir "${MANIFEST_DIR}" \
  --b40-m0-dir "${B40_M0_DIR}" \
  --b37-dir "${B37_DIR}" \
  --output-dir "${M2_DIR}"
"${PYTHON_BIN}" tasks/validate_bioaware_b40_m2_context_strata.py \
  --input-dir "${M2_DIR}"
echo "=== BioAware B40-M1/M2 frozen graph-completion audit completed: $(date) ==="
echo "M1 result: ${M1_DIR}"
echo "M2 result: ${M2_DIR}"
