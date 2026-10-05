#!/bin/bash
#SBATCH --job-name=bioaware_b40_m0
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=bioaware_b40_m0_%j.out
#SBATCH --error=bioaware_b40_m0_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
: "${B39_M0_DIR:?Set B39_M0_DIR to the completed B39-M0 directory containing visible_seed_contexts.csv.gz}"
OUTPUT_DIR="data/validation/bioaware_b40_m0_soft_graph_completion_${SLURM_JOB_ID}"

echo "=== BioAware B40-M0 soft graph completion started: $(date) ==="
"${PYTHON_BIN}" - <<'PY'
import importlib.util
for module in ("numpy", "pandas", "scipy"):
    if importlib.util.find_spec(module) is None:
        raise SystemExit(f"missing Python dependency: {module}")
print("[BioAware B40-M0 dependencies] PASS", flush=True)
PY
"${PYTHON_BIN}" tasks/test_bioaware_b40_soft_graph_completion.py
"${PYTHON_BIN}" tasks/build_bioaware_b40_soft_graph_completion_cache.py \
  --m0-dir "${B39_M0_DIR}" \
  --output-dir "${OUTPUT_DIR}"
"${PYTHON_BIN}" tasks/validate_bioaware_b40_soft_graph_completion.py \
  --input-dir "${OUTPUT_DIR}"
echo "=== BioAware B40-M0 soft graph completion completed: $(date) ==="
echo "Result: ${OUTPUT_DIR}"
