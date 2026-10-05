#!/bin/bash
#SBATCH --job-name=bioaware_b38_m1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=bioaware_b38_m1_%j.out
#SBATCH --error=bioaware_b38_m1_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
: "${M0_DIR:?Set M0_DIR to the completed B38-M0 output directory}"
B37_DIR="${B37_DIR:-data/validation/bioaware_b37_catalog_mechanism_2336398}"
OUTPUT_DIR="data/validation/bioaware_b38_m1_explicit_path_${SLURM_JOB_ID}"

echo "=== BioAware B38-M1 explicit-path incremental test started: $(date) ==="
"${PYTHON_BIN}" - <<'PY'
import importlib.util
for module in ("numpy", "pandas", "sklearn", "scipy"):
    if importlib.util.find_spec(module) is None:
        raise SystemExit(f"missing Python dependency: {module}")
print("[BioAware B38-M1 dependencies] PASS", flush=True)
PY

for required in \
  "${M0_DIR}/report.json" \
  "${M0_DIR}/candidate_path_coverage.csv.gz" \
  "${B37_DIR}/report.json" \
  "${B37_DIR}/catalog_ablation_transitions.csv.gz"; do
  test -s "${required}" || { echo "Missing required frozen input: ${required}" >&2; exit 2; }
done

"${PYTHON_BIN}" tasks/test_bioaware_b38_m1_explicit_path_incremental.py
"${PYTHON_BIN}" tasks/audit_bioaware_b38_m1_explicit_path_incremental.py \
  --m0-dir "${M0_DIR}" \
  --b37-dir "${B37_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --rewire-repeats 20 \
  --bootstrap-resamples 5000 \
  --seed 20260907
"${PYTHON_BIN}" tasks/validate_bioaware_b38_m1_explicit_path_incremental.py \
  --input-dir "${OUTPUT_DIR}"

echo "=== BioAware B38-M1 completed: $(date) ==="
echo "Result: ${OUTPUT_DIR}"
