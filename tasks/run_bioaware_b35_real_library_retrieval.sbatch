#!/bin/bash
#SBATCH --job-name=bioaware_b35_real
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=bioaware_b35_real_%j.out
#SBATCH --error=bioaware_b35_real_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS

PYTHON=/data02/home/scv7tsl/run/miniconda3/envs/dreams/bin/python
JOB_TAG="${SLURM_JOB_ID:-manual}"
OUTPUT="data/validation/bioaware_b35_real_library_retrieval_${JOB_TAG}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing Python interpreter: $PYTHON" >&2
  exit 2
fi
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite: $OUTPUT" >&2
  exit 3
fi

echo "=== BioAware B35 real-library retrieval benchmark started: $(date) ==="
"$PYTHON" -u tasks/test_bioaware_b35_real_library_retrieval.py
"$PYTHON" -u tasks/evaluate_bioaware_b35_real_library_retrieval.py \
  --output-dir "$OUTPUT" \
  --bootstrap-resamples 10000 \
  --seed 20260912
"$PYTHON" -u tasks/validate_bioaware_b35_real_library_retrieval.py \
  --output-dir "$OUTPUT"
echo "=== BioAware B35 completed: $(date) ==="
echo "Result: $OUTPUT/report.json"
