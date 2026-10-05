#!/bin/bash
#SBATCH --job-name=bioaware_b34_scale
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=bioaware_b34_scale_%j.out
#SBATCH --error=bioaware_b34_scale_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS

PYTHON=/data02/home/scv7tsl/run/miniconda3/envs/dreams/bin/python
JOB_TAG="${SLURM_JOB_ID:-manual}"
OUTPUT="data/validation/bioaware_b34_positive_polarity_scale_${JOB_TAG}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing Python interpreter: $PYTHON" >&2
  exit 2
fi
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite: $OUTPUT" >&2
  exit 3
fi

echo "=== BioAware B34 larger polarity-transfer audit started: $(date) ==="
"$PYTHON" -u tasks/audit_bioaware_b34_positive_polarity_scale.py \
  --output-dir "$OUTPUT" \
  --bootstrap-resamples 10000 \
  --model-seed 20260907 \
  --seed 20260912
"$PYTHON" -u tasks/validate_bioaware_b34_positive_polarity_scale.py \
  --output-dir "$OUTPUT"
echo "=== BioAware B34 completed: $(date) ==="
echo "Result: $OUTPUT/report.json"
