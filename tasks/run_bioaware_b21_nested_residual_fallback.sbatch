#!/bin/bash
#SBATCH --job-name=bioaware_b21_fallback
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=logs/bioaware_b21_fallback_%j.out
#SBATCH --error=logs/bioaware_b21_fallback_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
mkdir -p logs

PYTHON=/data02/home/scv7tsl/run/miniconda3/envs/dreams/bin/python
OUTPUT="data/validation/bioaware_b21_nested_residual_fallback_${SLURM_JOB_ID}"

"$PYTHON" -u tasks/test_bioaware_b21_nested_residual_fallback.py
set +e
"$PYTHON" -u tasks/audit_bioaware_b21_nested_residual_fallback.py \
  --output-dir "$OUTPUT"
STATUS=$?
set -e
"$PYTHON" -u tasks/validate_bioaware_b21_nested_residual_fallback.py "$OUTPUT"
exit "$STATUS"
