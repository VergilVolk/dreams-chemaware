#!/bin/bash
#SBATCH --job-name=bioaware_b22_selector
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=logs/bioaware_b22_selector_%j.out
#SBATCH --error=logs/bioaware_b22_selector_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
mkdir -p logs

PYTHON=/data02/home/scv7tsl/run/miniconda3/envs/dreams/bin/python
OUTPUT="data/validation/bioaware_b22_crossfit_action_selector_${SLURM_JOB_ID}"

"$PYTHON" -u tasks/test_bioaware_b22_crossfit_action_selector.py
set +e
"$PYTHON" -u tasks/audit_bioaware_b22_crossfit_action_selector.py \
  --output-dir "$OUTPUT"
STATUS=$?
set -e
"$PYTHON" -u tasks/validate_bioaware_b22_crossfit_action_selector.py "$OUTPUT"
exit "$STATUS"
