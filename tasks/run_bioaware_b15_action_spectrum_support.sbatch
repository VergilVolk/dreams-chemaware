#!/bin/bash
#SBATCH --job-name=bioaware_b15_support
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/bioaware_b15_support_%j.out
#SBATCH --error=logs/bioaware_b15_support_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
mkdir -p logs

PYTHON=/data02/home/scv7tsl/run/miniconda3/envs/dreams/bin/python
OUTPUT="data/validation/bioaware_b15_action_spectrum_support_${SLURM_JOB_ID}"

"$PYTHON" -u tasks/test_bioaware_b15_action_spectrum_support.py
"$PYTHON" -u tasks/audit_bioaware_b15_action_spectrum_support.py \
  --output-dir "$OUTPUT"
"$PYTHON" -u tasks/validate_bioaware_b15_action_spectrum_support.py "$OUTPUT"

echo "B15 result: $OUTPUT"
