#!/bin/bash
#SBATCH --job-name=bioaware_b20_manifest
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=bioaware_b20_manifest_%j.out
#SBATCH --error=bioaware_b20_manifest_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS

PYTHON=/data02/home/scv7tsl/run/miniconda3/envs/dreams/bin/python
OUTPUT="data/validation/bioaware_b20_direct_action_manifest_${SLURM_JOB_ID}"

"$PYTHON" -u tasks/test_bioaware_b20_direct_action_manifest.py
"$PYTHON" -u tasks/build_bioaware_b20_direct_action_manifest.py \
  --output-dir "$OUTPUT"
"$PYTHON" -u tasks/validate_bioaware_b20_direct_action_manifest.py "$OUTPUT"
