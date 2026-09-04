#!/bin/bash
#SBATCH --job-name=bio_smn_headroom
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=bioaware_smn_headroom_%j.out
#SBATCH --error=bioaware_smn_headroom_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS

echo "=== BioAware SMN incremental headroom started: $(date) ==="
python -u -m pytest -q tests/test_bioaware_metdna3_smn_headroom.py
python -u tasks/audit_bioaware_metdna3_smn_headroom.py
echo "=== BioAware SMN incremental headroom completed: $(date) ==="
