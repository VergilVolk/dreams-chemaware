#!/bin/bash
#SBATCH --job-name=bio_rt_headroom
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=bioaware_rt_headroom_%j.out
#SBATCH --error=bioaware_rt_headroom_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS

echo "=== BioAware identity-isolated RT headroom started: $(date) ==="
python -u -m pytest -q tests/test_bioaware_metdna3_rt_headroom.py
python -u tasks/audit_bioaware_metdna3_rt_headroom.py
echo "=== BioAware identity-isolated RT headroom completed: $(date) ==="
