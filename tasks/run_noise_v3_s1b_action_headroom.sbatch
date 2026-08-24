#!/bin/bash
#SBATCH --job-name=noise_v3_s1b
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=data/validation/noise_v3_s1b_%j.out
#SBATCH --error=data/validation/noise_v3_s1b_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

echo "=== Noise-v3 S1b action headroom started: $(date) ==="
python -u tasks/analyze_noise_v3_s1b_action_headroom.py
echo "=== Noise-v3 S1b action headroom completed: $(date) ==="
