#!/bin/bash
#SBATCH --job-name=noise_v3_s1c_fix
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=data/validation/noise_v3_s1c_redecision_%j.out
#SBATCH --error=data/validation/noise_v3_s1c_redecision_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"

echo "=== Noise-v3 S1c corrected decision started: $(date) ==="
python -u tasks/analyze_noise_v3_s1b_action_headroom.py \
  --matrix-dir data/validation/g8r_noise_v3_s1c_topk_matrix \
  --output data/validation/g8r_noise_v3_s1c_action_decision_v2.json \
  --status noise_v3_s1c_action_headroom_complete \
  --minimum-headroom-pp 2.0 \
  --minimum-recoverable-queries 600
echo "=== Noise-v3 S1c corrected decision completed: $(date) ==="
