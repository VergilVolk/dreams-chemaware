#!/bin/bash
#SBATCH --job-name=noise_e13_summary
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=data/validation/noise_e13_summary_%j.out
#SBATCH --error=data/validation/noise_e13_summary_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== E13 paired relaxed-transfer summary started: $(date) ==="
python -u tasks/summarize_noise_final_e13_relaxed_transfer.py \
  --bootstrap 5000 \
  --seed 20260905
echo "=== E13 paired relaxed-transfer summary completed: $(date) ==="
