#!/bin/bash
#SBATCH --job-name=chemaware_v2_g2_sum
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=data/validation/chemaware_v2_g2_summary_%j.out
#SBATCH --error=data/validation/chemaware_v2_g2_summary_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== ChemAware shared-v2 G2 paired summary started: $(date) ==="
python -u tasks/summarize_chemaware_shared_v2_g2.py
echo "=== ChemAware shared-v2 G2 paired summary completed: $(date) ==="
