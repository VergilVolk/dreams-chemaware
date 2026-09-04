#!/bin/bash
#SBATCH --job-name=bioaware_v4_sum
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/bioaware_v4_summary_%j.out
#SBATCH --error=data/validation/bioaware_v4_summary_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD/tasks:$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== BioAware V4 external summaries started: $(date) ==="
python -u tasks/summarize_bioaware_v4_external_7panel.py
python -u tasks/summarize_bioaware_v4_external_degree_decoys.py
echo "=== BioAware V4 external summaries completed: $(date) ==="
