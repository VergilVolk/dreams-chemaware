#!/bin/bash
#SBATCH --job-name=bioaware_v5_loso
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/bioaware_v5_loso_%j.out
#SBATCH --error=data/validation/bioaware_v5_loso_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD/tasks:$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== BioAware V5 nested leave-study-out started: $(date) ==="
python -u tasks/test_bioaware_v5_leave_study_out.py
python -u tasks/develop_bioaware_v5_leave_study_out.py
echo "=== BioAware V5 nested leave-study-out completed: $(date) ==="
