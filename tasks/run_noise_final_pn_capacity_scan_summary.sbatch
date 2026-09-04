#!/bin/bash
#SBATCH --job-name=noise_pn_cap_sum
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=data/validation/noise_pn_cap_summary_%j.out
#SBATCH --error=data/validation/noise_pn_cap_summary_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python -u tasks/summarize_noise_final_pn_capacity_scan.py
