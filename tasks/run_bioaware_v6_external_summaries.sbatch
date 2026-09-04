#!/bin/bash
#SBATCH --job-name=bio_v6_summary
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/bio_v6_summary_%j.out
#SBATCH --error=data/validation/bio_v6_summary_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD/tasks:$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python -u tasks/summarize_bioaware_v6_external_5panel.py
python -u tasks/summarize_bioaware_v6_external_degree_decoys.py
