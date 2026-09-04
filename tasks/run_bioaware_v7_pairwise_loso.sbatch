#!/bin/bash
#SBATCH --job-name=bio_v7_loso
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=bioaware_v7_loso_%j.out
#SBATCH --error=bioaware_v7_loso_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONUNBUFFERED=1

python -u tasks/test_bioaware_v7_pairwise_loso.py
python -u tasks/develop_bioaware_v7_pairwise_loso.py
