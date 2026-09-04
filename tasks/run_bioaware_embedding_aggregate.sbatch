#!/bin/bash
#SBATCH --job-name=bio_emb_aggregate
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=bioaware_embedding_aggregate_%j.out
#SBATCH --error=bioaware_embedding_aggregate_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
python -u tasks/aggregate_bioaware_embedding_adapter.py --seed 20260830

