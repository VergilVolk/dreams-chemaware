#!/bin/bash
#SBATCH --job-name=mona_bio_transfer
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=mona_bio_transfer_%j.out
#SBATCH --error=mona_bio_transfer_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

python -u tasks/evaluate_mona_bioaware_embedding_transfer.py --device cuda
