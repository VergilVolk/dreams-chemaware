#!/bin/bash
#SBATCH --job-name=bio_rhea_dir
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=bioaware_rhea_direction_%j.out
#SBATCH --error=bioaware_rhea_direction_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONUNBUFFERED=1

python -u tasks/test_bioaware_rhea_reactome_direction.py
python -u tasks/build_bioaware_rhea_reactome_direction_cache.py
