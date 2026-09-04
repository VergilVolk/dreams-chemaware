#!/bin/bash
#SBATCH --job-name=bioaware_m3_cache
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/bioaware_m3_cache_%j.out
#SBATCH --error=data/validation/bioaware_m3_cache_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD/tasks:$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== BioAware MetDNA3 DreaMS cache started: $(date) ==="
python -u tasks/build_bioaware_metdna3_dreams_cache.py
echo "=== BioAware MetDNA3 DreaMS cache completed: $(date) ==="
