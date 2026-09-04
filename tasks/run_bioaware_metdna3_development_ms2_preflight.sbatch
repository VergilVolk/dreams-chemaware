#!/bin/bash
#SBATCH --job-name=bioaware_m3_ms2
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=data/validation/bioaware_m3_ms2_%j.out
#SBATCH --error=data/validation/bioaware_m3_ms2_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD/tasks:$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== BioAware MetDNA3 development MS2 preflight started: $(date) ==="
python -u tasks/preflight_bioaware_metdna3_development_ms2.py
echo "=== BioAware MetDNA3 development MS2 preflight completed: $(date) ==="
