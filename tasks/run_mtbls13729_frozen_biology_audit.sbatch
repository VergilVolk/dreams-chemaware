#!/bin/bash
#SBATCH --job-name=mtbls_bioaudit
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=data/validation/mtbls_bioaudit_%j.out
#SBATCH --error=data/validation/mtbls_bioaudit_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== MTBLS13729 frozen biology result audit started: $(date) ==="
python -u tasks/audit_mtbls13729_frozen_biology_results.py
echo "=== MTBLS13729 frozen biology result audit completed: $(date) ==="
