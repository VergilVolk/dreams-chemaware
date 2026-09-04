#!/bin/bash
#SBATCH --job-name=mtbls_3way_audit
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=data/validation/mtbls_3way_audit_%j.out
#SBATCH --error=data/validation/mtbls_3way_audit_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== MTBLS13729 three-way application audit started: $(date) ==="
python -u tasks/audit_mtbls13729_threeway_application.py
echo "=== MTBLS13729 three-way application audit completed: $(date) ==="
