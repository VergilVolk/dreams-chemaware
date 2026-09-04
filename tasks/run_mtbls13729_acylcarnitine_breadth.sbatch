#!/bin/bash
#SBATCH --job-name=mtbls_lcac_breadth
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=mtbls_lcac_breadth_%j.out
#SBATCH --error=mtbls_lcac_breadth_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

echo "=== MTBLS13729 LCAC breadth audit started: $(date) ==="
python -u tasks/audit_mtbls13729_acylcarnitine_breadth.py
echo "=== MTBLS13729 LCAC breadth audit completed: $(date) ==="
