#!/bin/bash
#SBATCH --job-name=mtbls_candidate_value
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=data/validation/mtbls_candidate_value_%j.out
#SBATCH --error=data/validation/mtbls_candidate_value_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== MTBLS13729 integrated-candidate algorithm audit started: $(date) ==="
python -u tasks/test_mtbls13729_integrated_candidate_algorithm_value.py
python -u tasks/audit_mtbls13729_integrated_candidate_algorithm_value.py
echo "=== MTBLS13729 integrated-candidate algorithm audit completed: $(date) ==="
