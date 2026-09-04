#!/bin/bash
#SBATCH --job-name=chemaware_v2_cache
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/chemaware_v2_cache_%j.out
#SBATCH --error=data/validation/chemaware_v2_cache_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "=== ChemAware shared-v2 locked token cache started: $(date) ==="
python -u tasks/preflight_chemaware_shared_v2.py
python -u tasks/encode_chemaware_shared_v2_tokens.py
echo "=== ChemAware shared-v2 locked token cache completed: $(date) ==="
