#!/bin/bash
#SBATCH --job-name=kgmn_ext_contract
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/kgmn_external_contract_%j.out
#SBATCH --error=logs/kgmn_external_contract_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
mkdir -p logs
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

python -u tasks/fetch_kgmn_external_validation_sources.py
python -u tasks/freeze_kgmn_external_validation_contract.py

echo "=== KGMN external validation contract completed: $(date) ==="
