#!/bin/bash
#SBATCH --job-name=kgmn_oep_preflight
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/kgmn_oep_preflight_%j.out
#SBATCH --error=logs/kgmn_oep_preflight_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
mkdir -p logs
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

python -u tasks/preflight_kgmn_oep003284_inputs.py \
  --input-root data/reference/OEP003284 \
  --output data/validation/kgmn_oep003284_input_preflight_20260831.json \
  --require-ready \
  --require-dreams-edge \
  --allow-author-mapped-mzxml

echo "=== KGMN OEP003284 input preflight completed: $(date) ==="
