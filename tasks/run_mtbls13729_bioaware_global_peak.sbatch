#!/bin/bash
#SBATCH --job-name=bioaware_g1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=bioaware_g1_%j.out
#SBATCH --error=bioaware_g1_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

echo "=== BioAware G1 global peak network started: $(date) ==="
python -u tasks/requantify_mtbls13729_targeted_eic.py \
  --panels neg_rp pos_rp \
  --resolve-local-peaks \
  --max-apex-delta-sec 12
python -u tasks/build_mtbls13729_global_peak_network.py
echo "=== BioAware G1 global peak network completed: $(date) ==="
