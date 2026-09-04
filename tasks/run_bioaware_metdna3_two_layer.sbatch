#!/bin/bash
#SBATCH --job-name=bioaware_m3_2layer
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=data/validation/bioaware_m3_2layer_%j.out
#SBATCH --error=data/validation/bioaware_m3_2layer_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD/tasks:$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "=== BioAware MetDNA3 data-layer audit started: $(date) ==="
python -u tasks/encode_bioaware_metdna3_data_layer.py --device cuda --batch-size 128
python -u tasks/evaluate_bioaware_metdna3_two_layer.py
python -u tasks/audit_bioaware_metdna3_data_layer_calibration.py
python -u tasks/audit_bioaware_metdna3_raw_ms2_layer.py
echo "=== BioAware MetDNA3 data-layer audit completed: $(date) ==="
