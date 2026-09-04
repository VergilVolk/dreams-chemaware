#!/bin/bash
#SBATCH --job-name=bioaware_v2_hyper
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=data/validation/bioaware_v2_hyper_%j.out
#SBATCH --error=data/validation/bioaware_v2_hyper_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD/tasks:$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python -u tasks/test_mtbls13729_bioaware_v2_feature_graph.py
python -u tasks/audit_mtbls13729_bioaware_v2_hyperedge.py
