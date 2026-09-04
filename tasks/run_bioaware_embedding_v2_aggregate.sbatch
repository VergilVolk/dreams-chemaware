#!/bin/bash
#SBATCH --job-name=bio_emb_v2_agg
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=bioaware_embedding_v2_aggregate_%j.out
#SBATCH --error=bioaware_embedding_v2_aggregate_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONUNBUFFERED=1

python -u tasks/aggregate_bioaware_embedding_adapter.py \
  --root data/validation/bioaware_embedding_adapter_v2_spectral_only \
  --output data/validation/bioaware_embedding_adapter_v2_spectral_only_oof.json
python -u tasks/aggregate_bioaware_embedding_adapter.py \
  --root data/validation/bioaware_embedding_adapter_v2_biology_relation \
  --output data/validation/bioaware_embedding_adapter_v2_biology_relation_oof.json
python -u tasks/compare_bioaware_embedding_v2_ablation.py
