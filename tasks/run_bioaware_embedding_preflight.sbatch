#!/bin/bash
#SBATCH --job-name=bio_emb_preflight
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=bioaware_embedding_preflight_%j.out
#SBATCH --error=bioaware_embedding_preflight_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

if [[ ! -f data/validation/bioaware_embedding_relation_manifest/report.json ]]; then
  python -u tasks/build_bioaware_embedding_relation_manifest.py
fi
if [[ ! -f data/validation/bioaware_embedding_adapter_preflight.json ]]; then
  python -u tasks/preflight_bioaware_embedding_adapter.py
fi
