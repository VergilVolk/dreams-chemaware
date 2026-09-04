#!/bin/bash
#SBATCH --job-name=bio_emb_pilot
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=bioaware_embedding_pilot_%j.out
#SBATCH --error=bioaware_embedding_pilot_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

python -u tasks/train_bioaware_embedding_adapter.py \
  --outer-fold 0 \
  --seed 20260830 \
  --epochs 2 \
  --steps-per-epoch 75 \
  --output-root data/validation/bioaware_embedding_adapter_pilot

python -u tasks/validate_bioaware_embedding_pilot.py \
  --report data/validation/bioaware_embedding_adapter_pilot/fold_0/seed_20260830/report.json

