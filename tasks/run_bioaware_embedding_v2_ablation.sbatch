#!/bin/bash
#SBATCH --job-name=bio_emb_v2
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-9
#SBATCH --output=bioaware_embedding_v2_%A_%a.out
#SBATCH --error=bioaware_embedding_v2_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONUNBUFFERED=1

ARM=$((SLURM_ARRAY_TASK_ID / 5))
FOLD=$((SLURM_ARRAY_TASK_ID % 5))
if [[ "$ARM" -eq 0 ]]; then
  NAME=spectral_only
  LAMBDA=0.0
else
  NAME=biology_relation
  LAMBDA=0.10
fi
python -u tasks/train_bioaware_embedding_adapter.py \
  --manifest-dir data/validation/bioaware_embedding_relation_manifest_v2_20260830 \
  --preflight data/validation/bioaware_embedding_adapter_preflight_v2_20260830.json \
  --output-root "data/validation/bioaware_embedding_adapter_v2_${NAME}" \
  --outer-fold "$FOLD" --seed 20260830 --lambda-relation "$LAMBDA"
