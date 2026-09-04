#!/bin/bash
#SBATCH --job-name=bio_emb_final
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=bioaware_embedding_final_%A_%a.out
#SBATCH --error=bioaware_embedding_final_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

SEEDS=(20260830 20260831 20260832)
python -u tasks/train_bioaware_embedding_adapter.py \
  --outer-fold -1 \
  --seed "${SEEDS[$SLURM_ARRAY_TASK_ID]}"
