#!/bin/bash
#SBATCH --job-name=bio_emb_adapter
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=bioaware_embedding_%A_%a.out
#SBATCH --error=bioaware_embedding_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

python -u tasks/train_bioaware_embedding_adapter.py \
  --outer-fold "${SLURM_ARRAY_TASK_ID}" \
  --seed 20260830
