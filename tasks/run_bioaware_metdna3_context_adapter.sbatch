#!/bin/bash
#SBATCH --job-name=bio_m3_ctx
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=bioaware_metdna3_context_%j.out
#SBATCH --error=bioaware_metdna3_context_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

python -u tasks/test_bioaware_context_adapter.py
python -u tasks/test_metdna3_context_adapter_dataset.py
python -u tasks/encode_metdna3_context_seed_embeddings.py --device cuda
python -u tasks/build_metdna3_context_adapter_dataset.py
python -u tasks/train_metdna3_context_adapter_oof.py --device cuda
