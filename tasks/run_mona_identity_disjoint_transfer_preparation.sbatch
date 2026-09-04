#!/bin/bash
#SBATCH --job-name=mona_bio_transfer_prep
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=mona_bio_transfer_prep_%j.out
#SBATCH --error=mona_bio_transfer_prep_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

python -u tasks/test_mona_bioaware_embedding_transfer.py
python -u tasks/build_mona_identity_disjoint_transfer_panel.py
python -u tasks/encode_mona_transfer_peak_tokens.py --device cuda
