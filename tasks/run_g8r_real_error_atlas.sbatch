#!/bin/bash
#SBATCH --job-name=g8r_error_atlas
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/g8r_error_atlas_%j.out
#SBATCH --error=data/validation/g8r_error_atlas_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS

python -u tasks/build_g8r_real_error_atlas.py \
  --cache data/validation/g8r_error_atlas_listwise_cache.npz \
  --artifact data/validation/g8r_p2b_rank_fusion.json \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --p3-dir data/validation/g8r_p3_test \
  --output-dir data/validation/g8r_real_error_atlas
