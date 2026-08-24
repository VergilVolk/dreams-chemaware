#!/bin/bash
#SBATCH --job-name=g8r_atlas_cache
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/g8r_atlas_cache_%j.out
#SBATCH --error=data/validation/g8r_atlas_cache_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS

# Reuse the audited P2 cache builder, but include every valid real query row.
python -u tasks/build_g8r_p2_listwise_cache.py \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --pairs tasks/massspecgym_isomers/pairs.json \
  --p3-dir data/validation/g8r_p3_test \
  --preflight data/validation/g8r_p2_preflight.json \
  --base-ckpt data/e1/official_embedding_slim.pt \
  --architecture-ckpt dreams/models/pretrained/ssl_model_server.pt \
  --embedding-cache data/validation/g8r_p2_official_embeddings.npz \
  --output data/validation/g8r_error_atlas_listwise_cache.npz \
  --device cuda \
  --batch-size 128 \
  --max-queries 50000 \
  --queries-per-identity 100000 \
  --workers "${SLURM_CPUS_PER_TASK:-8}" \
  --seed 20260824

echo "Full real-query cache complete. Submit tasks/run_g8r_real_error_atlas.sbatch next."
