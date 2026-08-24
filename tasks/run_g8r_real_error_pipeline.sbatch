#!/bin/bash
#SBATCH --job-name=g8r_error_pipeline
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=36:00:00
#SBATCH --output=data/validation/g8r_error_pipeline_%j.out
#SBATCH --error=data/validation/g8r_error_pipeline_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

echo "=== real-error pipeline started: $(date) ==="
echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-8}"

required=(
  data/models/MassSpecGym_MurckoHist_split.hdf5
  data/e1/official_embedding_slim.pt
  dreams/models/pretrained/ssl_model_server.pt
  data/validation/g8r_p2_preflight.json
  data/validation/g8r_p2_official_embeddings.npz
  data/validation/g8r_p2_official_embeddings.json
  data/validation/g8r_p2b_rank_fusion.json
  data/validation/g8r_p3_test/p3_p2_allowed_training_ik14.json
  tasks/massspecgym_isomers/pairs.json
)
for file in "${required[@]}"; do
  [[ -f "$file" ]] || { echo "MISSING: $file" >&2; exit 2; }
done

for output in \
  data/validation/g8r_error_atlas_listwise_cache.npz \
  data/validation/g8r_real_error_atlas \
  data/validation/g8r_real_error_analysis; do
  [[ ! -e "$output" ]] || { echo "REFUSE existing output: $output" >&2; exit 3; }
done

echo "=== stage 1/3: full real-query cache ==="
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

echo "=== stage 2/3: DreaMS/P2b real-error graph ==="
python -u tasks/build_g8r_real_error_atlas.py \
  --cache data/validation/g8r_error_atlas_listwise_cache.npz \
  --artifact data/validation/g8r_p2b_rank_fusion.json \
  --data data/models/MassSpecGym_MurckoHist_split.hdf5 \
  --p3-dir data/validation/g8r_p3_test \
  --output-dir data/validation/g8r_real_error_atlas

echo "=== stage 3/3: error decomposition and priority cohorts ==="
python -u tasks/analyze_g8r_real_error_atlas.py \
  --atlas-dir data/validation/g8r_real_error_atlas \
  --output-dir data/validation/g8r_real_error_analysis

echo "=== real-error pipeline completed: $(date) ==="
