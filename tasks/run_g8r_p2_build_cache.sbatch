#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=g8r_p2_cache_%j.out
#SBATCH --error=g8r_p2_cache_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

required=(
  data/validation/g8r_p2_preflight.json
  data/validation/g8r_p3_test/p3_p2_allowed_training_ik14.json
  data/models/MassSpecGym_MurckoHist_split.hdf5
  tasks/massspecgym_isomers/pairs.json
  data/e1/official_embedding_slim.pt
  dreams/models/pretrained/ssl_model_server.pt
  tasks/build_g8r_p2_listwise_cache.py
)
for file in "${required[@]}"; do
  [[ -f "$file" ]] || { echo "MISSING: $file" >&2; exit 2; }
done
[[ ! -e data/validation/g8r_p2_listwise_cache.npz ]] || {
  echo "REFUSE: cache already exists; audit it instead of silently overwriting" >&2
  exit 3
}

python -u tasks/build_g8r_p2_listwise_cache.py \
  --device cuda \
  --batch-size 128 \
  --max-queries 20000 \
  --queries-per-identity 2

echo "=== P2 listwise cache completed: $(date) ==="
