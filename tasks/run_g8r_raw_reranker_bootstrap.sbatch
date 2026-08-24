#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=g8r_raw_reranker_bootstrap_%j.out
#SBATCH --error=g8r_raw_reranker_bootstrap_%j.err

# Formula-cluster paired bootstrap + McNemar for the RAW reranker's per-stratum
# net gain (companion to the 44/17 decomposition).
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

for f in tasks/massspecgym_isomers/g8r_locked/val.json \
  data/models/MassSpecGym_MurckoHist_split.hdf5 \
  data/validation/g8r_raw_reranker_cache.npz \
  data/validation/g8r_raw_reranker_cache_val.npz \
  tasks/audit_g8r_raw_reranker_bootstrap.py; do
  [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 2; }
done

python -u tasks/audit_g8r_raw_reranker_bootstrap.py \
  --output data/validation/g8r_raw_reranker_bootstrap.json

echo "=== bootstrap done: $(date) ==="
echo "Read data/validation/g8r_raw_reranker_bootstrap.json"
