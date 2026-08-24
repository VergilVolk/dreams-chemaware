#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=g8r_raw_reranker_44_17_%j.out
#SBATCH --error=g8r_raw_reranker_44_17_%j.err

# Decompose the RAW reranker's 44/17 Top-1 corrections/introductions by the MCES
# of the wrong candidate (conditional correction rate + introduced type + net
# gain + gate on/off counterfactual).
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

for f in tasks/massspecgym_isomers/g8r_locked/val.json \
  data/models/MassSpecGym_MurckoHist_split.hdf5 \
  data/validation/g8r_raw_reranker_cache.npz \
  data/validation/g8r_raw_reranker_cache_val.npz \
  tasks/audit_g8r_raw_reranker_44_17.py; do
  [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 2; }
done

python -u tasks/audit_g8r_raw_reranker_44_17.py \
  --output data/validation/g8r_raw_reranker_44_17.json

echo "=== 44/17 decomposition done: $(date) ==="
echo "Read data/validation/g8r_raw_reranker_44_17.json"
