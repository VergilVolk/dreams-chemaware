#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=g8r_raw_reranker_reconcile_%j.out
#SBATCH --error=g8r_raw_reranker_reconcile_%j.err

# Reconcile 45/18 (ungated) vs 44/17 (gated): output the two diff query IDs.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

for f in data/validation/g8r_raw_reranker_cache.npz \
  data/validation/g8r_raw_reranker_cache_val.npz \
  tasks/audit_g8r_raw_reranker_reconcile.py; do
  [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 2; }
done

python -u tasks/audit_g8r_raw_reranker_reconcile.py \
  --output data/validation/g8r_raw_reranker_reconcile.json

echo "=== reconcile done: $(date) ==="
echo "Read data/validation/g8r_raw_reranker_reconcile.json"
