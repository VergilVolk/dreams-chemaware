#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=g8r_raw_reranker_perquery_%j.out
#SBATCH --error=g8r_raw_reranker_perquery_%j.err

# Decisive per-query audit of the RAW reranker's near/mid effect (unified
# definition, transition table, formula-cluster paired bootstrap, and the
# D0/PSD "allneg" reconciliation).
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

for f in tasks/massspecgym_isomers/g8r_locked/val.json \
  data/models/MassSpecGym_MurckoHist_split.hdf5 \
  data/e1/official_embedding_slim.pt dreams/models/pretrained/ssl_model_server.pt \
  data/validation/g8r_raw_reranker_cache.npz \
  tasks/audit_g8r_raw_reranker_perquery.py; do
  [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 2; }
done

python -u tasks/audit_g8r_raw_reranker_perquery.py \
  --device cuda \
  --output-dir data/validation/g8r_raw_reranker_perquery_audit

echo "=== per-query audit done: $(date) ==="
echo "Read data/validation/g8r_raw_reranker_perquery_audit/summary.json and perquery.csv"
