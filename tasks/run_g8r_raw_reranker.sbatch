#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=g8r_raw_reranker_%j.out
#SBATCH --error=g8r_raw_reranker_%j.err

# Groupwise RAW reranker for g8r (cosine + RAW features only; no token, no old
# 8-peak panel).  Gate threshold is selected on formula-group OOF predictions
# (no in-sample leakage).  Writes feature cache so the model stage can be
# re-run cheaply after any fix.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

for f in tasks/massspecgym_isomers/g8r_locked/train.json \
  tasks/massspecgym_isomers/g8r_locked/val.json \
  data/models/MassSpecGym_MurckoHist_split.hdf5 \
  data/e1/official_embedding_slim.pt dreams/models/pretrained/ssl_model_server.pt \
  tasks/train_g8r_raw_reranker.py; do
  [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 2; }
done

python -u tasks/train_g8r_raw_reranker.py \
  --device cuda \
  --output-dir data/validation/g8r_raw_reranker \
  --c-values 0.0001 0.001 0.01

echo "=== g8r RAW reranker done: $(date) ==="
echo "Read data/validation/g8r_raw_reranker/report.json (g8r_val is DEV only)."
