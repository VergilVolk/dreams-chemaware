#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=g8r_raw_reranker_nearmid_%j.out
#SBATCH --error=g8r_raw_reranker_nearmid_%j.err

# Near/mid hardest-negative accuracy of the RAW reranker, using the val.json neg
# grades (near = MCES 0-2, mid = MCES 3-5).  Re-trains the ranker from the train
# feature cache; only embeds val anchors + neg rows for scoring.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

for f in tasks/massspecgym_isomers/g8r_locked/val.json \
  data/models/MassSpecGym_MurckoHist_split.hdf5 \
  data/e1/official_embedding_slim.pt dreams/models/pretrained/ssl_model_server.pt \
  data/validation/g8r_raw_reranker_cache.npz \
  tasks/eval_g8r_raw_reranker_nearmid.py; do
  [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 2; }
done

python -u tasks/eval_g8r_raw_reranker_nearmid.py \
  --device cuda \
  --output data/validation/g8r_raw_reranker_nearmid.json

echo "=== near/mid eval done: $(date) ==="
echo "Read data/validation/g8r_raw_reranker_nearmid.json"
