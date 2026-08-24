#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=g8r_raw_reranker_final_test_%j.out
#SBATCH --error=g8r_raw_reranker_final_test_%j.err

# ONE-SHOT final test of the frozen RAW reranker v1 on Test-A/B/C.
# No parameter may change after this runs.  Candidate library = full HDF5 minus
# g8r IK14.  Embedding is GPU; RAW peak features are CPU (~30-70 min).
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

for f in data/models/MassSpecGym_MurckoHist_split.hdf5 \
  data/e1/official_embedding_slim.pt dreams/models/pretrained/ssl_model_server.pt \
  data/validation/g8r_raw_reranker_cache.npz \
  data/validation/g8r_final_test_large/test_a_queries.csv \
  data/validation/g8r_final_test_large/test_b_queries.csv \
  data/validation/g8r_final_test_large/test_c_queries.csv \
  tasks/eval_g8r_raw_reranker_final_test.py; do
  [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 2; }
done

python -u tasks/eval_g8r_raw_reranker_final_test.py \
  --device cuda \
  --output data/validation/g8r_raw_reranker_final_test.json

echo "=== final test done: $(date) ==="
echo "Read data/validation/g8r_raw_reranker_final_test.json (ONE-SHOT result)"
