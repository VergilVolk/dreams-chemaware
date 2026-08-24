#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=g8r_raw_reranker_original_ab_%j.out
#SBATCH --error=g8r_raw_reranker_original_ab_%j.err

# Load the FROZEN RAW-v1 artifact (no freeze step here).  Verifies the artifact
# SHA256 + provenance, reproduces the dev +4.35pp, then evaluates the ORIGINAL
# locked Test-A/B.  If the artifact is missing or its hash mismatches, it exits
# before touching Test-A/B.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

ART=data/validation/g8r_raw_reranker_v1_artifact.json
for f in "$ART" "${ART%.json}.sha256" \
  data/models/MassSpecGym_MurckoHist_split.hdf5 \
  data/e1/official_embedding_slim.pt dreams/models/pretrained/ssl_model_server.pt \
  data/validation/g8r_raw_reranker_cache.npz \
  data/validation/g8r_raw_reranker_cache_val.npz \
  tasks/train_g8r_raw_reranker.py \
  data/validation/g8r_final_test/test_a_manifest.json \
  data/validation/g8r_final_test/test_b_manifest.json \
  tasks/eval_g8r_raw_reranker_original_ab.py; do
  [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 2; }
done

python -u tasks/eval_g8r_raw_reranker_original_ab.py \
  --device cuda \
  --artifact "$ART" \
  --output data/validation/g8r_raw_reranker_original_ab.json \
  --csv data/validation/g8r_raw_reranker_original_ab_perquery.csv

echo "=== original Test-A/B done: $(date) ==="
echo "Read data/validation/g8r_raw_reranker_original_ab.json + _perquery.csv"
