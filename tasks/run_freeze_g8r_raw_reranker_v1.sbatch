#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=freeze_g8r_raw_reranker_v1_%j.out
#SBATCH --error=freeze_g8r_raw_reranker_v1_%j.err

# ONE-TIME freeze: train once and save the RAW-v1 artifact + its SHA256 side
# file.  Run this exactly once; the eval sbatch only loads the artifact.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

for f in data/validation/g8r_raw_reranker_cache.npz \
  tasks/train_g8r_raw_reranker.py tasks/freeze_g8r_raw_reranker_v1.py; do
  [[ -f "$f" ]] || { echo "MISSING: $f" >&2; exit 2; }
done

python -u tasks/freeze_g8r_raw_reranker_v1.py \
  --output data/validation/g8r_raw_reranker_v1_artifact.json

echo "=== freeze done: $(date) ==="
echo "Record the printed artifact_sha256 and DO NOT re-run this before the eval."
