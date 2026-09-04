#!/bin/bash
#SBATCH --job-name=bio_emb_v2_pre
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=bioaware_embedding_v2_preflight_%j.out
#SBATCH --error=bioaware_embedding_v2_preflight_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONUNBUFFERED=1

MANIFEST=data/validation/bioaware_embedding_relation_manifest_v2_20260830
PREFLIGHT=data/validation/bioaware_embedding_adapter_preflight_v2_20260830.json
test ! -e "$MANIFEST" || { echo "fail-closed: $MANIFEST exists" >&2; exit 2; }
test ! -e "$PREFLIGHT" || { echo "fail-closed: $PREFLIGHT exists" >&2; exit 3; }
python -u tasks/build_bioaware_embedding_relation_manifest.py --output-dir "$MANIFEST"
python -u tasks/preflight_bioaware_embedding_adapter.py \
  --manifest-dir "$MANIFEST" --output "$PREFLIGHT"
