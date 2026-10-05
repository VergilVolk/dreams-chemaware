#!/bin/bash
#SBATCH --job-name=bioaware_b26_canary
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=bioaware_b26_canary_%j.out
#SBATCH --error=bioaware_b26_canary_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
PYTHON=/data02/home/scv7tsl/run/miniconda3/envs/dreams/bin/python
OUTPUT="data/validation/bioaware_b26_direct_shared_embedding_canary_${SLURM_JOB_ID}"
B20_DIR="${OUTPUT}/b20_manifest"
CANARY_DIR="${OUTPUT}/canary"

echo "=== BioAware B26 direct shared-embedding canary started: $(date) ==="
"$PYTHON" -u tasks/test_bioaware_b20_direct_action_manifest.py
"$PYTHON" -u tasks/test_bioaware_b26_direct_shared_embedding_canary.py
"$PYTHON" -u tasks/build_bioaware_b20_direct_action_manifest.py \
  --output-dir "$B20_DIR"
"$PYTHON" -u tasks/validate_bioaware_b20_direct_action_manifest.py "$B20_DIR"
"$PYTHON" -u tasks/train_bioaware_b26_direct_shared_embedding_canary.py \
  --manifest-dir "$B20_DIR" \
  --output "$CANARY_DIR"
"$PYTHON" -u tasks/validate_bioaware_b26_direct_shared_embedding_canary.py "$CANARY_DIR"
echo "=== BioAware B26 direct shared-embedding canary completed: $(date) ==="
echo "Result: $CANARY_DIR/report.json"
