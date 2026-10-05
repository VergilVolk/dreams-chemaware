#!/bin/bash
#SBATCH --job-name=bioaware_b32_canary
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=bioaware_b32_canary_%j.out
#SBATCH --error=bioaware_b32_canary_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS

PYTHON=/data02/home/scv7tsl/run/miniconda3/envs/dreams/bin/python
B32_DIR="data/validation/bioaware_b32_comprehensive_action_bank_frozen_20260907_v1"
OUTPUT="data/validation/bioaware_b32_direct_shared_embedding_canary_${SLURM_JOB_ID}"

echo "=== BioAware B32 comprehensive direct shared-embedding canary started: $(date) ==="
"$PYTHON" -u tasks/test_bioaware_b32_comprehensive_action_bank.py
"$PYTHON" -u tasks/test_bioaware_b26_direct_shared_embedding_canary.py
"$PYTHON" -u tasks/preflight_bioaware_b32_frozen_artifact.py "$B32_DIR"
"$PYTHON" -u tasks/validate_bioaware_b32_comprehensive_action_bank.py "$B32_DIR"
"$PYTHON" -u tasks/train_bioaware_b26_direct_shared_embedding_canary.py \
  --manifest-dir "$B32_DIR" \
  --output "$OUTPUT" \
  --steps 96 \
  --batch-corrective-identities 8 \
  --batch-safety-identities 4 \
  --unfreeze-blocks 1 \
  --backbone-lr 1e-6 \
  --head-lr 5e-6 \
  --lambda-corrective 1.0 \
  --lambda-safety 4.0 \
  --lambda-preserve 10.0
"$PYTHON" -u tasks/validate_bioaware_b26_direct_shared_embedding_canary.py "$OUTPUT"

echo "=== BioAware B32 comprehensive direct shared-embedding canary completed: $(date) ==="
echo "Result: $OUTPUT/report.json"
