#!/bin/bash
#SBATCH --job-name=bioaware_b31_canary
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=bioaware_b31_canary_%j.out
#SBATCH --error=bioaware_b31_canary_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS

PYTHON=/data02/home/scv7tsl/run/miniconda3/envs/dreams/bin/python
OUTPUT="data/validation/bioaware_b31_sink_safe_embedding_canary_${SLURM_JOB_ID}"
B20_DIR="data/validation/bioaware_b20_direct_action_manifest_localcheck_20260907_v1"
B30_DIR="${OUTPUT}/b30_sink_veto"
B31_DIR="${OUTPUT}/b31_action_bank"
CANARY_DIR="${OUTPUT}/canary"

echo "=== BioAware B31 sink-safe embedding canary started: $(date) ==="
"$PYTHON" -u tasks/test_bioaware_b30_cross_source_sink_veto.py
"$PYTHON" -u tasks/test_bioaware_b31_sink_safe_action_bank.py
"$PYTHON" -u tasks/test_bioaware_b26_direct_shared_embedding_canary.py

"$PYTHON" -u tasks/preflight_bioaware_b20_frozen_artifact.py "$B20_DIR"
"$PYTHON" -u tasks/validate_bioaware_b20_direct_action_manifest.py "$B20_DIR"
"$PYTHON" -u tasks/audit_bioaware_b30_cross_source_sink_veto.py \
  --b20-dir "$B20_DIR" \
  --output-dir "$B30_DIR"
"$PYTHON" -u tasks/validate_bioaware_b30_cross_source_sink_veto.py "$B30_DIR"
"$PYTHON" -u tasks/build_bioaware_b31_sink_safe_action_bank.py \
  --b20-dir "$B20_DIR" \
  --b30-dir "$B30_DIR" \
  --output-dir "$B31_DIR"
"$PYTHON" -u tasks/validate_bioaware_b31_sink_safe_action_bank.py "$B31_DIR"
"$PYTHON" -u tasks/train_bioaware_b26_direct_shared_embedding_canary.py \
  --manifest-dir "$B31_DIR" \
  --output "$CANARY_DIR"
"$PYTHON" -u tasks/validate_bioaware_b26_direct_shared_embedding_canary.py "$CANARY_DIR"

echo "=== BioAware B31 sink-safe embedding canary completed: $(date) ==="
echo "Action result: $B30_DIR/report.json"
echo "Action bank: $B31_DIR/report.json"
echo "Embedding canary: $CANARY_DIR/report.json"
