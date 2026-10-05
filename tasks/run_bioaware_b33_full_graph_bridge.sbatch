#!/bin/bash
#SBATCH --job-name=bioaware_b33_bridge
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --output=bioaware_b33_bridge_%j.out
#SBATCH --error=bioaware_b33_bridge_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS

PYTHON=/data02/home/scv7tsl/run/miniconda3/envs/dreams/bin/python
B32_DIR="data/validation/bioaware_b32_comprehensive_action_bank_frozen_20260907_v1"
GRAPH_DIR="data/validation/bioaware_b33_full_reference_graph_${SLURM_JOB_ID}"
OUTPUT="data/validation/bioaware_b33_full_graph_bridge_${SLURM_JOB_ID}"

echo "=== BioAware B33 full-reference shared-embedding bridge started: $(date) ==="
"$PYTHON" -u tasks/test_bioaware_b32_comprehensive_action_bank.py
"$PYTHON" -u tasks/test_bioaware_b33_full_graph_bridge.py
"$PYTHON" -u tasks/preflight_bioaware_b32_frozen_artifact.py "$B32_DIR"
"$PYTHON" -u tasks/validate_bioaware_b32_comprehensive_action_bank.py "$B32_DIR"

"$PYTHON" -u tasks/build_bioaware_b33_full_candidate_graph.py \
  --b32-dir "$B32_DIR" \
  --output-dir "$GRAPH_DIR"
"$PYTHON" -u tasks/validate_bioaware_b33_full_candidate_graph.py "$GRAPH_DIR"

"$PYTHON" -u tasks/train_bioaware_b33_full_graph_bridge.py \
  --graph-dir "$GRAPH_DIR" \
  --output "$OUTPUT" \
  --steps 96 \
  --batch-corrective-identities 8 \
  --batch-known-safety-identities 4 \
  --batch-broad-safety-identities 8 \
  --unfreeze-blocks 1 \
  --backbone-lr 1e-6 \
  --head-lr 5e-6 \
  --lambda-corrective 1.0 \
  --lambda-known-safety 4.0 \
  --lambda-broad-safety 2.0 \
  --lambda-preserve 10.0 \
  --grad-clip 1.0
"$PYTHON" -u tasks/validate_bioaware_b33_full_graph_bridge.py "$OUTPUT"

echo "=== BioAware B33 full-reference shared-embedding bridge completed: $(date) ==="
echo "Graph: $GRAPH_DIR/report.json"
echo "Result: $OUTPUT/report.json"
