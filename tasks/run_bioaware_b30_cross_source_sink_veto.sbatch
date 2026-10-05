#!/bin/bash
#SBATCH --job-name=bioaware_b30_sink
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=bioaware_b30_sink_%j.out
#SBATCH --error=bioaware_b30_sink_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS

PYTHON=/data02/home/scv7tsl/run/miniconda3/envs/dreams/bin/python
OUTPUT="data/validation/bioaware_b30_cross_source_sink_veto_${SLURM_JOB_ID}"
B20_DIR="data/validation/bioaware_b20_direct_action_manifest_localcheck_20260907_v1"
AUDIT_A_DIR="${OUTPUT}/audit_a"
AUDIT_B_DIR="${OUTPUT}/audit_b"

echo "=== BioAware B30 cross-source sink veto started: $(date) ==="
"$PYTHON" -u tasks/test_bioaware_b30_cross_source_sink_veto.py
"$PYTHON" -u tasks/test_bioaware_b30_reproducibility.py
"$PYTHON" -u tasks/preflight_bioaware_b20_frozen_artifact.py "$B20_DIR"
"$PYTHON" -u tasks/validate_bioaware_b20_direct_action_manifest.py "$B20_DIR"
"$PYTHON" -u tasks/audit_bioaware_b30_cross_source_sink_veto.py \
  --b20-dir "$B20_DIR" \
  --output-dir "$AUDIT_A_DIR"
"$PYTHON" -u tasks/validate_bioaware_b30_cross_source_sink_veto.py "$AUDIT_A_DIR"
"$PYTHON" -u tasks/audit_bioaware_b30_cross_source_sink_veto.py \
  --b20-dir "$B20_DIR" \
  --output-dir "$AUDIT_B_DIR"
"$PYTHON" -u tasks/validate_bioaware_b30_cross_source_sink_veto.py "$AUDIT_B_DIR"
"$PYTHON" -u tasks/validate_bioaware_b30_reproducibility.py \
  "$AUDIT_A_DIR" "$AUDIT_B_DIR" \
  --output "${OUTPUT}/reproducibility.json"
echo "=== BioAware B30 cross-source sink veto completed: $(date) ==="
echo "Primary result: $AUDIT_A_DIR/report.json"
echo "Replay result: $AUDIT_B_DIR/report.json"
echo "Reproducibility: ${OUTPUT}/reproducibility.json"
