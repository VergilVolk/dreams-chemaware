#!/bin/bash
#SBATCH --job-name=bioaware_b28_branch
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=bioaware_b28_branch_%j.out
#SBATCH --error=bioaware_b28_branch_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS

PYTHON=/data02/home/scv7tsl/run/miniconda3/envs/dreams/bin/python
OUTPUT="data/validation/bioaware_b28_branch_spectral_arbitration_${SLURM_JOB_ID}"
B20_DIR="${OUTPUT}/b20_manifest"
AUDIT_DIR="${OUTPUT}/audit"

echo "=== BioAware B28 branch spectral arbitration started: $(date) ==="
"$PYTHON" -u tasks/test_bioaware_b20_direct_action_manifest.py
"$PYTHON" -u tasks/test_bioaware_b28_branch_spectral_arbitration.py
"$PYTHON" -u tasks/build_bioaware_b20_direct_action_manifest.py \
  --output-dir "$B20_DIR"
"$PYTHON" -u tasks/validate_bioaware_b20_direct_action_manifest.py "$B20_DIR"
set +e
"$PYTHON" -u tasks/audit_bioaware_b28_branch_spectral_arbitration.py \
  --b20-dir "$B20_DIR" \
  --output-dir "$AUDIT_DIR"
AUDIT_STATUS=$?
set -e
"$PYTHON" -u tasks/validate_bioaware_b28_branch_spectral_arbitration.py "$AUDIT_DIR"
echo "=== BioAware B28 branch spectral arbitration completed: $(date) ==="
echo "Scientific audit exit status: $AUDIT_STATUS"
echo "Result: $AUDIT_DIR/report.json"
