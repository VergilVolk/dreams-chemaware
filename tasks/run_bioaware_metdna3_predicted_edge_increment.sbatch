#!/bin/bash
#SBATCH --job-name=bio_pred_edge
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=bioaware_predicted_edge_%j.out
#SBATCH --error=bioaware_predicted_edge_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS

echo "=== BioAware predicted-reaction edge audit started: $(date) ==="
python -u -m pytest -q tests/test_bioaware_metdna3_predicted_edge_increment.py
python -u tasks/audit_bioaware_metdna3_candidate_edge_ms2.py \
  --maximum-network-step 1 \
  --output-dir data/validation/bioaware_metdna3_candidate_edge_ms2_step1_v1
python -u tasks/summarize_bioaware_metdna3_predicted_edge_increment.py
echo "=== BioAware predicted-reaction edge audit completed: $(date) ==="
