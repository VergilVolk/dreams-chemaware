#!/bin/bash
#SBATCH --job-name=panredu_open_view
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/pan_redu_open_view_%j.out
#SBATCH --error=data/validation/pan_redu_open_view_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

MATRIX_ROOT="data/validation/pan_redu_fixed_capacity_response_transport_matrix_20261019_v5"
python -u tasks/build_pan_redu_open_retrieval_view.py \
  --matrix "$MATRIX_ROOT/ordered_contrast_transport_matrix.tsv.gz" \
  --matrix-report "$MATRIX_ROOT/report.json" \
  --output-dir data/validation/pan_redu_truth_blind_open_retrieval_20261019_v1
