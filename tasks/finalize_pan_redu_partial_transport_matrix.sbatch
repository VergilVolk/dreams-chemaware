#!/bin/bash
#SBATCH --job-name=panredu_ot_final
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/pan_redu_ot_final_%j.out
#SBATCH --error=data/validation/pan_redu_ot_final_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

python -u tasks/finalize_pan_redu_partial_transport_matrix.py \
  --pair-manifest data/validation/pan_redu_partial_transport_pairs_20261018_v2/panel_pairs.tsv \
  --pair-results-root data/validation/pan_redu_fixed_capacity_response_transport_20261019_v5 \
  --output-dir data/validation/pan_redu_fixed_capacity_response_transport_matrix_20261019_v5
