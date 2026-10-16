#!/bin/bash
#SBATCH --job-name=hfd_open
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=data/validation/hfd_open_%j.out
#SBATCH --error=data/validation/hfd_open_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

# This is deliberately separate from all pre-truth jobs.  Submit it only after
# the index, retrieval, null, and jackknife reports have been hash-frozen.
python -u tasks/evaluate_factorial_hfd_retrieval.py \
  --scores data/validation/factorial_hfd_pretruth_retrieval_official_v1_20261016/retrieval_scores.tsv.gz \
  --null data/validation/factorial_hfd_response_null_official_v1_20261016/response_assignment_null.npz \
  --jackknife data/validation/factorial_hfd_panel_jackknife_official_v1_20261016/single_panel_deletion_scores.tsv.gz \
  --output-dir data/validation/factorial_hfd_primary_adjudication_official_v1_20261016 \
  --world all
