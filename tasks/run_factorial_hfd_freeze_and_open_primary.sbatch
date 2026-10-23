#!/bin/bash
#SBATCH --job-name=hfd_freeze_open
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/hfd_freeze_open_%j.out
#SBATCH --error=data/validation/hfd_freeze_open_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

INDEX="data/validation/factorial_hfd_response_index_official_v1_20261016"
RETRIEVAL="data/validation/factorial_hfd_pretruth_retrieval_official_v1_20261016"
NULL_DIR="data/validation/factorial_hfd_response_null_official_v1_20261016"
JACKKNIFE="data/validation/factorial_hfd_panel_jackknife_official_v1_20261016"

python -u tasks/freeze_factorial_hfd_pretruth.py \
  --index-dir "$INDEX" \
  --retrieval-dir "$RETRIEVAL" \
  --null-dir "$NULL_DIR" \
  --jackknife-dir "$JACKKNIFE" \
  --output-dir data/validation/factorial_hfd_pretruth_freeze_official_v1_20261023

python -u tasks/evaluate_factorial_hfd_retrieval.py \
  --scores "$RETRIEVAL/retrieval_scores.tsv.gz" \
  --null "$NULL_DIR/response_assignment_null.npz" \
  --jackknife "$JACKKNIFE/single_panel_deletion_scores.tsv.gz" \
  --output-dir data/validation/factorial_hfd_primary_adjudication_official_v1_20261023 \
  --world all
