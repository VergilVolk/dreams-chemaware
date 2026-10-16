#!/bin/bash
#SBATCH --job-name=panredu_index
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/pan_redu_index_%j.out
#SBATCH --error=data/validation/pan_redu_index_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

OUT="data/validation/pan_redu_signed_response_index_official_20261016"
python -u tasks/assemble_pan_redu_signed_response_index.py \
  --panel-jobs data/validation/pan_redu_high_value_raw_panel_jobs_20261014/panel_jobs.tsv \
  --topology data/validation/pan_redu_high_value_raw_panel_jobs_20261014/contrast_topology_readiness.tsv \
  --effects-root data/validation/pan_redu_high_value_raw_panel_effects_20261014 \
  --embeddings-root data/validation/pan_redu_high_value_raw_panel_embeddings_official_20261016 \
  --output-dir "$OUT" \
  --minimum-entities 12

python -u tasks/query_signed_spectral_measure_sketches.py \
  --manifest "$OUT/contrast_manifest.tsv" \
  --sketches "$OUT/contrast_sketches.npz" \
  --output-dir "data/validation/pan_redu_signed_response_retrieval_official_20261016" \
  --all-queries \
  --minimum-entities 12
