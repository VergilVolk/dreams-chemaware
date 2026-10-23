#!/bin/bash
#SBATCH --job-name=hfd_index
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/hfd_index_%j.out
#SBATCH --error=data/validation/hfd_index_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

INDEX="data/validation/factorial_hfd_response_index_official_v1_20261016"
RETRIEVAL="data/validation/factorial_hfd_pretruth_retrieval_official_v1_20261016"
python -u tasks/assemble_factorial_hfd_response_index.py \
  --panel-jobs data/validation/factorial_hfd_panel_jobs_v1_20261016/panel_jobs.tsv \
  --panel-registry data/validation/factorial_hfd_wave_manifest_v5_20261016/panel_registry.tsv \
  --effects-root data/validation/factorial_hfd_panel_effects_v1_20261016 \
  --embeddings-root data/validation/factorial_hfd_panel_embeddings_official_v1_20261016 \
  --families-root data/validation/factorial_hfd_ion_families_v1_20261016 \
  --identity-table data/validation/factorial_hfd_identity_partition_v1_20261023/identity_partition.tsv.gz \
  --output-dir "$INDEX" \
  --minimum-entities 12

python -u tasks/query_factorial_hfd_response_index.py \
  --manifest "$INDEX/study_contrast_manifest.tsv" \
  --sketches "$INDEX/study_contrast_sketches.npz" \
  --output-dir "$RETRIEVAL" \
  --minimum-entities 12
