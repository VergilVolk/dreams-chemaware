#!/bin/bash
#SBATCH --job-name=hfd_local_response
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/hfd_local_response_%j.out
#SBATCH --error=data/validation/hfd_local_response_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

python -u tasks/query_factorial_hfd_local_directed_response.py \
  --panel-jobs data/validation/factorial_hfd_panel_jobs_v1_20261016/panel_jobs.tsv \
  --panel-registry data/validation/factorial_hfd_wave_manifest_v5_20261016/panel_registry.tsv \
  --effects-root data/validation/factorial_hfd_panel_effects_v1_20261016 \
  --embeddings-root data/validation/factorial_hfd_panel_embeddings_official_v1_20261016 \
  --families-root data/validation/factorial_hfd_ion_families_v1_20261016 \
  --identity-table data/validation/factorial_hfd_identity_partition_v1_20261023/identity_partition.tsv.gz \
  --output-dir data/validation/factorial_hfd_local_directed_official_v1_20261023 \
  --minimum-cosine 0.70 \
  --identity-ppm 10 \
  --minimum-local-edges 3 \
  --minimum-track-edges 1 \
  --block-size 1024 \
  --device cuda
