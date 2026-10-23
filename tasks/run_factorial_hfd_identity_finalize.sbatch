#!/bin/bash
#SBATCH --job-name=hfd_ident_final
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/hfd_ident_final_%j.out
#SBATCH --error=data/validation/hfd_ident_final_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

python -u tasks/finalize_pan_redu_panel_identity.py \
  --panel-jobs data/validation/factorial_hfd_panel_jobs_v1_20261016/panel_jobs.tsv \
  --partitions-root data/validation/factorial_hfd_panel_identity_v1_20261023 \
  --output-dir data/validation/factorial_hfd_identity_partition_v1_20261023
