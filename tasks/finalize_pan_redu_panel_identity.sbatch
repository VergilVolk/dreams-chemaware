#!/bin/bash
#SBATCH --job-name=panredu_ident_final
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/pan_redu_ident_final_%j.out
#SBATCH --error=data/validation/pan_redu_ident_final_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

python -u tasks/finalize_pan_redu_panel_identity.py \
  --panel-jobs data/validation/pan_redu_positive_panel_jobs_20261018_v2/panel_jobs.tsv \
  --partitions-root data/validation/pan_redu_high_value_raw_panel_identity_20261018_v2 \
  --output-dir data/validation/pan_redu_high_value_raw_identity_partition_20261018_v2
