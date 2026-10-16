#!/bin/bash
#SBATCH --job-name=panredu_hv_dl
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --array=0-192%12
#SBATCH --output=data/validation/pan_redu_hv_download_%A_%a.out
#SBATCH --error=data/validation/pan_redu_hv_download_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
mkdir -p data/validation/pan_redu_high_value_raw_download/reports

python -u tasks/download_pan_redu_wave_raw_batch.py \
  --manifest data/validation/pan_redu_high_value_raw_tranche_20261014/raw_acquisition_manifest.tsv \
  --version-map data/validation/pan_redu_systematic_wave_acquisition_20261014/ftps_version_map.json \
  --root data/external/pan_redu_high_value_raw_tranche_20261014 \
  --report-dir data/validation/pan_redu_high_value_raw_download/reports \
  --batch-index "${SLURM_ARRAY_TASK_ID}" \
  --batch-size 20
