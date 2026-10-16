#!/bin/bash
#SBATCH --job-name=hfd_raw_dl
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --array=0-12%4
#SBATCH --output=data/validation/hfd_raw_dl_%A_%a.out
#SBATCH --error=data/validation/hfd_raw_dl_%A_%a.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

python -u tasks/download_factorial_hfd_raw_batch.py \
  --manifest data/validation/factorial_hfd_wave_manifest_v5_20261016/phenotype_blind/acquisition_manifest.tsv \
  --report-dir data/validation/factorial_hfd_raw_download_v1_20261016/reports \
  --batch-index "${SLURM_ARRAY_TASK_ID}" \
  --batch-size 20
