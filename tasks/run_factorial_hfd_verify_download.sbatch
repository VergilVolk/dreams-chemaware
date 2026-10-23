#!/bin/bash
#SBATCH --job-name=hfd_verify
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=data/validation/hfd_verify_%j.out
#SBATCH --error=data/validation/hfd_verify_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

python -u tasks/verify_factorial_hfd_download.py \
  --manifest data/validation/factorial_hfd_wave_manifest_v5_20261016/phenotype_blind/acquisition_manifest.tsv \
  --report-dir data/validation/factorial_hfd_raw_download_v1_20261016/reports \
  --output-dir data/validation/factorial_hfd_raw_download_verified_v1_20261023
