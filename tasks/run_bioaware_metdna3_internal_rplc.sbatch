#!/bin/bash
#SBATCH --job-name=bioaware_m3_rplc
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=data/validation/bioaware_m3_rplc_%j.out
#SBATCH --error=data/validation/bioaware_m3_rplc_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD/tasks:$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

MANIFEST=data/validation/bioaware_metdna3_internal_rplc_manifest_v1
MZML=data/external/metdna3_2025/mzml/internal_rplc
PREFLIGHT=data/validation/bioaware_metdna3_internal_rplc_ms2_preflight.json
CACHE=data/validation/bioaware_metdna3_internal_rplc_dreams_cache_v1
SCORES=data/validation/bioaware_metdna3_internal_rplc_dreams_official_v1

echo "=== BioAware frozen internal RPLC preparation started: $(date) ==="
python -u tasks/download_bioaware_metdna3_internal_mzml.py
python -u tasks/preflight_bioaware_metdna3_development_ms2.py \
  --development-dir "$MANIFEST" --truth-name internal_level1.csv.gz \
  --mzml-dir "$MZML" --output "$PREFLIGHT" --scope internal_rplc --expected-files 16
python -u tasks/build_bioaware_metdna3_dreams_cache.py \
  --development-dir "$MANIFEST" --truth-name internal_level1.csv.gz \
  --preflight "$PREFLIGHT" --mzml-dir "$MZML" --output-dir "$CACHE" \
  --scope internal_rplc --minimum-identities 75 --minimum-queries 75
python -u tasks/encode_bioaware_metdna3_dreams.py \
  --cache-dir "$CACHE" --output-dir "$SCORES" --device cuda --batch-size 128
echo "=== BioAware frozen internal RPLC preparation completed: $(date) ==="
