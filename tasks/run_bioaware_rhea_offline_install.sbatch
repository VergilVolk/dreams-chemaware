#!/bin/bash
#SBATCH --job-name=bioaware_rhea_offline
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=data/validation/bioaware_rhea_offline_%j.out
#SBATCH --error=data/validation/bioaware_rhea_offline_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

ARCHIVE="bioaware_rhea_cache_20260827.zip"
OUTPUT="data/reference/bioaware_rhea_offline_20260827"
[[ -f "$ARCHIVE" ]] || { echo "Missing offline archive: $ARCHIVE" >&2; exit 3; }
[[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite: $OUTPUT" >&2; exit 4; }

echo "=== BioAware offline Rhea install started: $(date) ==="
python -u tasks/install_bioaware_rhea_offline.py --archive "$ARCHIVE" --output-dir "$OUTPUT"
python -u tasks/validate_bioaware_rhea_cache.py "$OUTPUT"
echo "=== BioAware offline Rhea install completed: $(date) ==="
