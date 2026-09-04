#!/bin/bash
#SBATCH --job-name=bioaware_rhea
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=data/validation/bioaware_rhea_%j.out
#SBATCH --error=data/validation/bioaware_rhea_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

OUTPUT="data/reference/bioaware_rhea"
[[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite: $OUTPUT" >&2; exit 4; }

echo "=== BioAware Rhea cache started: $(date) ==="
python -u tasks/build_bioaware_rhea_cache.py --output-dir "$OUTPUT"
python -u tasks/validate_bioaware_rhea_cache.py "$OUTPUT"
echo "=== BioAware Rhea cache completed: $(date) ==="
