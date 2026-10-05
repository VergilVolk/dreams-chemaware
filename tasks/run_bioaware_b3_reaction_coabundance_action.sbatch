#!/bin/bash
#SBATCH --job-name=bioaware_b3_coab
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=bioaware_b3_coabundance_%j.out
#SBATCH --error=bioaware_b3_coabundance_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

PYTHON_BIN=/data02/home/scv7tsl/run/miniconda3/envs/dreams/bin/python
OUTPUT="data/validation/bioaware_b3_reaction_coabundance_${SLURM_JOB_ID}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python: $PYTHON_BIN" >&2
  exit 2
fi
if [[ -e "$OUTPUT" ]]; then
  echo "Refusing to overwrite: $OUTPUT" >&2
  exit 3
fi

"$PYTHON_BIN" tasks/test_bioaware_b3_reaction_coabundance_action.py
"$PYTHON_BIN" tasks/audit_bioaware_b3_reaction_coabundance_action.py \
  --output-dir "$OUTPUT" \
  --bootstrap-resamples 5000 \
  --seed 20260906
"$PYTHON_BIN" tasks/validate_bioaware_b3_reaction_coabundance_action.py "$OUTPUT"

echo "BioAware B3 result: $OUTPUT/report.json"
