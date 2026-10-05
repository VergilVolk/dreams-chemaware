#!/bin/bash
#SBATCH --job-name=bioaware_b4_decomp
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:20:00
#SBATCH --output=bioaware_b4_decomposition_%j.out
#SBATCH --error=bioaware_b4_decomposition_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
PYTHON_BIN=/data02/home/scv7tsl/run/miniconda3/envs/dreams/bin/python
B3_SOURCE="${B3_SOURCE:-data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz}"
OUTPUT="data/validation/bioaware_b4_opportunity_decomposition_${SLURM_JOB_ID}"

[[ -x "$PYTHON_BIN" ]] || { echo "Missing Python: $PYTHON_BIN" >&2; exit 2; }
[[ -s "$B3_SOURCE" ]] || { echo "Missing B3 candidate cache: $B3_SOURCE" >&2; exit 3; }
[[ ! -e "$OUTPUT" ]] || { echo "Refusing to overwrite: $OUTPUT" >&2; exit 4; }

"$PYTHON_BIN" tasks/test_bioaware_b4_opportunity_decomposition.py
"$PYTHON_BIN" tasks/audit_bioaware_b4_opportunity_decomposition.py \
  --candidate-features "$B3_SOURCE" \
  --output-dir "$OUTPUT" \
  --bootstrap-resamples 5000 \
  --seed 20260906
"$PYTHON_BIN" tasks/validate_bioaware_b4_opportunity_decomposition.py "$OUTPUT"
echo "BioAware B4 result: $OUTPUT/report.json"
