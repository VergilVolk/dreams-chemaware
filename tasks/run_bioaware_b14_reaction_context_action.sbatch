#!/bin/bash
#SBATCH --job-name=bio_b14_rxn
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=bioaware_b14_reaction_context_%j.out
#SBATCH --error=bioaware_b14_reaction_context_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams

OUTPUT="data/validation/bioaware_b14_reaction_context_${SLURM_JOB_ID}"
if [ -e "$OUTPUT" ]; then
  echo "Refusing to overwrite: $OUTPUT" >&2
  exit 1
fi

python -u tasks/test_bioaware_b14_reaction_context_action.py
python -u tasks/audit_bioaware_b14_reaction_context_action.py \
  --output-dir "$OUTPUT" \
  --bootstrap-resamples 10000 \
  --seed 20260907
python -u tasks/validate_bioaware_b14_reaction_context_action.py "$OUTPUT"

echo "Result: $OUTPUT"
