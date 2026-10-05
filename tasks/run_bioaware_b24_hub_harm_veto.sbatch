#!/bin/bash
#SBATCH --job-name=bioaware_b24_hubveto
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=logs/bioaware_b24_hubveto_%j.out
#SBATCH --error=logs/bioaware_b24_hubveto_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
mkdir -p logs
OUTPUT="data/validation/bioaware_b24_hub_harm_veto_${SLURM_JOB_ID}"
if [ -e "${OUTPUT}" ]; then
  echo "Refusing to overwrite: ${OUTPUT}" >&2
  exit 2
fi

python -u tasks/test_bioaware_b24_hub_harm_veto.py
set +e
python -u tasks/audit_bioaware_b24_hub_harm_veto.py \
  --output-dir "${OUTPUT}" \
  --bootstrap-resamples 10000
STATUS=$?
set -e
python -u tasks/validate_bioaware_b24_hub_harm_veto.py "${OUTPUT}"
exit "${STATUS}"
