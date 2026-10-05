#!/bin/bash
#SBATCH --job-name=bioaware_b2_transform
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=bioaware_b2_transform_%j.out
#SBATCH --error=bioaware_b2_transform_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS

PYTHON=/data02/home/scv7tsl/run/miniconda3/envs/dreams/bin/python
STAMP="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
FINAL="data/validation/bioaware_b2_reaction_transform_${STAMP}"
WORK="${FINAL}.partial"

if [[ -e "${FINAL}" || -e "${WORK}" ]]; then
  echo "Refusing to overwrite: ${FINAL} or ${WORK}" >&2
  exit 2
fi

cleanup() {
  if [[ -d "${WORK}" ]]; then
    echo "Incomplete output retained for audit: ${WORK}" >&2
  fi
}
trap cleanup EXIT

echo "=== BioAware B2 reaction-transform action started: $(date) ==="
"${PYTHON}" tasks/test_bioaware_b2_reaction_transform_action.py
"${PYTHON}" -u tasks/audit_bioaware_b2_reaction_transform_action.py \
  --output-dir "${WORK}" \
  --bootstrap-resamples 5000 \
  --seed 20260906
"${PYTHON}" tasks/validate_bioaware_b2_reaction_transform_action.py \
  --output-dir "${WORK}"

mv "${WORK}" "${FINAL}"
trap - EXIT
echo "[sealed] ${FINAL}"
echo "=== BioAware B2 reaction-transform action completed: $(date) ==="
