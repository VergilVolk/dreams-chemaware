#!/bin/bash
#SBATCH --job-name=bioaware_b1_action
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=bioaware_b1_action_%j.out
#SBATCH --error=bioaware_b1_action_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS

PYTHON=/data02/home/scv7tsl/run/miniconda3/envs/dreams/bin/python
STAMP="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID}"
FINAL="data/validation/bioaware_b1_multisource_action_${STAMP}"
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

echo "=== BioAware B1 multisource action started: $(date) ==="
"${PYTHON}" tasks/test_bioaware_b1_multisource_action.py
"${PYTHON}" tasks/develop_bioaware_b1_multisource_action.py \
  --output-dir "${WORK}" \
  --bootstrap-resamples 5000 \
  --seed 20260906
"${PYTHON}" tasks/validate_bioaware_b1_multisource_action.py --output-dir "${WORK}"

mv "${WORK}" "${FINAL}"
trap - EXIT
echo "[sealed] ${FINAL}"
echo "=== BioAware B1 multisource action completed: $(date) ==="
