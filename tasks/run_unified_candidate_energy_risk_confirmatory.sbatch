#!/bin/bash
# Single risk-aligned confirmation on consumed GNPS development only.
# This job must not read, score, or open Enveda-180.
#SBATCH --job-name=unified_risk
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=06:00:00
#SBATCH --output=data/validation/unified_risk_%j.out
#SBATCH --error=data/validation/unified_risk_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

SOURCE_RUN="${UNIFIED_GNPS_SOURCE_RUN:-data/validation/unified_candidate_energy_run_2364137}"
IDENTITY="$SOURCE_RUN/evidence_identity_disjoint.npz"
FORMULA="$SOURCE_RUN/evidence_formula_disjoint.npz"
OUT="data/validation/unified_candidate_energy_risk_confirmatory_run_${SLURM_JOB_ID}"

for path in \
  "$IDENTITY" \
  "$FORMULA" \
  tasks/train_unified_candidate_energy_crossfit.py \
  tasks/train_unified_candidate_energy_risk_confirmatory.py \
  tasks/test_unified_candidate_energy.py
do
  [[ -s "$path" ]] || { echo "UNIFIED_RISK_MISSING_INPUT: $path" >&2; exit 4; }
done
[[ ! -e "$OUT" ]] || { echo "UNIFIED_RISK_OUTPUT_COLLISION: $OUT" >&2; exit 5; }

python -m py_compile \
  tasks/train_unified_candidate_energy_crossfit.py \
  tasks/train_unified_candidate_energy_risk_confirmatory.py
python -u tasks/test_unified_candidate_energy.py

python -u tasks/train_unified_candidate_energy_risk_confirmatory.py \
  --identity-bundle "$IDENTITY" \
  --formula-bundle "$FORMULA" \
  --output "$OUT" \
  --device cuda

echo "UNIFIED_RISK_CONFIRMATORY_COMPLETE_NO_ENVEDA_ACCESS: $OUT"
