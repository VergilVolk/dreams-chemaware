#!/bin/bash
# Sole post-A/B mechanism control: capacity-matched additive versus joint B.
# Reuses the completed GNPS A/B run and chemistry cache; never reads Enveda.
#SBATCH --job-name=wse_additive_ctl
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --output=data/validation/wse_additive_ctl_%j.out
#SBATCH --error=data/validation/wse_additive_ctl_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

PARENT_RUN="${WSE_AB_PARENT_RUN:-data/validation/wse_conditional_residual_run_2376648}"
SOURCE_RUN="${UNIFIED_GNPS_SOURCE_RUN:-data/validation/unified_candidate_energy_run_2364137}"
CHEM_CACHE="${CHEMAWARE_CONTINUOUS_CACHE:-$PARENT_RUN/continuous_chem_evidence}"
OUT="data/validation/wse_additive_mechanism_control_run_${SLURM_JOB_ID}"

required=(
  "$SOURCE_RUN/evidence_identity_disjoint.npz"
  "$SOURCE_RUN/evidence_formula_disjoint.npz"
  "$CHEM_CACHE/continuous_chem_identity_disjoint.npz"
  "$CHEM_CACHE/continuous_chem_formula_disjoint.npz"
  "$PARENT_RUN/modeling/report.json"
  "$PARENT_RUN/modeling/candidate_predictions.npz"
  tasks/conditional_null_energy_core.py
  tasks/train_wse_conditional_residual.py
  tasks/train_wse_additive_mechanism_control.py
  tasks/test_wse_additive_mechanism_control.py
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "WSE_ADDITIVE_CONTROL_MISSING: $path" >&2; exit 4; }
done
[[ ! -e "$OUT" ]] || { echo "WSE_ADDITIVE_CONTROL_OUTPUT_COLLISION: $OUT" >&2; exit 3; }

python -m py_compile tasks/train_wse_additive_mechanism_control.py
python -u tasks/test_wse_additive_mechanism_control.py

python -u tasks/train_wse_additive_mechanism_control.py \
  --identity-bundle "$SOURCE_RUN/evidence_identity_disjoint.npz" \
  --formula-bundle "$SOURCE_RUN/evidence_formula_disjoint.npz" \
  --identity-chem "$CHEM_CACHE/continuous_chem_identity_disjoint.npz" \
  --formula-chem "$CHEM_CACHE/continuous_chem_formula_disjoint.npz" \
  --parent-report "$PARENT_RUN/modeling/report.json" \
  --parent-predictions "$PARENT_RUN/modeling/candidate_predictions.npz" \
  --output "$OUT" \
  --device cuda

echo "WSE_ADDITIVE_MECHANISM_CONTROL_COMPLETE_NO_ENVEDA_ACCESS: $OUT"
