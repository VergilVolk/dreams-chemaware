#!/bin/bash
# Recover the already completed OOF B winner and fit one all-identity model.
# No OOF rerun, no hyperparameter search, and no Enveda access.
#SBATCH --job-name=wse_B_finalize
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/wse_B_finalize_%j.out
#SBATCH --error=data/validation/wse_B_finalize_%j.err

set -euo pipefail

cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

AB_RUN="${WSE_AB_PARENT_RUN:-data/validation/wse_conditional_residual_run_2376648}"
C_RUN="${WSE_ADDITIVE_CONTROL_RUN:-data/validation/wse_additive_mechanism_control_run_2377151}"
SOURCE_RUN="${UNIFIED_GNPS_SOURCE_RUN:-data/validation/unified_candidate_energy_run_2364137}"
CHEM_CACHE="${CHEMAWARE_CONTINUOUS_CACHE:-$AB_RUN/continuous_chem_evidence}"
OUT="data/validation/wse_unified_B_candidate_run_${SLURM_JOB_ID}"

required=(
  "$SOURCE_RUN/evidence_identity_disjoint.npz"
  "$SOURCE_RUN/evidence_formula_disjoint.npz"
  "$CHEM_CACHE/continuous_chem_identity_disjoint.npz"
  "$CHEM_CACHE/continuous_chem_formula_disjoint.npz"
  "$AB_RUN/modeling/report.json"
  "$AB_RUN/modeling/candidate_predictions.npz"
  "$C_RUN/report.json"
  tasks/conditional_null_energy_core.py
  tasks/train_wse_conditional_residual.py
  tasks/finalize_wse_unified_ranking_candidate.py
  tasks/test_wse_conditional_residual.py
  tasks/test_finalize_wse_unified_ranking_candidate.py
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "WSE_B_FINALIZE_MISSING: $path" >&2; exit 4; }
done
[[ ! -e "$OUT" ]] || { echo "WSE_B_FINALIZE_OUTPUT_COLLISION: $OUT" >&2; exit 3; }

python -m py_compile \
  tasks/train_wse_conditional_residual.py \
  tasks/finalize_wse_unified_ranking_candidate.py
python -u tasks/test_wse_conditional_residual.py
python -u tasks/test_finalize_wse_unified_ranking_candidate.py

python -u tasks/finalize_wse_unified_ranking_candidate.py \
  --identity-bundle "$SOURCE_RUN/evidence_identity_disjoint.npz" \
  --formula-bundle "$SOURCE_RUN/evidence_formula_disjoint.npz" \
  --identity-chem "$CHEM_CACHE/continuous_chem_identity_disjoint.npz" \
  --formula-chem "$CHEM_CACHE/continuous_chem_formula_disjoint.npz" \
  --parent-report "$AB_RUN/modeling/report.json" \
  --parent-predictions "$AB_RUN/modeling/candidate_predictions.npz" \
  --additive-control-report "$C_RUN/report.json" \
  --output "$OUT" \
  --device cuda

echo "WSE_UNIFIED_B_CANDIDATE_FINALIZED_NO_ENVEDA_ACCESS: $OUT"
