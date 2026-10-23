#!/bin/bash
# One-shot, cache-only repair after the run_2400157 complementarity diagnosis.
# Usage:
#   sbatch tasks/run_b_anchor_conditioned_family_repair.sbatch \
#     data/validation/B_dual_encoder_multireranker_run_2400157
#SBATCH --job-name=B_anchor_family
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=24:00:00
#SBATCH --output=data/validation/B_anchor_family_%j.out
#SBATCH --error=data/validation/B_anchor_family_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"

PREPARED_RUN="${1:-}"
[[ -n "$PREPARED_RUN" ]] || {
  echo "B_ANCHOR_FAMILY_USAGE: sbatch $0 data/validation/B_dual_encoder_multireranker_run_<JOBID>" >&2
  exit 2
}
case "$PREPARED_RUN" in
  data/validation/B_dual_encoder_multireranker_run_*) ;;
  *) echo "B_ANCHOR_FAMILY_INVALID_PREPARED_RUN: $PREPARED_RUN" >&2; exit 2 ;;
esac

SOURCE_RUN="${UNIFIED_GNPS_SOURCE_RUN:-data/validation/unified_candidate_energy_run_2364137}"
AB_RUN="${WSE_AB_PARENT_RUN:-data/validation/wse_conditional_residual_run_2376648}"
CHEM_CACHE="${CHEMAWARE_CONTINUOUS_CACHE:-$AB_RUN/continuous_chem_evidence}"
PRIOR="$PREPARED_RUN/modeling_nested_oof/candidate_predictions.npz"
OUT="$PREPARED_RUN/anchor_conditioned_family_nested_oof"

[[ ! -e "$OUT" ]] || { echo "B_ANCHOR_FAMILY_OUTPUT_COLLISION: $OUT" >&2; exit 3; }
required=(
  "$PREPARED_RUN/evidence_identity_disjoint.npz"
  "$PREPARED_RUN/evidence_formula_disjoint.npz"
  "$SOURCE_RUN/evidence_identity_disjoint.npz"
  "$SOURCE_RUN/evidence_formula_disjoint.npz"
  "$CHEM_CACHE/continuous_chem_identity_disjoint.npz"
  "$CHEM_CACHE/continuous_chem_formula_disjoint.npz"
  "$AB_RUN/modeling/report.json"
  "$AB_RUN/modeling/candidate_predictions.npz"
  "$PRIOR"
  tasks/train_b_anchor_conditioned_family_repair.py
  tasks/test_b_anchor_conditioned_family_repair.py
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "B_ANCHOR_FAMILY_MISSING: $path" >&2; exit 4; }
done

python -m py_compile tasks/train_b_anchor_conditioned_family_repair.py
python -u tasks/test_b_anchor_conditioned_family_repair.py
python -u tasks/train_b_anchor_conditioned_family_repair.py \
  --identity-bundle "$PREPARED_RUN/evidence_identity_disjoint.npz" \
  --formula-bundle "$PREPARED_RUN/evidence_formula_disjoint.npz" \
  --core-identity-bundle "$SOURCE_RUN/evidence_identity_disjoint.npz" \
  --core-formula-bundle "$SOURCE_RUN/evidence_formula_disjoint.npz" \
  --identity-chem "$CHEM_CACHE/continuous_chem_identity_disjoint.npz" \
  --formula-chem "$CHEM_CACHE/continuous_chem_formula_disjoint.npz" \
  --parent-report "$AB_RUN/modeling/report.json" \
  --parent-predictions "$AB_RUN/modeling/candidate_predictions.npz" \
  --prior-nested-predictions "$PRIOR" \
  --output "$OUT" --device cuda

echo "B_ANCHOR_CONDITIONED_FAMILY_REPAIR_COMPLETE_NO_EXTERNAL_ACCESS: $OUT"
