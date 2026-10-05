#!/bin/bash
#SBATCH --job-name=chem_outer_once
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=18:00:00
#SBATCH --output=data/validation/chem_outer_once_%j.out
#SBATCH --error=data/validation/chem_outer_once_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

RUN_ROOT="data/validation/chemaware_truthblind_outer_once/run_${SLURM_JOB_ID}"
NEW_POLICY_OUT="$RUN_ROOT/release_policy"
REUSABLE_POLICY="data/validation/chemaware_truthblind_outer_once/run_2338356/release_policy"
POLICY_OUT="$NEW_POLICY_OUT"
OUTER_OUT="$RUN_ROOT/outer_fold_4"
CANONICAL="data/validation/chemaware_truthblind_candidate_policy/run_2338337/policy"
SEAL="data/validation/chemaware_truthblind_candidate_policy/OUTER_FOLD_4_OPENED.lock"
[[ ! -e "$RUN_ROOT" ]] || { echo "Output collision: $RUN_ROOT" >&2; exit 3; }
[[ ! -e "$SEAL" ]] || { echo "ChemAware outer fold 4 was already opened: $SEAL" >&2; exit 4; }
mkdir -p "$RUN_ROOT"
trap 'echo "FAILED: ChemAware release replay or sealed outer evaluation stopped before completion" >&2' ERR

required_files=(
  tasks/chemaware_truthblind_candidate_core.py
  tasks/chemaware_outer_evaluation_core.py
  tasks/audit_chemaware_candidate_evidence_policy.py
  tasks/audit_chemaware_mass_kernel_embedding.py
  tasks/audit_chemaware_orthogonal_rule_residual_policy.py
  tasks/evaluate_chemaware_truthblind_outer_once.py
  tasks/test_chemaware_truthblind_candidate_core.py
  tasks/test_chemaware_truthblind_candidate_policy_contracts.py
  tasks/test_chemaware_outer_evaluation_core.py
  tasks/test_chemaware_truthblind_outer_once_contracts.py
  "$CANONICAL/truthblind_policy.joblib"
  "$CANONICAL/truthblind_policy.json"
  "$CANONICAL/report.json"
  "$CANONICAL/inner_policy.npz"
  data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz
  data/validation/chemaware_corrected_manifest_tokens_v1/rows.npy
  data/validation/chemaware_corrected_manifest_tokens_v1/official_embeddings_f32.npy
  dreams/models/chem_aware/chem_rules_data.json
)
for required_file in "${required_files[@]}"; do
  [[ -f "$required_file" ]] || {
    echo "Missing required input: $required_file" >&2
    exit 2
  }
done

verify_sha() {
  local path="$1"
  local expected="$2"
  local observed
  observed="$(sha256sum "$path" | awk '{print $1}')"
  [[ "$observed" == "$expected" ]] || {
    echo "Stale ChemAware runtime or artifact: $path" >&2
    echo "Expected SHA256: $expected" >&2
    echo "Observed SHA256: $observed" >&2
    exit 2
  }
}
verify_sha tasks/chemaware_truthblind_candidate_core.py bacc3e79d4f5c64fe8c7e63156e667b4ee59119ff158ec9b227e9b676cfa2593
verify_sha tasks/chemaware_outer_evaluation_core.py a25581cd5eebe2a8481b6af498f02a835801cfaab43f15bca916c9adaae9d764
verify_sha tasks/audit_chemaware_candidate_evidence_policy.py bebbaf1d18b0e5b51bb5e5497e3f42dcfd56492debf8b15ac31671dbfd4ee435
verify_sha tasks/audit_chemaware_mass_kernel_embedding.py e466bdafc12e9b9f2e9379f401dd4ec86c7c63bfb1295d0f88d57dd6ae16ab9a
verify_sha tasks/audit_chemaware_orthogonal_rule_residual_policy.py 1e24f499e90bba763ca518471f8590d0bfd958a55e3fc5c7c418499c49e88c4a
verify_sha tasks/evaluate_chemaware_truthblind_outer_once.py 5f1e82050d22e8855b4571238c2ffd6f4a67f8dda9b13c8d8c49a319de34c85c
verify_sha "$CANONICAL/truthblind_policy.joblib" 8402f32d07287a47008556bf25ad241d25c9e56e6d30fd1d68a407de1a2f2dee
verify_sha "$CANONICAL/report.json" 64a997a4fc05f442c04e2746a5ac6e1022272bb21417b237cde279ea66aa47bf
verify_sha "$CANONICAL/inner_policy.npz" 70e25e6d2459222be3597748a851f206c5121792cf920e8e745ed2caa6dcace3

python -c "import joblib, numpy, sklearn, torch"
python -m py_compile \
  tasks/chemaware_truthblind_candidate_core.py \
  tasks/chemaware_outer_evaluation_core.py \
  tasks/audit_chemaware_candidate_evidence_policy.py \
  tasks/audit_chemaware_mass_kernel_embedding.py \
  tasks/audit_chemaware_orthogonal_rule_residual_policy.py \
  tasks/evaluate_chemaware_truthblind_outer_once.py \
  tasks/test_chemaware_truthblind_candidate_core.py \
  tasks/test_chemaware_truthblind_candidate_policy_contracts.py \
  tasks/test_chemaware_outer_evaluation_core.py \
  tasks/test_chemaware_truthblind_outer_once_contracts.py
python tasks/test_chemaware_truthblind_candidate_core.py
python tasks/test_chemaware_truthblind_candidate_policy_contracts.py
python tasks/test_chemaware_outer_evaluation_core.py
python tasks/test_chemaware_truthblind_outer_once_contracts.py

# Reuse the fully written release policy from the failed 2338356 evaluation.
# If it is absent, refit only on the already-consumed roles 0-2.  In either
# case the evaluator refuses to open role 4 unless all three policies exactly
# reproduce run 2338337 on role 3.
if [[ -s "$REUSABLE_POLICY/report.json" \
   && -s "$REUSABLE_POLICY/inner_policy.npz" \
   && -s "$REUSABLE_POLICY/truthblind_policy.joblib" \
   && -s "$REUSABLE_POLICY/truthblind_policy.json" ]]; then
  POLICY_OUT="$REUSABLE_POLICY"
  echo "Reusing completed release policy from run 2338356"
else
  python -u tasks/audit_chemaware_orthogonal_rule_residual_policy.py \
    --output "$POLICY_OUT"
fi

test -s "$POLICY_OUT/report.json"
test -s "$POLICY_OUT/inner_policy.npz"
test -s "$POLICY_OUT/truthblind_policy.joblib"
test -s "$POLICY_OUT/truthblind_policy.json"
python -c 'import joblib,sys; b=joblib.load(sys.argv[1]); assert set(b["baselines"])=={"same_feature_direct","nuisance_only"}' "$POLICY_OUT/truthblind_policy.joblib"

python -u tasks/evaluate_chemaware_truthblind_outer_once.py \
  --policy-dir "$POLICY_OUT" \
  --canonical-policy-dir "$CANONICAL" \
  --seal "$SEAL" \
  --output "$OUTER_OUT"

test -s "$OUTER_OUT/report.json"
test -s "$OUTER_OUT/predictions_truthblind.npz"
test -s "$OUTER_OUT/evaluation_with_truth.npz"
grep -q 'CHEMAWARE_TRUTHBLIND_OUTER_ONCE_COMPLETE' "$OUTER_OUT/report.json"
echo "COMPLETE: ChemAware exact release replay and one-time sealed outer evaluation"
