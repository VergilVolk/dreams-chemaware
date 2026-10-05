#!/bin/bash
#SBATCH --job-name=chem_truthblind
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/chem_truthblind_%j.out
#SBATCH --error=data/validation/chem_truthblind_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

RUN_ROOT="data/validation/chemaware_truthblind_candidate_policy/run_${SLURM_JOB_ID}"
POLICY_OUT="$RUN_ROOT/policy"
REPLAY_OUT="$RUN_ROOT/truthblind_replay"
[[ ! -e "$RUN_ROOT" ]] || { echo "Output collision: $RUN_ROOT" >&2; exit 3; }
mkdir -p "$RUN_ROOT"
trap 'echo "FAILED: ChemAware truth-blind candidate policy stopped before completion" >&2' ERR

required_files=(
  tasks/chemaware_truthblind_candidate_core.py
  tasks/audit_chemaware_candidate_evidence_policy.py
  tasks/audit_chemaware_mass_kernel_embedding.py
  tasks/audit_chemaware_orthogonal_rule_residual_policy.py
  tasks/replay_chemaware_orthogonal_policy_truthblind.py
  tasks/test_chemaware_truthblind_candidate_core.py
  tasks/test_chemaware_truthblind_candidate_policy_contracts.py
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
    echo "Stale ChemAware runtime: $path" >&2
    echo "Expected SHA256: $expected" >&2
    echo "Observed SHA256: $observed" >&2
    exit 2
  }
}
verify_sha tasks/chemaware_truthblind_candidate_core.py bacc3e79d4f5c64fe8c7e63156e667b4ee59119ff158ec9b227e9b676cfa2593
verify_sha tasks/audit_chemaware_candidate_evidence_policy.py bebbaf1d18b0e5b51bb5e5497e3f42dcfd56492debf8b15ac31671dbfd4ee435
verify_sha tasks/audit_chemaware_mass_kernel_embedding.py e466bdafc12e9b9f2e9379f401dd4ec86c7c63bfb1295d0f88d57dd6ae16ab9a
verify_sha tasks/audit_chemaware_orthogonal_rule_residual_policy.py 1e24f499e90bba763ca518471f8590d0bfd958a55e3fc5c7c418499c49e88c4a
verify_sha tasks/replay_chemaware_orthogonal_policy_truthblind.py bf07a6bdb6a488734d2d320120eada88d1cd765b83e84dbfbcbed6ec4c7d0873

python -c "import joblib, numpy, sklearn, torch"
python -m py_compile \
  tasks/chemaware_truthblind_candidate_core.py \
  tasks/audit_chemaware_candidate_evidence_policy.py \
  tasks/audit_chemaware_mass_kernel_embedding.py \
  tasks/audit_chemaware_orthogonal_rule_residual_policy.py \
  tasks/replay_chemaware_orthogonal_policy_truthblind.py \
  tasks/test_chemaware_truthblind_candidate_core.py \
  tasks/test_chemaware_truthblind_candidate_policy_contracts.py
python tasks/test_chemaware_truthblind_candidate_core.py
python tasks/test_chemaware_truthblind_candidate_policy_contracts.py

python -u tasks/audit_chemaware_orthogonal_rule_residual_policy.py \
  --output "$POLICY_OUT"

test -s "$POLICY_OUT/report.json"
test -s "$POLICY_OUT/inner_policy.npz"
test -s "$POLICY_OUT/truthblind_policy.joblib"
test -s "$POLICY_OUT/truthblind_policy.json"

python -u tasks/replay_chemaware_orthogonal_policy_truthblind.py \
  --policy-dir "$POLICY_OUT" \
  --output "$REPLAY_OUT"

test -s "$REPLAY_OUT/report.json"
test -s "$REPLAY_OUT/predictions.npz"
grep -q 'CHEMAWARE_TRUTHBLIND_POLICY_REPLAY_PASS' "$REPLAY_OUT/report.json"
echo "COMPLETE: ChemAware truth-blind candidate policy and mechanism ablations"
