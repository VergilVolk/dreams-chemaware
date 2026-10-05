#!/bin/bash
#SBATCH --job-name=chem_b2_rules
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=data/validation/chem_b2_rules_%j.out
#SBATCH --error=data/validation/chem_b2_rules_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

RUN_ROOT="data/validation/chemaware_domain_conditioned_action_rules/run_${SLURM_JOB_ID}"
RULE_OUT="$RUN_ROOT/rules"
ASSET_ROOT="$RUN_ROOT/generated_assets"
PREDICATES="$ASSET_ROOT/chem_parent_predicates_rdkit_v1.json"
OBSERVATIONS="$ASSET_ROOT/chem_observation_channels_v2.json"
[[ ! -e "$RUN_ROOT" ]] || { echo "Output collision: $RUN_ROOT" >&2; exit 3; }
mkdir -p "$RUN_ROOT"
trap 'echo "FAILED: ChemAware B2 domain-conditioned rule mining stopped" >&2' ERR

python -c "import h5py, numpy, rdkit"
python -m py_compile \
  tasks/build_chemaware_parent_predicate_registry.py \
  tasks/build_chemaware_observation_channel_registry.py \
  tasks/mine_chemaware_domain_conditioned_action_rules.py \
  tasks/test_chemaware_parent_predicate_registry.py \
  tasks/test_chemaware_observation_channel_registry.py \
  tasks/test_chemaware_domain_conditioned_action_rules_contract.py
python tasks/test_chemaware_parent_predicate_registry.py
python tasks/test_chemaware_observation_channel_registry.py
python tasks/test_chemaware_domain_conditioned_action_rules_contract.py
python -u tasks/build_chemaware_parent_predicate_registry.py \
  --output "$PREDICATES" \
  --report "$ASSET_ROOT/parent_predicate_build/report.json"
python -u tasks/build_chemaware_observation_channel_registry.py \
  --legacy-library "$PWD/dreams/models/chem_aware/chem_rules_data.json" \
  --output "$OBSERVATIONS" \
  --report "$ASSET_ROOT/observation_channel_build/report.json"
test -s "$PREDICATES"
test -s "$OBSERVATIONS"
python -u tasks/mine_chemaware_domain_conditioned_action_rules.py \
  --predicates "$PREDICATES" --observations "$OBSERVATIONS" \
  --output "$RULE_OUT" \
  --draws 10000 \
  --discovery-folds 0 1 --confirmation-fold 2 \
  --preflight-only

# This is rule/action discovery only.  It neither loads nor updates DreaMS.
python -u tasks/mine_chemaware_domain_conditioned_action_rules.py \
  --predicates "$PREDICATES" --observations "$OBSERVATIONS" \
  --output "$RULE_OUT" \
  --draws 10000 \
  --discovery-folds 0 1 --confirmation-fold 2

test -f "$RULE_OUT/report.json"
test -f "$RULE_OUT/conditioned_rules.json"
test -f "$RULE_OUT/identity_domain_evidence.npz"
echo "COMPLETE: ChemAware B2 domain-conditioned action rules"
