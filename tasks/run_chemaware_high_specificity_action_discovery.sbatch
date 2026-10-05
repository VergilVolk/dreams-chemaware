#!/bin/bash
#SBATCH --job-name=chem_actions
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=data/validation/chem_actions_%j.out
#SBATCH --error=data/validation/chem_actions_%j.err

set -euo pipefail
cd /data02/run01/scv7tsl/DreaMS
source /data02/home/scv7tsl/run/miniconda3/etc/profile.d/conda.sh
conda activate dreams
export PYTHONPATH="$PWD:$PWD/tasks:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_ROOT="data/validation/chemaware_high_specificity_action_discovery/run_${SLURM_JOB_ID}"
B1J_OUT="$RUN_ROOT/b1j_jacobian_intersection"
B1_OUT="$RUN_ROOT/b1_boundary_consensus"
B2_OUT="$RUN_ROOT/b2_domain_rules"
B3_OUT="$RUN_ROOT/b3_stratified_meta_rules"
BPMT_OUT="$RUN_ROOT/bpmt_transfer_manifest"
ASSET_ROOT="$RUN_ROOT/generated_assets"
PREDICATES="$ASSET_ROOT/chem_parent_predicates_rdkit_v1.json"
OBSERVATIONS="$ASSET_ROOT/chem_observation_channels_v2.json"
[[ ! -e "$RUN_ROOT" ]] || { echo "Output collision: $RUN_ROOT" >&2; exit 3; }
mkdir -p "$RUN_ROOT"
trap 'echo "FAILED: integrated ChemAware action discovery stopped before completion" >&2' ERR

# One fail-fast contract gate for the entire action-discovery suite.
python -c "import h5py, numpy, rdkit, torch"
python -m py_compile \
  tasks/build_chemaware_parent_predicate_registry.py \
  tasks/build_chemaware_observation_channel_registry.py \
  tasks/chemaware_boundary_consensus_action_core.py \
  tasks/audit_chemaware_boundary_consensus_action_atlas.py \
  tasks/chemaware_jacobian_intersection_action_core.py \
  tasks/audit_chemaware_jacobian_intersection_actions.py \
  tasks/mine_chemaware_domain_conditioned_action_rules.py \
  tasks/mine_chemaware_stratified_meta_action_rules.py \
  tasks/chemaware_boundary_pmt_core.py \
  tasks/build_chemaware_boundary_pmt_manifest.py \
  tasks/test_chemaware_boundary_consensus_action_core.py \
  tasks/test_chemaware_boundary_consensus_action_atlas_contract.py \
  tasks/test_chemaware_jacobian_intersection_action_core.py \
  tasks/test_chemaware_jacobian_intersection_action_contract.py \
  tasks/test_chemaware_jacobian_intersection_runtime.py \
  tasks/test_chemaware_parent_predicate_registry.py \
  tasks/test_chemaware_observation_channel_registry.py \
  tasks/test_chemaware_domain_conditioned_action_rules_contract.py \
  tasks/test_chemaware_stratified_meta_action_rules.py \
  tasks/test_chemaware_boundary_pmt_core.py \
  tasks/test_build_chemaware_boundary_pmt_manifest.py \
  tasks/test_chemaware_high_specificity_action_discovery_sbatch.py
python tasks/test_chemaware_parent_predicate_registry.py
python tasks/test_chemaware_observation_channel_registry.py
python tasks/test_chemaware_boundary_consensus_action_core.py
python tasks/test_chemaware_boundary_consensus_action_atlas_contract.py
python tasks/test_chemaware_jacobian_intersection_action_core.py
python tasks/test_chemaware_jacobian_intersection_action_contract.py
python tasks/test_chemaware_jacobian_intersection_runtime.py
python tasks/test_chemaware_domain_conditioned_action_rules_contract.py
python tasks/test_chemaware_stratified_meta_action_rules.py
python tasks/test_chemaware_boundary_pmt_core.py
python tasks/test_build_chemaware_boundary_pmt_manifest.py
python tasks/test_chemaware_high_specificity_action_discovery_sbatch.py

# Rebuild both B2 registries from executable, versioned sources.  The job does
# not depend on manually copying generated JSON files from another machine.
python -u tasks/build_chemaware_parent_predicate_registry.py \
  --output "$PREDICATES" \
  --report "$ASSET_ROOT/parent_predicate_build/report.json"
python -u tasks/build_chemaware_observation_channel_registry.py \
  --legacy-library "$PWD/dreams/models/chem_aware/chem_rules_data.json" \
  --output "$OBSERVATIONS" \
  --report "$ASSET_ROOT/observation_channel_build/report.json"
test -s "$PREDICATES"
test -s "$OBSERVATIONS"

# Validate every input and frozen protocol before the first expensive pass.
python -u tasks/mine_chemaware_domain_conditioned_action_rules.py \
  --predicates "$PREDICATES" --observations "$OBSERVATIONS" \
  --output "$B2_OUT" \
  --draws 10000 \
  --discovery-folds 0 1 --confirmation-fold 2 \
  --preflight-only
python -u tasks/audit_chemaware_jacobian_intersection_actions.py \
  --output "$B1J_OUT" \
  --device cuda --batch-size 16 --torch-threads 8 \
  --bootstrap-draws 10000 \
  --discovery-folds 0 1 --confirmation-fold 2 \
  --embedding-evaluation-fold 3 --reserve-fold 4 \
  --preflight-only
python -u tasks/audit_chemaware_boundary_consensus_action_atlas.py \
  --output "$B1_OUT" \
  --device cuda --batch-size 64 --torch-threads 8 \
  --bootstrap-draws 10000 \
  --discovery-folds 0 1 --confirmation-fold 2 \
  --embedding-evaluation-fold 3 --reserve-fold 4 \
  --preflight-only

# The three stages discover and audit actions only.  They do not train or
# update the shared embedding, and one stage cannot silently select another.
python -u tasks/audit_chemaware_jacobian_intersection_actions.py \
  --output "$B1J_OUT" \
  --device cuda --batch-size 16 --torch-threads 8 \
  --bootstrap-draws 10000 \
  --discovery-folds 0 1 --confirmation-fold 2 \
  --embedding-evaluation-fold 3 --reserve-fold 4
test -f "$B1J_OUT/report.json"
test -f "$B1J_OUT/screen.npz"

python -u tasks/audit_chemaware_boundary_consensus_action_atlas.py \
  --output "$B1_OUT" \
  --device cuda --batch-size 64 --torch-threads 8 \
  --bootstrap-draws 10000 \
  --discovery-folds 0 1 --confirmation-fold 2 \
  --embedding-evaluation-fold 3 --reserve-fold 4
test -f "$B1_OUT/report.json"
test -f "$B1_OUT/screen.npz"

# Compile the passed same-boundary action into a non-negative clean-margin
# transfer ledger.  A failed B1 or insufficient strict-positive boundary
# coverage writes a fail-closed report and cannot launch training.
python -u tasks/build_chemaware_boundary_pmt_manifest.py \
  --action-dir "$B1_OUT" \
  --output "$BPMT_OUT" \
  --activation-margin 0.05 --advantage-cap 0.05
test -f "$BPMT_OUT/report.json"

python -u tasks/mine_chemaware_domain_conditioned_action_rules.py \
  --predicates "$PREDICATES" --observations "$OBSERVATIONS" \
  --output "$B2_OUT" \
  --draws 10000 \
  --discovery-folds 0 1 --confirmation-fold 2
test -f "$B2_OUT/report.json"
test -f "$B2_OUT/conditioned_rules.json"
test -f "$B2_OUT/identity_domain_evidence.npz"

# B3 does not pool spectra across acquisition domains.  It meta-analyses
# formula-balanced within-domain contrasts and requires multi-domain
# replication before an action is admitted.
python -u tasks/mine_chemaware_stratified_meta_action_rules.py \
  --domain-rule-dir "$B2_OUT" \
  --predicates "$PREDICATES" --observations "$OBSERVATIONS" \
  --output "$B3_OUT" \
  --draws 10000 \
  --discovery-folds 0 1 --confirmation-fold 2 \
  --preflight-only
python -u tasks/mine_chemaware_stratified_meta_action_rules.py \
  --domain-rule-dir "$B2_OUT" \
  --predicates "$PREDICATES" --observations "$OBSERVATIONS" \
  --output "$B3_OUT" \
  --draws 10000 \
  --discovery-folds 0 1 --confirmation-fold 2
test -f "$B3_OUT/report.json"
test -f "$B3_OUT/stratified_meta_rules.json"

echo "COMPLETE: integrated ChemAware high-specificity action discovery"
