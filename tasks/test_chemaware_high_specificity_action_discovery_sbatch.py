from pathlib import Path


SBATCH = Path(__file__).with_name(
    "run_chemaware_high_specificity_action_discovery.sbatch"
)
source = SBATCH.read_text(encoding="utf-8")

assert source.count("#SBATCH --partition=gpu") == 1
assert source.count("#SBATCH --gpus=1") == 1
assert "#SBATCH --mem" not in source
assert "#SBATCH --mem-per-cpu" not in source
assert 'run_${SLURM_JOB_ID}' in source
assert "run_233" not in source
assert "sbatch " not in source

entries = (
    "tasks/audit_chemaware_jacobian_intersection_actions.py",
    "tasks/audit_chemaware_boundary_consensus_action_atlas.py",
    "tasks/mine_chemaware_domain_conditioned_action_rules.py",
)
for entry in entries:
    assert source.count(entry) == 3, entry  # compile, preflight, full execution

assert source.count("--preflight-only") == 4
assert 'B1J_OUT="$RUN_ROOT/b1j_jacobian_intersection"' in source
assert 'B1_OUT="$RUN_ROOT/b1_boundary_consensus"' in source
assert 'B2_OUT="$RUN_ROOT/b2_domain_rules"' in source
assert 'B3_OUT="$RUN_ROOT/b3_stratified_meta_rules"' in source
assert 'BPMT_OUT="$RUN_ROOT/bpmt_transfer_manifest"' in source
assert 'PREDICATES="$ASSET_ROOT/chem_parent_predicates_rdkit_v1.json"' in source
assert 'OBSERVATIONS="$ASSET_ROOT/chem_observation_channels_v2.json"' in source
assert source.count("tasks/build_chemaware_parent_predicate_registry.py") == 2
assert source.count("tasks/build_chemaware_observation_channel_registry.py") == 2
assert source.count('--predicates "$PREDICATES" --observations "$OBSERVATIONS"') == 4
assert source.count('tasks/mine_chemaware_stratified_meta_action_rules.py') == 3
assert source.count('--domain-rule-dir "$B2_OUT"') == 2
assert source.count("tasks/build_chemaware_boundary_pmt_manifest.py") == 2
assert '--action-dir "$B1_OUT"' in source
assert source.index("tasks/build_chemaware_parent_predicate_registry.py") < source.index(
    "--preflight-only"
)
assert source.index("tasks/build_chemaware_observation_channel_registry.py") < source.index(
    "--preflight-only"
)
executions = [
    line.removeprefix("python -u ").removesuffix(" \\")
    for line in source.splitlines()
    if line.startswith("python -u tasks/")
]
assert executions == [
    "tasks/build_chemaware_parent_predicate_registry.py",
    "tasks/build_chemaware_observation_channel_registry.py",
    "tasks/mine_chemaware_domain_conditioned_action_rules.py",
    "tasks/audit_chemaware_jacobian_intersection_actions.py",
    "tasks/audit_chemaware_boundary_consensus_action_atlas.py",
    "tasks/audit_chemaware_jacobian_intersection_actions.py",
    "tasks/audit_chemaware_boundary_consensus_action_atlas.py",
    "tasks/build_chemaware_boundary_pmt_manifest.py",
    "tasks/mine_chemaware_domain_conditioned_action_rules.py",
    "tasks/mine_chemaware_stratified_meta_action_rules.py",
    "tasks/mine_chemaware_stratified_meta_action_rules.py",
]

print("PASS: integrated ChemAware action-discovery sbatch contracts")
