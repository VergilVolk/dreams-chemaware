"""Static fail-closed contracts for the B-PMT manifest compiler."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "tasks/build_chemaware_boundary_pmt_manifest.py").read_text()
ast.parse(source)
required = (
    "CHEMAWARE_BOUNDARY_CONSENSUS_ACTION_CONFIRMATION_PASS",
    "pass_to_action_transfer_audit",
    "strict_action_advantage",
    "active_transfer_weights",
    "margin_bin_formula_derangement",
    "matched_formula_deranged_source_index",
    "matched_formula_deranged_source_formula",
    "control_gradient_subtraction\": False",
    "nonactive_corrective_weight\": 0.0",
    "embedding_evaluation_queries_untouched",
    "reserve_queries_untouched",
)
missing = [item for item in required if item not in source]
assert not missing, missing
assert "correct - controls" not in source
print("PASS: ChemAware B-PMT manifest contracts")
