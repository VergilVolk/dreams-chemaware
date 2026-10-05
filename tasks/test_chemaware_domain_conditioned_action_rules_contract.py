"""Static contracts for the condition-aware rule-mining route."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MINER = ROOT / "tasks/mine_chemaware_domain_conditioned_action_rules.py"
SBATCH = ROOT / "tasks/run_chemaware_domain_conditioned_action_rules.sbatch"
LEGACY_LIBRARY = ROOT / "dreams/models/chem_aware/chem_rules_data.json"


def main() -> None:
    source = MINER.read_text(encoding="utf-8")
    ast.parse(source)
    required = (
        "[M+H]+",
        "[M+Na]+",
        "Orbitrap",
        "QTOF",
        "very_low_le_10",
        "low_10_20",
        "medium_low_20_30",
        "medium_high_30_40",
        "high_40_60",
        "very_high_gt_60",
        "missing",
        "excluded_unknown_or_missing_domains",
        "unknown_instrument_rule_mining",
        "missing_collision_energy_rule_mining",
        "within_formula_structure_controls",
        "formula_disjoint_confirmation",
        "negative_association_compiled_as_conflict",
        "negative_or_absence_action_authorized",
        "support_boost_observed_match",
        "formal_training_authorized",
        "fold_protocol",
    )
    missing = [value for value in required if value not in source]
    assert not missing, missing
    assert '"negative_association_compiled_as_conflict": False' in source
    assert '"negative_or_absence_action_authorized": False' in source
    assert '"unknown_instrument_rule_mining": False' in source
    assert '"missing_collision_energy_rule_mining": False' in source
    assert "default=10_000" in source
    assert "default=12" in source
    assert "default=24" in source
    assert LEGACY_LIBRARY.is_file(), LEGACY_LIBRARY

    sbatch = SBATCH.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpu" in sbatch
    assert "#SBATCH --gpus=1" in sbatch
    assert "#SBATCH --gpus=" not in sbatch.replace("#SBATCH --gpus=1", "")
    assert "--mem" not in sbatch and "--mem-per-cpu" not in sbatch
    assert "run_${SLURM_JOB_ID}" in sbatch
    assert "--preflight-only" in sbatch
    print("PASS: ChemAware B2 domain-rule and one-GPU sbatch contracts")


if __name__ == "__main__":
    main()
