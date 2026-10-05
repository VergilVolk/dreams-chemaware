from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    audit = (ROOT / "tasks/audit_chemaware_direct_first_backoff.py").read_text(
        encoding="utf-8"
    )
    serializer = (ROOT / "tasks/audit_chemaware_orthogonal_rule_residual_policy.py").read_text(
        encoding="utf-8"
    )
    core = (ROOT / "tasks/chemaware_truthblind_candidate_core.py").read_text(
        encoding="utf-8"
    )
    assert "--manifest" not in audit
    assert '"outer_role_4_accessed": False' in audit
    assert '"model_fit": False' in audit
    assert '"threshold_tuned": False' in audit
    assert "same_feature_direct_candidate_utility=direct_validation_utilities" in serializer
    assert "nuisance_only_candidate_utility=nuisance_validation_utility" in serializer
    assert "def predict_truthblind_direct_first_backoff(" in core
    assert "arbitrate_truthblind_predictions(baselines[\"same_feature_direct\"], residual)" in core
    print("PASS: ChemAware direct-first chemical-backoff contracts")


if __name__ == "__main__":
    main()
