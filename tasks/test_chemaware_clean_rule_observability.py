"""Synthetic contract tests for the clean-rule observability gate."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from audit_chemaware_clean_rule_observability import audit_arrays


def ledger(features: np.ndarray, formula_prefix: str, signal: bool, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed); count = len(features)
    formula = np.asarray([f"{formula_prefix}_{index // 2}" for index in range(count)], dtype=str)
    positive = features[:, 0] > 1.0 if signal else rng.random(count) < 0.16
    harmful = features[:, 0] < -1.0 if signal else rng.random(count) < 0.10
    harmful &= ~positive
    mass_hit = rng.random(count) < 0.65
    mass_hit[positive] = False; mass_hit[harmful] = True
    rule_hit = mass_hit.copy(); rule_hit[positive] = True; rule_hit[harmful] = False
    return {"formula": formula, "positive": positive, "harmful": harmful,
            "mass_hit": mass_hit, "rule_hit": rule_hit}


def run(signal: bool) -> dict:
    rng = np.random.default_rng(4)
    discovery_features = rng.normal(size=(800, 8))
    confirmation_features = rng.normal(size=(400, 8))
    args = SimpleNamespace(
        folds=5, seed=9, min_selected_formulas=10, bootstrap_draws=1000,
        permutation_controls=20, min_discovery_positive=30, min_discovery_harmful=20,
        min_confirmation_positive=15, min_confirmation_harmful=8,
    )
    report, _ = audit_arrays(
        discovery_features, ledger(discovery_features, "d", signal, 5),
        confirmation_features, ledger(confirmation_features, "c", signal, 6), args,
    )
    return report


def main() -> None:
    positive = run(True)
    assert positive["status"] == "CLEAN_RULE_OBSERVABILITY_PASS", positive
    assert positive["pass_to_gpu_training"] is True
    null = run(False)
    assert null["status"] == "CLEAN_RULE_OBSERVABILITY_FAIL", null
    assert null["pass_to_gpu_training"] is False
    print("clean-rule observability positive and no-signal contracts passed")


if __name__ == "__main__":
    main()
