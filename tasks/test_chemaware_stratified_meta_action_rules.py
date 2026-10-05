"""CPU contracts for stratified multi-domain ChemAware actions."""

from __future__ import annotations

import numpy as np

from mine_chemaware_stratified_meta_action_rules import (
    domain_replication,
    stratified_formula_contrasts,
)


def main() -> None:
    observation = np.asarray(
        [[1.0], [0.0], [0.8], [0.0], [0.6], [0.0], [0.4], [0.0]]
    )
    predicate = np.asarray([True, False] * 4)
    formula = np.asarray(["A", "A", "B", "B", "A", "A", "B", "B"])
    identity = np.arange(8)
    domain = np.asarray(["d1"] * 4 + ["d2"] * 4)
    fold = np.zeros(8, dtype=np.int16)
    valid = np.ones(8, dtype=bool)
    result = stratified_formula_contrasts(
        observation,
        predicate,
        formula,
        identity,
        domain,
        fold,
        valid,
        (0,),
        ("d1", "d2"),
    )
    assert result["formulas"].tolist() == ["A", "B"]
    assert np.allclose(result["matrix"][:, 0], [0.8, 0.6])
    assert result["positive_identities"] == 4
    replication = domain_replication(result, 0, 2, 0.0)
    assert replication["replicating_domains"] == 2
    assert replication["positive_domains"] == 2
    assert replication["nonnegative_domain_fraction"] == 1.0
    valid[-2:] = False
    filtered = stratified_formula_contrasts(
        observation,
        predicate,
        formula,
        identity,
        domain,
        fold,
        valid,
        (0,),
        ("d1", "d2"),
    )
    assert np.allclose(filtered["matrix"][:, 0], [0.8, 0.8])
    assert filtered["positive_identities"] == 3
    print("PASS: ChemAware stratified meta-action contracts")


if __name__ == "__main__":
    main()
