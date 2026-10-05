"""CPU contracts for query- and candidate-exact ChemAware triplets."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from build_chemaware_exact_boundary_residual_triplets import (  # noqa: E402
    ALTERNATE_METRICS,
    BASELINE_METRICS,
    diverse_safety,
    exact_boundary_proof,
    metric_columns,
)


def evidence_fixture() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    names = tuple(dict.fromkeys(BASELINE_METRICS + ALTERNATE_METRICS))
    metric = np.zeros((4, 1, 2, len(names)), dtype=np.float32)
    columns = {name: names.index(name) for name in names}
    # Truth slot beats all nulls against the baseline candidate.
    for name in BASELINE_METRICS:
        metric[0, 0, 0, columns[name]] = 1.0
    # For an alternate false candidate, correct favors truth while every null
    # favors the false candidate.  The proof must stay candidate-specific.
    for name in ALTERNATE_METRICS:
        metric[0, 0, 0, columns[name]] = 1.0
        metric[0, 0, 1, columns[name]] = 0.0
        metric[1:, 0, 0, columns[name]] = 0.0
        metric[1:, 0, 1, columns[name]] = 1.0
    evidence = {
        "metric_names": np.asarray(names),
        "arm_names": np.asarray(["correct", "null_a", "null_b", "null_c"]),
        "arm_metric": metric,
        "valid": np.asarray([[True, True]]),
        "proposed_candidate": np.asarray([[0, 2]], dtype=np.int16),
        "baseline_candidate": np.asarray([1], dtype=np.int16),
        "query": np.asarray([0], dtype=np.int64),
    }
    manifest = {
        "query_ptr": np.asarray([0, 3], dtype=np.int64),
        "molecule_label": np.asarray([1, 0, 0], dtype=np.int8),
        "query_formula": np.asarray(["F0"]),
    }
    return evidence, manifest


def main() -> None:
    evidence, manifest = evidence_fixture()
    columns = metric_columns(evidence)
    baseline = exact_boundary_proof(evidence, manifest, 0, 1, columns, 2, 3)
    assert baseline is not None and baseline["kind"] == "baseline"
    assert baseline["false_candidate"] == 1 and baseline["strict"] is True

    alternate = exact_boundary_proof(evidence, manifest, 0, 2, columns, 2, 3)
    assert alternate is not None and alternate["kind"] == "alternate"
    assert alternate["false_candidate"] == 2 and alternate["strict"] is True
    assert exact_boundary_proof(evidence, manifest, 0, 3, columns, 2, 3) is None

    safety = [
        {"query": 0, "margin": 0.02},
        {"query": 1, "margin": 0.01},
        {"query": 2, "margin": 0.03},
    ]
    safety_manifest = {"query_formula": np.asarray(["A", "A", "B"])}
    selected = diverse_safety(safety, safety_manifest, 2)
    assert [int(row["query"]) for row in selected] == [1, 2]
    print("PASS: ChemAware exact-boundary residual-triplet contracts")


if __name__ == "__main__":
    main()
