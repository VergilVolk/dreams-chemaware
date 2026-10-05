"""CPU contracts for the ChemAware MotifDB evidence source."""
from __future__ import annotations

import inspect
import math

import numpy as np

import acquire_chemaware_public_fragmentation_sources as acquire
import build_chemaware_motifdb_candidate_source_ledger as score
import build_chemaware_motifdb_evidence_corpus as corpus


def main() -> None:
    assert len(acquire.MS2LDA_COMMIT) == 40
    assert len(acquire.MOTIFDB_SHA256) == 64
    assert acquire.MS2LDA_COMMIT in acquire.MOTIFDB_URL

    features = corpus.finite_features(
        [100.0, math.nan, 80.0, 120.0], [0.2, 1.0, 0.001, 0.8], 0.005, 2,
    )
    assert features == [{"mz": 120.0, "weight": 0.8}, {"mz": 100.0, "weight": 0.2}]

    observed_mz = np.asarray([50.0, 100.005, 182.0])
    observed_intensity = np.asarray([0.2, 1.0, 0.4])
    strength, matches = score.feature_match(
        observed_mz, observed_intensity, 200.0,
        [{"mz": 100.0, "weight": 1.0}], False, 0.005, 20.0, 0.01,
    )
    assert matches == 1 and strength > 0.99
    loss_strength, loss_matches = score.feature_match(
        observed_mz, observed_intensity, 200.0,
        [{"mz": 18.0, "weight": 1.0}], True, 0.005, 20.0, 0.01,
    )
    assert loss_matches == 1 and 0.39 < loss_strength < 0.41

    records = [
        {"charge": 1, "consensus_morgan_bits": [1, 2, 3], "source_structure_ik14": ["SOURCE"]},
        {"charge": 1, "consensus_morgan_bits": [4, 5, 6], "source_structure_ik14": []},
        {"charge": -1, "consensus_morgan_bits": [7, 8, 9], "source_structure_ik14": []},
    ]
    controls = score.cyclic_predicates(records)
    assert len(controls) == len(records)
    assert controls[0] != frozenset(records[0]["consensus_morgan_bits"])

    values, suppressed, active = score.score_candidates(
        [frozenset({1, 2, 3}), frozenset({4, 5, 6})],
        ["SOURCE", "OTHER"], records[:2], np.asarray([1.0, 0.8]),
        [frozenset({1, 2, 3}), frozenset({4, 5, 6})], 0.5, True,
    )
    assert suppressed == 1
    assert active == 1
    assert values[1] > values[0]

    rotated = score.cyclic_candidate_scores(
        np.asarray([0.1, 0.8, 0.3]), ["A", "A", "B"],
    )
    assert not np.array_equal(rotated, np.asarray([0.1, 0.8, 0.3]))

    source = inspect.getsource(score)
    assert 'default="cross_formula"' in source
    assert "molecule_label" not in source
    assert "native DreaMS triplets" in source
    print("PASS: ChemAware MotifDB acquisition, corpus and truth-blind source contracts")


if __name__ == "__main__":
    main()
