"""CPU contracts for recurrent MassBank fragment and neutral-loss evidence."""
from __future__ import annotations

import inspect

import numpy as np

import build_chemaware_massbank_annotated_fragment_corpus as corpus
import build_chemaware_massbank_candidate_source_ledger as score


def main() -> None:
    record = """ACCESSION: TEST0001
RECORD_TITLE: Mellein; LC-ESI-MS2; [M+H]+
CH$FORMULA: C10H10O3
CH$SMILES: CC1CC2=C(C(=CC=C2)O)C(=O)O1
CH$LINK: INCHIKEY TESTABCDEFGHIJ-AA-B
AC$MASS_SPECTROMETRY: ION_MODE POSITIVE
AC$MASS_SPECTROMETRY: IONIZATION ESI
MS$FOCUSED_ION: PRECURSOR_M/Z 179.0697
MS$FOCUSED_ION: PRECURSOR_TYPE [M+H]+
PK$ANNOTATION: m/z tentative_formula mass_error(ppm)
  133.0643 C9H9O1+ -3.74
  151.0751 C9H11O2+ -1.72
PK$NUM_PEAK: 2
//
"""
    parsed = corpus.parse_record(
        record, "MassBank-MassBank-data-85e5599/TEST/TEST0001.txt",
    )
    assert parsed is not None
    assert parsed.contributor == "TEST"
    assert parsed.parent_formula == "C10H10O3"
    assert parsed.annotations == ((133.0643, "C9H9O1"), (151.0751, "C9H11O2"))
    assert corpus.parse_formula("C9H9O1+") == {"C": 9, "H": 9, "O": 1}
    assert corpus.parse_formula("[13C]H4") is None
    no_smiles = record.replace("CC1CC2=C(C(=CC=C2)O)C(=O)O1", "N/A")
    assert corpus.parse_record(no_smiles, "x/y.txt") is None

    observations = score.matched_observations(
        np.asarray([133.0644, 151.2]),
        np.asarray([1.0, 0.5]),
        np.asarray([133.0643, 151.0751]),
        ppm=20.0,
        floor_da=0.01,
    )
    assert np.allclose(observations, [1.0, 0.0])

    values, suppressed, active = score.score_rules(
        observations=np.asarray([1.0]),
        candidates_formula=[{"C": 10, "H": 11, "O": 3}, {"C": 8, "H": 9, "O": 2}],
        candidates_bits=[frozenset({1, 2, 3}), frozenset({4, 5, 6})],
        candidates_ik14=["SOURCEIDENTITY", "OTHERIDENTITY"],
        rule_formula=[{"C": 9, "H": 9, "O": 1}],
        rule_bits=[frozenset({1, 2, 3})],
        rule_source_identities=[{"SOURCEIDENTITY"}],
        rule_weight=np.asarray([1.0]),
        minimum_coverage=0.5,
        suppress_source_identity=True,
    )
    assert suppressed == 1
    assert active == 0
    assert np.all(values == 0)

    values, suppressed, active = score.score_rules(
        observations=np.asarray([1.0]),
        candidates_formula=[{"C": 10, "H": 11, "O": 3}, {"C": 10, "H": 11, "O": 3}],
        candidates_bits=[frozenset({1, 2, 3}), frozenset({4, 5, 6})],
        candidates_ik14=["A", "B"],
        rule_formula=[{"C": 9, "H": 9, "O": 1}],
        rule_bits=[frozenset({1, 2, 3})],
        rule_source_identities=[set()],
        rule_weight=np.asarray([1.0]),
        minimum_coverage=0.5,
        suppress_source_identity=True,
    )
    assert suppressed == 0 and active == 1
    assert values[0] > values[1]

    source_corpus = inspect.getsource(corpus)
    source_score = inspect.getsource(score)
    assert score.Counter is not None
    assert "len(by_identity) < args.minimum_identities" in source_corpus
    assert "args.minimum_identities < 3" in source_corpus
    assert "candidate_truth_not_used" in source_corpus
    assert "molecule_label" not in source_score
    assert "formula-disjoint qualification" in source_score
    print("PASS: ChemAware recurrent MassBank fragment-source contracts")


if __name__ == "__main__":
    main()
