#!/usr/bin/env python
"""Unit tests for the v2 exact-action-event corpus and rotation schedule."""
import re
from pathlib import Path

import numpy as np
from types import SimpleNamespace

from build_noise_relation_t1_t3_corpus import build_split
from build_noise_relation_t1_t3_corpus_v2 import (
    audit_against_clean_corpus,
    extract_action_events,
)
from noise_relation_t1_t3_core import proportional_interleave


def _pool(registry_rows, anchors, positive_ptr, positive_idx, negative_ptr,
          negative_idx, event_kind, event_query, event_action):
    kinds, sources = zip(*registry_rows)
    return {
        "registry_kind": np.asarray(kinds, dtype=np.int8),
        "registry_source_index": np.asarray(sources, dtype=np.int64),
        "anchor_idx": np.asarray(anchors, dtype=np.int64),
        "positive_ptr": np.asarray(positive_ptr, dtype=np.int64),
        "positive_idx": np.asarray(positive_idx, dtype=np.int64),
        "negative_ptr": np.asarray(negative_ptr, dtype=np.int64),
        "negative_idx": np.asarray(negative_idx, dtype=np.int64),
        "event_kind": np.asarray(event_kind, dtype=np.int8),
        "event_query": np.asarray(event_query, dtype=np.int64),
        "event_action_index": np.asarray(event_action, dtype=np.int64),
    }


def test_extract_action_events_keeps_exact_bridges() -> None:
    # registry: 0=clean anchor row10 | 1=ACTION 500 | 2=row11 | 3=row20
    #           4=row12 | 5=ACTION 501 | 6=row30 | 7=row31
    pool = _pool(
        registry_rows=[(0, 10), (1, 500), (0, 11), (0, 20), (0, 12), (1, 501), (0, 30), (0, 31)],
        anchors=[0, 1, 5, 1, 0, 1],
        positive_ptr=[0, 1, 2, 3, 4, 5, 7],
        positive_idx=[2, 4, 6, 2, 4, 4, 6],
        negative_ptr=[0, 1, 2, 3, 4, 5, 6, 7],
        negative_idx=[3, 7, 7, 3, 3, 3, 3],
        event_kind=[0, 2, 2, 2, 2, 2],
        event_query=[0, 0, 1, 1, 0, 1],
        event_action=[-1, 500, 501, 500, 500, 500],
    )
    query_formula = np.asarray(["F", "G"])
    folds = np.asarray([2, 1], dtype=np.int8)
    representable = np.zeros(1000, dtype=bool)
    representable[500] = True
    events, audit = extract_action_events(
        pool, query_formula, folds, representable, outer_fold=0, validation_fold=1,
    )
    assert audit["action_events_in_pool"] == 5
    assert audit["rejected_bad_registry"] == 1          # event 4: HDF5 anchor
    assert audit["rejected_unrepresentable"] == 1       # event 2: action 501
    assert audit["training_fold_events"] == 1           # event 1 -> fold 2
    assert audit["validation_fold_events"] == 1         # event 3 -> fold 1
    assert len(events["pool_event"]) == 2
    training = events["fold"] != 1
    assert list(events["query"][training]) == [0]
    assert list(events["action_index"][training]) == [500]
    assert list(events["positive_row"][training]) == [12]
    assert list(events["negative_row"][training]) == [31]
    assert list(events["query_formula"][training]) == ["F"]


def test_extract_action_events_aborts_on_outer_fold_leak() -> None:
    pool = _pool(
        registry_rows=[(0, 10), (1, 900), (0, 11), (0, 20)],
        anchors=[0, 1],
        positive_ptr=[0, 1, 2],
        positive_idx=[2, 2],
        negative_ptr=[0, 1, 2],
        negative_idx=[3, 3],
        event_kind=[0, 2],
        event_query=[0, 0],
        event_action=[-1, 900],
    )
    representable = np.zeros(1000, dtype=bool)
    representable[900] = True
    try:
        extract_action_events(
            pool, np.asarray(["F"]), np.asarray([0], dtype=np.int8),
            representable, outer_fold=0, validation_fold=1,
        )
    except RuntimeError as error:
        assert "outer-held" in str(error)
    else:
        raise AssertionError("fold-0 leak did not abort")


def test_audit_against_clean_corpus_measures_v1_supervision_loss() -> None:
    graph = SimpleNamespace(
        query_ptr=np.asarray([0, 3, 5]),
        molecule_ptr=np.asarray([0, 2, 4, 5, 7, 9]),
        pair_candidate_row=np.asarray([10, 11, 20, 21, 22, 30, 31, 40, 41]),
        query_row=np.asarray([10, 30]),
        query_formula=np.asarray(["F", "G"]),
        molecule_formula=np.asarray(["F", "F", "X", "G", "G"]),
        molecule_mces_grade=np.asarray([-1, 1, 3, -1, 2]),
    )
    pair = np.asarray([1.0, .5, .8, .7, .9, 1.0, .4, .6, .5])
    molecule = np.asarray([1.0, .8, .9, 1.0, .6])
    arrays, _ = build_split(
        graph, pair, molecule, np.asarray([0]), {0: []},
        max_positive_refs=4, max_negative_molecules=4,
        additional_boundary_molecules=2, max_negative_refs=3,
    )
    # Reference layout established by the builder: positives first, then
    # each negative molecule's hardest references by descending score.
    assert list(arrays["reference_row"]) == [11, 20, 21, 22]
    events = {
        "query": np.asarray([0, 0]),
        "positive_row": np.asarray([11, 99]),
        "negative_row": np.asarray([21, 21]),
    }
    audit = audit_against_clean_corpus(events, arrays)
    assert audit["events_with_clean_counterpart"] == 2
    assert audit["exact_positive_in_clean_pool_fraction"] == 0.5
    # Both exact negatives (row 21) fall inside the selected negative pool.
    assert audit["exact_negative_in_clean_pool_fraction"] == 1.0
    assert audit["exact_bridge_fully_in_clean_pool_fraction"] == 0.5


def test_proportional_interleave_is_deterministic_and_dose_exact() -> None:
    schedule = proportional_interleave(7, 3)
    assert int(np.sum(schedule == 0)) == 7
    assert int(np.sum(schedule == 1)) == 3
    assert list(schedule) == list(proportional_interleave(7, 3))
    # Stream order is preserved inside each kind: positions of a kind appear
    # in their original sequence, so batch/event cursors advance monotonically.
    assert list(np.flatnonzero(schedule == 0)) == sorted(np.flatnonzero(schedule == 0))
    assert list(proportional_interleave(0, 4)) == [1, 1, 1, 1]
    assert list(proportional_interleave(4, 0)) == [0, 0, 0, 0]


def test_entry_scripts_reference_only_declared_arguments() -> None:
    # A stale args.<attribute> reference (argument removed from the parser
    # but its guard left behind) crashes only at cluster runtime, after the
    # expensive corpus build.  Catch that class statically here.
    for name in (
        "build_noise_relation_t1_t3_corpus_v2.py",
        "train_noise_relation_t1_t3_v2.py",
        "score_fold1_noise_relation_checkpoint.py",
        "summarize_noise_relation_t1_t3_v2_fold1.py",
    ):
        source = Path(__file__).with_name(name).read_text(encoding="utf-8")
        declared = {
            match.replace("-", "_")
            for match in re.findall(r'add_argument\(\s*"--([A-Za-z0-9-]+)"', source)
        }
        referenced = set(re.findall(r"\bargs\.([A-Za-z_][A-Za-z0-9_]*)", source))
        stale = referenced - declared
        assert not stale, f"{name} references undeclared args attributes: {sorted(stale)}"


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"[test_noise_relation_t1_t3_v2] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
