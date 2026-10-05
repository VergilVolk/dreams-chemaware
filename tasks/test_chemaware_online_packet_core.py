"""CPU contracts for continuous-Adam online ChemAware query packets."""
from __future__ import annotations

import numpy as np

from build_chemaware_action_hard_native_triplets import OFFICIAL_HARD, SPECIFIC_HARD
from chemaware_online_packet_core import (
    CHEMICAL_SLOT,
    DREAMS_NATIVE_REPLAY,
    ERROR_QUERY_PACKET,
    SAFE_QUERY_PACKET,
    allocate_role_steps,
    build_online_packet_pool,
    phasea_role_fractions,
    role_entries,
)


def main() -> None:
    manifest = {
        "query_row": np.asarray([0, 10], dtype=np.int64),
        "query_ptr": np.asarray([0, 3, 6], dtype=np.int64),
        "molecule_label": np.asarray([1, 0, 0, 1, 0, 0], dtype=bool),
        "molecule_ptr": np.arange(7, dtype=np.int64),
        "pair_candidate_row": np.asarray([1, 2, 3, 11, 12, 13], dtype=np.int64),
    }
    evidence = {
        "query": np.asarray([0, 1], dtype=np.int64),
        "baseline_rank": np.asarray([2, 1], dtype=np.int32),
    }
    base = {
        "source_query": np.asarray([0, 0, 1, 1], dtype=np.int64),
        "negative_candidate": np.asarray([1, 2, 2, 1], dtype=np.int32),
        "source_tag": np.asarray(
            [OFFICIAL_HARD, SPECIFIC_HARD, OFFICIAL_HARD, SPECIFIC_HARD],
            dtype=np.int16,
        ),
    }
    replay = {
        "anchor_idx": np.asarray([20], dtype=np.int64),
        "positive_ptr": np.asarray([0, 2], dtype=np.int64),
        "positive_idx": np.asarray([21, 22], dtype=np.int64),
        "negative_ptr": np.asarray([0, 2], dtype=np.int64),
        "negative_idx": np.asarray([23, 24], dtype=np.int64),
    }
    rows = np.asarray([0, 1, 2, 3, 10, 11, 12, 13], dtype=np.int64)
    vectors = np.asarray([
        [1.0, 0.0], [0.8, 0.6], [0.9, 0.4358899], [0.75, 0.6614378],
        [0.0, 1.0], [0.0, 1.0], [0.9797959, 0.2], [0.8660254, 0.5],
    ], dtype=np.float32)
    pool, report = build_online_packet_pool(
        manifest=manifest, evidence=evidence, base=base, replay=replay,
        embedding_rows=rows, embeddings=vectors, margin=0.1,
        packet_width=3, replay_events=1, seed=3407,
    )
    assert report["status"] == "CHEMAWARE_ONLINE_QUERY_PACKET_COMPLETE"
    assert report["queries"] == 2
    assert report["current_error_queries"] == 1
    assert report["current_correct_queries"] == 1
    assert report["packets_with_distinct_chemical_negative"] == 1
    assert report["packets_with_distinct_specific_negative"] == 1
    assert report["packets_with_distinct_action_only_negative"] == 0
    assert len(pool["anchor_idx"]) == 3
    assert np.array_equal(
        pool["event_kind"],
        np.asarray([ERROR_QUERY_PACKET, SAFE_QUERY_PACKET, DREAMS_NATIVE_REPLAY]),
    )
    assert int(np.sum(pool["packet_slot_role"][0] == CHEMICAL_SLOT)) == 1
    assert np.all(pool["packet_candidate"][1] == pool["winner_candidate"][1])
    assert np.array_equal(pool["source_query"][:2], np.asarray([0, 1]))
    assert all(report["gates"].values())
    entries = role_entries(pool)
    assert {name: len(rows) for name, rows in entries.items()} == {
        "safe": 1, "error": 1, "chemical": 1, "replay": 1,
    }

    phasea = {
        "anchor_idx": np.arange(10),
        "curriculum_role": np.asarray([1] * 6 + [2, 3] + [4] * 2),
    }
    fractions = phasea_role_fractions(phasea)
    assert fractions == {"safe": 0.6, "error": 0.1, "chemical": 0.1, "replay": 0.2}
    assert allocate_role_steps(fractions, 10) == {
        "safe": 6, "error": 1, "chemical": 1, "replay": 2,
    }

    # Removing query 0 chemistry may change only query 0's packet.  It cannot
    # create a chemical slot for query 1 through identity/query broadcast.
    control = {key: value.copy() for key, value in base.items()}
    control["source_tag"][1] = 0
    control_pool, control_report = build_online_packet_pool(
        manifest=manifest, evidence=evidence, base=control, replay=replay,
        embedding_rows=rows, embeddings=vectors, margin=0.1,
        packet_width=3, replay_events=1, seed=3407,
    )
    assert control_report["packets_with_distinct_chemical_negative"] == 0
    assert np.array_equal(pool["packet_candidate"][1], control_pool["packet_candidate"][1])
    print("PASS: ChemAware online query-packet core contracts")


if __name__ == "__main__":
    main()
