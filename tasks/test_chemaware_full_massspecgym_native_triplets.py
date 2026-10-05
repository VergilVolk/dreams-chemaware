from __future__ import annotations

import numpy as np

from build_chemaware_full_massspecgym_native_triplets import (
    CHEMICAL_WINNER_EVENT,
    construct_pool,
    edge_slice,
    merge_official_pools,
    unchanged_native_events,
    validate_native_pool,
)


def synthetic_pool() -> dict[str, np.ndarray]:
    return {
        "anchor_idx": np.asarray([10, 20, 30], dtype=np.int64),
        "positive_ptr": np.asarray([0, 2, 3, 5], dtype=np.int64),
        "positive_idx": np.asarray([11, 12, 21, 31, 32], dtype=np.int64),
        "negative_ptr": np.asarray([0, 2, 4, 5], dtype=np.int64),
        "negative_idx": np.asarray([13, 14, 22, 23, 33], dtype=np.int64),
    }


def main() -> None:
    official = synthetic_pool()
    validate_native_pool(official, "synthetic")
    overrides = {
        20: {
            "query": 7,
            "truth": 0,
            "candidate": 2,
            "role": CHEMICAL_WINNER_EVENT,
            "event": {
                "positive_row": 21,
                "negative_rows": np.asarray([23], dtype=np.int64),
                "hinges": np.asarray([0.04], dtype=np.float64),
            },
            "verdict": {
                "pair_class": "unanimous_admitted_style",
                "supports": [("massbank", 0.5)],
            },
        }
    }
    output, ledger = construct_pool(official, overrides)
    assert np.array_equal(output["anchor_idx"], official["anchor_idx"])
    assert len(np.unique(output["anchor_idx"])) == len(official["anchor_idx"])
    assert np.array_equal(edge_slice(output, 0, "positive"), np.asarray([11, 12]))
    assert np.array_equal(edge_slice(output, 0, "negative"), np.asarray([13, 14]))
    assert np.array_equal(edge_slice(output, 1, "positive"), np.asarray([21]))
    assert np.array_equal(edge_slice(output, 1, "negative"), np.asarray([23]))
    assert np.array_equal(edge_slice(output, 2, "positive"), np.asarray([31, 32]))
    assert np.array_equal(edge_slice(output, 2, "negative"), np.asarray([33]))
    assert unchanged_native_events(official, output, {20})
    assert output["curriculum_role"].tolist() == [0, CHEMICAL_WINNER_EVENT, 0]
    assert len(ledger) == 1 and ledger[0]["anchor_row"] == 20

    broken = synthetic_pool()
    broken["anchor_idx"] = np.asarray([10, 10, 30], dtype=np.int64)
    try:
        validate_native_pool(broken, "broken")
    except RuntimeError:
        pass
    else:
        raise AssertionError("duplicate anchors must fail closed")

    second = {
        "anchor_idx": np.asarray([40], dtype=np.int64),
        "positive_ptr": np.asarray([0, 1], dtype=np.int64),
        "positive_idx": np.asarray([41], dtype=np.int64),
        "negative_ptr": np.asarray([0, 1], dtype=np.int64),
        "negative_idx": np.asarray([42], dtype=np.int64),
    }
    merged = merge_official_pools([official, second])
    validate_native_pool(merged, "merged")
    assert merged["anchor_idx"].tolist() == [10, 20, 30, 40]
    assert merged["positive_ptr"].tolist() == [0, 2, 3, 5, 6]
    assert merged["positive_idx"].tolist() == [11, 12, 21, 31, 32, 41]
    assert merged["negative_ptr"].tolist() == [0, 2, 4, 5, 6]
    assert merged["negative_idx"].tolist() == [13, 14, 22, 23, 33, 42]

    overlapped = merge_official_pools([official, official])
    try:
        validate_native_pool(overlapped, "overlapped")
    except RuntimeError:
        pass
    else:
        raise AssertionError("pool overlap must fail closed after merge")
    print("PASS: ChemAware full-MassSpecGym native-triplet contracts")


if __name__ == "__main__":
    main()
