from __future__ import annotations

import pandas as pd

from tasks.freeze_kgmn_external_validation_contract import build_hidden_seed_splits


def _synthetic_seed_rows() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for index in range(5):
        rows.append({"inchikey1": f"NEG{index:011d}", "polarity": "negative"})
    for index in range(31):
        identity = f"BOTH{index:010d}"
        rows.extend(
            [
                {"inchikey1": identity, "polarity": "negative"},
                {"inchikey1": identity, "polarity": "positive"},
            ]
        )
    for index in range(6):
        rows.append({"inchikey1": f"POS{index:011d}", "polarity": "positive"})
    return pd.DataFrame(rows)


def test_hidden_seed_splits_are_stratified_balanced_and_reproducible() -> None:
    seeds = _synthetic_seed_rows()
    first = build_hidden_seed_splits(seeds, repeats=10, fraction=0.30, seed=20260831)
    second = build_hidden_seed_splits(seeds, repeats=10, fraction=0.30, seed=20260831)
    pd.testing.assert_frame_equal(first, second)

    counts = first.groupby(["repeat", "role"]).size().unstack(fill_value=0)
    assert counts["seed"].eq(13).all()
    assert counts["hidden_validation"].eq(29).all()

    appearances = (
        first.loc[first["role"].eq("seed")]
        .groupby(["polarity_presence", "inchikey1"])
        .size()
    )
    all_identities = first[["polarity_presence", "inchikey1"]].drop_duplicates()
    balanced = (
        all_identities.set_index(["polarity_presence", "inchikey1"])
        .assign(seed_appearances=appearances)
        ["seed_appearances"]
        .fillna(0)
        .astype(int)
    )
    assert balanced.min() > 0
    assert balanced.max() < 10
    assert balanced.groupby(level=0).agg(lambda values: values.max() - values.min()).le(1).all()


def test_hidden_seed_splits_change_with_seed_but_keep_balance() -> None:
    seeds = _synthetic_seed_rows()
    first = build_hidden_seed_splits(seeds, repeats=10, fraction=0.30, seed=1)
    second = build_hidden_seed_splits(seeds, repeats=10, fraction=0.30, seed=2)
    assert not first.equals(second)
    assert first.groupby("repeat")["inchikey1"].nunique().eq(42).all()
    assert second.groupby("repeat")["inchikey1"].nunique().eq(42).all()
