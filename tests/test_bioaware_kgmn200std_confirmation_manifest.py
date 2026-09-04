import pandas as pd

from tasks.build_bioaware_kgmn200std_confirmation_manifest import (
    balanced_splits,
    normalize_name,
    strict_unique_top,
)


def test_name_normalization_is_exact_but_punctuation_insensitive() -> None:
    assert normalize_name("L-Serine") == normalize_name("L Serine")
    assert normalize_name("D-Serine") != normalize_name("L-Serine")


def test_balanced_hidden_seed_splits() -> None:
    frame = balanced_splits([f"IK{index:02d}" for index in range(21)], 10, 0.3, 7)
    assert frame.groupby("repeat")["ik14"].nunique().eq(21).all()
    hidden = frame[frame["role"].eq("hidden_validation")]
    assert hidden.groupby("repeat").size().eq(6).all()
    appearances = hidden.groupby("ik14").size()
    assert int(appearances.max() - appearances.min()) <= 1


def test_strict_top_rejects_tie() -> None:
    frame = pd.DataFrame({
        "candidate_id": ["A", "B"],
        "score": [0.5, 0.5],
    })
    candidate, unique = strict_unique_top(frame, "score")
    assert candidate == "A"
    assert not unique
