import pandas as pd

from run_contrast_atlas_count_matched_unresolved_control import (
    sample_unresolved_like_known,
)


def test_sample_unresolved_like_known_is_deterministic_and_count_matched():
    identity = pd.DataFrame(
        {
            "study_id": ["A"] * 6 + ["B"] * 4,
            "entity_id": [f"a{i}" for i in range(6)] + [f"b{i}" for i in range(4)],
            "identity_status": (
                ["library_supported"] * 2 + ["unresolved_entity"] * 4
                + ["library_supported"] + ["unresolved_entity"] * 3
            ),
        }
    )
    first, audit = sample_unresolved_like_known(identity, seed=7)
    second, _ = sample_unresolved_like_known(identity, seed=7)
    assert first.equals(second)
    assert first.groupby("study_id").size().to_dict() == {"A": 2, "B": 1}
    assert set(first["identity_status"]) == {"unresolved_entity"}
    assert len(audit) == 3
