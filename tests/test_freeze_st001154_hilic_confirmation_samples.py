import pandas as pd

from tasks.freeze_st001154_hilic_confirmation_samples import evenly_spaced_samples


def test_evenly_spaced_samples_are_deterministic_and_cover_ends():
    frame = pd.DataFrame(
        {
            "Order": [5, 1, 4, 2, 3],
            "FileName": ["e", "a", "d", "b", "c"],
        }
    )
    selected = evenly_spaced_samples(frame, 3)
    assert selected["Order"].tolist() == [1, 3, 5]
    assert selected["FileName"].tolist() == ["a", "c", "e"]
