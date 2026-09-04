import pandas as pd

from tasks.freeze_st001154_hilic_confirmation_samples import evenly_spaced_samples


def test_extension_even_spacing_is_deterministic_and_unique():
    frame = pd.DataFrame({
        "Order": list(range(20)),
        "FileName": [f"f{i}" for i in range(20)],
    })
    first = evenly_spaced_samples(frame, 8)
    second = evenly_spaced_samples(frame.sample(frac=1, random_state=3), 8)
    assert first["FileName"].tolist() == second["FileName"].tolist()
    assert first["FileName"].nunique() == 8
