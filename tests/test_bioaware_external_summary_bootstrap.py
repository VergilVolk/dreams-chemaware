import numpy as np
import pandas as pd

from tasks.summarize_bioaware_v4_external_7panel import formula_bootstrap
from tasks.summarize_bioaware_v6_external_5panel import bootstrap


def _slow_cluster_bootstrap(frame: pd.DataFrame, repeats: int, seed: int) -> np.ndarray:
    groups = [group for _, group in frame.groupby("global_formula", sort=True)]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        draw = rng.integers(0, len(groups), size=len(groups))
        sample = pd.concat([groups[index] for index in draw], ignore_index=True)
        values.append(float(sample.delta.mean()))
    return np.asarray(values)


def test_vectorized_cluster_bootstrap_matches_concatenation_semantics() -> None:
    frame = pd.DataFrame(
        {
            "global_formula": ["A", "A", "B", "C", "C", "C"],
            "delta": [1, 0, -1, 1, 1, 0],
        }
    )
    repeats, seed = 200, 17
    expected = _slow_cluster_bootstrap(frame, repeats, seed)
    for result in (
        formula_bootstrap(frame, repeats, seed),
        bootstrap(frame, repeats, seed),
    ):
        assert result["mean"] == float(frame.delta.mean())
        assert result["clusters"] == 3
        assert result["ci_low"] == float(np.quantile(expected, 0.025))
        assert result["ci_high"] == float(np.quantile(expected, 0.975))
