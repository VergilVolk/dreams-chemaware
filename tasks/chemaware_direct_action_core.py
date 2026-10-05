"""CPU-only contracts shared by ChemAware direct-action audits and tests."""
from __future__ import annotations

import numpy as np


ROLE_CODE = {
    "uncertain": 0,
    "corrective_rank": 1,
    "corrective_margin": 2,
    "robustness": 3,
    "harmful": 4,
}


def classify_action_roles(
    old_rank: np.ndarray,
    new_rank: np.ndarray,
    delta_margin: np.ndarray,
    advantage_swap: np.ndarray,
    advantage_permuted: np.ndarray,
    minimum_margin: float = 0.005,
    harmful_margin: float = 0.010,
) -> np.ndarray:
    """Assign a role without ever treating a non-causal gain as corrective."""
    arrays = [
        np.asarray(value)
        for value in (old_rank, new_rank, delta_margin, advantage_swap, advantage_permuted)
    ]
    if len({len(value) for value in arrays}) != 1:
        raise ValueError("action role arrays are not aligned")
    role = np.full(len(arrays[0]), ROLE_CODE["uncertain"], dtype=np.int8)
    causal = (arrays[3] > 0) & (arrays[4] > 0)
    rank_better = arrays[1] < arrays[0]
    rank_worse = arrays[1] > arrays[0]
    role[causal & rank_better] = ROLE_CODE["corrective_rank"]
    role[
        causal & ~rank_better & ~rank_worse & (arrays[2] >= minimum_margin)
    ] = ROLE_CODE["corrective_margin"]
    robust = (
        ~rank_worse
        & (arrays[2] >= -minimum_margin)
        & (role == ROLE_CODE["uncertain"])
    )
    role[robust] = ROLE_CODE["robustness"]
    harmful = rank_worse | (arrays[2] <= -harmful_margin)
    role[harmful] = ROLE_CODE["harmful"]
    return role


def formula_bootstrap(
    values: np.ndarray, formula: np.ndarray, seed: int, draws: int
) -> dict:
    """Bootstrap formula-macro means so repeated spectra are not pseudo-replicates."""
    values = np.asarray(values, dtype=np.float64)
    formula = np.asarray(formula).astype(str)
    if len(values) != len(formula) or not len(values):
        raise ValueError("formula bootstrap requires non-empty aligned arrays")
    unique, inverse = np.unique(formula, return_inverse=True)
    macro = np.asarray([np.mean(values[inverse == i]) for i in range(len(unique))])
    rng = np.random.default_rng(seed)
    boot = np.empty(draws)
    for draw in range(draws):
        boot[draw] = np.mean(macro[rng.integers(0, len(macro), len(macro))])
    return {
        "formula_macro_mean": float(np.mean(macro)),
        "formula_cluster_bootstrap_95ci": [
            float(v) for v in np.quantile(boot, (0.025, 0.975))
        ],
        "formula_clusters": int(len(unique)),
        "draws": int(draws),
    }
