from __future__ import annotations

import numpy as np

from estimate_factorial_hfd_panel_contrasts import fit_feature_hc3


def test_factorial_effects_recover_main_and_interaction() -> None:
    genotype = np.repeat([0, 0, 1, 1], 6)
    diet = np.repeat([0, 1, 0, 1], 6)
    residual = np.tile(np.array([-0.2, -0.1, 0.0, 0.0, 0.1, 0.2]), 4)
    y = 10 + 2 * genotype + 3 * diet - 1.5 * genotype * diet + residual
    result = fit_feature_hc3(y, genotype, diet, minimum_cell_n=4)
    assert result is not None
    assert np.isclose(result["diet_reference"][0], 3.0)
    assert np.isclose(result["diet_modified"][0], 1.5)
    assert np.isclose(result["genotype_control"][0], 2.0)
    assert np.isclose(result["genotype_hfd"][0], 0.5)
    assert np.isclose(result["genotype_by_hfd"][0], -1.5)


def test_factorial_effect_fails_when_one_cell_is_too_small() -> None:
    genotype = np.array([0] * 6 + [0] * 6 + [1] * 6 + [1] * 3)
    diet = np.array([0] * 6 + [1] * 6 + [0] * 6 + [1] * 3)
    y = np.arange(len(genotype), dtype=float)
    assert fit_feature_hc3(y, genotype, diet, minimum_cell_n=4) is None
