from __future__ import annotations

import numpy as np
import pandas as pd

from build_msv000092887_paired_response_object import (
    apply_frozen_metadata_correction,
    paired_effects,
)


def test_paired_effects_keep_abundance_and_detection_separate() -> None:
    # Features x samples; COVID indices are first two, controls are last two.
    abundance = np.asarray([
        [8.0, 16.0, 4.0, 4.0],   # +1 and +2 log2 differences
        [5.0, 0.0, 0.0, 7.0],    # one COVID-only, one control-only
        [2.0, 4.0, 0.0, 0.0],    # two COVID-only detections
    ])
    result = paired_effects(
        abundance, covid_indices=[0, 1], control_indices=[2, 3],
        minimum_both_present_pairs=2, minimum_discordant_pairs=2,
    )
    assert np.isclose(result.loc[0, "abundance_beta_log2_covid_vs_control"], 1.5)
    assert result.loc[0, "abundance_qc_pass"]
    assert result.loc[1, "detection_n_discordant_pairs"] == 2
    assert np.isclose(result.loc[1, "detection_beta_log_odds_covid_vs_control"], 0.0)
    assert result.loc[2, "detection_beta_log_odds_covid_vs_control"] > 0
    assert result.loc[2, "detection_qc_pass"]


def test_zero_is_not_imputed_into_abundance_effect() -> None:
    abundance = np.asarray([[8.0, 0.0, 4.0, 4.0]])
    result = paired_effects(
        abundance, covid_indices=[0, 1], control_indices=[2, 3],
        minimum_both_present_pairs=1, minimum_discordant_pairs=1,
    )
    assert result.loc[0, "abundance_n_both_present_pairs"] == 1
    assert np.isnan(result.loc[0, "abundance_se"])
    assert not result.loc[0, "abundance_qc_pass"]
    assert result.loc[0, "detection_n_control_only_pairs"] == 1


def test_only_frozen_c43_assay_typo_is_corrected() -> None:
    design = pd.DataFrame({
        "biological_unit_id": ["C31", "C43"],
        "assay_name": ["21FP", "21FP"],
        "pair_id": ["21", "24"],
        "group": ["covid", "covid"],
    })
    corrected = apply_frozen_metadata_correction(design)
    assert corrected.assay_name.tolist() == ["21FP", "24FP"]
    assert corrected.metadata_correction.eq("").tolist() == [True, False]
    assert "21FP corrected to 24FP" in corrected.loc[1, "metadata_correction"]
