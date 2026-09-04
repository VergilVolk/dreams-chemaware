from __future__ import annotations

import numpy as np
import pytest

from tasks.kgmn_edge_calibration_core import crossfit_edge_scores


def test_component_crossfit_prefers_informative_dreams_arm() -> None:
    rng = np.random.default_rng(9)
    n = 250
    folds = np.repeat(np.arange(5), n // 5)
    weights = np.ones(n)
    components = np.asarray([f"c{fold}_{i // 5}" for i, fold in enumerate(folds)], dtype=object)
    formulas = np.asarray([f"f{i // 3}" for i in range(n)], dtype=object)
    author_positive = rng.normal(0.55, 0.12, n)
    author_negative = rng.normal(0.50, 0.12, n)
    dreams_positive = rng.normal(0.75, 0.08, n)
    dreams_negative = rng.normal(0.30, 0.08, n)
    report, probabilities, artifact = crossfit_edge_scores(
        {
            "author_dp": (author_positive, author_negative),
            "official_dreams": (dreams_positive, dreams_negative),
        },
        folds,
        weights,
        components,
        formulas,
        bootstrap_resamples=300,
    )
    assert report["official_dreams"]["paired_accuracy"] > report["author_dp"]["paired_accuracy"]
    assert report["official_dreams"]["component_bootstrap_accuracy_delta_vs_author"]["ci_low"] > 0
    assert np.isfinite(probabilities["author_official_intersection"][0]).all()
    assert artifact["official_dreams"]["coefficient"] > 0
    assert artifact["evaluation_thresholds_from_nested_oof"]["official_dreams"]["fdr_0.05"] is not None
    assert artifact["deployment_thresholds_full_refit"]["official_dreams"]["fdr_0.05"] is not None


def test_thresholds_are_fit_without_outer_fold_and_all_oof_scores_are_present() -> None:
    rng = np.random.default_rng(3)
    n = 100
    folds = np.repeat(np.arange(5), 20)
    weights = np.ones(n)
    raw = rng.normal(size=n)
    report, probabilities, artifact = crossfit_edge_scores(
        {
            "author_dp": (raw + 0.3, raw),
            "official_dreams": (raw + 0.5, raw - 0.1),
        },
        folds,
        weights,
        np.asarray([f"c{fold}_{i}" for i, fold in enumerate(folds)], dtype=object),
        np.asarray([f"f{i // 2}" for i in range(n)], dtype=object),
        bootstrap_resamples=100,
    )
    assert len(artifact["outer_thresholds"]["official_dreams"]) == 15
    for positive, negative in probabilities.values():
        assert np.isfinite(positive).all()
        assert np.isfinite(negative).all()
    assert "0.10" in report["official_dreams"]["fixed_fdr"]


def test_component_leakage_across_outer_folds_fails_closed() -> None:
    rng = np.random.default_rng(14)
    n = 100
    folds = np.repeat(np.arange(5), 20)
    raw = rng.normal(size=n)
    components = np.asarray([f"c{i % 10}" for i in range(n)], dtype=object)
    with pytest.raises(RuntimeError, match="component leakage"):
        crossfit_edge_scores(
            {
                "author_dp": (raw + 0.2, raw),
                "official_dreams": (raw + 0.4, raw - 0.1),
            },
            folds,
            np.ones(n),
            components,
            np.asarray([f"f{i // 2}" for i in range(n)], dtype=object),
            bootstrap_resamples=20,
        )
