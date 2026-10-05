"""Contracts for the pinned public-model pair-score adapter."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

import GLM_build_gnps_public_model_pair_scores as target  # noqa: E402


def test_module_imports_without_heavy_dependencies() -> None:
    """matchms/gensim/ms2deepscore must stay lazily imported."""
    import sys as _sys

    assert "matchms" not in _sys.modules
    assert "gensim" not in _sys.modules
    assert "ms2deepscore" not in _sys.modules


def test_model_pinning_covers_both_methods() -> None:
    for method in ("spec2vec_gnps_2019", "ms2deepscore_dual_2024"):
        pin = target.MODEL_PINNING[method]
        assert pin["main_file"] in pin["files_md5"]
        assert pin["zenodo"].startswith("10.5281/zenodo.")
        assert all(len(md5) == 32 for md5 in pin["files_md5"].values())


def test_md5_verification_fails_closed_on_tamper() -> None:
    with tempfile.TemporaryDirectory() as directory:
        model_dir = Path(directory)
        (model_dir / "ms2deepscore_model.pt").write_bytes(b"model-bytes")
        (model_dir / "settings.json").write_bytes(b"{}")
        with pytest.raises(RuntimeError, match="md5 mismatch"):
            target.verify_model_dir("ms2deepscore_dual_2024", model_dir)
        with pytest.raises(FileNotFoundError):
            (model_dir / "ms2deepscore_model.pt").unlink()
            target.verify_model_dir("ms2deepscore_dual_2024", model_dir)


def test_real_panel_edges_match_frozen_pair_counts() -> None:
    benchmark = (
        ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1"
    )
    identity_panel = benchmark / "panel_identity_disjoint.npz"
    if not identity_panel.is_file():
        pytest.skip(
            "local sealed benchmark copy lost panel_identity_disjoint.npz; "
            "restore it from the server before trusting local panel tests"
        )
    edges = target.panel_edge_rows(benchmark)
    assert set(edges) == {"identity_disjoint", "formula_disjoint"}
    assert len(edges["identity_disjoint"][0]) == 175171
    assert len(edges["formula_disjoint"][0]) == 47724
    for queries, candidates in edges.values():
        assert len(queries) == len(candidates)
        assert queries.min() >= 0 and candidates.min() >= 0


def test_panel_scoring_preserves_frozen_order() -> None:
    class Stub:
        def pair(self, spectrum_a, spectrum_b) -> float:
            return float(len(spectrum_a) + len(spectrum_b))

    class FakeSpectrum(list):
        pass

    spectra = {
        10: FakeSpectrum([1]),
        20: FakeSpectrum([2, 3]),
        30: FakeSpectrum([4]),
    }
    queries = np.asarray([10, 20, 20], dtype=np.int64)
    candidates = np.asarray([30, 30, 10], dtype=np.int64)
    scores = target.score_panel_edges(
        Stub(), spectra, queries, candidates, "stub",
    )
    assert scores.tolist() == [2.0, 3.0, 3.0]
    with pytest.raises(RuntimeError, match="non-finite"):
        class Broken:
            def pair(self, a, b) -> float:
                return float("nan")

        target.score_panel_edges(Broken(), spectra, queries, candidates, "broken")


def test_cached_embedding_scoring_preserves_edge_order() -> None:
    embeddings = target.unit_rows(np.asarray([
        [2.0, 0.0],
        [0.0, 4.0],
        [1.0, 1.0],
    ], dtype=np.float32))
    positions = {10: 0, 20: 1, 30: 2}
    scores = target.pair_scores_from_embeddings(
        embeddings,
        positions,
        np.asarray([10, 20, 20]),
        np.asarray([30, 30, 10]),
    )
    assert np.allclose(scores, [2 ** -0.5, 2 ** -0.5, 0.0])


def test_spec2vec_article_preprocessing_is_frozen_in_adapter() -> None:
    text = (ROOT / "tasks/GLM_build_gnps_public_model_pair_scores.py").read_text(
        encoding="utf-8",
    )
    for token in (
        "normalize_intensities",
        "select_by_mz",
        "require_minimum_number_of_peaks",
        "reduce_to_number_of_peaks",
        "ratio_desired=0.5",
        "select_by_relative_intensity",
        "intensity_from=0.001",
        "compute_losses(loss_mz_from=5.0, loss_mz_to=200.0)",
    ):
        assert token in text
    assert "PROTON_MASS_DA" in text
    assert '"parent_mass": float(precursor) - PROTON_MASS_DA' in text
