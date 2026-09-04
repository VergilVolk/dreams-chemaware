"""Fast contract tests for the MTBLS13729 frozen-P2b application path."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from encode_hdf5_shared_dreams import order_hdf5_by_manifest
from encode_unified_library_for_p2b import iter_mgf, top_peak_array
from infer_mtbls13729_p2b_vs_dreams import nearest_target_links, summarize_features
from infer_mtbls13729_embedding_retrieval import assert_query_alignment


def test_mgf_order_and_peak_cache() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "x.mgf"
        path.write_text(
            "BEGIN IONS\nNAME=a\nPEPMASS=100\nINCHIKEY=AAAAAAAAAAAAAA-X\n"
            "60 1\n50 4\n70 2\nEND IONS\n",
            encoding="utf-8",
        )
        records = list(iter_mgf(path))
        assert len(records) == 1 and records[0][0]["NAME"] == "a"
        cached = top_peak_array(records[0][2], 2)
        assert cached.shape == (2, 2)
        assert np.allclose(cached[0], [50, 70])


def test_nearest_target_is_mass_and_rt_constrained() -> None:
    manifest = pd.DataFrame({
        "precursor_mz": [100.0005, 100.0005, 150.0],
        "RT": [2.0, 9.0, 2.0],  # minutes
    })
    targets = pd.DataFrame({
        "feature_id": [7, 8], "mz": [100.0, 100.0008], "rt_sec": [121.0, 500.0]
    })
    links = nearest_target_links(manifest, targets, ppm=10.0, rt_seconds=20.0)
    assert links.query_idx.tolist() == [0]
    assert links.feature_id.tolist() == [7]


def test_feature_summary_uses_replicate_consensus() -> None:
    rows = pd.DataFrame({
        "feature_id": [1, 1, 1],
        "query_file": ["a", "b", "c"],
        "query_scan": [1, 1, 1],
        "dreams_ik14": ["A" * 14, "A" * 14, "B" * 14],
        "dreams_inchikey": ["A" * 14 + "-X", "A" * 14 + "-X", "B" * 14 + "-X"],
        "dreams_name": ["a", "a", "b"],
        "dreams_smiles": ["C", "C", "CC"],
        "dreams_dreams_similarity": [0.8, 0.9, 0.95],
        "dreams_score": [0.8, 0.9, 0.95],
    })
    summary = summarize_features(rows, "dreams")
    assert len(summary) == 1
    assert summary.iloc[0].ik14 == "A" * 14
    assert summary.iloc[0].n_support_samples == 2
    assert np.isclose(summary.iloc[0].agreement_fraction, 2 / 3)


def test_experimental_query_cache_must_be_row_aligned() -> None:
    official = pd.DataFrame({
        "file_name": ["a", "b"], "scan_number": [1, 2],
        "precursor_mz": [100.0, 200.0], "row_in_file": [0, 0],
    })
    assert_query_alignment(official.copy(), official)
    broken = official.iloc[::-1].reset_index(drop=True)
    try:
        assert_query_alignment(broken, official)
    except RuntimeError:
        pass
    else:
        raise AssertionError("misaligned experimental cache was accepted")


def test_hdf5_inputs_follow_reference_manifest_order() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        paths = [root / "a.hdf5", root / "b.hdf5"]
        for path in paths:
            path.touch()
        manifest = root / "manifest.csv"
        pd.DataFrame({"file_name": ["b", "b", "a"]}).to_csv(manifest, index=False)
        ordered = order_hdf5_by_manifest(paths, manifest)
        assert [path.stem for path in ordered] == ["b", "a"]


if __name__ == "__main__":
    test_mgf_order_and_peak_cache()
    test_nearest_target_is_mass_and_rt_constrained()
    test_feature_summary_uses_replicate_consensus()
    test_experimental_query_cache_must_be_row_aligned()
    test_hdf5_inputs_follow_reference_manifest_order()
    print("[test_mtbls13729_p2b_application] PASS")
