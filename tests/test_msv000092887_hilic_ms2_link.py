from __future__ import annotations

from pathlib import Path

import pandas as pd

from tasks.link_msv000092887_hilic_ms2 import assay_role, summarize_features
from tasks.prepare_msv000092887_open_response_source import response_channel
from tasks.stage_msv000092887_hilic_mzxml import download_candidates


ROOT = Path(__file__).resolve().parents[1]


def test_stager_has_official_frontend_fallbacks_without_changing_file_key() -> None:
    url = (
        "https://massive.ucsd.edu/ProteoSAFe/DownloadResultFile?"
        "file=f.MSV000092887%2Fpeak%2F10CP.mzXML"
    )
    candidates = download_candidates(url)
    assert [candidate.split("/")[2] for candidate in candidates] == [
        "massive.ucsd.edu", "gnps.ucsd.edu", "proteomics.ucsd.edu"
    ]
    assert len({candidate.split("?", 1)[1] for candidate in candidates}) == 1


def test_frozen_acquisition_roles_do_not_use_covid_control_suffixes() -> None:
    manifest = pd.read_csv(
        ROOT
        / "data/validation/msv000092887_hilic_positive_mzxml_manifest_v2_20261019"
        / "mzxml_manifest.tsv",
        sep="\t",
        dtype=str,
    )
    counts = manifest.assay_name.map(assay_role).value_counts().to_dict()
    assert counts == {"biological": 78, "blank": 40, "pooled_qc": 11}
    assert assay_role("1CP") == assay_role("1FP") == "biological"


def test_ambiguous_and_blank_only_ms2_do_not_enter_usable_partition() -> None:
    features = pd.DataFrame(
        {
            "entity_id": ["known", "unknown", "blank", "ambiguous"],
            "precursor_mz": [100.0, 200.0, 300.0, 400.0],
            "retention_time_seconds": [10.0, 20.0, 30.0, 40.0],
            "reference_spectrum_evidence": [True, False, False, False],
        }
    )
    links = pd.DataFrame(
        [
            {
                "nearest_feature_id": "known",
                "candidate_feature_ids": "known",
                "unique_feature_link": True,
                "assay_role": "pooled_qc",
                "assay_name": "Q1",
            },
            {
                "nearest_feature_id": "unknown",
                "candidate_feature_ids": "unknown",
                "unique_feature_link": True,
                "assay_role": "biological",
                "assay_name": "1CP",
            },
            {
                "nearest_feature_id": "blank",
                "candidate_feature_ids": "blank",
                "unique_feature_link": True,
                "assay_role": "blank",
                "assay_name": "b1",
            },
            {
                "nearest_feature_id": "ambiguous",
                "candidate_feature_ids": "unknown|ambiguous",
                "unique_feature_link": False,
                "assay_role": "biological",
                "assay_name": "2CP",
            },
        ]
    )
    representatives = {
        feature_id: {"assay_role": role}
        for feature_id, role in (
            ("known", "pooled_qc"),
            ("unknown", "biological"),
            ("blank", "blank"),
        )
    }
    result = summarize_features(features, links, representatives).set_index("entity_id")

    assert result.loc["known", "source_local_ms2_partition"] == "library_supported"
    assert result.loc["unknown", "source_local_ms2_partition"] == "unresolved_source_local"
    assert result.loc["blank", "source_local_ms2_partition"] == "blank_only"
    assert result.loc["ambiguous", "source_local_ms2_partition"] == "no_unique_raw_ms2"
    assert not bool(result.loc["blank", "has_usable_raw_ms2"])
    assert not bool(result.loc["ambiguous", "has_usable_raw_ms2"])
    assert int(result.loc["ambiguous", "ambiguous_candidate_spectra"]) == 1


def test_abundance_and_detection_channels_are_filtered_without_collapsing() -> None:
    frame = pd.DataFrame(
        {
            "entity_id": ["both", "abundance", "detection", "no_ms2"],
            "has_usable_raw_ms2": [True, True, True, False],
            "abundance_qc_pass": [True, True, False, True],
            "abundance_beta": [1.0, 2.0, 3.0, 4.0],
            "abundance_se": [0.5, 0.5, 0.5, 0.5],
            "detection_qc_pass": [True, False, True, True],
            "detection_beta": [-1.0, -2.0, -3.0, -4.0],
            "detection_se": [0.4, 0.4, 0.4, 0.4],
        }
    )
    prevalence = pd.Series([1.0, 0.9, 0.8, 0.7])
    abundance = response_channel(
        frame,
        qc_column="abundance_qc_pass",
        beta_column="abundance_beta",
        se_column="abundance_se",
        prevalence=prevalence,
        model="abundance",
    )
    detection = response_channel(
        frame,
        qc_column="detection_qc_pass",
        beta_column="detection_beta",
        se_column="detection_se",
        prevalence=prevalence,
        model="detection",
    )

    assert set(abundance.entity_id) == {"both", "abundance"}
    assert set(detection.entity_id) == {"both", "detection"}
    assert abundance.set_index("entity_id").loc["both", "beta"] == 1.0
    assert detection.set_index("entity_id").loc["both", "beta"] == -1.0
