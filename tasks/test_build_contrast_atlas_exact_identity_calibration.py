from __future__ import annotations

import pandas as pd

from build_contrast_atlas_exact_identity_calibration import (
    directed_cross_source_pairs,
    normalize_entity_id,
    resolve_unambiguous_evidence,
    strict_planar_inchikey,
)


def test_strict_planar_key_accepts_only_real_key_shapes() -> None:
    assert strict_planar_inchikey("ABCDEFGHIJKLMN-UHFFFAOYSA-N") == "ABCDEFGHIJKLMN"
    assert strict_planar_inchikey("ABCDEFGHIJKLMN") == "ABCDEFGHIJKLMN"
    assert strict_planar_inchikey("N/A") is None
    assert strict_planar_inchikey("") is None
    assert strict_planar_inchikey("abcdefghijklmn-UHFFFAOYSA-N") is None


def test_normalize_entity_id_only_removes_numeric_csv_suffix() -> None:
    assert normalize_entity_id("178.0") == "178"
    assert normalize_entity_id("A001.0") == "A001.0"


def test_conflicting_planar_keys_exclude_entity() -> None:
    frame = pd.DataFrame([
        {"raw_entity_id": "x", "planar_inchikey": "ABCDEFGHIJKLMN", "evidence_source": "a"},
        {"raw_entity_id": "x", "planar_inchikey": "NOPQRSTUVWXYZA", "evidence_source": "b"},
        {"raw_entity_id": "y", "planar_inchikey": "ABCDEFGHIJKLMN", "evidence_source": "a"},
    ])
    resolved, ambiguous = resolve_unambiguous_evidence(frame)
    assert ambiguous == 1
    assert resolved.raw_entity_id.tolist() == ["y"]


def test_directed_pairs_are_cross_source_and_bidirectional() -> None:
    members = pd.DataFrame([
        {"planar_inchikey": "ABCDEFGHIJKLMN", "study_id": "a", "source_family": "fa", "entity_id": "a1", "raw_entity_id": "1"},
        {"planar_inchikey": "ABCDEFGHIJKLMN", "study_id": "b", "source_family": "fb", "entity_id": "b1", "raw_entity_id": "1"},
        {"planar_inchikey": "ABCDEFGHIJKLMN", "study_id": "a", "source_family": "fa", "entity_id": "a2", "raw_entity_id": "2"},
    ])
    pairs = directed_cross_source_pairs(members)
    assert len(pairs) == 4
    assert set(zip(pairs.left_source_family, pairs.right_source_family)) == {("fa", "fb"), ("fb", "fa")}
    assert set(pairs.truth_scope) == {"posttruth_spectral_entity_retrieval_calibration_only"}
