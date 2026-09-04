from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tasks import build_e1_triplet_pool as e1
from tasks import build_reference_library as reference
from tasks.audit_chemaware_spectral_consensus_applicability import (
    RAW_VIEWS,
    build_query_table,
)
from tasks.evaluate_chemaware_frozen_spectral_gate_test import probability


@pytest.mark.skipif(not reference.HAS_RDKIT, reason="RDKit is required by the reference builder")
def test_massspecgym_membership_is_retained_not_filtered(tmp_path: Path):
    base = {
        "PRECURSOR_MZ": "47.0497",
        "SMILES": "CN",
        "ADDUCT": "[M+H]+",
        "INSTRUMENT_TYPE": "Orbitrap",
        "COLLISION_ENERGY": "20.0",
        "FOLD": "train",
    }
    records = []
    for membership, identifier in (("True", "member"), ("False", "nonmember")):
        fields = dict(base, SIMULATION_CHALLENGE=membership, IDENTIFIER=identifier)
        record = reference.normalize_massspecgym(fields, [(20.0, 1.0), (30.0, 0.5), (40.0, 0.2)])
        assert record is not None
        assert record["simulation_challenge"] == membership
        records.append(record)

    output = tmp_path / "membership.mgf"
    assert reference.write_mgf(records, output) == 2
    text = output.read_text(encoding="utf-8")
    assert text.count("BEGIN IONS") == 2
    assert "SIMULATION_CHALLENGE=True" in text
    assert "SIMULATION_CHALLENGE=False" in text
    assert "INSTRUMENT_TYPE=Orbitrap" in text
    assert "COLLISION_ENERGY=20.0" in text
    assert "FOLD=train" in text


def write_allow(path: Path, **updates) -> Path:
    body = {
        "simulation_challenge_semantics": (
            "spectrum-simulation benchmark subset membership; not spectrum provenance"
        ),
        "p3_query_overlap": 0,
        "train_primary_all": {"rows": [1, 3, 5]},
    }
    body.update(updates)
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_corrected_allow_list_accepts_both_memberships_by_schema(tmp_path: Path):
    path = write_allow(tmp_path / "allow.json")
    rows, audit = e1.load_corrected_allow_list(path, total_rows=6)
    assert np.array_equal(rows, np.asarray([1, 3, 5]))
    assert audit["schema"] == "train_primary_all.rows"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"train_primary_all": {"rows": [1, 1]}}, "duplicate"),
        ({"train_primary_all": {"rows": [6]}}, "out-of-range"),
        ({"p3_query_overlap": 1}, "zero P3 identity overlap"),
        ({"simulation_challenge_semantics": "True means simulated"}, "corrected"),
        ({"train_primary_all": None, "real_train_primary": {"rows": [1]}}, "train_primary_all"),
    ],
)
def test_corrected_allow_list_fails_closed(tmp_path: Path, updates: dict, message: str):
    path = write_allow(tmp_path / "bad.json", **updates)
    with pytest.raises(RuntimeError, match=message):
        e1.load_corrected_allow_list(path, total_rows=6)


def test_frozen_gate_probability_matches_declared_linear_pipeline():
    frame = pd.DataFrame({"a": [1.0, 3.0], "b": [2.0, -2.0]})
    gate = {
        "features": ["a", "b"],
        "standard_scaler_mean": [1.0, 0.0],
        "standard_scaler_scale": [2.0, 4.0],
        "logistic_coefficient": [0.5, -1.5],
        "logistic_intercept": 0.25,
    }
    standardized = (frame[["a", "b"]].to_numpy() - [1.0, 0.0]) / [2.0, 4.0]
    expected = 1.0 / (1.0 + np.exp(-(standardized @ [0.5, -1.5] + 0.25)))
    assert np.allclose(probability(frame, gate), expected, rtol=0, atol=1e-12)


def test_query_table_treats_dreams_top_tie_as_wrong_and_raw_ties_as_abstentions(
    tmp_path: Path,
):
    manifest = pd.DataFrame(
        {
            "ik14": ["AAAAAAAAAAAAAA", "AAAAAAAAAAAAAA", "BBBBBBBBBBBBBB"],
            "formula": ["C2H6O", "C2H6O", "C2H6O"],
        }
    )
    pairs = pd.DataFrame(
        {
            "split": ["unit", "unit"],
            "left": [0, 0],
            "right": [1, 2],
            "dreams_similarity": [0.8, 0.8],
            **{name: [0.5, 0.5] for name in RAW_VIEWS},
        }
    )
    manifest_path = tmp_path / "manifest.csv"
    pair_path = tmp_path / "pairs.csv"
    manifest.to_csv(manifest_path, index=False)
    pairs.to_csv(pair_path, index=False)

    table = build_query_table(pair_path, manifest_path, "unit")
    assert len(table) == 1
    row = table.iloc[0]
    assert not bool(row["dreams_correct"])
    assert int(row["consensus_votes"]) == 0
    assert not bool(row["route_candidate"])
