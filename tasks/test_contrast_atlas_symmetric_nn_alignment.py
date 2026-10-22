from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


def _fixture(root: Path) -> tuple[Path, Path]:
    collection, embeddings = root / "collection", root / "embeddings"
    collection.mkdir()
    rows, contrasts, sources = [], [], []
    directions = {"S1": 1.0, "S2": 1.0, "S3": -1.0}
    # Deliberately non-lexicographic to verify canonical source-pair keys.
    for study, contrast in (("S2", "C2"), ("S1", "C1"), ("S3", "C3")):
        contrasts.append(
            {"study_id": study, "source_family": study, "contrast_id": contrast}
        )
        sources.append({"study_id": study, "biological_units": 8})
        direction = directions[study]
        for suffix, base_beta, identity in (
            ("kp", 1.0, "library_supported"),
            ("kn", -1.0, "library_supported"),
            ("up", 1.0, "unresolved_entity"),
            ("un", -1.0, "unresolved_entity"),
        ):
            rows.append(
                {
                    "study_id": study,
                    "source_family": study,
                    "entity_id": f"{study}::{suffix}",
                    "contrast_id": contrast,
                    "beta": direction * base_beta,
                    "se": 0.5,
                    "identity_status": identity,
                }
            )
        directory = embeddings / study
        directory.mkdir(parents=True)
        ids = np.asarray([f"{study}::{x}" for x in ("kp", "kn", "up", "un")])
        vectors = np.eye(4, dtype=np.float32)
        np.savez_compressed(directory / "entity_embeddings.npz", entity_id=ids, embeddings=vectors)
    pd.DataFrame(rows).to_csv(
        collection / "response_entities.tsv.gz", sep="\t", index=False, compression="gzip"
    )
    pd.DataFrame(contrasts).to_csv(collection / "contrast_manifest.tsv", sep="\t", index=False)
    pd.DataFrame(sources).to_csv(collection / "source_manifest.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "relation_direction": "registered",
                "left_study_id": "S1",
                "left_contrast_id": "C1",
                "right_study_id": "S2",
                "right_contrast_id": "C2",
                "relation_class": "synthetic",
                "expected_signed_direction": "unspecified",
                "evidence_boundary": "synthetic",
            }
        ]
    ).to_csv(collection / "registered_relations.tsv", sep="\t", index=False)
    (collection / "report.json").write_text(json.dumps({"status": "synthetic"}), encoding="utf-8")
    return collection, embeddings


def test_symmetric_nn_alignment_freezes_graph_and_preserves_candidate_universe(
    tmp_path: Path,
) -> None:
    collection, embeddings = _fixture(tmp_path)
    output = tmp_path / "output"
    script = Path(__file__).with_name("score_contrast_atlas_symmetric_nn_alignment.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--collection",
            str(collection),
            "--embeddings-root",
            str(embeddings),
            "--method-id",
            "synthetic",
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    scores = pd.read_csv(output / "fixed_candidate_scores.tsv.gz", sep="\t")
    assert len(scores) == 18
    assert set(scores["adjudication_status"]) == {"unadjudicated"}
    query = scores.loc[
        scores["query_study_id"].eq("S1")
        & scores["query_contrast_id"].eq("C1")
        & scores["evidence_world"].eq("all")
    ]
    same = query.loc[query["candidate_study_id"].eq("S2")].iloc[0]
    opposite = query.loc[query["candidate_study_id"].eq("S3")].iloc[0]
    assert same["similar_response_score"] > 0
    assert same["opposite_response_score"] == 0
    assert same["similar_response_absolute_rank"] == 1
    assert opposite["opposite_response_score"] > 0
    assert opposite["similar_response_score"] == 0
    assert opposite["opposite_response_absolute_rank"] == 1
    assert same["query_to_candidate_edge_coverage"] == 1
    assert same["candidate_to_query_edge_coverage"] == 1

    sensitivity = pd.read_csv(
        output / "deterministic_profile_reassignment_scores.tsv.gz", sep="\t"
    )
    assert len(sensitivity) == len(scores)
    mapping = pd.read_csv(output / "deterministic_profile_reassignment.tsv.gz", sep="\t")
    assert not mapping["fixed_point"].astype(bool).any()
    assert not mapping.duplicated(["study_id", "recipient_entity_id"]).any()
    relation = pd.read_csv(output / "registered_relation_absolute_ranks.tsv", sep="\t")
    assert len(relation) == 6  # observed + sensitivity, each in three worlds.
    assert set(relation["pass_fail"]) == {"NOT_ADJUDICATED"}
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["parameters"]["neighbors_each_direction"] == 1
    assert report["parameters"]["minimum_cosine"] == 0.70
    assert "non-calibrated sensitivity" in report["sensitivity_boundary"]


def test_no_thresholded_edge_is_retained_as_zero_not_deleted(tmp_path: Path) -> None:
    collection, embeddings = _fixture(tmp_path)
    archive = embeddings / "S3" / "entity_embeddings.npz"
    with np.load(archive, allow_pickle=False) as packed:
        ids = np.asarray(packed["entity_id"])
    # Orthogonal to every vector in the four-dimensional S1/S2 universe.
    vectors = np.zeros((4, 5), dtype=np.float32)
    vectors[:, 4] = 1.0
    # Expand the other studies to the same dimension.
    for study in ("S1", "S2"):
        path = embeddings / study / "entity_embeddings.npz"
        with np.load(path, allow_pickle=False) as packed:
            other_ids = np.asarray(packed["entity_id"])
            old = np.asarray(packed["embeddings"])
        expanded = np.column_stack([old, np.zeros(len(old), dtype=np.float32)])
        np.savez_compressed(path, entity_id=other_ids, embeddings=expanded)
    np.savez_compressed(archive, entity_id=ids, embeddings=vectors)
    output = tmp_path / "output"
    script = Path(__file__).with_name("score_contrast_atlas_symmetric_nn_alignment.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--collection",
            str(collection),
            "--embeddings-root",
            str(embeddings),
            "--method-id",
            "synthetic_no_edge",
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    scores = pd.read_csv(output / "fixed_candidate_scores.tsv.gz", sep="\t")
    pair = scores.loc[
        scores["query_study_id"].eq("S1")
        & scores["candidate_study_id"].eq("S3")
        & scores["evidence_world"].eq("all")
    ].iloc[0]
    assert bool(pair["score_available"])
    assert pair["query_to_candidate_edge_coverage"] == 0
    assert pair["candidate_to_query_edge_coverage"] == 0
    assert pair["unsigned_response_score"] == 0
    assert pair["composition_score"] == 0


def test_spectrally_ineligible_entity_remains_in_response_denominator(tmp_path: Path) -> None:
    collection, embeddings = _fixture(tmp_path)
    archive = embeddings / "S1" / "entity_embeddings.npz"
    with np.load(archive, allow_pickle=False) as packed:
        ids = np.asarray(packed["entity_id"])
        vectors = np.asarray(packed["embeddings"])
    np.savez_compressed(archive, entity_id=ids[:3], embeddings=vectors[:3])
    output = tmp_path / "output"
    script = Path(__file__).with_name("score_contrast_atlas_symmetric_nn_alignment.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--collection",
            str(collection),
            "--embeddings-root",
            str(embeddings),
            "--method-id",
            "synthetic_ineligible",
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    scores = pd.read_csv(output / "fixed_candidate_scores.tsv.gz", sep="\t")
    pair = scores.loc[
        scores["query_study_id"].eq("S1")
        & scores["candidate_study_id"].eq("S2")
        & scores["evidence_world"].eq("all")
    ].iloc[0]
    assert pair["query_universe_entities"] == 4
    assert pair["query_to_candidate_edge_coverage"] == 0.75
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    s1 = next(source for source in report["embedding_sources"] if source["study_id"] == "S1")
    assert s1["required_response_entities"] == 4
    assert s1["embedded_response_entities"] == 3
    assert s1["spectrally_ineligible_response_entities"] == 1
