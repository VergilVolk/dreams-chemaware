from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


def _write_collection(root: Path) -> None:
    root.mkdir()
    response_rows: list[dict[str, object]] = []
    contrasts: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    for study, contrast in (("S1", "C1"), ("S2", "C2"), ("S3", "C3")):
        contrasts.append(
            {
                "study_id": study,
                "source_family": study,
                "contrast_id": contrast,
                "independent_units": 8,
            }
        )
        sources.append({"study_id": study, "biological_units": 8})
        for suffix, beta, identity in (
            ("kp", 1.0, "library_supported"),
            ("kn", -1.0, "library_supported"),
            ("up", 1.0, "unresolved_entity"),
            ("un", -1.0, "unresolved_entity"),
        ):
            response_rows.append(
                {
                    "study_id": study,
                    "source_family": study,
                    "entity_id": f"{study}::{suffix}",
                    "contrast_id": contrast,
                    "beta": beta,
                    "se": 0.5,
                    "identity_status": identity,
                }
            )
    pd.DataFrame(response_rows).to_csv(
        root / "response_entities.tsv.gz", sep="\t", index=False, compression="gzip"
    )
    pd.DataFrame(contrasts).to_csv(root / "contrast_manifest.tsv", sep="\t", index=False)
    pd.DataFrame(sources).to_csv(root / "source_manifest.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "relation_direction": "registered",
                "left_study_id": "S1",
                "left_contrast_id": "C1",
                "right_study_id": "S2",
                "right_contrast_id": "C2",
                "relation_class": "synthetic_same_response",
                "expected_signed_direction": "unspecified",
                "evidence_boundary": "synthetic test only",
            }
        ]
    ).to_csv(root / "registered_relations.tsv", sep="\t", index=False)
    (root / "report.json").write_text(json.dumps({"status": "synthetic"}), encoding="utf-8")


def _write_embeddings(root: Path) -> None:
    for study in ("S1", "S2", "S3"):
        directory = root / study
        directory.mkdir(parents=True)
        entity_id = np.asarray(
            [f"{study}::kp", f"{study}::kn", f"{study}::up", f"{study}::un"]
        )
        if study == "S3":
            embedding = np.asarray([[0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.float32)
        else:
            embedding = np.asarray([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
        np.savez_compressed(directory / "entity_embeddings.npz", entity_id=entity_id, embeddings=embedding)


def test_precomputed_embeddings_use_fixed_candidates_and_open_truth_last(tmp_path: Path) -> None:
    collection = tmp_path / "collection"
    embeddings = tmp_path / "embeddings"
    output = tmp_path / "output"
    _write_collection(collection)
    _write_embeddings(embeddings)
    script = Path(__file__).with_name("score_contrast_atlas_precomputed_embedding_measures.py")
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--collection",
            str(collection),
            "--embeddings-root",
            str(embeddings),
            "--method-id",
            "synthetic_unit_encoder",
            "--output-dir",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    scores = pd.read_csv(output / "fixed_candidate_scores.tsv.gz", sep="\t")
    assert len(scores) == 18  # 3 queries x 2 source-disjoint candidates x 3 worlds.
    assert set(scores["adjudication_status"]) == {"unadjudicated"}
    assert not (scores["query_source_family"] == scores["candidate_source_family"]).any()
    query = scores.loc[
        scores["query_study_id"].eq("S1")
        & scores["query_contrast_id"].eq("C1")
        & scores["evidence_world"].eq("all")
    ]
    similar = query.loc[query["candidate_study_id"].eq("S2")].iloc[0]
    opposite = query.loc[query["candidate_study_id"].eq("S3")].iloc[0]
    assert similar["signed_similar_absolute_rank"] == 1
    assert opposite["signed_opposite_absolute_rank"] == 1
    relation = pd.read_csv(output / "registered_relation_absolute_ranks.tsv", sep="\t")
    assert len(relation) == 3
    assert set(relation["pass_fail"]) == {"NOT_ADJUDICATED"}
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["method_id"] == "synthetic_unit_encoder"
    assert report["method"]["spectral_encoder_execution"] == "none; precomputed embeddings only"


def test_precomputed_embeddings_reject_missing_response_entity(tmp_path: Path) -> None:
    collection = tmp_path / "collection"
    embeddings = tmp_path / "embeddings"
    output = tmp_path / "output"
    _write_collection(collection)
    _write_embeddings(embeddings)
    archive = embeddings / "S2" / "entity_embeddings.npz"
    with np.load(archive, allow_pickle=False) as packed:
        ids = np.asarray(packed["entity_id"][:-1])
        vectors = np.asarray(packed["embeddings"][:-1])
    np.savez_compressed(archive, entity_id=ids, embeddings=vectors)
    script = Path(__file__).with_name("score_contrast_atlas_precomputed_embedding_measures.py")
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--collection",
            str(collection),
            "--embeddings-root",
            str(embeddings),
            "--method-id",
            "synthetic_missing_entity",
            "--output-dir",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "response entities lack embeddings" in completed.stderr
