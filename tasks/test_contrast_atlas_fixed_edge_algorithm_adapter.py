from pathlib import Path

import numpy as np
import pandas as pd

from encode_fixed_edge_endpoints_checkpoint import requested_entities
from run_contrast_atlas_open_local_edge_enrichment import apply_algorithm_edge_scores
from score_contrast_atlas_fixed_edges_embeddings import load_method


KEYS = [
    "left_study_id",
    "right_study_id",
    "evidence_world",
    "left_entity_id",
    "right_entity_id",
]


def edge_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "left_study_id": ["A", "A"],
            "right_study_id": ["B", "B"],
            "evidence_world": ["all", "unresolved"],
            "left_entity_id": ["A::1", "A::1"],
            "right_entity_id": ["B::2", "B::2"],
            "chemical_similarity": [0.8, 0.8],
        }
    )


def test_requested_entities_uses_exact_frozen_axis(tmp_path: Path) -> None:
    path = tmp_path / "edges.tsv"
    edge_table().to_csv(path, sep="\t", index=False)
    assert requested_entities(path) == {"A": {"A::1"}, "B": {"B::2"}}


def test_algorithm_scores_reorder_clip_and_zero_missing(tmp_path: Path) -> None:
    scores = edge_table()[KEYS].iloc[::-1].reset_index(drop=True)
    scores["model_score"] = [np.nan, -0.2]
    path = tmp_path / "scores.tsv"
    scores.to_csv(path, sep="\t", index=False)
    replaced, provenance = apply_algorithm_edge_scores(
        edge_table(), path, "model_score", "zero"
    )
    assert replaced["chemical_similarity"].tolist() == [0.0, 0.0]
    assert replaced["base_chemical_similarity"].tolist() == [0.8, 0.8]
    assert replaced["algorithm_score_available"].tolist() == [True, False]
    assert provenance["edge_rows_available"] == 1


def test_embedding_cache_merge_accepts_identical_overlap(tmp_path: Path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    np.savez_compressed(
        first,
        entity_id=np.asarray(["A::1"]),
        embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
    )
    np.savez_compressed(
        second,
        entity_id=np.asarray(["A::1", "B::2"]),
        embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    cache = load_method([first, second])
    assert sorted(cache) == ["A::1", "B::2"]
