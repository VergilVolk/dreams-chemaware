from __future__ import annotations

import numpy as np
import pandas as pd

from build_signed_spectral_measure_sketches import build, validate_input
from query_signed_spectral_measure_sketches import score_query


def test_direction_is_carried_by_response_not_composition() -> None:
    embeddings = np.eye(3, dtype=np.float32)
    rows = []
    # Studies S1 and S2 have the same response direction. S3 reverses it.
    for study, source, reverse in (("S1", "F1", False), ("S2", "F2", False), ("S3", "F3", True)):
        for group in ("A", "B"):
            for replicate in range(5):
                sample = f"{study}_{group}_{replicate}"
                for entity, embedding_row, status in (
                    ("e0", 0, "unresolved_entity"),
                    ("e1", 1, "library_supported"),
                    ("e2", 2, "unresolved_entity"),
                ):
                    sign = -1 if reverse else 1
                    shift = sign * (1 if entity == "e0" else -1 if entity == "e1" else 0)
                    base = 10 + replicate * 0.01
                    value = base + (shift if group == "B" else 0)
                    rows.append({
                        "study_id": study, "source_family": source,
                        "contrast_id": "B_vs_A", "sample_id": sample, "group": group,
                        "entity_id": entity, "value": value,
                        "embedding_row": embedding_row, "identity_status": status,
                        "qc_pass": True,
                    })
    frame = validate_input(pd.DataFrame(rows), embeddings)
    effects, sketches, manifest_rows = build(
        frame, embeddings, transform="none", min_group_n=5,
        min_prevalence=0.2, se_floor=1e-6,
    )
    manifest = pd.DataFrame(manifest_rows)
    scores = score_query(
        manifest, sketches, query_row=0, allow_same_source=False, minimum_entities=1,
    )
    all_world = scores.loc[scores["evidence_world"].eq("all")]
    same = all_world.loc[all_world["study_id"].eq("S2"), "signed_response_score"].iloc[0]
    reverse = all_world.loc[all_world["study_id"].eq("S3"), "signed_response_score"].iloc[0]
    assert same > reverse
    assert len(effects) == 9
    assert manifest["n_entities_unresolved"].eq(2).all()


def test_empty_known_world_abstains_instead_of_receiving_a_rank() -> None:
    manifest = pd.DataFrame([
        {"study_id": "S1", "source_family": "F1", "contrast_id": "c1",
         "n_entities_all": 20, "n_entities_known": 0, "n_entities_unresolved": 20},
        {"study_id": "S2", "source_family": "F2", "contrast_id": "c2",
         "n_entities_all": 20, "n_entities_known": 0, "n_entities_unresolved": 20},
    ])
    sketches = {}
    for world in ("all", "known", "unresolved"):
        value = np.ones((2, 2), dtype=np.float32) if world != "known" else np.zeros((2, 2), dtype=np.float32)
        sketches[f"positive_{world}"] = value
        sketches[f"negative_{world}"] = value
    scores = score_query(manifest, sketches, query_row=0, allow_same_source=False)
    known = scores.loc[scores["evidence_world"].eq("known")]
    assert not known["answerable"].any()
    assert known["rank"].isna().all()
    assert known["signed_response_score"].isna().all()
