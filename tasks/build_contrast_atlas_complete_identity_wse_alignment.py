#!/usr/bin/env python3
"""Adapt complete-axis WSE scores to the frozen Atlas edge interface.

This stage is deliberately response blind.  It re-ranks every complete
query-by-target-study identity block separately in three descriptive evidence
worlds, emits one directional unique Top-1 decision when positive spectral
evidence exists, and leaves reciprocity filtering to the downstream response
retrieval implementation.  No beta, standard error, phenotype, contrast label,
or registered relation is read here.

The three worlds are diagnostics, not a formal unknown-information test:

* ``all``: every physicochemically compatible candidate;
* ``known``: both endpoints are strict common-library-supported entities;
* ``unresolved``: both endpoints are unresolved under that same policy.

Known and unresolved candidates are re-ranked inside their own worlds.  The
implementation never selects the all-world winner and filters it afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


WORLDS = ("all", "known", "unresolved")
KNOWN = "library_supported"
UNRESOLVED = "unresolved_entity"
EDGE_COLUMNS = [
    "left_study_id",
    "right_study_id",
    "evidence_world",
    "direction",
    "left_entity_id",
    "right_entity_id",
    "chemical_similarity",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def world_mask(pairs: pd.DataFrame, world: str) -> np.ndarray:
    query_status = pairs["query_identity_status"].astype(str).to_numpy()
    candidate_status = pairs["candidate_identity_status"].astype(str).to_numpy()
    if world == "all":
        return np.ones(len(pairs), dtype=bool)
    if world == "known":
        return (query_status == KNOWN) & (candidate_status == KNOWN)
    if world == "unresolved":
        return (query_status == UNRESOLVED) & (candidate_status == UNRESOLVED)
    raise ValueError(world)


def select_directional_top1(
    pairs: pd.DataFrame,
    score: np.ndarray,
    matched: np.ndarray,
    available: np.ndarray,
    world: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Select response-blind unique WSE Top-1 decisions inside one world."""
    if not (len(pairs) == len(score) == len(matched) == len(available)):
        raise ValueError("pair and WSE arrays are not aligned")
    eligible = world_mask(pairs, world) & available
    work = pairs.loc[eligible, [
        "query_block_index",
        "query_study_id",
        "query_entity_id",
        "candidate_study_id",
        "candidate_entity_id",
    ]].copy()
    work["chemical_similarity"] = score[eligible]
    work["matched_fragments"] = matched[eligible]

    rows: list[dict[str, object]] = []
    tied = 0
    no_positive_evidence = 0
    singleton = 0
    multiple = 0
    for _block_index, block in work.groupby("query_block_index", sort=False):
        if len(block) == 1:
            singleton += 1
        else:
            multiple += 1
        values = block["chemical_similarity"].to_numpy(float)
        top = float(np.max(values))
        winners = np.flatnonzero(values >= top - 1e-12)
        if len(winners) != 1:
            tied += 1
            continue
        chosen = block.iloc[int(winners[0])]
        if top <= 0 or int(chosen["matched_fragments"]) <= 0:
            no_positive_evidence += 1
            continue

        query_study = str(chosen["query_study_id"])
        candidate_study = str(chosen["candidate_study_id"])
        query_entity = str(chosen["query_entity_id"])
        candidate_entity = str(chosen["candidate_entity_id"])
        if query_study < candidate_study:
            left_study, right_study = query_study, candidate_study
            left_entity, right_entity = query_entity, candidate_entity
            direction = "left_to_right"
        else:
            left_study, right_study = candidate_study, query_study
            left_entity, right_entity = candidate_entity, query_entity
            direction = "right_to_left"
        rows.append({
            "left_study_id": left_study,
            "right_study_id": right_study,
            "evidence_world": world,
            "direction": direction,
            "left_entity_id": left_entity,
            "right_entity_id": right_entity,
            "chemical_similarity": top,
        })

    output = pd.DataFrame(rows, columns=EDGE_COLUMNS)
    summary = {
        "eligible_candidate_pairs": int(eligible.sum()),
        "nonempty_world_blocks": int(work["query_block_index"].nunique()),
        "singleton_world_blocks": singleton,
        "multiple_world_blocks": multiple,
        "tied_top1_abstentions": tied,
        "no_positive_spectral_evidence_abstentions": no_positive_evidence,
        "directional_top1_edges": int(len(output)),
    }
    return output, summary


def reciprocal_edge_summary(edges: pd.DataFrame) -> dict[str, int]:
    if edges.empty:
        return {"canonical_mutual_edges": 0}
    key = [
        "left_study_id",
        "right_study_id",
        "evidence_world",
        "left_entity_id",
        "right_entity_id",
    ]
    counts = edges.groupby(key, sort=False)["direction"].nunique()
    mutual = counts[counts.eq(2)]
    return {
        "canonical_mutual_edges": int(len(mutual)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--axis-dir",
        type=Path,
        default=Path(
            "data/validation/contrast_atlas_model_independent_identity_axis_v5_20261027"
        ),
    )
    parser.add_argument(
        "--wse-dir",
        type=Path,
        default=Path(
            "data/validation/contrast_atlas_complete_identity_wse_release_v1_20261029"
        ),
    )
    parser.add_argument(
        "--collection",
        type=Path,
        default=Path(
            "data/validation/contrast_atlas_open_response_collection_v9_frozenbase_chemprop_20261023"
        ),
    )
    parser.add_argument(
        "--source-alignment-report",
        type=Path,
        default=Path(
            "data/validation/contrast_atlas_sparse_local_alignment_v3_chemprop_20261023/report.json"
        ),
        help="Phenotype-blind source/MGF manifest reused only for endpoint lineage.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    paths = {
        "axis_report": args.axis_dir / "report.json",
        "pairs": args.axis_dir / "directed_complete_candidate_pairs.tsv.gz",
        "blocks": args.axis_dir / "query_target_source_blocks.tsv.gz",
        "registry": args.axis_dir / "observed_ion_registry.tsv.gz",
        "wse_report": args.wse_dir / "report.json",
        "wse": args.wse_dir / "wse_scores_aligned.npz",
        "response_entities": args.collection / "response_entities.tsv.gz",
        "source_alignment_report": args.source_alignment_report,
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    axis_report = json.loads(paths["axis_report"].read_text(encoding="utf-8"))
    wse_report = json.loads(paths["wse_report"].read_text(encoding="utf-8"))
    source_report = json.loads(
        paths["source_alignment_report"].read_text(encoding="utf-8")
    )
    if sha256(paths["pairs"]) != wse_report["alignment_contract"]["pair_axis_sha256"]:
        raise RuntimeError("WSE/pair-axis checksum mismatch")
    if sha256(paths["blocks"]) != wse_report["alignment_contract"]["block_axis_sha256"]:
        raise RuntimeError("WSE/block-axis checksum mismatch")
    if sha256(paths["response_entities"]) != source_report["provenance"]["response_entities_sha256"]:
        raise RuntimeError("source manifest and frozen response collection disagree")

    pairs = pd.read_csv(paths["pairs"], sep="\t", low_memory=False)
    compact = np.load(paths["wse"], allow_pickle=False)
    score = compact["wse_score"].astype(np.float64)
    matched = compact["wse_matched_fragments"].astype(np.int64)
    available = compact["wse_availability"].astype(bool)
    if len(pairs) != int(axis_report["denominators"]["directed_candidate_pairs"]):
        raise RuntimeError("complete candidate-pair denominator drift")
    if len(pairs) != len(score):
        raise RuntimeError("WSE array is not aligned to the complete pair axis")

    world_frames: list[pd.DataFrame] = []
    world_summaries: list[dict[str, object]] = []
    for world in WORLDS:
        selected, summary = select_directional_top1(
            pairs, score, matched, available, world
        )
        reciprocal = reciprocal_edge_summary(selected)
        world_frames.append(selected)
        world_summaries.append({"evidence_world": world, **summary, **reciprocal})
    edges = pd.concat(world_frames, ignore_index=True)
    if edges.duplicated([
        "left_study_id",
        "right_study_id",
        "evidence_world",
        "direction",
        "left_entity_id",
        "right_entity_id",
    ]).any():
        raise RuntimeError("duplicate directional edge keys were produced")

    registry = pd.read_csv(paths["registry"], sep="\t", dtype=str)
    identity_counts = {
        str(key): int(value)
        for key, value in registry["identity_status"].value_counts().items()
    }

    # Report endpoint-status composition of reciprocal all-world edges without
    # consulting identity truth beyond the already frozen strict status field.
    status = registry[["study_id", "entity_id", "identity_status"]].drop_duplicates()
    status_lookup = status.set_index(["study_id", "entity_id"])["identity_status"]
    key = [
        "left_study_id", "right_study_id", "evidence_world",
        "left_entity_id", "right_entity_id",
    ]
    mutual_keys = (
        edges.groupby(key, sort=False)["direction"].nunique().loc[lambda x: x.eq(2)]
        .reset_index()[key]
    )
    if len(mutual_keys):
        mutual_keys["left_identity_status"] = [
            status_lookup.loc[(str(study), str(entity))]
            for study, entity in zip(
                mutual_keys["left_study_id"], mutual_keys["left_entity_id"]
            )
        ]
        mutual_keys["right_identity_status"] = [
            status_lookup.loc[(str(study), str(entity))]
            for study, entity in zip(
                mutual_keys["right_study_id"], mutual_keys["right_entity_id"]
            )
        ]
        mutual_keys["endpoint_composition"] = np.where(
            mutual_keys["left_identity_status"].eq(KNOWN)
            & mutual_keys["right_identity_status"].eq(KNOWN),
            "known_known",
            np.where(
                mutual_keys["left_identity_status"].eq(UNRESOLVED)
                & mutual_keys["right_identity_status"].eq(UNRESOLVED),
                "unresolved_unresolved",
                "known_unresolved",
            ),
        )
        composition = (
            mutual_keys.groupby(["evidence_world", "endpoint_composition"])
            .size().rename("canonical_mutual_edges").reset_index().to_dict("records")
        )
    else:
        composition = []

    args.output_dir.mkdir(parents=True)
    edge_path = args.output_dir / "source_pair_frozen_edges.tsv.gz"
    edges.to_csv(
        edge_path,
        sep="\t",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    report = {
        "schema": "contrast_atlas_complete_identity_wse_alignment_v1",
        "status": "DEVELOPMENT_CONSUMED_COMPLETE_IDENTITY_WSE_ALIGNMENT_FROZEN",
        "data_role": "development_consumed",
        "chemical_graph": (
            "complete model-independent observed-precursor 10-ppm axis; "
            "unique positive-evidence WSE Top-1 selected separately per evidence world; "
            "downstream response retrieval retains only reciprocal decisions"
        ),
        "method": {
            "score": "weighted_spectral_entropy",
            "worlds": {
                "all": "all physicochemically compatible candidate endpoints",
                "known": "both endpoints common-policy library_supported",
                "unresolved": "both endpoints common-policy unresolved_entity",
            },
            "selection": (
                "unique maximum WSE within the world; exact score ties abstain; "
                "top WSE and matched-fragment count must both be positive"
            ),
            "response_fields_read": False,
            "registered_relations_read": False,
            "phenotype_read": False,
        },
        "denominators": {
            "studies": int(axis_report["denominators"]["studies"]),
            "entities": int(axis_report["denominators"]["entities"]),
            "query_target_source_blocks": int(
                axis_report["denominators"]["query_target_source_blocks"]
            ),
            "complete_directed_candidate_pairs": int(len(pairs)),
            "identity_status": identity_counts,
            "frozen_directional_edges": int(len(edges)),
            "worlds": world_summaries,
            "mutual_endpoint_composition": composition,
        },
        "sources": source_report["sources"],
        "claim_boundary": [
            "These are response-blind same-observed-precursor identity hypotheses, not confirmed molecular identities.",
            "The all/known/unresolved worlds are descriptive and do not by themselves establish an unresolved-information increment.",
            "Strict common-policy library support is sparse, so known and unresolved worlds have very different evidence opportunities.",
            "No different-mass chemical relation is admitted by this identity track.",
            "All sources are development/consumed; this is not sealed external biological validation.",
        ],
        "provenance": {
            "edges": sha256(edge_path),
            "response_entities_sha256": sha256(paths["response_entities"]),
            "axis_report_sha256": sha256(paths["axis_report"]),
            "complete_pair_axis_sha256": sha256(paths["pairs"]),
            "query_block_axis_sha256": sha256(paths["blocks"]),
            "observed_ion_registry_sha256": sha256(paths["registry"]),
            "wse_report_sha256": sha256(paths["wse_report"]),
            "wse_compact_sha256": sha256(paths["wse"]),
            "source_alignment_report_sha256": sha256(paths["source_alignment_report"]),
            "implementation_sha256": sha256(Path(__file__)),
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "checksums.sha256").write_text(
        f"{sha256(edge_path)}  {edge_path.name}\n"
        f"{sha256(report_path)}  {report_path.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(report["denominators"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
