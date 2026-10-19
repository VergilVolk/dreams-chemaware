#!/usr/bin/env python3
"""Score a frozen Contrast Atlas collection using precomputed embeddings only.

The scorer never loads a spectral model.  It accepts one unit-normalised entity
embedding archive per study and applies the same signed-measure mathematics and
fixed source-disjoint candidate universe used by the transparent fragment-bin
baseline.  A single interface therefore supports official DreaMS, Noise V1,
or another already-frozen encoder without changing the response collection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from run_contrast_atlas_sparse_fragment_measure_baseline import (
    WORLDS,
    registered_relation_audit,
    sha256,
    truth_blind_scores,
    unweighted_centroid,
    weighted_centroid,
)


def normalise_entity_id(value: object) -> str:
    return str(value).strip()


def resolve_embedding_jobs(args: argparse.Namespace, studies: list[str]) -> pd.DataFrame:
    if bool(args.embedding_manifest) == bool(args.embeddings_root):
        raise ValueError("provide exactly one of --embedding-manifest and --embeddings-root")
    if args.embedding_manifest:
        jobs = pd.read_csv(args.embedding_manifest, sep="\t", dtype=str)
        required = {"study_id", "embedding_path", "method_id"}
        if missing := required - set(jobs):
            raise ValueError(f"embedding manifest lacks {sorted(missing)}")
        jobs = jobs.copy()
        jobs["study_id"] = jobs["study_id"].map(normalise_entity_id)
        if jobs["study_id"].duplicated().any():
            raise ValueError("embedding manifest has duplicate study_id rows")
        if jobs["method_id"].nunique() != 1:
            raise ValueError("one scoring run must contain exactly one method_id")
    else:
        if not args.method_id:
            raise ValueError("--method-id is required with --embeddings-root")
        jobs = pd.DataFrame(
            {
                "study_id": studies,
                "embedding_path": [
                    str(args.embeddings_root / study / "entity_embeddings.npz")
                    for study in studies
                ],
                "method_id": args.method_id,
            }
        )
    if set(jobs["study_id"]) != set(studies):
        missing = sorted(set(studies) - set(jobs["study_id"]))
        extra = sorted(set(jobs["study_id"]) - set(studies))
        raise ValueError(f"embedding studies differ from collection; missing={missing}, extra={extra}")
    return jobs.set_index("study_id").loc[studies].reset_index()


def load_embedding_archive(
    path: Path,
    required_entities: set[str],
    norm_tolerance: float,
) -> tuple[list[str], np.ndarray, dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as packed:
        if set(packed.files) != {"entity_id", "embeddings"}:
            raise ValueError(
                f"{path} must contain exactly entity_id and embeddings, got {sorted(packed.files)}"
            )
        entity_ids = [normalise_entity_id(value) for value in packed["entity_id"]]
        embeddings = np.asarray(packed["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(entity_ids):
        raise ValueError(f"embedding shape does not match entity IDs in {path}")
    if len(entity_ids) != len(set(entity_ids)):
        raise ValueError(f"duplicate entity_id values in {path}")
    if embeddings.shape[1] < 1 or not np.isfinite(embeddings).all():
        raise ValueError(f"invalid embedding matrix in {path}")
    norms = np.linalg.norm(embeddings, axis=1)
    maximum_norm_error = float(np.max(np.abs(norms - 1.0))) if len(norms) else float("inf")
    if maximum_norm_error > norm_tolerance:
        raise ValueError(
            f"embeddings are not unit-normalised in {path}; max error={maximum_norm_error}"
        )
    lookup = {entity: row for row, entity in enumerate(entity_ids)}
    missing = required_entities - set(lookup)
    if missing:
        raise ValueError(
            f"{len(missing)} frozen response entities lack embeddings in {path}: "
            f"{sorted(missing)[:10]}"
        )
    selected_ids = sorted(required_entities)
    selected = embeddings[[lookup[entity] for entity in selected_ids]]
    return selected_ids, selected, {
        "embedding_path": str(path),
        "embedding_sha256": sha256(path),
        "archive_entities": len(entity_ids),
        "required_response_entities": len(required_entities),
        "extra_archive_entities_not_scored": len(entity_ids) - len(required_entities),
        "embedding_dimensions": int(embeddings.shape[1]),
        "maximum_unit_norm_error": maximum_norm_error,
    }


def build_sketches(
    contrasts: pd.DataFrame,
    entities: pd.DataFrame,
    study_embeddings: dict[str, tuple[list[str], np.ndarray]],
) -> dict[tuple[int, str], dict[str, sparse.csr_matrix | int]]:
    sketches: dict[tuple[int, str], dict[str, sparse.csr_matrix | int]] = {}
    for contrast_index, contrast in contrasts.iterrows():
        study_id = str(contrast["study_id"])
        contrast_id = str(contrast["contrast_id"])
        embedding_ids, embedding = study_embeddings[study_id]
        row_lookup = {entity: row for row, entity in enumerate(embedding_ids)}
        block = entities.loc[
            entities["study_id"].astype(str).eq(study_id)
            & entities["contrast_id"].astype(str).eq(contrast_id)
        ].copy()
        if block.empty:
            raise ValueError(f"contrast has no response entities: {study_id}/{contrast_id}")
        if block["entity_id"].duplicated().any():
            raise ValueError(f"duplicate response entity: {study_id}/{contrast_id}")
        block["embedding_row"] = block["entity_id"].map(row_lookup)
        if block["embedding_row"].isna().any():
            raise ValueError(f"effect-to-embedding join incomplete: {study_id}/{contrast_id}")
        beta = block["beta"].to_numpy(float)
        se = block["se"].to_numpy(float)
        denominator = np.sqrt(beta * beta + se * se)
        signed_weight = np.divide(
            beta, denominator, out=np.zeros_like(beta), where=denominator > 0
        )
        for world in WORLDS:
            if world == "all":
                mask = np.ones(len(block), dtype=bool)
            elif world == "known":
                mask = block["identity_status"].astype(str).eq("library_supported").to_numpy()
            else:
                mask = block["identity_status"].astype(str).eq("unresolved_entity").to_numpy()
            rows = block.loc[mask, "embedding_row"].to_numpy(int)
            selected = sparse.csr_matrix(embedding[rows], dtype=np.float32)
            weights = signed_weight[mask]
            sketches[(contrast_index, world)] = {
                "n_entities": int(mask.sum()),
                "positive": weighted_centroid(selected, np.maximum(weights, 0)),
                "negative": weighted_centroid(selected, np.maximum(-weights, 0)),
                "composition": unweighted_centroid(selected),
            }
    return sketches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--embedding-manifest", type=Path)
    source.add_argument("--embeddings-root", type=Path)
    parser.add_argument("--method-id")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--unit-norm-tolerance", type=float, default=2e-5)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.unit_norm_tolerance <= 0:
        raise ValueError("unit norm tolerance must be positive")

    entities_path = args.collection / "response_entities.tsv.gz"
    contrast_path = args.collection / "contrast_manifest.tsv"
    source_path = args.collection / "source_manifest.tsv"
    relations_path = args.collection / "registered_relations.tsv"
    collection_report_path = args.collection / "report.json"
    for path in (entities_path, contrast_path, source_path, relations_path, collection_report_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    entities = pd.read_csv(entities_path, sep="\t", dtype={"entity_id": str})
    entities["entity_id"] = entities["entity_id"].map(normalise_entity_id)
    contrasts = pd.read_csv(contrast_path, sep="\t").reset_index(drop=True)
    studies = list(contrasts["study_id"].astype(str).drop_duplicates())
    jobs = resolve_embedding_jobs(args, studies)
    method_id = str(jobs["method_id"].iloc[0])

    study_embeddings: dict[str, tuple[list[str], np.ndarray]] = {}
    embedding_reports: list[dict[str, object]] = []
    dimensions: set[int] = set()
    for job in jobs.to_dict("records"):
        study_id = str(job["study_id"])
        required = set(entities.loc[entities["study_id"].astype(str).eq(study_id), "entity_id"])
        ids, embedding, report = load_embedding_archive(
            Path(job["embedding_path"]), required, args.unit_norm_tolerance
        )
        dimensions.add(int(embedding.shape[1]))
        study_embeddings[study_id] = (ids, embedding)
        embedding_reports.append(
            {
                "study_id": study_id,
                "method_id": method_id,
                **{
                    key: value
                    for key, value in job.items()
                    if key not in {"study_id", "embedding_path", "method_id"}
                    and pd.notna(value)
                },
                **report,
            }
        )
    if len(dimensions) != 1:
        raise ValueError(f"embedding dimensions differ across studies: {sorted(dimensions)}")

    sketches = build_sketches(contrasts, entities, study_embeddings)
    # Complete truth-blind scoring before reading any registered relation.
    scores = truth_blind_scores(contrasts, sketches)
    args.output_dir.mkdir(parents=True)
    scores_path = args.output_dir / "fixed_candidate_scores.tsv.gz"
    scores.to_csv(scores_path, sep="\t", index=False, compression="gzip")

    relations = pd.read_csv(relations_path, sep="\t", dtype=str)
    relation_audit = registered_relation_audit(scores, relations)
    relation_path = args.output_dir / "registered_relation_absolute_ranks.tsv"
    relation_audit.to_csv(relation_path, sep="\t", index=False)

    sources = pd.read_csv(source_path, sep="\t")
    available = scores["score_available"].astype(bool)
    report = {
        "schema": "contrast_atlas_precomputed_embedding_measure_score_v1",
        "status": "DEVELOPMENT_CONSUMED_PRECOMPUTED_EMBEDDING_SCORE_COMPLETE",
        "data_role": "development_consumed",
        "method_id": method_id,
        "method": {
            "spectral_encoder_execution": "none; precomputed embeddings only",
            "required_npz_arrays": ["entity_id", "embeddings"],
            "entity_embedding_requirement": "finite and unit-normalised",
            "response_weight": "beta/sqrt(beta^2+se^2)",
            "positive_negative_measure": "absolute response-weighted mean of unit embeddings; centroids are not renormalised",
            "scores": "signed similar, signed opposite, unsigned response, and composition",
        },
        "denominators": {
            "independent_studies": int(contrasts["study_id"].nunique()),
            "source_families": int(contrasts["source_family"].nunique()),
            "study_level_contrasts": len(contrasts),
            "biological_units_across_sources": int(sources["biological_units"].sum()),
            "eligible_response_rows": len(entities),
            "fixed_source_disjoint_candidate_rows_per_world": int(len(scores) / len(WORLDS)),
            "fixed_source_disjoint_candidate_rows_total": len(scores),
            "score_available_rows": int(available.sum()),
            "score_unavailable_rows_retained": int((~available).sum()),
            "registered_directed_relevance_relations": len(relations),
            "registered_signed_direction_truth_relations": int(
                (~relations["expected_signed_direction"].str.lower().eq("unspecified")).sum()
            ),
            "embedding_dimensions": next(iter(dimensions)),
        },
        "truth_boundary": {
            "scoring": "registered relations were not read until after all fixed-candidate scores and ranks were computed",
            "unregistered_pairs": "unadjudicated, never negative",
            "registered_unspecified_direction": "descriptive absolute ranks only; no PASS/FAIL",
        },
        "embedding_sources": embedding_reports,
        "input_sha256": {
            str(entities_path): sha256(entities_path),
            str(contrast_path): sha256(contrast_path),
            str(source_path): sha256(source_path),
            str(relations_path): sha256(relations_path),
            str(collection_report_path): sha256(collection_report_path),
            **(
                {str(args.embedding_manifest): sha256(args.embedding_manifest)}
                if args.embedding_manifest
                else {}
            ),
        },
        "claim_boundary": [
            "This scorer compares already-frozen embedding methods; it does not encode spectra or train a model.",
            "The frozen response collection and source-disjoint candidate graph are unchanged across methods.",
            "All collection sources are development/consumed.",
            "Unresolved means not library-supported, not a unique unknown metabolite.",
            "No unregistered pair is a negative example.",
        ],
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    checksums = {
        scores_path.name: sha256(scores_path),
        relation_path.name: sha256(relation_path),
        report_path.name: sha256(report_path),
    }
    (args.output_dir / "checksums.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in checksums.items()),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
