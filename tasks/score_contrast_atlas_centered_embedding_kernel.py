#!/usr/bin/env python3
"""Apply the centred response kernel to frozen, precomputed spectral embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from run_contrast_atlas_centered_response_kernel import (
    WORLDS,
    audit_registered,
    centred_response_vector,
    score_candidates,
    stable_seed,
    stratified_permutation,
)
from run_contrast_atlas_sparse_fragment_measure_baseline import sha256, unweighted_centroid
from score_contrast_atlas_precomputed_embedding_measures import (
    load_embedding_archive,
    normalise_entity_id,
    resolve_embedding_jobs,
)


def build_prepared(
    contrasts: pd.DataFrame,
    entities: pd.DataFrame,
    embeddings: dict[str, tuple[list[str], np.ndarray]],
) -> tuple[
    dict[tuple[int, str], dict[str, object]],
    dict[tuple[int, str], tuple[sparse.csr_matrix, np.ndarray, np.ndarray]],
]:
    sketches: dict[tuple[int, str], dict[str, object]] = {}
    prepared: dict[tuple[int, str], tuple[sparse.csr_matrix, np.ndarray, np.ndarray]] = {}
    for ci, contrast in contrasts.iterrows():
        study_id = str(contrast["study_id"])
        contrast_id = str(contrast["contrast_id"])
        ids, embedding = embeddings[study_id]
        lookup = {entity: index for index, entity in enumerate(ids)}
        block = entities.loc[
            entities["study_id"].astype(str).eq(study_id)
            & entities["contrast_id"].astype(str).eq(contrast_id)
        ].copy()
        block["embedding_row"] = block["entity_id"].map(lookup)
        if block.empty or block["embedding_row"].isna().any() or block["entity_id"].duplicated().any():
            raise ValueError(f"invalid effect-to-embedding join: {study_id}/{contrast_id}")
        beta = block["beta"].to_numpy(float)
        se = block["se"].to_numpy(float)
        denominator = np.sqrt(beta * beta + se * se)
        weights = np.divide(beta, denominator, out=np.zeros_like(beta), where=denominator > 0)
        for world in WORLDS:
            if world == "all":
                mask = np.ones(len(block), dtype=bool)
            elif world == "known":
                mask = block["identity_status"].astype(str).eq("library_supported").to_numpy()
            else:
                mask = block["identity_status"].astype(str).eq("unresolved_entity").to_numpy()
            selected = sparse.csr_matrix(
                embedding[block.loc[mask, "embedding_row"].to_numpy(int)], dtype=np.float32
            )
            response, raw_norm, effective_n = centred_response_vector(selected, weights[mask])
            sketches[(ci, world)] = {
                "n_entities": int(mask.sum()),
                "effective_n": effective_n,
                "raw_response_norm": raw_norm,
                "response": response,
                "composition": unweighted_centroid(selected),
            }
            prepared[(ci, world)] = (
                selected,
                weights[mask],
                block.loc[mask, "identity_status"].astype(str).to_numpy(),
            )
    return sketches, prepared


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection", type=Path,
        default=Path("data/validation/contrast_atlas_open_response_collection_v2_20261019"),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--embedding-manifest", type=Path)
    source.add_argument("--embeddings-root", type=Path)
    parser.add_argument("--method-id")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--unit-norm-tolerance", type=float, default=2e-5)
    parser.add_argument("--minimum-entities", type=int, default=10)
    parser.add_argument("--minimum-effective-n", type=float, default=5.0)
    parser.add_argument("--null-replicates", type=int, default=31)
    parser.add_argument("--null-seed", type=int, default=20261022)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    entities_path = args.collection / "response_entities.tsv.gz"
    contrasts_path = args.collection / "contrast_manifest.tsv"
    relations_path = args.collection / "registered_relations.tsv"
    entities = pd.read_csv(entities_path, sep="\t", dtype={"entity_id": str})
    entities["entity_id"] = entities["entity_id"].map(normalise_entity_id)
    contrasts = pd.read_csv(contrasts_path, sep="\t").reset_index(drop=True)
    studies = list(contrasts["study_id"].astype(str).drop_duplicates())
    jobs = resolve_embedding_jobs(args, studies)
    method_id = str(jobs["method_id"].iloc[0])

    study_embeddings = {}
    embedding_reports = []
    dimensions = set()
    for job in jobs.to_dict("records"):
        study_id = str(job["study_id"])
        required = set(entities.loc[entities["study_id"].astype(str).eq(study_id), "entity_id"])
        ids, embedding, details = load_embedding_archive(
            Path(job["embedding_path"]), required, args.unit_norm_tolerance
        )
        study_embeddings[study_id] = (ids, embedding)
        dimensions.add(int(embedding.shape[1]))
        embedding_reports.append({"study_id": study_id, "method_id": method_id, **details})
    if len(dimensions) != 1:
        raise ValueError("embedding dimensions differ across sources")

    sketches, prepared = build_prepared(contrasts, entities, study_embeddings)
    scores = score_candidates(
        contrasts, sketches, args.minimum_entities, args.minimum_effective_n
    )
    args.output_dir.mkdir(parents=True)
    scores_path = args.output_dir / "fixed_candidate_scores.tsv.gz"
    scores.to_csv(scores_path, sep="\t", index=False, compression="gzip")
    score_hash_before_truth = sha256(scores_path)

    null_frames = []
    for replicate in range(args.null_replicates):
        null_sketches = {}
        for (ci, world), (selected, weights, strata) in prepared.items():
            rng = np.random.default_rng(stable_seed(args.null_seed, replicate, ci, world))
            permuted = stratified_permutation(weights, strata, rng)
            response, raw_norm, effective_n = centred_response_vector(selected, permuted)
            null_sketches[(ci, world)] = {
                "n_entities": int(selected.shape[0]),
                "effective_n": effective_n,
                "raw_response_norm": raw_norm,
                "response": response,
                "composition": sketches[(ci, world)]["composition"],
            }
        frame = score_candidates(
            contrasts, null_sketches, args.minimum_entities, args.minimum_effective_n
        )
        frame.insert(0, "null_replicate", replicate)
        null_frames.append(frame)
    null_scores = pd.concat(null_frames, ignore_index=True)
    null_path = args.output_dir / "stratified_response_reassignment_scores.tsv.gz"
    null_scores.to_csv(null_path, sep="\t", index=False, compression="gzip")

    # Truth is opened only after observed and reassigned score tables are frozen.
    relations = pd.read_csv(relations_path, sep="\t")
    audit = audit_registered(scores, relations)
    audit_path = args.output_dir / "registered_relation_absolute_ranks.tsv"
    audit.to_csv(audit_path, sep="\t", index=False)
    report = {
        "schema": "contrast_atlas_centered_precomputed_embedding_kernel_v1",
        "status": "DEVELOPMENT_CONSUMED_TRUTH_BLIND_RESULT",
        "method_id": method_id,
        "method": {
            "spectral_encoder_execution": "none; frozen precomputed embeddings only",
            "response_weight": "beta/sqrt(beta^2+se^2)",
            "centering": "within contrast and evidence world",
            "kernel": "cosine between unit centred response-associated embedding vectors",
            "null_replicates": args.null_replicates,
            "null_seed": args.null_seed,
        },
        "denominators": {
            "source_families": int(contrasts["source_family"].nunique()),
            "study_level_contrasts": int(len(contrasts)),
            "fixed_candidate_rows_per_world": int(len(scores) / len(WORLDS)),
            "score_available_rows": int(scores["score_available"].sum()),
            "score_unavailable_rows": int((~scores["score_available"]).sum()),
            "embedding_dimensions": next(iter(dimensions)),
        },
        "truth_boundary": {
            "score_sha256_before_registered_relations": score_hash_before_truth,
            "registered_signed_direction_truth_relations": int(
                (~relations["expected_signed_direction"].astype(str).str.lower().eq("unspecified")).sum()
            ),
            "unregistered_pairs": "unadjudicated, never negative",
        },
        "embedding_sources": embedding_reports,
        "claim_boundary": [
            "All response studies are development/consumed.",
            "The encoder was frozen and no spectral model was loaded by this scorer.",
            "Response reassignment is a sensitivity, not a calibrated p-value.",
            "Unresolved means not library-supported, not a unique molecule.",
        ],
        "input_sha256": {
            str(path): sha256(path)
            for path in (entities_path, contrasts_path, relations_path, args.embedding_manifest)
            if path is not None
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "checksums.sha256").write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in (scores_path, null_path, audit_path, report_path)
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
