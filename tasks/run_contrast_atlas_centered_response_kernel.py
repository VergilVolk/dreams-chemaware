#!/usr/bin/env python3
"""Truth-blind centred spectral-response kernel for an open response collection.

The operator targets response-associated fragment evidence rather than chemical
composition alone.  Within each study contrast and evidence world, uncertainty-
shrunk response weights are mean-centred before spectra are aggregated.  The
resulting signed fragment vector is L2-normalised and compared across studies.
Registered biological relations are opened only after the complete fixed-score
table has been written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from build_contrast_atlas_open_response_collection import load_contract
from run_contrast_atlas_sparse_fragment_measure_baseline import (
    WORLDS,
    load_study_spectra,
    normalise_entity_id,
    pad_columns,
    sha256,
    unweighted_centroid,
)


SCORES = (
    "signed_similar_score",
    "signed_opposite_score",
    "response_alignment_strength_score",
    "composition_score",
)


def l2_normalise(row: sparse.csr_matrix) -> tuple[sparse.csr_matrix, float]:
    norm = float(np.sqrt(row.multiply(row).sum()))
    if norm <= 0:
        return sparse.csr_matrix(row.shape, dtype=np.float32), 0.0
    return sparse.csr_matrix(row / norm, dtype=np.float32), norm


def centred_response_vector(
    matrix: sparse.csr_matrix, weights: np.ndarray
) -> tuple[sparse.csr_matrix, float, float]:
    """Return unit response-associated fragment vector, raw norm, effective n."""
    if matrix.shape[0] == 0:
        return sparse.csr_matrix((1, matrix.shape[1]), dtype=np.float32), 0.0, 0.0
    centred = np.asarray(weights, dtype=np.float64) - float(np.mean(weights))
    sum_squares = float(np.dot(centred, centred))
    if sum_squares <= 0:
        return sparse.csr_matrix((1, matrix.shape[1]), dtype=np.float32), 0.0, 0.0
    effective_n = float(np.square(np.abs(centred).sum()) / sum_squares)
    vector = sparse.csr_matrix(centred.astype(np.float32)[None, :] @ matrix)
    unit, raw_norm = l2_normalise(vector)
    return unit, raw_norm, effective_n


def dot(left: sparse.csr_matrix, right: sparse.csr_matrix) -> float:
    return float(left.multiply(right).sum())


def stable_seed(base_seed: int, replicate: int, contrast_index: int, world: str) -> int:
    payload = f"{base_seed}|{replicate}|{contrast_index}|{world}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def stratified_permutation(
    weights: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    result = np.asarray(weights, dtype=float).copy()
    for stratum in sorted(set(strata.astype(str))):
        indices = np.flatnonzero(strata.astype(str) == stratum)
        if len(indices) > 1:
            result[indices] = weights[rng.permutation(indices)]
    return result


def score_candidates(
    manifest: pd.DataFrame,
    sketches: dict[tuple[int, str], dict[str, object]],
    minimum_entities: int,
    minimum_effective_n: float,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for qi, query in manifest.iterrows():
        for ti, target in manifest.iterrows():
            if str(query["source_family"]) == str(target["source_family"]):
                continue
            for world in WORLDS:
                q = sketches[(qi, world)]
                t = sketches[(ti, world)]
                available = (
                    int(q["n_entities"]) >= minimum_entities
                    and int(t["n_entities"]) >= minimum_entities
                    and float(q["effective_n"]) >= minimum_effective_n
                    and float(t["effective_n"]) >= minimum_effective_n
                    and float(q["raw_response_norm"]) > 0
                    and float(t["raw_response_norm"]) > 0
                )
                row = {
                    "query_study_id": str(query["study_id"]),
                    "query_source_family": str(query["source_family"]),
                    "query_contrast_id": str(query["contrast_id"]),
                    "candidate_study_id": str(target["study_id"]),
                    "candidate_source_family": str(target["source_family"]),
                    "candidate_contrast_id": str(target["contrast_id"]),
                    "evidence_world": world,
                    "query_entities": int(q["n_entities"]),
                    "candidate_entities": int(t["n_entities"]),
                    "query_effective_n": float(q["effective_n"]),
                    "candidate_effective_n": float(t["effective_n"]),
                    "query_raw_response_norm": float(q["raw_response_norm"]),
                    "candidate_raw_response_norm": float(t["raw_response_norm"]),
                    "score_available": bool(available),
                    "unavailability_reason": "" if available else "insufficient_response_measure",
                }
                if available:
                    signed = dot(q["response"], t["response"])
                    row.update(
                        {
                            "signed_similar_score": signed,
                            "signed_opposite_score": -signed,
                            "response_alignment_strength_score": abs(signed),
                            "composition_score": dot(q["composition"], t["composition"]),
                        }
                    )
                else:
                    row.update({name: np.nan for name in SCORES})
                records.append(row)

    scores = pd.DataFrame(records)
    groups = scores.groupby(
        ["query_study_id", "query_contrast_id", "evidence_world"], sort=False
    )
    for score in SCORES:
        rank = score.replace("_score", "_absolute_rank")
        scores[rank] = np.nan
        for _, indices in groups.groups.items():
            values = scores.loc[indices, score]
            present = values.notna()
            scores.loc[values.index[present], rank] = values.loc[present].rank(
                ascending=False, method="min"
            )
    return scores


def audit_registered(scores: pd.DataFrame, relations: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, object]] = []
    for relation in relations.to_dict("records"):
        mask = (
            scores["query_study_id"].eq(str(relation["left_study_id"]))
            & scores["query_contrast_id"].eq(str(relation["left_contrast_id"]))
            & scores["candidate_study_id"].eq(str(relation["right_study_id"]))
            & scores["candidate_contrast_id"].eq(str(relation["right_contrast_id"]))
        )
        matched = scores.loc[mask]
        if len(matched) != len(WORLDS):
            raise ValueError(f"registered relation does not match three worlds: {relation}")
        for row in matched.to_dict("records"):
            record = dict(relation)
            record.update(row)
            record["adjudication_status"] = (
                "DESCRIPTIVE_DESIGN_RELEVANCE_ONLY_SIGNED_DIRECTION_UNSPECIFIED"
            )
            record["pass_fail"] = "NOT_ADJUDICATED"
            output.append(record)
    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract", type=Path,
        default=Path("tasks/contrast_atlas_open_collection_ibd_extension_v1.json"),
    )
    parser.add_argument(
        "--collection", type=Path,
        default=Path("data/validation/contrast_atlas_open_response_collection_v3_ibd_20261019"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/contrast_atlas_centered_response_kernel_v2_20261020"),
    )
    parser.add_argument("--bin-width-da", type=float, default=0.1)
    parser.add_argument("--minimum-entities", type=int, default=10)
    parser.add_argument("--minimum-effective-n", type=float, default=5.0)
    parser.add_argument("--null-replicates", type=int, default=31)
    parser.add_argument("--null-seed", type=int, default=20261020)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    root = Path(__file__).resolve().parents[1]
    contract, contract_paths = load_contract(args.contract.resolve(), root)
    entities_path = args.collection / "response_entities.tsv.gz"
    contrasts_path = args.collection / "contrast_manifest.tsv"
    relations_path = args.collection / "registered_relations.tsv"
    entities = pd.read_csv(entities_path, sep="\t", dtype={"raw_entity_id": str})
    entities["raw_entity_id"] = entities["raw_entity_id"].map(normalise_entity_id)
    contrasts = pd.read_csv(contrasts_path, sep="\t").reset_index(drop=True)
    source_contract = {str(item["study_id"]): item for item in contract["sources"]}
    if set(contrasts["study_id"].astype(str)) != set(source_contract):
        raise ValueError("contract and contrast manifest contain different studies")

    per_source: dict[str, tuple[list[str], sparse.csr_matrix]] = {}
    source_reports = []
    dimensions = 0
    for study_id in contrasts["study_id"].astype(str).drop_duplicates():
        required = set(entities.loc[entities["study_id"].astype(str).eq(study_id), "raw_entity_id"])
        mgf_path = Path(source_contract[study_id]["mgf_path"])
        ids, matrix, details = load_study_spectra(mgf_path, required, args.bin_width_da)
        per_source[study_id] = (ids, matrix)
        dimensions = max(dimensions, matrix.shape[1])
        source_reports.append({"study_id": study_id, "mgf_sha256": sha256(mgf_path), **details})

    sketches: dict[tuple[int, str], dict[str, object]] = {}
    prepared: dict[tuple[int, str], tuple[sparse.csr_matrix, np.ndarray, np.ndarray]] = {}
    for ci, contrast in contrasts.iterrows():
        study_id = str(contrast["study_id"])
        contrast_id = str(contrast["contrast_id"])
        ids, raw_matrix = per_source[study_id]
        matrix = pad_columns(raw_matrix, dimensions)
        lookup = {entity: index for index, entity in enumerate(ids)}
        block = entities.loc[
            entities["study_id"].astype(str).eq(study_id)
            & entities["contrast_id"].astype(str).eq(contrast_id)
        ].copy()
        block["spectrum_row"] = block["raw_entity_id"].map(lookup)
        if block["spectrum_row"].isna().any() or block["raw_entity_id"].duplicated().any():
            raise ValueError(f"invalid effect-to-spectrum join: {study_id}/{contrast_id}")
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
            selected = matrix[block.loc[mask, "spectrum_row"].to_numpy(int)]
            response, raw_norm, effective_n = centred_response_vector(selected, weights[mask])
            prepared[(ci, world)] = (
                selected,
                weights[mask],
                block.loc[mask, "identity_status"].astype(str).to_numpy(),
            )
            sketches[(ci, world)] = {
                "n_entities": int(mask.sum()),
                "effective_n": effective_n,
                "raw_response_norm": raw_norm,
                "response": response,
                "composition": unweighted_centroid(selected),
            }

    # Freeze every candidate score before opening the registered relation table.
    scores = score_candidates(
        contrasts, sketches, args.minimum_entities, args.minimum_effective_n
    )
    args.output_dir.mkdir(parents=True)
    scores_path = args.output_dir / "fixed_candidate_scores.tsv.gz"
    scores.to_csv(scores_path, sep="\t", index=False, compression="gzip")
    score_hash_before_truth = sha256(scores_path)

    null_frames = []
    for replicate in range(args.null_replicates):
        null_sketches: dict[tuple[int, str], dict[str, object]] = {}
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
        null_scores = score_candidates(
            contrasts, null_sketches, args.minimum_entities, args.minimum_effective_n
        )
        null_scores.insert(0, "null_replicate", replicate)
        null_frames.append(null_scores)
    null_scores = pd.concat(null_frames, ignore_index=True)
    null_scores_path = args.output_dir / "stratified_response_reassignment_scores.tsv.gz"
    null_scores.to_csv(null_scores_path, sep="\t", index=False, compression="gzip")

    relations = pd.read_csv(relations_path, sep="\t")
    audit = audit_registered(scores, relations)
    audit_path = args.output_dir / "registered_relation_absolute_ranks.tsv"
    audit.to_csv(audit_path, sep="\t", index=False)

    relation_keys = relations.rename(
        columns={
            "left_study_id": "query_study_id",
            "left_contrast_id": "query_contrast_id",
            "right_study_id": "candidate_study_id",
            "right_contrast_id": "candidate_contrast_id",
        }
    )[
        ["query_study_id", "query_contrast_id", "candidate_study_id", "candidate_contrast_id"]
    ]
    null_registered = null_scores.merge(
        relation_keys,
        on=["query_study_id", "query_contrast_id", "candidate_study_id", "candidate_contrast_id"],
        how="inner",
        validate="many_to_one",
    )
    null_registered_path = args.output_dir / "registered_relation_reassignment_ranks.tsv.gz"
    null_registered.to_csv(null_registered_path, sep="\t", index=False, compression="gzip")

    sensitivity_rows = []
    identity = [
        "query_study_id", "query_contrast_id", "candidate_study_id",
        "candidate_contrast_id", "evidence_world",
    ]
    for _, observed in audit.iterrows():
        mask = np.ones(len(null_registered), dtype=bool)
        for column in identity:
            mask &= null_registered[column].astype(str).eq(str(observed[column])).to_numpy()
        block = null_registered.loc[mask]
        observed_rank = observed["response_alignment_strength_absolute_rank"]
        null_ranks = block["response_alignment_strength_absolute_rank"].dropna().to_numpy(float)
        sensitivity_rows.append(
            {
                **{column: observed[column] for column in identity},
                "observed_alignment_strength_rank": observed_rank,
                "null_replicates_available": int(len(null_ranks)),
                "null_rank_median": float(np.median(null_ranks)) if len(null_ranks) else np.nan,
                "null_rank_q05": float(np.quantile(null_ranks, 0.05)) if len(null_ranks) else np.nan,
                "null_rank_q95": float(np.quantile(null_ranks, 0.95)) if len(null_ranks) else np.nan,
                "fraction_null_rank_as_good_or_better": (
                    float(np.mean(null_ranks <= float(observed_rank)))
                    if len(null_ranks) and pd.notna(observed_rank) else np.nan
                ),
                "interpretation": "NONCALIBRATED_RESPONSE_REASSIGNMENT_SENSITIVITY",
            }
        )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity_path = args.output_dir / "registered_relation_reassignment_sensitivity.tsv"
    sensitivity.to_csv(sensitivity_path, sep="\t", index=False)

    top1 = {}
    for world, group in audit.groupby("evidence_world", sort=True):
        top1[str(world)] = {
            "registered_relations": int(len(group)),
            "score_available": int(group["score_available"].sum()),
            "signed_similar_top1": int(group["signed_similar_absolute_rank"].eq(1).sum()),
            "signed_opposite_top1": int(group["signed_opposite_absolute_rank"].eq(1).sum()),
            "alignment_strength_top1": int(
                group["response_alignment_strength_absolute_rank"].eq(1).sum()
            ),
            "composition_top1": int(group["composition_absolute_rank"].eq(1).sum()),
        }

    report = {
        "schema": "contrast_atlas_centered_response_kernel_v1",
        "status": "DEVELOPMENT_CONSUMED_TRUTH_BLIND_RESULT",
        "method": {
            "response_weight": "beta/sqrt(beta^2+se^2)",
            "centering": "subtract within-contrast/world mean response weight before spectral aggregation",
            "kernel": "cosine between L2-normalised centred response-associated fragment vectors",
            "signed_opposite": "negative signed-similar kernel",
            "alignment_strength": "absolute signed kernel; direction-agnostic but response-dependent",
            "composition_control": "dot product of unweighted spectrum centroids",
            "bin_width_da": args.bin_width_da,
            "minimum_entities": args.minimum_entities,
            "minimum_effective_n": args.minimum_effective_n,
            "parameter_selection": "fixed from statistical identifiability, not registered-relation ranks",
            "response_reassignment": (
                "deterministic seeded permutations within study, contrast and identity stratum; "
                "sensitivity only, not a randomisation p-value"
            ),
            "null_replicates": args.null_replicates,
            "null_seed": args.null_seed,
        },
        "denominators": {
            "source_families": int(contrasts["source_family"].nunique()),
            "study_level_contrasts": int(len(contrasts)),
            "fixed_candidate_rows_per_world": int(len(scores) / len(WORLDS)),
            "score_available_rows": int(scores["score_available"].sum()),
            "score_unavailable_rows": int((~scores["score_available"]).sum()),
            "registered_directed_relevance_relations": int(len(relations)),
            "registered_signed_direction_truth_relations": int(
                (~relations["expected_signed_direction"].astype(str).str.lower().eq("unspecified")).sum()
            ),
            "fragment_dimensions": dimensions,
        },
        "registered_relation_descriptive_top1": top1,
        "registered_relation_reassignment_sensitivity": sensitivity.to_dict("records"),
        "truth_boundary": {
            "score_sha256_before_registered_relations": score_hash_before_truth,
            "unregistered_pairs": "unadjudicated, never negative",
            "pass_fail": "not adjudicated because signed-direction truth is absent",
            "unknown_increment": "not established by descriptive ranks",
        },
        "source_reports": source_reports,
        "input_sha256": {
            str(args.contract): sha256(args.contract),
            **{f"contract_lineage::{p.relative_to(root)}": sha256(p) for p in contract_paths},
            str(entities_path): sha256(entities_path),
            str(contrasts_path): sha256(contrasts_path),
            str(relations_path): sha256(relations_path),
        },
        "claim_boundary": [
            f"All {contrasts['source_family'].nunique()} sources are development/consumed.",
            "Registered relations specify design relevance, not signed response truth.",
            "This is a relation-layer operator and never merges distinct entities as one molecule.",
            "Unresolved means not library-supported, not a unique metabolite.",
            "A better descriptive rank is not an independently verified biological discovery.",
        ],
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "checksums.sha256").write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in (
                scores_path, null_scores_path, audit_path, null_registered_path,
                sensitivity_path, report_path,
            )
        ),
        encoding="utf-8",
    )
    print(json.dumps({"top1": top1, "denominators": report["denominators"]}, indent=2))


if __name__ == "__main__":
    main()
