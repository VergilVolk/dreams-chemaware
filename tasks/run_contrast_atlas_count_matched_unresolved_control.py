#!/usr/bin/env python3
"""Count-matched unresolved control for Contrast Atlas response retrieval.

For each frozen resample, each study contributes exactly as many unresolved
entities as it has strict-common-library-supported entities.  Spectral graphs
are rebuilt within the sampled unresolved universe before response effects or
registered relations are opened.  Whole response profiles are then permuted
within study to calibrate candidate-specific response scores.

This is a development control for evidence quantity.  It does not make the
sampled unresolved and known entities chemically or analytically equivalent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_contrast_atlas_open_response_collection import load_contract
from calibrate_contrast_atlas_sparse_local_response import (
    METRICS,
    add_calibrated_ranks,
    build_profile_matrices,
    candidate_coordinates,
    permutation_indices,
    prepare_graphs,
    score_candidate_rows,
)
from run_contrast_atlas_sparse_fragment_measure_baseline import (
    normalise_entity_id,
    sha256,
)
from run_contrast_atlas_sparse_local_alignment import (
    freeze_sparse_graphs,
    load_sparse_sources,
)
from score_contrast_atlas_symmetric_nn_alignment import (
    frozen_source_universes,
    response_profiles,
    score_all,
)


def sample_unresolved_like_known(
    identity: pd.DataFrame,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Draw a study-stratified unresolved universe with strict-known counts."""
    rng = np.random.default_rng(seed)
    selected_blocks: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for study, block in identity.groupby("study_id", sort=True):
        known_count = int(block["identity_status"].eq("library_supported").sum())
        unresolved = np.asarray(
            sorted(block.loc[
                block["identity_status"].eq("unresolved_entity"), "entity_id"
            ].astype(str)),
            dtype=str,
        )
        if known_count <= 0 or len(unresolved) < known_count:
            raise ValueError(
                f"cannot count-match {study}: known={known_count}, unresolved={len(unresolved)}"
            )
        positions = np.sort(rng.choice(len(unresolved), size=known_count, replace=False))
        chosen = unresolved[positions]
        chosen_frame = pd.DataFrame({
            "study_id": str(study),
            "entity_id": chosen,
            "identity_status": "unresolved_entity",
        })
        selected_blocks.append(chosen_frame)
        audit_rows.extend(
            {
                "study_id": str(study),
                "entity_id": str(entity),
                "known_count_target": known_count,
                "unresolved_available": int(len(unresolved)),
            }
            for entity in chosen
        )
    return pd.concat(selected_blocks, ignore_index=True), pd.DataFrame(audit_rows)


def calibrate_one_resample(
    selected_identity: pd.DataFrame,
    entities: pd.DataFrame,
    contrasts: pd.DataFrame,
    studies: list[str],
    full_spectra: dict[str, tuple[np.ndarray, object]],
    permutations: int,
    permutation_seeds: np.ndarray,
    chunk_rows: int,
    minimum_cosine: float,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    selected_lookup = {
        study: set(block["entity_id"].astype(str))
        for study, block in selected_identity.groupby("study_id", sort=False)
    }
    spectra = {}
    for study in studies:
        ids, matrix = full_spectra[study]
        keep = np.asarray([str(value) in selected_lookup[study] for value in ids])
        spectra[study] = (ids[keep], matrix[keep].tocsr())
        if int(keep.sum()) != len(selected_lookup[study]):
            raise RuntimeError(f"selected spectrum coverage drift for {study}")
    universes = frozen_source_universes(selected_identity)
    frozen_graphs, edge_table = freeze_sparse_graphs(
        studies, universes, spectra, chunk_rows, minimum_cosine
    )

    selected_keys = pd.MultiIndex.from_frame(
        selected_identity[["study_id", "entity_id"]].astype(str)
    )
    entity_keys = pd.MultiIndex.from_frame(
        entities[["study_id", "entity_id"]].astype(str)
    )
    selected_entities = entities.loc[entity_keys.isin(selected_keys)].copy()
    profiles = response_profiles(selected_entities)
    observed = score_all(contrasts, frozen_graphs, profiles, "observed").loc[
        lambda frame: frame["evidence_world"].eq("all")
    ].reset_index(drop=True)
    matrices, entity_ids, contrast_ids, statuses = build_profile_matrices(
        selected_entities, contrasts, selected_identity
    )
    edges = edge_table.loc[edge_table["evidence_world"].eq("all")].copy()
    coordinates = candidate_coordinates(observed, contrast_ids)
    graphs = prepare_graphs(edges, entity_ids, statuses, set(coordinates))
    recomputed = score_candidate_rows(
        matrices, graphs, coordinates, row_count=len(observed), score_mode="entity_mean"
    )
    observed_values = {
        metric: observed[f"{metric}_response_score"].to_numpy(float)
        for metric in METRICS
    }
    for metric in METRICS:
        if not np.allclose(
            recomputed[metric], observed_values[metric], rtol=1e-10, atol=1e-12
        ):
            raise RuntimeError(f"vectorized {metric} score drift in matched control")

    totals = {metric: np.zeros(len(observed)) for metric in METRICS}
    squares = {metric: np.zeros(len(observed)) for metric in METRICS}
    exceed = {metric: np.zeros(len(observed), dtype=np.int64) for metric in METRICS}
    for seed in permutation_seeds:
        donors = permutation_indices(statuses, int(seed))
        permuted = {study: matrix[donors[study]] for study, matrix in matrices.items()}
        values = score_candidate_rows(
            permuted, graphs, coordinates, row_count=len(observed), score_mode="entity_mean"
        )
        for metric in METRICS:
            totals[metric] += values[metric]
            squares[metric] += values[metric] * values[metric]
            exceed[metric] += values[metric] >= observed_values[metric] - 1e-15

    calibrated = observed.copy()
    for metric in METRICS:
        mean = totals[metric] / permutations
        variance = np.maximum(
            (squares[metric] - permutations * mean * mean) / (permutations - 1), 0.0
        )
        sd = np.sqrt(variance)
        difference = observed_values[metric] - mean
        z = np.divide(difference, sd, out=np.zeros_like(difference), where=sd > 1e-15)
        z[(sd <= 1e-15) & (difference > 1e-15)] = np.inf
        z[(sd <= 1e-15) & (difference < -1e-15)] = -np.inf
        calibrated[f"{metric}_response_null_mean"] = mean
        calibrated[f"{metric}_response_null_sd"] = sd
        calibrated[f"{metric}_response_null_z"] = z
        calibrated[f"{metric}_response_null_empirical_p"] = (
            1.0 + exceed[metric]
        ) / (permutations + 1.0)
    calibrated = add_calibrated_ranks(calibrated)
    return calibrated, edge_table, len(selected_entities)


def audit_relations_one_world(
    calibrated: pd.DataFrame,
    relations: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for relation in relations.to_dict("records"):
        matched = calibrated.loc[
            calibrated["query_study_id"].eq(str(relation["left_study_id"]))
            & calibrated["query_contrast_id"].eq(str(relation["left_contrast_id"]))
            & calibrated["candidate_study_id"].eq(str(relation["right_study_id"]))
            & calibrated["candidate_contrast_id"].eq(str(relation["right_contrast_id"]))
        ]
        if len(matched) != 1:
            raise ValueError("registered relation does not match one candidate row")
        row = matched.iloc[0]
        output = dict(relation)
        output.update({
            "candidate_universe_size": int(len(calibrated.loc[
                calibrated["query_study_id"].eq(row["query_study_id"])
                & calibrated["query_contrast_id"].eq(row["query_contrast_id"])
            ])),
            "evidence_world": "unresolved_count_matched",
        })
        for column in calibrated.columns:
            if (
                column.endswith("_response_score")
                or "response_null" in column
                or column in {"composition_score", "composition_absolute_rank"}
            ):
                output[column] = row[column]
        rows.append(output)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--identity-partition", required=True, type=Path)
    parser.add_argument("--alignment-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resamples", type=int, default=16)
    parser.add_argument("--permutations", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--chunk-rows", type=int, default=128)
    parser.add_argument("--minimum-cosine", type=float, default=0.70)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.resamples < 2 or args.permutations < 2:
        raise ValueError("at least two resamples and permutations are required")

    root = Path(__file__).resolve().parents[1]
    contract, contract_paths = load_contract(args.contract.resolve(), root)
    paths = {
        "entities": args.collection / "response_entities.tsv.gz",
        "contrasts": args.collection / "contrast_manifest.tsv",
        "relations": args.collection / "registered_relations.tsv",
        "sources": args.collection / "source_manifest.tsv",
        "alignment_report": args.alignment_dir / "report.json",
    }
    for path in (*paths.values(), args.identity_partition):
        if not path.is_file():
            raise FileNotFoundError(path)
    alignment_report = json.loads(paths["alignment_report"].read_text(encoding="utf-8"))
    if alignment_report["provenance"]["response_entities_sha256"] != sha256(paths["entities"]):
        raise RuntimeError("alignment/collection response provenance drift")
    if alignment_report["provenance"]["identity_partition_sha256"] != sha256(args.identity_partition):
        raise RuntimeError("alignment/common identity provenance drift")

    identity = pd.read_csv(args.identity_partition, sep="\t", dtype=str)
    entities = pd.read_csv(paths["entities"], sep="\t", low_memory=False)
    entities["study_id"] = entities["study_id"].astype(str)
    entities["entity_id"] = entities["entity_id"].astype(str)
    contrasts = pd.read_csv(paths["contrasts"], sep="\t", dtype=str)
    sources = pd.read_csv(paths["sources"], sep="\t", dtype=str)
    studies = sources["study_id"].astype(str).tolist()

    raw_registry = pd.read_csv(
        paths["entities"], sep="\t",
        usecols=["study_id", "raw_entity_id", "entity_id"], dtype=str,
    ).drop_duplicates()
    raw_registry["raw_entity_id"] = raw_registry["raw_entity_id"].map(normalise_entity_id)
    registry = raw_registry.merge(
        identity, on=["study_id", "entity_id"], how="left", validate="one_to_one"
    )
    if registry["identity_status"].isna().any():
        raise RuntimeError("identity partition does not cover response registry")
    full_spectra, source_reports, _dimensions = load_sparse_sources(
        contract, registry, studies, bin_width=0.1
    )

    master = np.random.default_rng(args.seed)
    selection_seeds = master.integers(
        0, np.iinfo(np.int64).max, size=args.resamples, dtype=np.int64
    )
    all_scores: list[pd.DataFrame] = []
    all_selections: list[pd.DataFrame] = []
    edge_counts: list[int] = []
    response_rows: list[int] = []
    for replicate, selection_seed in enumerate(selection_seeds):
        selected_identity, selection = sample_unresolved_like_known(
            identity, int(selection_seed)
        )
        permutation_rng = np.random.default_rng(int(selection_seed) ^ args.seed)
        permutation_seeds = permutation_rng.integers(
            0, np.iinfo(np.int64).max, size=args.permutations, dtype=np.int64
        )
        calibrated, edges, selected_response_rows = calibrate_one_resample(
            selected_identity, entities, contrasts, studies, full_spectra,
            args.permutations, permutation_seeds, args.chunk_rows,
            args.minimum_cosine,
        )
        calibrated.insert(0, "resample", replicate)
        calibrated.insert(1, "selection_seed", int(selection_seed))
        calibrated["evidence_world"] = "unresolved_count_matched"
        selection.insert(0, "resample", replicate)
        selection.insert(1, "selection_seed", int(selection_seed))
        all_scores.append(calibrated)
        all_selections.append(selection)
        edge_counts.append(len(edges.loc[edges["evidence_world"].eq("all")]))
        response_rows.append(selected_response_rows)

    calibrated_all = pd.concat(all_scores, ignore_index=True)
    selection_all = pd.concat(all_selections, ignore_index=True)

    # Biological relations are first opened after every resample and null score
    # has been frozen.
    relations = pd.read_csv(paths["relations"], sep="\t", dtype=str)
    relation_audits = []
    for replicate, block in calibrated_all.groupby("resample", sort=True):
        audit = audit_relations_one_world(block, relations)
        audit.insert(0, "resample", int(replicate))
        relation_audits.append(audit)
    relation_audit = pd.concat(relation_audits, ignore_index=True)
    canonical_drugs = relation_audit.loc[
        relation_audit["relation_direction"].eq("registered")
        & relation_audit["relation_class"].eq("consumed_exact_drug_cross_design")
    ].copy()
    minimum_p = 1.0 / (args.permutations + 1.0)
    replicate_summary = canonical_drugs.groupby("resample").agg(
        exact_drug_relations=("relation_class", "size"),
        similar_top1=("similar_response_null_z_absolute_rank", lambda x: int((x == 1).sum())),
        similar_top5=("similar_response_null_z_absolute_rank", lambda x: int((x <= 5).sum())),
        median_similar_rank=("similar_response_null_z_absolute_rank", "median"),
        similar_p_at_minimum=("similar_response_null_empirical_p", lambda x: int((x <= minimum_p).sum())),
        unsigned_top1=("unsigned_response_null_z_absolute_rank", lambda x: int((x == 1).sum())),
        median_unsigned_rank=("unsigned_response_null_z_absolute_rank", "median"),
    ).reset_index()

    args.output_dir.mkdir(parents=True)
    outputs = {
        "scores": args.output_dir / "count_matched_calibrated_candidate_scores.tsv.gz",
        "selections": args.output_dir / "count_matched_unresolved_selections.tsv.gz",
        "relations": args.output_dir / "registered_relation_count_matched_ranks.tsv.gz",
        "summary": args.output_dir / "exact_drug_resample_summary.tsv",
    }
    compression = {"method": "gzip", "mtime": 0}
    calibrated_all.to_csv(outputs["scores"], sep="\t", index=False, compression=compression)
    selection_all.to_csv(outputs["selections"], sep="\t", index=False, compression=compression)
    relation_audit.to_csv(outputs["relations"], sep="\t", index=False, compression=compression)
    replicate_summary.to_csv(outputs["summary"], sep="\t", index=False)

    report = {
        "schema": "contrast_atlas_count_matched_unresolved_control_v1",
        "status": "DEVELOPMENT_CONSUMED_COUNT_MATCHED_CONTROL_COMPLETE",
        "data_role": "development_consumed",
        "method": {
            "resamples": args.resamples,
            "permutations_per_resample": args.permutations,
            "seed": args.seed,
            "selection": "within each study sample unresolved entities without replacement to equal strict-known entity count",
            "graph": "rebuild bidirectional top-1 fragment-cosine graph inside every sampled universe before opening responses or relations",
            "null": "whole response-profile permutation within sampled study",
        },
        "denominators": {
            "studies": len(studies),
            "contrasts": len(contrasts),
            "selected_entities_per_resample": int(
                identity["identity_status"].eq("library_supported").sum()
            ),
            "selected_response_rows_per_resample": sorted(set(response_rows)),
            "candidate_rows_per_resample": int(len(calibrated_all) / args.resamples),
            "exact_drug_relations": int(canonical_drugs.loc[canonical_drugs["resample"].eq(0)].shape[0]),
            "frozen_edges_per_resample_range": [int(min(edge_counts)), int(max(edge_counts))],
        },
        "exact_drug_resample_result": {
            "similar_p_at_minimum_median": float(replicate_summary["similar_p_at_minimum"].median()),
            "similar_p_at_minimum_range": [
                int(replicate_summary["similar_p_at_minimum"].min()),
                int(replicate_summary["similar_p_at_minimum"].max()),
            ],
            "similar_top5_median": float(replicate_summary["similar_top5"].median()),
            "similar_top5_range": [
                int(replicate_summary["similar_top5"].min()),
                int(replicate_summary["similar_top5"].max()),
            ],
            "median_similar_rank_across_resamples": float(
                replicate_summary["median_similar_rank"].median()
            ),
            "unsigned_top1_total_across_resamples": int(
                replicate_summary["unsigned_top1"].sum()
            ),
        },
        "leakage_status": "selection and every chemical graph were frozen from common identity status and MS/MS only; beta/SE were opened after graphs; registered relations were opened after all observed and null candidate scores",
        "decision_boundary": "development evidence-count control only; it does not match chemical class, ion redundancy, platform coverage or independent project source",
        "provenance": {
            "contract_sha256": sha256(args.contract),
            "contract_lineage": {
                str(path.relative_to(root)): sha256(path) for path in contract_paths
            },
            "response_entities_sha256": sha256(paths["entities"]),
            "identity_partition_sha256": sha256(args.identity_partition),
            "alignment_report_sha256": sha256(paths["alignment_report"]),
            **{name: sha256(path) for name, path in outputs.items()},
            "implementation_sha256": sha256(Path(__file__)),
        },
        "sources": source_reports,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
