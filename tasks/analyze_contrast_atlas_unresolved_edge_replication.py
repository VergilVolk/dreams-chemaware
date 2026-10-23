#!/usr/bin/env python3
"""Systematic unresolved-edge response replication for registered relations.

The chemical graph is frozen upstream without effects.  This analysis asks
whether both endpoints of an unresolved spectral edge show FDR-qualified,
same-direction effects for a registered cross-accession perturbation.  A
10-ppm precursor-compatible track is kept separate from the broader chemical
relation track.  Precursor compatibility is an identity hypothesis, not proof
of molecular identity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from build_contrast_atlas_open_response_collection import load_contract
from run_contrast_atlas_sparse_fragment_measure_baseline import (
    iter_mgf,
    normalise_entity_id,
    sha256,
    spectrum_id,
)


TRACK_LABELS = (
    "all_edges",
    "precursor_compatible_10ppm",
    "chemical_relation_only",
    "mutual_top1_all",
    "mutual_top1_precursor_compatible_10ppm",
    "mutual_top1_chemical_relation_only",
)


def support_masks(edge_block: pd.DataFrame) -> list[np.ndarray]:
    mass = edge_block["precursor_compatible_10ppm"].to_numpy(bool)
    mutual = edge_block["mutual_top1"].to_numpy(bool)
    return [
        np.ones(len(edge_block), dtype=bool),
        mass,
        ~mass,
        mutual,
        mutual & mass,
        mutual & ~mass,
    ]


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    output = np.full(len(values), np.nan, dtype=float)
    finite = np.isfinite(values)
    selected = values[finite]
    if not len(selected):
        return output
    order = np.argsort(selected, kind="stable")
    ranked = selected[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty(len(selected), dtype=float)
    restored[order] = np.minimum(adjusted, 1.0)
    output[finite] = restored
    return output


def add_effect_statistics(entities: pd.DataFrame) -> pd.DataFrame:
    frame = entities.copy()
    beta = frame["beta"].to_numpy(float)
    se = frame["se"].to_numpy(float)
    z = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    frame["effect_z"] = z
    frame["effect_p"] = 2.0 * norm.sf(np.abs(z))
    frame["effect_q"] = np.nan
    for indices in frame.groupby(["study_id", "contrast_id"], sort=False).groups.values():
        frame.loc[indices, "effect_q"] = bh_adjust(frame.loc[indices, "effect_p"].to_numpy(float))
    frame["effect_sign"] = np.sign(beta).astype(np.int8)
    denominator = np.sqrt(beta * beta + se * se)
    frame["response_weight"] = np.divide(
        beta, denominator, out=np.zeros_like(beta), where=denominator > 0
    )
    return frame


def precursor_by_entity(
    contract: dict[str, object],
    registry: pd.DataFrame,
    studies: set[str],
) -> dict[tuple[str, str], float]:
    sources = {str(item["study_id"]): item for item in contract["sources"]}
    output: dict[tuple[str, str], float] = {}
    for study in sorted(studies):
        block = registry.loc[registry["study_id"].eq(study)]
        raw_to_entity = dict(zip(block["raw_entity_id"], block["entity_id"]))
        for metadata, _peaks in iter_mgf(Path(sources[study]["mgf_path"])):
            raw_id = spectrum_id(metadata)
            entity = raw_to_entity.get(raw_id)
            if entity is None:
                continue
            value = metadata.get("PEPMASS", "").split()
            if not value:
                continue
            try:
                precursor = float(value[0])
            except ValueError:
                continue
            if np.isfinite(precursor) and precursor > 0:
                output[(study, entity)] = precursor
    return output


def canonical_edge_table(edges: pd.DataFrame) -> pd.DataFrame:
    unresolved = edges.loc[edges["evidence_world"].eq("unresolved")].copy()
    grouped = unresolved.groupby(
        ["left_study_id", "right_study_id", "left_entity_id", "right_entity_id"],
        sort=False,
    )
    table = grouped.agg(
        chemical_similarity=("chemical_similarity", "max"),
        directional_support=("direction", "nunique"),
    ).reset_index()
    table["mutual_top1"] = table["directional_support"].eq(2)
    return table


def build_profiles(
    effects: pd.DataFrame,
    studies: set[str],
) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    entity_ids: dict[str, list[str]] = {}
    contrast_ids: dict[str, list[str]] = {}
    significant: dict[str, np.ndarray] = {}
    signs: dict[str, np.ndarray] = {}
    weights: dict[str, np.ndarray] = {}
    for study in sorted(studies):
        block = effects.loc[effects["study_id"].eq(study)]
        entities = sorted(block["entity_id"].astype(str).unique())
        contrasts = sorted(block["contrast_id"].astype(str).unique())
        erow = {value: index for index, value in enumerate(entities)}
        ccol = {value: index for index, value in enumerate(contrasts)}
        sig = np.zeros((len(entities), len(contrasts)), dtype=bool)
        sign = np.zeros((len(entities), len(contrasts)), dtype=np.int8)
        weight = np.zeros((len(entities), len(contrasts)), dtype=float)
        for row in block.itertuples(index=False):
            i, j = erow[str(row.entity_id)], ccol[str(row.contrast_id)]
            sig[i, j] = bool(float(row.effect_q) <= 0.05)
            sign[i, j] = int(row.effect_sign)
            weight[i, j] = float(row.response_weight)
        entity_ids[study] = entities
        contrast_ids[study] = contrasts
        significant[study] = sig
        signs[study] = sign
        weights[study] = weight
    return entity_ids, contrast_ids, significant, signs, weights


def relation_edge_indices(
    edge_table: pd.DataFrame,
    relations: pd.DataFrame,
    entity_ids: dict[str, list[str]],
    contrast_ids: dict[str, list[str]],
) -> list[dict[str, object]]:
    entity_lookup = {
        study: {entity: index for index, entity in enumerate(values)}
        for study, values in entity_ids.items()
    }
    contrast_lookup = {
        study: {contrast: index for index, contrast in enumerate(values)}
        for study, values in contrast_ids.items()
    }
    output = []
    for relation in relations.to_dict("records"):
        left_study = str(relation["left_study_id"])
        right_study = str(relation["right_study_id"])
        low, high = sorted((left_study, right_study))
        block = edge_table.loc[
            edge_table["left_study_id"].eq(low)
            & edge_table["right_study_id"].eq(high)
        ].copy()
        if left_study == low:
            left_entities = block["left_entity_id"].astype(str)
            right_entities = block["right_entity_id"].astype(str)
        else:
            left_entities = block["right_entity_id"].astype(str)
            right_entities = block["left_entity_id"].astype(str)
        output.append({
            "relation": relation,
            "edges": block,
            "left_rows": np.asarray([entity_lookup[left_study][x] for x in left_entities], dtype=np.int64),
            "right_rows": np.asarray([entity_lookup[right_study][x] for x in right_entities], dtype=np.int64),
            "left_column": contrast_lookup[left_study][str(relation["left_contrast_id"])],
            "right_column": contrast_lookup[right_study][str(relation["right_contrast_id"])],
            "left_entities": left_entities.to_numpy(str),
            "right_entities": right_entities.to_numpy(str),
        })
    return output


def count_support(
    item: dict[str, object],
    significant: dict[str, np.ndarray],
    signs: dict[str, np.ndarray],
    donors: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    relation = item["relation"]
    left_study = str(relation["left_study_id"])
    right_study = str(relation["right_study_id"])
    left_rows = item["left_rows"]
    right_rows = item["right_rows"]
    if donors is not None:
        left_rows = donors[left_study][left_rows]
        right_rows = donors[right_study][right_rows]
    left_sig = significant[left_study][left_rows, int(item["left_column"])]
    right_sig = significant[right_study][right_rows, int(item["right_column"])]
    left_sign = signs[left_study][left_rows, int(item["left_column"])]
    right_sign = signs[right_study][right_rows, int(item["right_column"])]
    dual = left_sig & right_sig & (left_sign != 0) & (left_sign == right_sign)
    return dual, left_sign * right_sign


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--identity-partition", required=True, type=Path)
    parser.add_argument("--alignment-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--permutations", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--precursor-ppm", type=float, default=10.0)
    parser.add_argument(
        "--relation-class",
        action="append",
        dest="relation_classes",
        default=None,
        help="registered relation class to audit; repeat for multiple classes",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    root = Path(__file__).resolve().parents[1]
    contract, contract_paths = load_contract(args.contract.resolve(), root)
    paths = {
        "entities": args.collection / "response_entities.tsv.gz",
        "relations": args.collection / "registered_relations.tsv",
        "edges": args.alignment_dir / "source_pair_frozen_edges.tsv.gz",
        "alignment_report": args.alignment_dir / "report.json",
    }
    for path in (*paths.values(), args.identity_partition):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(paths["alignment_report"].read_text(encoding="utf-8"))
    if report["provenance"]["edges"] != sha256(paths["edges"]):
        raise RuntimeError("frozen edge checksum drift")
    if report["provenance"]["response_entities_sha256"] != sha256(paths["entities"]):
        raise RuntimeError("response collection checksum drift")

    relation_classes = args.relation_classes or ["consumed_exact_drug_cross_design"]
    # The relation registry is already frozen.  Only explicitly requested
    # canonical development relations are queried; none is learned from effects.
    all_relations = pd.read_csv(paths["relations"], sep="\t", dtype=str)
    relations = all_relations.loc[
        all_relations["relation_direction"].eq("registered")
        & all_relations["relation_class"].isin(relation_classes)
    ].reset_index(drop=True)
    if not len(relations):
        raise RuntimeError(f"no canonical relations found for {relation_classes}")
    studies = set(relations["left_study_id"]) | set(relations["right_study_id"])

    identity = pd.read_csv(args.identity_partition, sep="\t", dtype=str)
    effects = pd.read_csv(paths["entities"], sep="\t", low_memory=False)
    effects["study_id"] = effects["study_id"].astype(str)
    effects["entity_id"] = effects["entity_id"].astype(str)
    effects = effects.drop(columns=["identity_status"], errors="ignore").merge(
        identity[["study_id", "entity_id", "identity_status"]],
        on=["study_id", "entity_id"], how="left", validate="many_to_one",
    )
    if effects["identity_status"].isna().any():
        raise RuntimeError("common identity partition does not cover effect rows")
    effects = effects.loc[
        effects["study_id"].isin(studies)
        & effects["identity_status"].eq("unresolved_entity")
    ].copy()
    effects = add_effect_statistics(effects)
    edges = canonical_edge_table(pd.read_csv(paths["edges"], sep="\t", dtype=str).assign(
        chemical_similarity=lambda x: pd.to_numeric(x["chemical_similarity"])
    ))

    registry = pd.read_csv(
        paths["entities"], sep="\t",
        usecols=["study_id", "raw_entity_id", "entity_id"], dtype=str,
    ).drop_duplicates()
    registry["raw_entity_id"] = registry["raw_entity_id"].map(normalise_entity_id)
    precursors = precursor_by_entity(contract, registry, studies)
    entity_ids, contrast_ids, significant, signs, weights = build_profiles(effects, studies)
    indexed = relation_edge_indices(edges, relations, entity_ids, contrast_ids)
    for item in indexed:
        relation = item["relation"]
        left_study = str(relation["left_study_id"])
        right_study = str(relation["right_study_id"])
        compatibility = []
        for left_entity, right_entity in zip(
            item["left_entities"], item["right_entities"]
        ):
            left_mz = precursors.get((left_study, str(left_entity)), np.nan)
            right_mz = precursors.get((right_study, str(right_entity)), np.nan)
            ppm = (
                abs(left_mz - right_mz) / ((left_mz + right_mz) / 2.0) * 1e6
                if np.isfinite(left_mz) and np.isfinite(right_mz) else np.nan
            )
            compatibility.append(bool(np.isfinite(ppm) and ppm <= args.precursor_ppm))
        item["edges"] = item["edges"].copy()
        item["edges"]["precursor_compatible_10ppm"] = compatibility

    effect_q_lookup = {
        (str(row.study_id), str(row.contrast_id), str(row.entity_id)): float(row.effect_q)
        for row in effects.itertuples(index=False)
    }

    rng = np.random.default_rng(args.seed)
    permutation_seeds = rng.integers(
        0, np.iinfo(np.int64).max, size=args.permutations, dtype=np.int64
    )
    null_counts = np.zeros(
        (len(indexed), args.permutations, len(TRACK_LABELS)), dtype=np.int32
    )
    for pindex, seed in enumerate(permutation_seeds):
        prng = np.random.default_rng(int(seed))
        donors = {
            study: prng.permutation(len(entity_ids[study])) for study in sorted(studies)
        }
        for rindex, item in enumerate(indexed):
            dual, _direction = count_support(item, significant, signs, donors)
            edge_block = item["edges"]
            null_counts[rindex, pindex] = [
                int((dual & mask).sum()) for mask in support_masks(edge_block)
            ]

    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for rindex, item in enumerate(indexed):
        relation = item["relation"]
        block = item["edges"].copy().reset_index(drop=True)
        left_study = str(relation["left_study_id"])
        right_study = str(relation["right_study_id"])
        left_rows, right_rows = item["left_rows"], item["right_rows"]
        left_column, right_column = int(item["left_column"]), int(item["right_column"])
        dual, direction_product = count_support(item, significant, signs)
        left_weight = weights[left_study][left_rows, left_column]
        right_weight = weights[right_study][right_rows, right_column]
        for edge_index, edge in block.iterrows():
            left_entity = str(item["left_entities"][edge_index])
            right_entity = str(item["right_entities"][edge_index])
            left_mz = precursors.get((left_study, left_entity), np.nan)
            right_mz = precursors.get((right_study, right_entity), np.nan)
            ppm = (
                abs(left_mz - right_mz) / ((left_mz + right_mz) / 2.0) * 1e6
                if np.isfinite(left_mz) and np.isfinite(right_mz) else np.nan
            )
            detail_rows.append({
                **relation,
                "left_entity_id": left_entity,
                "right_entity_id": right_entity,
                "chemical_similarity": float(edge["chemical_similarity"]),
                "mutual_top1": bool(edge["mutual_top1"]),
                "left_precursor_mz": left_mz,
                "right_precursor_mz": right_mz,
                "precursor_ppm_difference": ppm,
                "precursor_compatible_10ppm": bool(np.isfinite(ppm) and ppm <= args.precursor_ppm),
                "left_effect_q": effect_q_lookup.get(
                    (left_study, str(relation["left_contrast_id"]), left_entity), 1.0
                ),
                "right_effect_q": effect_q_lookup.get(
                    (right_study, str(relation["right_contrast_id"]), right_entity), 1.0
                ),
                "same_direction": bool(direction_product[edge_index] > 0),
                "dual_fdr_same_direction": bool(dual[edge_index]),
                "signed_edge_contribution": float(
                    edge["chemical_similarity"] * left_weight[edge_index] * right_weight[edge_index]
                ),
            })
        observed = np.asarray([
            int((dual & mask).sum()) for mask in support_masks(block)
        ])
        result = {**relation, "frozen_unresolved_edges": len(block)}
        for index, label in enumerate(TRACK_LABELS):
            result[f"{label}_dual_fdr_same_direction"] = int(observed[index])
            result[f"{label}_null_mean"] = float(null_counts[rindex, :, index].mean())
            result[f"{label}_empirical_p"] = float(
                (1 + (null_counts[rindex, :, index] >= observed[index]).sum())
                / (args.permutations + 1)
            )
        summary_rows.append(result)

    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)
    args.output_dir.mkdir(parents=True)
    outputs = {
        "detail": args.output_dir / "unresolved_edge_response_evidence.tsv.gz",
        "summary": args.output_dir / "registered_relation_edge_replication_summary.tsv",
        "seeds": args.output_dir / "permutation_seeds.tsv",
    }
    compression = {"method": "gzip", "mtime": 0}
    detail.to_csv(outputs["detail"], sep="\t", index=False, compression=compression)
    summary.to_csv(outputs["summary"], sep="\t", index=False)
    pd.DataFrame({"permutation_index": range(args.permutations), "seed": permutation_seeds}).to_csv(
        outputs["seeds"], sep="\t", index=False
    )

    result_report = {
        "schema": "contrast_atlas_unresolved_edge_response_replication_v1",
        "status": "DEVELOPMENT_CONSUMED_EDGE_REPLICATION_COMPLETE",
        "data_role": "development_consumed",
        "denominators": {
            "registered_relations": len(relations),
            "relation_classes": relation_classes,
            "unresolved_effect_rows": len(effects),
            "relation_edge_evaluations": len(detail),
            "unique_spectral_edges_involved": int(len(detail.drop_duplicates([
                "left_study_id", "right_study_id", "left_entity_id", "right_entity_id"
            ]))),
            "permutations": args.permutations,
        },
        "result": {
            "relations_with_any_dual_fdr_same_direction_edge": int(
                summary["all_edges_dual_fdr_same_direction"].gt(0).sum()
            ),
            "total_dual_fdr_same_direction_edges": int(
                summary["all_edges_dual_fdr_same_direction"].sum()
            ),
            "precursor_compatible_dual_edges": int(
                summary["precursor_compatible_10ppm_dual_fdr_same_direction"].sum()
            ),
            "chemical_relation_only_dual_edges": int(
                summary["chemical_relation_only_dual_fdr_same_direction"].sum()
            ),
            "relations_at_minimum_empirical_p": int(
                summary["all_edges_empirical_p"].le(1 / (args.permutations + 1)).sum()
            ),
            "mutual_top1_dual_edges": int(
                summary["mutual_top1_all_dual_fdr_same_direction"].sum()
            ),
            "relations_with_mutual_top1_support": int(
                summary["mutual_top1_all_dual_fdr_same_direction"].gt(0).sum()
            ),
            "relations_with_mutual_top1_minimum_empirical_p": int(
                summary["mutual_top1_all_empirical_p"].le(1 / (args.permutations + 1)).sum()
            ),
            "mutual_top1_precursor_compatible_dual_edges": int(
                summary["mutual_top1_precursor_compatible_10ppm_dual_fdr_same_direction"].sum()
            ),
            "mutual_top1_chemical_relation_only_dual_edges": int(
                summary["mutual_top1_chemical_relation_only_dual_fdr_same_direction"].sum()
            ),
            "relations_with_mutual_top1_precursor_compatible_minimum_empirical_p": int(
                summary["mutual_top1_precursor_compatible_10ppm_empirical_p"].le(
                    1 / (args.permutations + 1)
                ).sum()
            ),
            "relations_with_mutual_top1_chemical_relation_only_minimum_empirical_p": int(
                summary["mutual_top1_chemical_relation_only_empirical_p"].le(
                    1 / (args.permutations + 1)
                ).sum()
            ),
        },
        "leakage_status": "spectral edges and the requested registered-relation subset were frozen upstream; no edge was selected from effect direction or significance; null reassigns whole entity response profiles within each study",
        "claim_boundary": [
            "Both-endpoint FDR and same direction establish cross-accession response support, not molecular mechanism.",
            "Ten-ppm precursor compatibility is an identity hypothesis, not proof against isomers or in-source/adduct ambiguity.",
            "Non-mass-compatible edges are chemical relation evidence and their beta values are never meta-analysed as one entity.",
            "Every queried source is development-consumed; independent accessions do not by themselves constitute sealed external validation.",
            "Biological comparability is limited to the evidence boundaries frozen in the registered relation table.",
        ],
        "registered_evidence_boundaries": sorted(
            relations["evidence_boundary"].astype(str).unique().tolist()
        ),
        "provenance": {
            "contract_sha256": sha256(args.contract),
            "contract_lineage": {
                str(path.relative_to(root)): sha256(path) for path in contract_paths
            },
            "response_entities_sha256": sha256(paths["entities"]),
            "identity_partition_sha256": sha256(args.identity_partition),
            "frozen_edges_sha256": sha256(paths["edges"]),
            **{name: sha256(path) for name, path in outputs.items()},
            "implementation_sha256": sha256(Path(__file__)),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(result_report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result_report, indent=2))


if __name__ == "__main__":
    main()
