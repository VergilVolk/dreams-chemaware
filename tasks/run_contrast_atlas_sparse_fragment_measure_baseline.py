#!/usr/bin/env python3
"""Truth-blind sparse fragment-bin baseline for the open response collection.

This is deliberately a transparent algorithmic baseline, not the final Contrast
Atlas method.  Every spectrum is represented by 0.1 Da fragment bins after a
per-peak square-root transform and per-spectrum L2 normalisation.  Study-level
positive and negative response measures use beta / sqrt(beta^2 + se^2).

Cross-study registered relations are opened only after the complete fixed-
candidate score table has been written.  Unregistered pairs remain
unadjudicated and are never converted into negative examples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from build_contrast_atlas_open_response_collection import load_contract


WORLDS = ("all", "known", "unresolved")
SCORE_COLUMNS = (
    "signed_similar_score",
    "signed_opposite_score",
    "unsigned_response_score",
    "composition_score",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise_entity_id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            number = float(text)
            if number.is_integer():
                return str(int(number))
        except ValueError:
            pass
    return text


def iter_mgf(path: Path) -> Iterable[tuple[dict[str, str], list[tuple[float, float]]]]:
    metadata: dict[str, str] | None = None
    peaks: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "BEGIN IONS":
                if metadata is not None:
                    raise ValueError(f"nested BEGIN IONS in {path}")
                metadata, peaks = {}, []
            elif line == "END IONS":
                if metadata is None:
                    raise ValueError(f"END IONS without BEGIN IONS in {path}")
                yield metadata, peaks
                metadata, peaks = None, []
            elif metadata is not None and line:
                if "=" in line:
                    key, value = line.split("=", 1)
                    metadata[key.strip().upper()] = value.strip()
                else:
                    fields = line.replace("\t", " ").split()
                    if len(fields) >= 2:
                        try:
                            peaks.append((float(fields[0]), float(fields[1])))
                        except ValueError:
                            continue
    if metadata is not None:
        raise ValueError(f"unterminated spectrum in {path}")


def spectrum_id(metadata: dict[str, str]) -> str:
    for key in ("FEATURE_ID", "SCANS"):
        value = metadata.get(key)
        if value:
            return normalise_entity_id(value)
    raise ValueError("spectrum lacks FEATURE_ID and SCANS")


def sparse_spectrum(
    peaks: list[tuple[float, float]], bin_width: float
) -> tuple[list[int], list[float]]:
    binned: dict[int, float] = defaultdict(float)
    for mz, intensity in peaks:
        if not (math.isfinite(mz) and math.isfinite(intensity)):
            continue
        if mz < 0 or intensity <= 0:
            continue
        # Half-open bins [k*w, (k+1)*w); transform each peak before summing.
        index = int(math.floor(mz / bin_width + 1e-10))
        binned[index] += math.sqrt(intensity)
    if not binned:
        return [], []
    norm = math.sqrt(sum(value * value for value in binned.values()))
    if norm <= 0:
        return [], []
    indices = sorted(binned)
    return indices, [binned[index] / norm for index in indices]


def load_study_spectra(
    mgf_path: Path,
    required_ids: set[str],
    bin_width: float,
) -> tuple[list[str], sparse.csr_matrix, dict[str, int]]:
    rows: list[str] = []
    row_indices: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    seen: set[str] = set()
    spectra_total = 0
    spectra_nonempty = 0
    max_bin = -1
    for metadata, peaks in iter_mgf(mgf_path):
        spectra_total += 1
        entity = spectrum_id(metadata)
        if entity not in required_ids:
            continue
        if entity in seen:
            raise ValueError(f"duplicate representative spectrum {entity} in {mgf_path}")
        seen.add(entity)
        indices, intensities = sparse_spectrum(peaks, bin_width)
        if not indices:
            continue
        row = len(rows)
        rows.append(entity)
        spectra_nonempty += 1
        row_indices.extend([row] * len(indices))
        columns.extend(indices)
        values.extend(intensities)
        max_bin = max(max_bin, indices[-1])
    missing = required_ids - set(rows)
    if missing:
        preview = sorted(missing)[:10]
        raise ValueError(
            f"{len(missing)} response entities lack nonempty representative spectra in "
            f"{mgf_path}: {preview}"
        )
    matrix = sparse.csr_matrix(
        (np.asarray(values, dtype=np.float32), (row_indices, columns)),
        shape=(len(rows), max_bin + 1),
        dtype=np.float32,
    )
    return rows, matrix, {
        "mgf_spectra_total": spectra_total,
        "response_spectra_nonempty": spectra_nonempty,
        "fragment_nonzero_bins": int(matrix.nnz),
        "maximum_bin_index": max_bin,
    }


def pad_columns(matrix: sparse.csr_matrix, dimensions: int) -> sparse.csr_matrix:
    if matrix.shape[1] == dimensions:
        return matrix
    if matrix.shape[1] > dimensions:
        raise ValueError("cannot shrink sparse spectrum matrix")
    return sparse.hstack(
        [matrix, sparse.csr_matrix((matrix.shape[0], dimensions - matrix.shape[1]))],
        format="csr",
        dtype=np.float32,
    )


def weighted_centroid(matrix: sparse.csr_matrix, weights: np.ndarray) -> sparse.csr_matrix:
    total = float(weights.sum())
    if total <= 0 or matrix.shape[0] == 0:
        return sparse.csr_matrix((1, matrix.shape[1]), dtype=np.float32)
    return sparse.csr_matrix(weights.astype(np.float32)[None, :] @ matrix / total)


def unweighted_centroid(matrix: sparse.csr_matrix) -> sparse.csr_matrix:
    if matrix.shape[0] == 0:
        return sparse.csr_matrix((1, matrix.shape[1]), dtype=np.float32)
    return sparse.csr_matrix(matrix.mean(axis=0), dtype=np.float32)


def dot(left: sparse.csr_matrix, right: sparse.csr_matrix) -> float:
    return float(left.multiply(right).sum())


def truth_blind_scores(
    manifest: pd.DataFrame,
    sketches: dict[tuple[int, str], dict[str, sparse.csr_matrix | int]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for query_index, query in manifest.iterrows():
        for target_index, target in manifest.iterrows():
            if str(query["source_family"]) == str(target["source_family"]):
                continue
            for world in WORLDS:
                q = sketches[(query_index, world)]
                t = sketches[(target_index, world)]
                q_count, t_count = int(q["n_entities"]), int(t["n_entities"])
                score_available = q_count > 0 and t_count > 0
                record: dict[str, object] = {
                    "query_study_id": str(query["study_id"]),
                    "query_source_family": str(query["source_family"]),
                    "query_contrast_id": str(query["contrast_id"]),
                    "candidate_study_id": str(target["study_id"]),
                    "candidate_source_family": str(target["source_family"]),
                    "candidate_contrast_id": str(target["contrast_id"]),
                    "evidence_world": world,
                    "query_entities": q_count,
                    "candidate_entities": t_count,
                    "score_available": score_available,
                    "adjudication_status": "unadjudicated",
                }
                if score_available:
                    pp = dot(q["positive"], t["positive"])
                    nn = dot(q["negative"], t["negative"])
                    pn = dot(q["positive"], t["negative"])
                    np_ = dot(q["negative"], t["positive"])
                    signed = 0.5 * ((pp + nn) - (pn + np_))
                    record.update(
                        {
                            "positive_positive": pp,
                            "negative_negative": nn,
                            "positive_negative": pn,
                            "negative_positive": np_,
                            "signed_similar_score": signed,
                            "signed_opposite_score": -signed,
                            "unsigned_response_score": 0.5 * ((pp + nn) + (pn + np_)),
                            "composition_score": dot(q["composition"], t["composition"]),
                        }
                    )
                else:
                    for column in (
                        "positive_positive",
                        "negative_negative",
                        "positive_negative",
                        "negative_positive",
                        *SCORE_COLUMNS,
                    ):
                        record[column] = np.nan
                rows.append(record)
    scores = pd.DataFrame(rows)
    for score in SCORE_COLUMNS:
        rank_column = score.replace("_score", "_absolute_rank")
        scores[rank_column] = np.nan
        grouped = scores.groupby(
            ["query_study_id", "query_contrast_id", "evidence_world"], sort=False
        )
        for _, indices in grouped.groups.items():
            block = scores.loc[indices, score]
            present = block.notna()
            scores.loc[block.index[present], rank_column] = block.loc[present].rank(
                ascending=False, method="min"
            )
    return scores


def registered_relation_audit(scores: pd.DataFrame, relations: pd.DataFrame) -> pd.DataFrame:
    relation_keys = [
        "left_study_id",
        "left_contrast_id",
        "right_study_id",
        "right_contrast_id",
    ]
    if relations.empty:
        return pd.DataFrame()
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
            raise ValueError(f"registered relation did not match three worlds: {relation}")
        for row in matched.to_dict("records"):
            result = {key: relation[key] for key in relation_keys}
            result.update(
                {
                    "relation_direction": relation.get("relation_direction", "registered"),
                    "relation_class": relation.get("relation_class", ""),
                    "expected_signed_direction": relation.get("expected_signed_direction", ""),
                    "evidence_world": row["evidence_world"],
                    "candidate_universe_size": int(
                        len(
                            scores.loc[
                                scores["query_study_id"].eq(row["query_study_id"])
                                & scores["query_contrast_id"].eq(row["query_contrast_id"])
                                & scores["evidence_world"].eq(row["evidence_world"])
                            ]
                        )
                    ),
                    "score_available": row["score_available"],
                    "adjudication_status": (
                        "DESCRIPTIVE_DESIGN_RELEVANCE_ONLY_SIGNED_DIRECTION_UNSPECIFIED"
                    ),
                    "pass_fail": "NOT_ADJUDICATED",
                    "evidence_boundary": relation.get("evidence_boundary", ""),
                }
            )
            for score in SCORE_COLUMNS:
                result[score] = row[score]
                result[score.replace("_score", "_absolute_rank")] = row[
                    score.replace("_score", "_absolute_rank")
                ]
            output.append(result)
    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument(
        "--identity-partition",
        type=Path,
        help="Complete common-policy identity partition replacing source-native labels.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bin-width-da", type=float, default=0.1)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if not args.bin_width_da > 0:
        raise ValueError("bin width must be positive")

    root = Path(__file__).resolve().parents[1]
    contract, contract_paths = load_contract(args.contract.resolve(), root)
    entities_path = args.collection / "response_entities.tsv.gz"
    contrast_path = args.collection / "contrast_manifest.tsv"
    relations_path = args.collection / "registered_relations.tsv"
    entities = pd.read_csv(entities_path, sep="\t", dtype={"raw_entity_id": str})
    entities["raw_entity_id"] = entities["raw_entity_id"].map(normalise_entity_id)
    identity_provenance: dict[str, str] = {}
    if args.identity_partition is not None:
        identity_report_path = args.identity_partition.parent / "report.json"
        if not args.identity_partition.is_file() or not identity_report_path.is_file():
            raise FileNotFoundError(args.identity_partition)
        identity_report = json.loads(identity_report_path.read_text(encoding="utf-8"))
        if identity_report.get("status") != "PHENOTYPE_BLIND_COMMON_STRICT_LIBRARY_PARTITIONS_COMPLETE":
            raise RuntimeError("identity partition is not a complete common-policy artifact")
        if identity_report.get("provenance", {}).get("combined_partition_sha256") != sha256(
            args.identity_partition
        ):
            raise RuntimeError("common identity partition checksum drift")
        identity = pd.read_csv(
            args.identity_partition,
            sep="\t",
            usecols=["study_id", "entity_id", "identity_status"],
            dtype=str,
        )
        if identity.duplicated(["study_id", "entity_id"]).any():
            raise RuntimeError("common identity partition contains duplicate entity keys")
        original_columns = list(entities.columns)
        base = entities.drop(columns="identity_status")
        base_keys = base[["study_id", "entity_id"]].drop_duplicates()
        if len(identity) != len(base_keys) or set(map(tuple, identity[["study_id", "entity_id"]].to_numpy())) != set(
            map(tuple, base_keys[["study_id", "entity_id"]].to_numpy())
        ):
            raise RuntimeError("common identity partition does not match response entities")
        entities = base.merge(
            identity, on=["study_id", "entity_id"], how="left", validate="many_to_one"
        )[original_columns]
        identity_provenance = {
            str(args.identity_partition): sha256(args.identity_partition),
            str(identity_report_path): sha256(identity_report_path),
        }
    contrasts = pd.read_csv(contrast_path, sep="\t")
    source_contract = {str(item["study_id"]): item for item in contract["sources"]}
    if set(contrasts["study_id"].astype(str)) != set(source_contract):
        raise ValueError("contract sources and frozen contrast manifest differ")
    if not np.isfinite(entities[["beta", "se"]].to_numpy(float)).all():
        raise ValueError("response entities contain non-finite beta or se")
    if (entities["se"].to_numpy(float) < 0).any():
        raise ValueError("response entities contain negative standard errors")

    per_source: dict[str, tuple[list[str], sparse.csr_matrix]] = {}
    source_reports: list[dict[str, object]] = []
    dimensions = 0
    for study_id in contrasts["study_id"].astype(str).drop_duplicates():
        block = entities.loc[entities["study_id"].astype(str).eq(study_id)]
        required = set(block["raw_entity_id"])
        mgf_path = Path(source_contract[study_id]["mgf_path"])
        ids, matrix, details = load_study_spectra(mgf_path, required, args.bin_width_da)
        per_source[study_id] = (ids, matrix)
        dimensions = max(dimensions, matrix.shape[1])
        source_reports.append(
            {
                "study_id": study_id,
                "mgf_path": str(mgf_path),
                "mgf_sha256": sha256(mgf_path),
                "required_response_entities": len(required),
                **details,
            }
        )

    sketches: dict[tuple[int, str], dict[str, sparse.csr_matrix | int]] = {}
    for contrast_index, contrast in contrasts.reset_index(drop=True).iterrows():
        study_id = str(contrast["study_id"])
        contrast_id = str(contrast["contrast_id"])
        ids, raw_matrix = per_source[study_id]
        matrix = pad_columns(raw_matrix, dimensions)
        row_lookup = {entity: row for row, entity in enumerate(ids)}
        block = entities.loc[
            entities["study_id"].astype(str).eq(study_id)
            & entities["contrast_id"].astype(str).eq(contrast_id)
        ].copy()
        if block["raw_entity_id"].duplicated().any():
            raise ValueError(f"duplicate effect entity in {study_id}/{contrast_id}")
        block["spectrum_row"] = block["raw_entity_id"].map(row_lookup)
        if block["spectrum_row"].isna().any():
            raise ValueError(f"effect-to-spectrum join incomplete in {study_id}/{contrast_id}")
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
            rows = block.loc[mask, "spectrum_row"].to_numpy(int)
            selected = matrix[rows]
            weights = signed_weight[mask]
            sketches[(contrast_index, world)] = {
                "n_entities": int(mask.sum()),
                "positive": weighted_centroid(selected, np.maximum(weights, 0)),
                "negative": weighted_centroid(selected, np.maximum(-weights, 0)),
                "composition": unweighted_centroid(selected),
            }

    # Truth-blind scoring is complete before the registered-relation table is read.
    frozen_manifest = contrasts.reset_index(drop=True)
    scores = truth_blind_scores(frozen_manifest, sketches)
    args.output_dir.mkdir(parents=True)
    scores_path = args.output_dir / "fixed_candidate_scores.tsv.gz"
    scores.to_csv(scores_path, sep="\t", index=False, compression="gzip")

    relations = pd.read_csv(relations_path, sep="\t", dtype=str)
    relation_audit = registered_relation_audit(scores, relations)
    relation_path = args.output_dir / "registered_relation_absolute_ranks.tsv"
    relation_audit.to_csv(relation_path, sep="\t", index=False)

    available = scores["score_available"].astype(bool)
    report = {
        "schema": "contrast_atlas_sparse_fragment_measure_baseline_v1",
        "status": "DEVELOPMENT_CONSUMED_TRUTH_BLIND_ALGORITHMIC_BASELINE_COMPLETE",
        "data_role": "development_consumed",
        "method": {
            "fragment_bin_width_da": args.bin_width_da,
            "fragment_bin_rule": "half-open floor(mz/bin_width) bins",
            "spectrum_transform": "sqrt(each positive peak intensity), sum within bin, L2-normalise each spectrum",
            "response_weight": "beta/sqrt(beta^2+se^2)",
            "positive_negative_measure": "absolute response-weighted mean of unit-L2 spectrum vectors; centroids are not renormalised",
            "signed_similar": "0.5*((positive-positive + negative-negative) - (positive-negative + negative-positive))",
            "signed_opposite": "negative signed_similar",
            "unsigned": "0.5*((positive-positive + negative-negative) + (positive-negative + negative-positive))",
            "composition": "dot product of unweighted mean unit-L2 spectrum vectors; beta and se unused",
            "identity_policy": (
                "complete common GNPS Gold/Silver strict partition"
                if args.identity_partition is not None
                else "source-native development partitions"
            ),
        },
        "denominators": {
            "independent_studies": int(frozen_manifest["study_id"].nunique()),
            "source_families": int(frozen_manifest["source_family"].nunique()),
            "study_level_contrasts": len(frozen_manifest),
            "biological_units_across_sources": int(
                pd.read_csv(args.collection / "source_manifest.tsv", sep="\t")[
                    "biological_units"
                ].sum()
            ),
            "eligible_response_rows": len(entities),
            "fixed_source_disjoint_candidate_rows_per_world": int(len(scores) / len(WORLDS)),
            "fixed_source_disjoint_candidate_rows_total": len(scores),
            "score_available_rows": int(available.sum()),
            "score_unavailable_rows_retained": int((~available).sum()),
            "registered_directed_relevance_relations": len(relations),
            "registered_signed_direction_truth_relations": int(
                (~relations["expected_signed_direction"].str.lower().eq("unspecified")).sum()
            ),
            "fragment_dimensions": dimensions,
        },
        "truth_boundary": {
            "scoring": "registered relations were not read until after all fixed-candidate scores and ranks were computed",
            "unregistered_pairs": "unadjudicated, never negative",
            "registered_design_relations": "design relevance only; absolute ranks are descriptive",
            "pass_fail": "not adjudicated because expected signed direction is unspecified",
        },
        "sources": source_reports,
        "input_sha256": {
            str(args.contract): sha256(args.contract),
            **{
                f"contract_lineage::{path.relative_to(root)}": sha256(path)
                for path in contract_paths
            },
            str(entities_path): sha256(entities_path),
            str(contrast_path): sha256(contrast_path),
            str(relations_path): sha256(relations_path),
            **identity_provenance,
        },
        "claim_boundary": [
            "This sparse fragment-bin implementation is a transparent baseline, not the final Contrast Atlas method.",
            "All collection sources and their prior pairwise analyses are development/consumed.",
            "Unresolved means not library-supported under the frozen common partition, not a unique unknown metabolite.",
            "Feature rows may include adduct and isotope redundancy.",
            "No unregistered pair is a negative example.",
            "No registered relation is a signed-response PASS/FAIL endpoint unless signed direction truth was preregistered.",
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
