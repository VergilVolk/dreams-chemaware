#!/usr/bin/env python3
"""Compare the legacy fixed graph with the frozen complete identity axis.

This ledger records candidate-axis changes separately from model decisions.
Full23 decisions remain explicitly pending until complete-block application
scores are delivered; missing scores are never backfilled from another method.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def finite_float(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def canonical_key(
    study_a: str, entity_a: str, study_b: str, entity_b: str
) -> tuple[str, str, str, str]:
    left = (study_a, entity_a)
    right = (study_b, entity_b)
    return (*left, *right) if left < right else (*right, *left)


def stable_id(prefix: str, *parts: str) -> str:
    return f"{prefix}_{hashlib.sha256(chr(9).join(parts).encode('utf-8')).hexdigest()}"


def read_gzip_tsv(path: Path):
    handle = gzip.open(path, "rt", encoding="utf-8-sig", newline="")
    return handle, csv.DictReader(handle, delimiter="\t")


def write_gzip_tsv(path: Path, fields: list[str]):
    raw = gzip.GzipFile(filename=str(path), mode="wb", mtime=0)
    text = __import__("io").TextIOWrapper(raw, encoding="utf-8", newline="")
    writer = csv.DictWriter(text, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    return text, writer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old-handoff",
        default="data/validation/contrast_atlas_open_local_edge_enrichment_v5_20261027/frozen_mutual_edge_algorithm_handoff.tsv.gz",
    )
    parser.add_argument(
        "--new-axis",
        default="data/validation/contrast_atlas_model_independent_identity_axis_v5_20261027",
    )
    parser.add_argument(
        "--official-scores",
        default="data/validation/contrast_atlas_fixed_edge_embedding_scores_official_v1_20261027/fixed_edge_embedding_scores.tsv.gz",
    )
    parser.add_argument(
        "--stage1-scores",
        default="data/validation/contrast_atlas_fixed_edge_embedding_scores_stage1_v2_20261027/fixed_edge_embedding_scores.tsv.gz",
    )
    parser.add_argument(
        "--collection-dir",
        default="data/validation/contrast_atlas_open_response_collection_v9_frozenbase_chemprop_20261023",
    )
    parser.add_argument(
        "--identity-partition",
        default="data/validation/contrast_atlas_open_v9_frozenbase_common_identity_20261023/identity_partition.tsv.gz",
    )
    parser.add_argument(
        "--output-dir",
        default="data/validation/contrast_atlas_entity_change_ledger_v7_20261027",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    old_path = (root / args.old_handoff).resolve()
    axis_dir = (root / args.new_axis).resolve()
    new_pair_path = axis_dir / "directed_complete_candidate_pairs.tsv.gz"
    official_path = (root / args.official_scores).resolve()
    stage1_path = (root / args.stage1_scores).resolve()
    collection = (root / args.collection_dir).resolve()
    response_path = collection / "response_entities.tsv.gz"
    relations_path = collection / "registered_relations.tsv"
    identity_partition_path = (root / args.identity_partition).resolve()
    output = (root / args.output_dir).resolve()
    required = [
        old_path,
        axis_dir / "report.json",
        new_pair_path,
        official_path,
        stage1_path,
        response_path,
        relations_path,
        identity_partition_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing input(s): " + "; ".join(missing))
    if output.exists():
        raise FileExistsError(output)

    axis_report = json.loads((axis_dir / "report.json").read_text(encoding="utf-8"))
    if axis_report.get("status") != "DEVELOPMENT_CONSUMED_MODEL_INDEPENDENT_IDENTITY_AXIS_FROZEN":
        raise ValueError("new candidate axis is not frozen")

    old: dict[tuple[str, str, str, str], dict[str, str]] = {}
    handle, reader = read_gzip_tsv(old_path)
    try:
        for row in reader:
            if row["evidence_world"] != "all":
                continue
            key = canonical_key(
                row["left_study_id"], row["left_entity_id"],
                row["right_study_id"], row["right_entity_id"],
            )
            if key in old:
                raise ValueError(f"duplicate all-world old edge: {key}")
            old[key] = row
    finally:
        handle.close()

    new: dict[tuple[str, str, str, str], dict[str, str]] = {}
    handle, reader = read_gzip_tsv(new_pair_path)
    try:
        for row in reader:
            key = canonical_key(
                row["query_study_id"], row["query_entity_id"],
                row["candidate_study_id"], row["candidate_entity_id"],
            )
            if key in new:
                # The file is directed; the reverse orientation must be chemically identical.
                previous = new[key]
                if previous["physicochemical_compatibility_grade"] != row["physicochemical_compatibility_grade"]:
                    raise ValueError(f"asymmetric candidate chemistry: {key}")
                continue
            new[key] = row
    finally:
        handle.close()
    if len(new) * 2 != int(axis_report["denominators"]["directed_candidate_pairs"]):
        raise ValueError("new directed axis is not exactly symmetric")

    score_maps: dict[str, dict[tuple[str, str, str, str], float | None]] = {}
    for method, path, column in (
        ("official_dreams", official_path, "official_dreams_score"),
        ("chemaware_stage1", stage1_path, "chemaware_stage1_score"),
    ):
        mapping: dict[tuple[str, str, str, str], float | None] = {}
        handle, reader = read_gzip_tsv(path)
        try:
            for row in reader:
                if row["evidence_world"] != "all":
                    continue
                key = canonical_key(
                    row["left_study_id"], row["left_entity_id"],
                    row["right_study_id"], row["right_entity_id"],
                )
                mapping[key] = finite_float(row.get(column, ""))
        finally:
            handle.close()
        if set(mapping) != set(old):
            raise ValueError(f"{method} score keys differ from old all-world axis")
        score_maps[method] = mapping

    identity_status: dict[tuple[str, str], str] = {}
    handle, reader = read_gzip_tsv(identity_partition_path)
    try:
        for row in reader:
            key = (row["study_id"], row["entity_id"])
            if key in identity_status:
                raise ValueError(f"duplicate common identity row: {key}")
            identity_status[key] = row["identity_status"]
    finally:
        handle.close()
    if len(identity_status) != 54861:
        raise ValueError(f"unexpected common identity denominator: {len(identity_status)}")

    effects: dict[tuple[str, str, str], tuple[float, float, str]] = {}
    handle, reader = read_gzip_tsv(response_path)
    try:
        for row in reader:
            beta, se = finite_float(row["beta"]), finite_float(row["se"])
            if beta is None or se is None or se <= 0:
                raise ValueError("frozen response collection contains invalid beta/SE")
            key = (row["study_id"], row["contrast_id"], row["entity_id"])
            entity_key = (row["study_id"], row["entity_id"])
            if entity_key not in identity_status:
                raise ValueError(f"response entity absent from common identity partition: {entity_key}")
            effects[key] = (beta, se, identity_status[entity_key])
    finally:
        handle.close()

    with relations_path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_relations = list(csv.DictReader(handle, delimiter="\t"))
    canonical_relations: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in raw_relations:
        left = (row["left_study_id"], row["left_contrast_id"])
        right = (row["right_study_id"], row["right_contrast_id"])
        key = (*left, *right) if left < right else (*right, *left)
        canonical_relations.setdefault(key, row)
    relation_by_study_pair: dict[tuple[str, str], list[tuple[tuple[str, str, str, str], dict[str, str]]]] = defaultdict(list)
    for relation_key, row in canonical_relations.items():
        relation_by_study_pair[(relation_key[0], relation_key[2])].append((relation_key, row))

    old_identity_degree: Counter[tuple[str, str]] = Counter()
    new_identity_degree: Counter[tuple[str, str]] = Counter()
    for key, row in old.items():
        if truthy(row["precursor_compatible_10ppm"]):
            old_identity_degree[(key[0], key[1])] += 1
            old_identity_degree[(key[2], key[3])] += 1
    for key in new:
        new_identity_degree[(key[0], key[1])] += 1
        new_identity_degree[(key[2], key[3])] += 1

    output.mkdir(parents=True)
    ledger_path = output / "entity_change_ledger.tsv.gz"
    response_long_path = output / "registered_response_evidence.tsv.gz"
    ledger_fields = [
        "canonical_pair_id", "left_to_right_query_block_id", "right_to_left_query_block_id",
        "left_to_right_directed_pair_id", "right_to_left_directed_pair_id",
        "left_study_id", "left_entity_id", "right_study_id", "right_entity_id",
        "left_identity_status", "right_identity_status", "old_axis_present",
        "old_axis_track", "new_identity_axis_present", "candidate_axis_change",
        "old_fragment_cosine", "official_dreams_score", "chemaware_stage1_score",
        "noise_left_to_right_score", "noise_right_to_left_score",
        "wse_left_to_right_score", "wse_right_to_left_score",
        "full23_v1_left_to_right_score", "full23_v1_right_to_left_score",
        "full23_v11_left_to_right_score", "full23_v11_right_to_left_score",
        "noise_score_availability", "wse_score_availability",
        "full23_v1_score_availability", "full23_v11_score_availability",
        "v2_structure_evidence_status",
        "model_decision_change", "physicochemical_compatibility_grade",
        "left_precursor_mz", "right_precursor_mz", "precursor_ppm",
        "old_identity_degree_left", "old_identity_degree_right",
        "new_identity_degree_left", "new_identity_degree_right",
        "registered_response_contexts", "same_sign_contexts", "opposite_sign_contexts",
        "both_abs_z_ge_1_96_contexts", "independent_validation_status",
        "contradiction_status", "adjudication_status",
    ]
    response_fields = [
        "left_study_id", "left_entity_id", "left_contrast_id", "left_beta", "left_se", "left_z",
        "right_study_id", "right_entity_id", "right_contrast_id", "right_beta", "right_se", "right_z",
        "relation_class", "eligible_primary_relevance", "signed_truth_claimed",
        "response_sign_relation", "candidate_axis_change", "evidence_role",
    ]
    ledger_text, ledger_writer = write_gzip_tsv(ledger_path, ledger_fields)
    response_text, response_writer = write_gzip_tsv(response_long_path, response_fields)
    status_counts: Counter[str] = Counter()
    response_rows = 0
    union = sorted(set(old) | set(new))
    try:
        for key in union:
            left_study, left_entity, right_study, right_entity = key
            old_row, new_row = old.get(key), new.get(key)
            old_track = "absent"
            if old_row is not None:
                old_track = (
                    "same_precursor_10ppm_identity_hypothesis"
                    if truthy(old_row["precursor_compatible_10ppm"])
                    else "nonmass_chemical_relation_hypothesis"
                )
            if old_row is not None and new_row is not None:
                axis_change = "retained_identity_candidate"
            elif new_row is not None:
                axis_change = "gained_complete_block_identity_candidate"
            elif old_track == "same_precursor_10ppm_identity_hypothesis":
                axis_change = "lost_identity_candidate_explicit_ion_conflict"
            else:
                axis_change = "relation_track_only_not_identity_loss"
            status_counts[axis_change] += 1

            registered_contexts = same_sign = opposite_sign = both_sig = 0
            for relation_key, relation in relation_by_study_pair.get((left_study, right_study), []):
                left_contrast, right_contrast = relation_key[1], relation_key[3]
                left_effect = effects.get((left_study, left_contrast, left_entity))
                right_effect = effects.get((right_study, right_contrast, right_entity))
                if left_effect is None or right_effect is None:
                    continue
                lb, ls, _ = left_effect
                rb, rs, _ = right_effect
                lz, rz = lb / ls, rb / rs
                sign_relation = "same" if lb * rb > 0 else "opposite" if lb * rb < 0 else "zero_involved"
                registered_contexts += 1
                same_sign += int(sign_relation == "same")
                opposite_sign += int(sign_relation == "opposite")
                both_sig += int(abs(lz) >= 1.96 and abs(rz) >= 1.96)
                response_writer.writerow(
                    {
                        "left_study_id": left_study,
                        "left_entity_id": left_entity,
                        "left_contrast_id": left_contrast,
                        "left_beta": lb,
                        "left_se": ls,
                        "left_z": lz,
                        "right_study_id": right_study,
                        "right_entity_id": right_entity,
                        "right_contrast_id": right_contrast,
                        "right_beta": rb,
                        "right_se": rs,
                        "right_z": rz,
                        "relation_class": relation["relation_class"],
                        "eligible_primary_relevance": relation["eligible_primary_relevance"],
                        "signed_truth_claimed": relation["signed_truth_claimed"],
                        "response_sign_relation": sign_relation,
                        "candidate_axis_change": axis_change,
                        "evidence_role": "consumed_descriptive_not_independent_validation",
                    }
                )
                response_rows += 1

            left_mz = (
                new_row["query_precursor_mz"] if new_row and new_row["query_study_id"] == left_study
                else new_row["candidate_precursor_mz"] if new_row
                else old_row["left_precursor_mz"]
            )
            right_mz = (
                new_row["candidate_precursor_mz"] if new_row and new_row["query_study_id"] == left_study
                else new_row["query_precursor_mz"] if new_row
                else old_row["right_precursor_mz"]
            )
            ppm = (
                new_row["symmetric_ppm"] if new_row
                else old_row["precursor_ppm_difference"]
            )
            ledger_writer.writerow(
                {
                    "canonical_pair_id": stable_id(
                        "canonical_pair", left_study, left_entity, right_study, right_entity
                    ),
                    "left_to_right_query_block_id": "" if new_row is None else stable_id(
                        "block", left_study, left_entity, right_study
                    ),
                    "right_to_left_query_block_id": "" if new_row is None else stable_id(
                        "block", right_study, right_entity, left_study
                    ),
                    "left_to_right_directed_pair_id": "" if new_row is None else stable_id(
                        "pair", left_study, left_entity, right_study, right_entity
                    ),
                    "right_to_left_directed_pair_id": "" if new_row is None else stable_id(
                        "pair", right_study, right_entity, left_study, left_entity
                    ),
                    "left_study_id": left_study,
                    "left_entity_id": left_entity,
                    "right_study_id": right_study,
                    "right_entity_id": right_entity,
                    "left_identity_status": identity_status[(left_study, left_entity)],
                    "right_identity_status": identity_status[(right_study, right_entity)],
                    "old_axis_present": str(old_row is not None),
                    "old_axis_track": old_track,
                    "new_identity_axis_present": str(new_row is not None),
                    "candidate_axis_change": axis_change,
                    "old_fragment_cosine": "" if old_row is None else old_row["chemical_similarity"],
                    "official_dreams_score": "" if old_row is None or score_maps["official_dreams"][key] is None else score_maps["official_dreams"][key],
                    "chemaware_stage1_score": "" if old_row is None or score_maps["chemaware_stage1"][key] is None else score_maps["chemaware_stage1"][key],
                    "noise_left_to_right_score": "",
                    "noise_right_to_left_score": "",
                    "wse_left_to_right_score": "",
                    "wse_right_to_left_score": "",
                    "full23_v1_left_to_right_score": "",
                    "full23_v1_right_to_left_score": "",
                    "full23_v11_left_to_right_score": "",
                    "full23_v11_right_to_left_score": "",
                    "noise_score_availability": "not_delivered_on_complete_block_axis",
                    "wse_score_availability": "not_delivered_on_complete_block_axis",
                    "full23_v1_score_availability": "not_delivered_on_complete_block_axis",
                    "full23_v11_score_availability": "experimental_not_delivered_on_complete_block_axis",
                    "v2_structure_evidence_status": "not_evaluated_requires_legal_structure_availability",
                    "model_decision_change": "pending_complete_block_scores",
                    "physicochemical_compatibility_grade": "not_identity_axis" if new_row is None else new_row["physicochemical_compatibility_grade"],
                    "left_precursor_mz": left_mz,
                    "right_precursor_mz": right_mz,
                    "precursor_ppm": ppm,
                    "old_identity_degree_left": old_identity_degree[(left_study, left_entity)],
                    "old_identity_degree_right": old_identity_degree[(right_study, right_entity)],
                    "new_identity_degree_left": new_identity_degree[(left_study, left_entity)],
                    "new_identity_degree_right": new_identity_degree[(right_study, right_entity)],
                    "registered_response_contexts": registered_contexts,
                    "same_sign_contexts": same_sign,
                    "opposite_sign_contexts": opposite_sign,
                    "both_abs_z_ge_1_96_contexts": both_sig,
                    "independent_validation_status": "absent",
                    "contradiction_status": "descriptive_mixed_signs" if same_sign and opposite_sign else "none_observed_not_truth" if registered_contexts else "not_adjudicated",
                    "adjudication_status": "unjudged_candidate" if new_row is not None else "relation_hypothesis_only",
                }
            )
    finally:
        ledger_text.close()
        response_text.close()

    report = {
        "schema": "contrast_atlas_entity_change_ledger_v1",
        "status": "CANDIDATE_AXIS_LEDGER_COMPLETE_FULL23_APPLICATION_PENDING",
        "data_role": "development_consumed",
        "denominators": {
            "old_all_world_unique_edges": len(old),
            "new_canonical_identity_candidates": len(new),
            "union_entity_pairs": len(union),
            "candidate_axis_change": dict(status_counts),
            "registered_response_evidence_rows": response_rows,
        },
        "method_availability": {
            "old_fragment_cosine": "available_on_old_fixed_edges",
            "official_dreams": "available_on_old_fixed_edges",
            "chemaware_stage1": "available_on_old_fixed_edges",
            "noise": "not_delivered_on_new_complete_blocks",
            "wse": "not_delivered_on_new_complete_blocks",
            "full23_v1": "not_delivered_on_new_complete_blocks",
            "full23_v1_1": "experimental_not_delivered_on_new_complete_blocks",
        },
        "interpretation": [
            "Gained/lost/retained refer to candidate-axis membership, not a correct or incorrect model decision.",
            "Legacy nonmass edges remain relation hypotheses and are not counted as lost molecular identities.",
            "Registered response signs are descriptive consumed evidence; no relation supplies signed entity-level truth.",
            "The ledger is ready for an exact-key append of complete-block WSE, Noise, Full23 v1 and later v1.1 scores.",
            "Complete-block method scores are directional because block normalization and rank context can differ by query direction; they must never be collapsed into one undirected value.",
        ],
        "provenance": {
            "old_handoff_sha256": sha256(old_path),
            "new_axis_report_sha256": sha256(axis_dir / "report.json"),
            "new_axis_pairs_sha256": sha256(new_pair_path),
            "official_scores_sha256": sha256(official_path),
            "stage1_scores_sha256": sha256(stage1_path),
            "response_entities_sha256": sha256(response_path),
            "common_identity_partition_sha256": sha256(identity_partition_path),
            "registered_relations_sha256": sha256(relations_path),
            "implementation_sha256": sha256(Path(__file__)),
        },
        "outputs": {
            "entity_change_ledger.tsv.gz": sha256(ledger_path),
            "registered_response_evidence.tsv.gz": sha256(response_long_path),
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    checksum_path = output / "checksums.sha256"
    checksum_path.write_text(
        f"{sha256(ledger_path)}  {ledger_path.name}\n"
        f"{sha256(response_long_path)}  {response_long_path.name}\n"
        f"{sha256(report_path)}  {report_path.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
