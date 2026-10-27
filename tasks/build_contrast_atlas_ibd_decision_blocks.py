#!/usr/bin/env python3
"""Freeze IBD decision blocks and evidence without selecting on model scores.

The handoff contains every model-independent candidate in each relevant
query-by-target-study block.  Response effects are attached only after block
membership is frozen and are never used to admit or remove a candidate.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bh_qvalues(pvalues: list[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    out = [1.0] * len(pvalues)
    running = 1.0
    n = len(pvalues)
    for rank_index in range(n - 1, -1, -1):
        index = order[rank_index]
        rank = rank_index + 1
        running = min(running, pvalues[index] * n / rank)
        out[index] = min(1.0, running)
    return out


def gzip_rows(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_gzip(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    raw = gzip.GzipFile(filename=str(path), mode="wb", mtime=0)
    text = __import__("io").TextIOWrapper(raw, encoding="utf-8", newline="")
    try:
        writer = csv.DictWriter(text, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)
    finally:
        text.close()


def canonical(a_study: str, a_entity: str, b_study: str, b_entity: str) -> tuple[str, str, str, str]:
    left, right = (a_study, a_entity), (b_study, b_entity)
    return (*left, *right) if left < right else (*right, *left)


def registered_pair_role(
    query_entity: str,
    candidate_entity: str,
    shared_sets: list[str],
    definitions: dict[str, dict[str, Any]],
) -> str:
    pair = {query_entity, candidate_entity}
    for set_id in shared_sets:
        definition = definitions[set_id]
        expected = set(definition.get("expected_identity_pair", []))
        if expected and pair == expected:
            return "expected_identity_pair"
        competitor = definition.get("known_competitor")
        if competitor in pair:
            return "known_same_mass_competitor_pair"
    return "registered_response_hypothesis_pair" if shared_sets else "within_block_competitor"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="tasks/contrast_atlas_ibd_decision_blocks_v1.json")
    parser.add_argument(
        "--collection-dir",
        default="data/validation/contrast_atlas_open_response_collection_v9_frozenbase_chemprop_20261023",
    )
    parser.add_argument(
        "--identity-partition",
        default="data/validation/contrast_atlas_open_v9_frozenbase_common_identity_20261023/identity_partition.tsv.gz",
    )
    parser.add_argument(
        "--candidate-axis",
        default="data/validation/contrast_atlas_model_independent_identity_axis_v5_20261027",
    )
    parser.add_argument(
        "--legacy-handoff",
        default="data/validation/contrast_atlas_open_local_edge_enrichment_v5_20261027/frozen_mutual_edge_algorithm_handoff.tsv.gz",
    )
    parser.add_argument(
        "--output-dir",
        default="data/validation/contrast_atlas_ibd_decision_blocks_v5_20261027",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    registry_path = (root / args.registry).resolve()
    collection = (root / args.collection_dir).resolve()
    identity_path = (root / args.identity_partition).resolve()
    axis_dir = (root / args.candidate_axis).resolve()
    pair_path = axis_dir / "directed_complete_candidate_pairs.tsv.gz"
    block_path = axis_dir / "query_target_source_blocks.tsv.gz"
    legacy_path = (root / args.legacy_handoff).resolve()
    output = (root / args.output_dir).resolve()
    required = [registry_path, collection / "response_entities.tsv.gz", identity_path,
                axis_dir / "report.json", pair_path, block_path, legacy_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing input(s): " + "; ".join(missing))
    if output.exists():
        raise FileExistsError(output)

    axis_report = json.loads((axis_dir / "report.json").read_text(encoding="utf-8"))
    if axis_report.get("status") != "DEVELOPMENT_CONSUMED_MODEL_INDEPENDENT_IDENTITY_AXIS_FROZEN":
        raise ValueError("candidate axis is not frozen")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    supporting_evidence_hashes: dict[str, str] = {}
    for source in registry.get("supporting_evidence_sources", []):
        source_path = (root / source["path"]).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"registered supporting evidence is missing: {source_path}")
        observed_hash = sha256(source_path)
        if observed_hash != source["sha256"]:
            raise ValueError(
                f"supporting-evidence hash mismatch for {source['evidence_id']}: "
                f"expected {source['sha256']}, observed {observed_hash}"
            )
        supporting_evidence_hashes[source["path"]] = observed_hash
    sets = {row["candidate_set_id"]: row for row in registry["candidate_sets"]}
    entity_to_sets: dict[str, list[str]] = defaultdict(list)
    for set_id, definition in sets.items():
        for entity in definition["entities"]:
            entity_to_sets[entity].append(set_id)

    identities = {(row["study_id"], row["entity_id"]): row["identity_status"] for row in gzip_rows(identity_path)}
    response_rows = gzip_rows(collection / "response_entities.tsv.gz")
    by_contrast: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in response_rows:
        by_contrast[(row["study_id"], row["contrast_id"])].append(row)
    q_by_effect: dict[tuple[str, str, str], float] = {}
    for (study, contrast), rows in by_contrast.items():
        pvalues = [math.erfc(abs(float(row["beta"]) / float(row["se"])) / math.sqrt(2.0)) for row in rows]
        for row, qvalue in zip(rows, bh_qvalues(pvalues)):
            q_by_effect[(study, contrast, row["entity_id"])] = qvalue

    selected_effects: dict[str, dict[str, str]] = {}
    for row in response_rows:
        if row["entity_id"] in entity_to_sets:
            selected_effects[row["entity_id"]] = row
    missing_entities = sorted(set(entity_to_sets) - set(selected_effects))
    if missing_entities:
        raise ValueError(f"registered entities absent from response collection: {missing_entities}")
    selected_contrast_by_study: dict[str, str] = {}
    for row in selected_effects.values():
        previous = selected_contrast_by_study.setdefault(row["study_id"], row["contrast_id"])
        if previous != row["contrast_id"]:
            raise ValueError(f"multiple selected IBD contrasts for {row['study_id']}")
    effect_lookup = {
        (row["study_id"], row["contrast_id"], row["entity_id"]): row
        for row in response_rows
    }

    legacy: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in gzip_rows(legacy_path):
        if row["evidence_world"] != "all":
            continue
        key = canonical(row["left_study_id"], row["left_entity_id"], row["right_study_id"], row["right_entity_id"])
        legacy[key] = row

    # Every directed block from each frozen candidate entity to every other
    # study represented in the same candidate set is retained in full.
    requested_blocks: set[tuple[str, str, str]] = set()
    for definition in sets.values():
        studies = {entity.split("::", 1)[0] for entity in definition["entities"]}
        for entity in definition["entities"]:
            query_study = entity.split("::", 1)[0]
            for target_study in studies - {query_study}:
                requested_blocks.add((query_study, entity, target_study))

    block_manifest = [row for row in gzip_rows(block_path)
                      if (row["query_study_id"], row["query_entity_id"], row["candidate_study_id"]) in requested_blocks]
    if { (r["query_study_id"], r["query_entity_id"], r["candidate_study_id"]) for r in block_manifest } != requested_blocks:
        raise ValueError("one or more requested query-target blocks are absent")

    handoff_rows: list[dict[str, Any]] = []
    pair_evidence_rows: list[dict[str, Any]] = []
    for row in gzip_rows(pair_path):
        block_key = (row["query_study_id"], row["query_entity_id"], row["candidate_study_id"])
        if block_key not in requested_blocks:
            continue
        query_effect = selected_effects[row["query_entity_id"]]
        candidate_contrast = selected_contrast_by_study[row["candidate_study_id"]]
        candidate_effect = effect_lookup.get(
            (row["candidate_study_id"], candidate_contrast, row["candidate_entity_id"])
        )
        key = canonical(row["query_study_id"], row["query_entity_id"], row["candidate_study_id"], row["candidate_entity_id"])
        old = legacy.get(key)
        candidate_sets = sorted(set(entity_to_sets.get(row["query_entity_id"], [])) & set(entity_to_sets.get(row["candidate_entity_id"], [])))
        # Algorithm handoff is deliberately response- and adjudication-blind.
        # It contains the exact parent-axis row and nothing derived from the
        # candidate registry or biological effects.
        handoff_rows.append(dict(row))
        pair_evidence_rows.append({
            "query_block_id": row["query_block_id"],
            "directed_pair_id": row["directed_pair_id"],
            "query_study_id": row["query_study_id"],
            "query_entity_id": row["query_entity_id"],
            "candidate_study_id": row["candidate_study_id"],
            "candidate_entity_id": row["candidate_entity_id"],
            "registered_candidate_sets": ";".join(candidate_sets) or "none",
            "registered_pair_role": registered_pair_role(
                row["query_entity_id"], row["candidate_entity_id"], candidate_sets, sets
            ),
            "legacy_edge_present": str(old is not None),
            "legacy_fragment_cosine": "" if old is None else old["chemical_similarity"],
            "query_beta_descriptive_only": query_effect["beta"],
            "query_se_descriptive_only": query_effect["se"],
            "candidate_beta_descriptive_only": "" if candidate_effect is None else candidate_effect["beta"],
            "candidate_se_descriptive_only": "" if candidate_effect is None else candidate_effect["se"],
        })

    evidence_rows: list[dict[str, Any]] = []
    for entity, row in sorted(selected_effects.items()):
        study = row["study_id"]
        common_status = identities[(study, entity)]
        evidence_rows.append({
            "candidate_set_ids": ";".join(entity_to_sets[entity]),
            "study_id": study,
            "entity_id": entity,
            "contrast_id": row["contrast_id"],
            "common_identity_status": common_status,
            "source_native_identity_status": row["identity_status"],
            "beta": row["beta"],
            "se": row["se"],
            "z": float(row["beta"]) / float(row["se"]),
            "two_sided_bh_q_within_contrast": q_by_effect[(study, row["contrast_id"], entity)],
            "effect_model": row["effect_model"],
            "n_positive": row["n_positive"],
            "n_negative": row["n_negative"],
            "biospecimen": row["biospecimen"],
            "data_role": row["data_role"],
        })

    output.mkdir(parents=True)
    handoff_path = output / "complete_block_algorithm_handoff.tsv.gz"
    pair_evidence_path = output / "pair_adjudication_evidence.tsv.gz"
    block_out = output / "query_target_block_manifest.tsv"
    evidence_path = output / "candidate_response_evidence.tsv"
    write_gzip(handoff_path, handoff_rows, list(handoff_rows[0]))
    write_gzip(pair_evidence_path, pair_evidence_rows, list(pair_evidence_rows[0]))
    with block_out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(block_manifest[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(block_manifest)
    with evidence_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(evidence_rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(evidence_rows)

    stercobilin = [row for row in evidence_rows if "stercobilin_like" in row["candidate_set_ids"]]
    report = {
        "schema": "contrast_atlas_ibd_decision_blocks_v1",
        "status": "DEVELOPMENT_CANDIDATE_BLOCKS_FROZEN_ALGORITHM_SCORES_PENDING",
        "data_role": "development_consumed",
        "denominators": {
            "candidate_sets": len(sets),
            "registered_entities": len(evidence_rows),
            "directed_query_target_blocks": len(block_manifest),
            "complete_block_candidate_rows": len(handoff_rows),
            "stercobilin_like_sources": len(stercobilin),
            "stercobilin_like_bh_q_lt_0_05": sum(float(row["two_sided_bh_q_within_contrast"]) < 0.05 for row in stercobilin),
        },
        "hard_result": [
            "No IBD entity currently qualifies as a claim-ready flagship biological discovery.",
            "The stercobilin-like group is a three-source, same-direction unresolved hypothesis; only one source passes within-contrast BH q<0.05.",
            "The m/z 528.332 5-ASA group is an identity-disambiguation control and is medication-confounded, not an IBD-mechanism result.",
            "WSE, Noise and Full23 v1 complete-block scores have not been delivered; no score columns are emitted in the algorithm handoff."
        ],
        "algorithm_handoff_isolation": {
            "response_columns_present": False,
            "candidate_set_or_expected_answer_columns_present": False,
            "score_columns_present": False,
            "adjudication_file_separate": pair_evidence_path.name,
        },
        "method_score_availability": {
            "wse": "not_delivered",
            "noise": "not_delivered",
            "full23_v1": "not_delivered",
            "full23_v1_1": "experimental_not_delivered"
        },
        "claim_boundary": registry["claim_boundary"],
        "provenance": {
            "registry_sha256": sha256(registry_path),
            "response_entities_sha256": sha256(collection / "response_entities.tsv.gz"),
            "common_identity_partition_sha256": sha256(identity_path),
            "candidate_axis_report_sha256": sha256(axis_dir / "report.json"),
            "candidate_pairs_sha256": sha256(pair_path),
            "block_manifest_sha256": sha256(block_path),
            "legacy_handoff_sha256": sha256(legacy_path),
            "supporting_evidence_sha256_by_path": supporting_evidence_hashes,
            "implementation_sha256": sha256(Path(__file__)),
        },
        "outputs": {
            handoff_path.name: sha256(handoff_path),
            pair_evidence_path.name: sha256(pair_evidence_path),
            block_out.name: sha256(block_out),
            evidence_path.name: sha256(evidence_path),
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "checksums.sha256").write_text(
        "".join(
            f"{sha256(path)}  {path.name}\n"
            for path in (handoff_path, pair_evidence_path, block_out, evidence_path, report_path)
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
