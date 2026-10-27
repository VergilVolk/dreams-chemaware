#!/usr/bin/env python3
"""Build a fail-closed design/eligibility registry for the frozen response atlas.

The output deliberately separates a computable response object from a valid
biological truth unit.  It never infers missing acquisition or study-design
metadata from a study name, and it never treats unregistered pairs as negative
examples.
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
from typing import Any, Iterable


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def finite(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def load_contract_sources(root: Path, contract_path: Path) -> dict[str, dict[str, Any]]:
    seen: set[Path] = set()
    merged: dict[str, dict[str, Any]] = {}

    def visit(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            raise ValueError(f"contract cycle: {resolved}")
        seen.add(resolved)
        obj = json.loads(resolved.read_text(encoding="utf-8"))
        base = obj.get("base_contract")
        if base:
            visit((root / str(base)).resolve())
        for source in obj.get("sources", []):
            sid = str(source["study_id"])
            if sid in merged:
                raise ValueError(f"duplicate study_id across contract chain: {sid}")
            merged[sid] = source

    visit(contract_path)
    return merged


def path_state(root: Path, source: dict[str, Any], *keys: str) -> tuple[str, str]:
    for key in keys:
        value = source.get(key)
        if value:
            rel = str(value).replace("\\", "/")
            return rel, "present" if (root / rel).exists() else "missing"
    return "not_registered", "not_registered"


def truth_targets(relations: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    out: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in relations:
        out[(row["left_study_id"], row["left_contrast_id"])].append(row)
    return out


def load_common_identity(path: Path) -> dict[tuple[str, str], str]:
    identity: dict[tuple[str, str], str] = {}
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (row["study_id"], row["entity_id"])
            if key in identity:
                raise ValueError(f"duplicate common identity row: {key}")
            identity[key] = row["identity_status"]
    return identity


def response_stats(
    path: Path, common_identity: dict[tuple[str, str], str]
) -> dict[tuple[str, str], dict[str, Any]]:
    stats: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "rows": 0,
            "finite_beta": 0,
            "finite_se": 0,
            "positive_se": 0,
            "effect_models": set(),
            "n_positive": set(),
            "n_negative": set(),
            "identity": Counter(),
            "native_identity": Counter(),
        }
    )
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (row["study_id"], row["contrast_id"])
            cur = stats[key]
            cur["rows"] += 1
            cur["finite_beta"] += int(finite(row.get("beta", "")))
            cur["finite_se"] += int(finite(row.get("se", "")))
            if finite(row.get("se", "")) and float(row["se"]) > 0:
                cur["positive_se"] += 1
            cur["effect_models"].add(row.get("effect_model", "not_recorded"))
            if finite(row.get("n_positive", "")):
                cur["n_positive"].add(str(int(float(row["n_positive"]))))
            if finite(row.get("n_negative", "")):
                cur["n_negative"].add(str(int(float(row["n_negative"]))))
            entity_key = (row["study_id"], row["entity_id"])
            if entity_key not in common_identity:
                raise ValueError(f"response entity absent from common identity partition: {entity_key}")
            cur["identity"][common_identity[entity_key]] += 1
            cur["native_identity"][row.get("identity_status", "not_recorded")] += 1
    return stats


def exact_effect_input(
    root: Path,
    source: dict[str, Any],
    contrast_id: str,
    override: dict[str, Any],
) -> tuple[str, str]:
    value = override.get("effect_input_override")
    if value:
        rel = str(value).replace("\\", "/")
        return rel, "present" if (root / rel).exists() else "missing"
    if source.get("effects_path"):
        rel = str(source["effects_path"]).replace("\\", "/")
        return rel, "present" if (root / rel).exists() else "missing"
    if source.get("response_root"):
        base = str(source["response_root"]).replace("\\", "/").rstrip("/")
        rel = f"{base}/{contrast_id}/response_object.tsv.gz"
        return rel, "present" if (root / rel).exists() else "missing"
    return "not_registered", "not_registered"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection-dir",
        default="data/validation/contrast_atlas_open_response_collection_v9_frozenbase_chemprop_20261023",
    )
    parser.add_argument("--contract", default="tasks/contrast_atlas_open_collection_chemprop_extension_v1.json")
    parser.add_argument("--design-registry", default="tasks/contrast_atlas_design_registry_v1.json")
    parser.add_argument(
        "--identity-partition",
        default="data/validation/contrast_atlas_open_v9_frozenbase_common_identity_20261023/identity_partition.tsv.gz",
    )
    parser.add_argument(
        "--output-dir",
        default="data/validation/contrast_atlas_contrast_eligibility_v1_20261027",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    collection = (root / args.collection_dir).resolve()
    output = (root / args.output_dir).resolve()
    contract_path = (root / args.contract).resolve()
    registry_path = (root / args.design_registry).resolve()
    identity_partition_path = (root / args.identity_partition).resolve()

    required = [
        collection / "report.json",
        collection / "contrast_manifest.tsv",
        collection / "source_manifest.tsv",
        collection / "registered_relations.tsv",
        collection / "response_entities.tsv.gz",
        contract_path,
        registry_path,
        identity_partition_path,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required input(s): " + "; ".join(missing))

    collection_report = json.loads((collection / "report.json").read_text(encoding="utf-8"))
    if collection_report.get("status") != "DEVELOPMENT_CONSUMED_FROZEN_BASE_EXTENSION_COMPLETE":
        raise ValueError("refusing non-frozen-base response collection")
    if int(collection_report["denominators"]["eligible_response_rows"]) != 238412:
        raise ValueError("unexpected frozen response denominator")

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    source_design = registry["sources"]
    contract_sources = load_contract_sources(root, contract_path)
    contrasts = read_tsv(collection / "contrast_manifest.tsv")
    sources = read_tsv(collection / "source_manifest.tsv")
    relations = read_tsv(collection / "registered_relations.tsv")
    rel_by_query = truth_targets(relations)
    common_identity = load_common_identity(identity_partition_path)
    if len(common_identity) != 54861:
        raise ValueError(f"unexpected common identity denominator: {len(common_identity)}")
    identity_counts = Counter(common_identity.values())
    expected_identity = Counter({"unresolved_entity": 53882, "library_supported": 979})
    if identity_counts != expected_identity:
        raise ValueError(f"unexpected common identity partition: {identity_counts}")
    stats = response_stats(collection / "response_entities.tsv.gz", common_identity)

    source_ids = {row["study_id"] for row in sources}
    if len(contrasts) != 51 or len(source_ids) != 15:
        raise ValueError(f"unexpected collection shape: {len(source_ids)} sources / {len(contrasts)} contrasts")
    if source_ids != set(source_design):
        raise ValueError(
            "design registry/source mismatch: missing="
            + repr(sorted(source_ids - set(source_design)))
            + " extra="
            + repr(sorted(set(source_design) - source_ids))
        )

    duplicate_keys = [key for key, count in Counter((r["study_id"], r["contrast_id"]) for r in contrasts).items() if count > 1]
    if duplicate_keys:
        raise ValueError(f"duplicate contrast keys: {duplicate_keys}")

    default_measurement = registry["default_measurement_metadata"]
    overrides = registry.get("contrast_overrides", {})
    project_groups = registry.get("project_dependence_groups", {})
    rows: list[dict[str, Any]] = []
    for contrast in contrasts:
        sid, cid = contrast["study_id"], contrast["contrast_id"]
        key = (sid, cid)
        if key not in stats:
            raise ValueError(f"contrast has no response rows: {sid}::{cid}")
        st = stats[key]
        if st["rows"] != int(contrast["eligible_response_entities"]):
            raise ValueError(f"response denominator mismatch for {sid}::{cid}")
        source_contract = contract_sources.get(sid, {})
        design = source_design[sid]
        contrast_override = overrides.get(f"{sid}::{cid}", {})
        effect_path, effect_state = exact_effect_input(root, source_contract, cid, contrast_override)
        if contrast_override.get("effect_report_override"):
            report_rel = str(contrast_override["effect_report_override"]).replace("\\", "/")
            effect_report_path = report_rel
            effect_report_state = "present" if (root / report_rel).exists() else "missing"
        else:
            effect_report_path, effect_report_state = path_state(root, source_contract, "effect_report_path")
        mgf_path, mgf_state = path_state(root, source_contract, "mgf_path")
        identity_path, identity_state = path_state(root, source_contract, "identity_path")
        targets = rel_by_query.get(key, [])
        eligible_targets = [r for r in targets if parse_bool(r.get("eligible_primary_relevance", False))]
        signed_targets = [r for r in eligible_targets if parse_bool(r.get("signed_truth_claimed", False))]
        primary_classes = sorted({r["relation_class"] for r in eligible_targets})
        project_group = project_groups.get(sid, contrast["source_family"])
        small_n = int(contrast["independent_units"]) < 4
        observational = design["causal_status"].lower().startswith("observational")
        exclusions = ["development_consumed", "no_external_validation", "no_confirmed_signed_response_truth"]
        if small_n:
            exclusions.append("fewer_than_four_independent_units")
        if observational:
            exclusions.append("observational_no_causal_claim")
        if sid in {"MSV000094899", "MSV000096724", "MSV000093571"}:
            exclusions.append("shared_chemprop_parent_project")
        if sid == "IBD200_GROUP":
            exclusions.append("plate3_sensitivity_not_independent_cohort")
        if sid == "MSV000096564":
            exclusions.append("broad_active_or_remission_crohn_group")
        rows.append(
            {
                "study_id": sid,
                "source_family": contrast["source_family"],
                "project_dependence_group": project_group,
                "contrast_id": cid,
                "species": contrast["species"],
                "biospecimen": contrast["biospecimen"],
                "condition_family": contrast["condition_family"],
                "estimand": contrast["design"],
                "biological_unit": design["biological_unit"],
                "unit_identifier_traceability": design["unit_identifier_traceability"],
                "study_design": design["study_design"],
                "repeated_measures": design["repeated_measures"],
                "independence_note": contrast_override.get("independence_note", "independent-unit count comes from the frozen contrast manifest"),
                "independent_units": contrast["independent_units"],
                "time_or_dose": design["time_or_dose"],
                "covariates": design["covariates"],
                "intentional_drug_exposure": str(bool(design["intentional_drug_exposure"])),
                "causal_status": design["causal_status"],
                "polarity": default_measurement["polarity"],
                "instrument": default_measurement["instrument"],
                "chromatography": default_measurement["chromatography"],
                "effect_model": "; ".join(sorted(st["effect_models"])),
                "response_rows": st["rows"],
                "finite_beta_rows": st["finite_beta"],
                "finite_se_rows": st["finite_se"],
                "positive_se_rows": st["positive_se"],
                "n_positive_values": ";".join(sorted(st["n_positive"])) or "not_carried",
                "n_negative_values": ";".join(sorted(st["n_negative"])) or "not_carried",
                "library_supported_rows": st["identity"].get("library_supported", 0),
                "unresolved_rows": st["identity"].get("unresolved_entity", 0),
                "source_native_library_supported_rows": st["native_identity"].get("library_supported", 0),
                "source_native_unresolved_rows": st["native_identity"].get("unresolved_entity", 0),
                "source_disjoint_candidate_contrasts": contrast["source_disjoint_candidate_contrasts"],
                "registered_primary_targets": len(eligible_targets),
                "registered_primary_target_keys": ";".join(f'{r["right_study_id"]}::{r["right_contrast_id"]}' for r in eligible_targets) or "none",
                "registered_relation_classes": ";".join(primary_classes) or "none",
                "signed_truth_targets": len(signed_targets),
                "eligible_method_diagnostic": "True",
                "eligible_unsigned_registered_relevance": str(bool(eligible_targets)),
                "eligible_signed_response_accuracy": str(bool(signed_targets)),
                "eligible_same_entity_replication": "conditional_on_separately_calibrated_identity_link",
                "eligible_chemical_family_relation": "hypothesis_only_no_beta_or_se_pooling",
                "eligible_occurrence_analysis": "descriptive_only",
                "eligible_primary_biological_discovery": "False",
                "external_validation_eligible": "False",
                "effect_input_path": effect_path,
                "effect_input_state": effect_state,
                "effect_report_path": effect_report_path,
                "effect_report_state": effect_report_state,
                "ms2_path": mgf_path,
                "ms2_state": mgf_state,
                "identity_input_path": identity_path,
                "identity_input_state": identity_state,
                "design_provenance": design["design_provenance"],
                "exclusion_or_limit_reasons": ";".join(exclusions),
            }
        )

    relation_units: list[dict[str, Any]] = []
    seen_units: set[tuple[str, str, str, str]] = set()
    for rel in relations:
        left = (rel["left_study_id"], rel["left_contrast_id"])
        right = (rel["right_study_id"], rel["right_contrast_id"])
        canonical = (*left, *right) if left <= right else (*right, *left)
        if canonical in seen_units:
            continue
        seen_units.add(canonical)
        left_group = project_groups.get(left[0], next(r["source_family"] for r in contrasts if (r["study_id"], r["contrast_id"]) == left))
        right_group = project_groups.get(right[0], next(r["source_family"] for r in contrasts if (r["study_id"], r["contrast_id"]) == right))
        relation_units.append(
            {
                "left_key": f"{left[0]}::{left[1]}",
                "right_key": f"{right[0]}::{right[1]}",
                "relation_class": rel["relation_class"],
                "eligible_primary_relevance": rel["eligible_primary_relevance"],
                "signed_truth_claimed": rel["signed_truth_claimed"],
                "left_dependence_group": left_group,
                "right_dependence_group": right_group,
                "source_independent": str(left_group != right_group),
                "evidence_boundary": rel["evidence_boundary"],
            }
        )

    primary_relation_units = [r for r in relation_units if parse_bool(r["eligible_primary_relevance"])]
    cross_project_primary_edges = [r for r in primary_relation_units if parse_bool(r["source_independent"])]

    def component_count(edges: list[dict[str, Any]]) -> int:
        parent: dict[str, str] = {}

        def find(node: str) -> str:
            parent.setdefault(node, node)
            if parent[node] != node:
                parent[node] = find(parent[node])
            return parent[node]

        for edge in edges:
            left, right = edge["left_key"], edge["right_key"]
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root
        return len({find(node) for node in parent})

    output.mkdir(parents=True, exist_ok=True)
    eligibility_path = output / "contrast_eligibility.tsv"
    relation_path = output / "registered_relation_independence.tsv"
    write_tsv(eligibility_path, rows, list(rows[0]))
    write_tsv(relation_path, relation_units, list(relation_units[0]))

    report = {
        "schema": "contrast_atlas_contrast_eligibility_v1",
        "status": "DEVELOPMENT_CONSUMED_CONTRAST_ELIGIBILITY_COMPLETE",
        "data_role": "development_consumed",
        "frozen_collection": str(collection.relative_to(root)),
        "denominators": {
            "source_families": len(source_ids),
            "study_level_contrasts": len(rows),
            "response_rows": sum(int(r["response_rows"]) for r in rows),
            "canonical_registered_relation_units": len(relation_units),
            "canonical_primary_relation_units": len(primary_relation_units),
            "cross_project_primary_pair_edges": len(cross_project_primary_edges),
            "primary_relation_connected_components": component_count(primary_relation_units),
            "cross_project_primary_connected_components": component_count(cross_project_primary_edges),
            "registered_directional_rows": len(relations),
            "contrasts_with_primary_registered_target": sum(parse_bool(r["eligible_unsigned_registered_relevance"]) for r in rows),
            "contrasts_with_signed_truth": sum(parse_bool(r["eligible_signed_response_accuracy"]) for r in rows),
            "external_validation_contrasts": 0,
            "primary_biological_discovery_contrasts": 0,
            "chemprop_directional_rows": sum(r["project_dependence_group"] == "CHEMPROP_SHARED_PARENT_PROJECT" for r in rows),
            "chemprop_unique_drug_units": len({r["condition_family"] for r in rows if r["project_dependence_group"] == "CHEMPROP_SHARED_PARENT_PROJECT"}),
        },
        "hard_findings": [
            "The collection contains 51 contrast rows, not 51 independent studies or validation units.",
            "The 24 directional ChemProp retrieval relations arise from 12 shared-drug units within one broader project.",
            "The five cross-project primary pair edges collapse to two connected source-family components (HFD and IBD); they are not five independent validations.",
            "The frozen-base collection has 238,412 response rows and preserves the 662-entity Plate3 adult-IBD correction.",
            "Known/unresolved counts use the common GNPS Gold/Silver partition (979/53,882), never source-native labels.",
            "No registered relation supplies confirmed signed metabolite-level truth.",
            "Acquisition polarity, instrument, and chromatography are not harmonized in the frozen response collection and cannot be used implicitly as matched design variables.",
        ],
        "claim_boundary": registry["claim_boundary"] + [
            "A registered relevance relation is an unsigned retrieval target, not a molecular-identity or mechanistic truth.",
            "Unregistered cross-study pairs are unadjudicated and never counted as confirmed negatives.",
            "Same-entity response pooling remains conditional on a separately calibrated identity link; chemical-relation edges never pool beta or SE.",
        ],
        "provenance": {
            "collection_report_sha256": sha256(collection / "report.json"),
            "response_entities_sha256": sha256(collection / "response_entities.tsv.gz"),
            "contrast_manifest_sha256": sha256(collection / "contrast_manifest.tsv"),
            "source_manifest_sha256": sha256(collection / "source_manifest.tsv"),
            "registered_relations_sha256": sha256(collection / "registered_relations.tsv"),
            "common_identity_partition_sha256": sha256(identity_partition_path),
            "design_registry_sha256": sha256(registry_path),
            "contract_sha256": sha256(contract_path),
            "implementation_sha256": sha256(Path(__file__)),
        },
        "outputs": {
            "contrast_eligibility.tsv": sha256(eligibility_path),
            "registered_relation_independence.tsv": sha256(relation_path),
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    checksum_path = output / "checksums.sha256"
    checksum_path.write_text(
        f"{sha256(eligibility_path)}  contrast_eligibility.tsv\n"
        f"{sha256(relation_path)}  registered_relation_independence.tsv\n"
        f"{sha256(report_path)}  report.json\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
