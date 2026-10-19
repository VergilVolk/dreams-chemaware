#!/usr/bin/env python3
"""Freeze a multi-source, open-candidate Contrast Atlas response collection.

The builder standardizes already-consumed within-study beta/SE objects and
phenotype-blind library partitions.  It does not calculate cross-study scores
or infer truth from condition labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


IDENTITY_STATES = {"library_supported", "unresolved_entity"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(
    contract_path: Path, root: Path
) -> tuple[dict[str, Any], list[Path]]:
    """Load either a complete contract or an append-only extension contract."""
    document = json.loads(contract_path.read_text(encoding="utf-8"))
    schema = document.get("schema")
    if schema == "contrast_atlas_open_collection_contract_v1":
        return document, [contract_path]
    if schema != "contrast_atlas_open_collection_extension_v1":
        raise RuntimeError(f"unsupported collection contract schema: {schema!r}")
    base_path = root / document["base_contract"]
    base, paths = load_contract(base_path, root)
    if document.get("data_role", base.get("data_role")) != base.get("data_role"):
        raise RuntimeError("extension data role differs from base contract")
    merged = dict(base)
    merged["sources"] = [*base["sources"], *document.get("sources", [])]
    merged["registered_relations"] = [
        *base.get("registered_relations", []),
        *document.get("registered_relations", []),
    ]
    study_ids = [str(source["study_id"]) for source in merged["sources"]]
    if len(study_ids) != len(set(study_ids)):
        raise RuntimeError("extension creates duplicate study IDs")
    relation_keys = [
        (
            str(row["left_study_id"]), str(row["left_contrast_id"]),
            str(row["right_study_id"]), str(row["right_contrast_id"]),
        )
        for row in merged["registered_relations"]
    ]
    if len(relation_keys) != len(set(relation_keys)):
        raise RuntimeError("extension creates duplicate registered relations")
    return merged, [*paths, contract_path]


def parse_bool(series: pd.Series, label: str) -> pd.Series:
    mapping = {"true": True, "false": False, "1": True, "0": False}
    values = series.astype(str).str.strip().str.lower()
    if not values.isin(mapping).all():
        raise RuntimeError(f"{label} contains invalid booleans")
    return values.map(mapping).astype(bool)


def load_mgf_ids(path: Path) -> list[str]:
    ids: list[str] = []
    fields: dict[str, str] = {}
    inside = False
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "BEGIN IONS":
                if inside:
                    raise RuntimeError(f"nested BEGIN IONS in {path}")
                inside, fields = True, {}
            elif line == "END IONS":
                if not inside:
                    raise RuntimeError(f"END IONS without BEGIN IONS in {path}")
                entity_id = fields.get("FEATURE_ID") or fields.get("TITLE") or fields.get("SCANS")
                if not entity_id:
                    raise RuntimeError(f"spectrum lacks FEATURE_ID/TITLE/SCANS in {path}")
                ids.append(str(entity_id))
                inside = False
            elif inside and "=" in line:
                key, value = line.split("=", 1)
                fields[key.upper()] = value
    if inside or not ids or len(ids) != len(set(ids)):
        raise RuntimeError(f"invalid or duplicate MGF registry in {path}")
    return ids


def read_report(path: Path, expected_status: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != expected_status:
        raise RuntimeError(
            f"status drift for {path}: {report.get('status')!r} != {expected_status!r}"
        )
    return report


def load_identity(source: dict[str, Any], root: Path) -> tuple[pd.DataFrame, Path]:
    path = root / source["identity_path"]
    sep = source.get("identity_sep", ",")
    frame = pd.read_csv(path, sep=sep, dtype={source["identity_entity_column"]: str})
    for column, value in source.get("identity_filter", {}).items():
        frame = frame.loc[frame[column].astype(str).eq(str(value))].copy()
    entity_column = source["identity_entity_column"]
    if source.get("identity_supported_column"):
        supported = parse_bool(frame[source["identity_supported_column"]], "library_supported")
        status = np.where(supported, "library_supported", "unresolved_entity")
    else:
        raw = frame[source["identity_status_column"]].astype(str)
        mapping = source["identity_status_map"]
        if not raw.isin(mapping).all():
            raise RuntimeError(f"unmapped identity states in {path}")
        status = raw.map(mapping).to_numpy()
    result = pd.DataFrame({
        "raw_entity_id": frame[entity_column].astype(str),
        "identity_status": status,
    })
    if result.raw_entity_id.duplicated().any() or not set(result.identity_status).issubset(IDENTITY_STATES):
        raise RuntimeError(f"invalid identity partition in {path}")
    return result, path


def standardize_effects(source: dict[str, Any], root: Path) -> tuple[pd.DataFrame, list[Path]]:
    paths: list[Path] = []
    if source["effect_adapter"] == "flat_csv":
        path = root / source["effects_path"]
        frame = pd.read_csv(
            path,
            sep=source.get("effects_sep", ","),
            dtype={source["entity_column"]: str, "contrast_id": str},
        )
        paths.append(path)
    elif source["effect_adapter"] == "response_directories":
        blocks = []
        response_root = root / source["response_root"]
        for contrast_id in source["contrasts"]:
            path = response_root / contrast_id / "response_object.tsv.gz"
            block = pd.read_csv(path, sep="\t", dtype={source["entity_column"]: str})
            block["contrast_id"] = contrast_id
            blocks.append(block)
            paths.append(path)
        frame = pd.concat(blocks, ignore_index=True)
    else:
        raise RuntimeError(f"unsupported effect adapter: {source['effect_adapter']}")

    if source.get("qc_column"):
        frame = frame.loc[parse_bool(frame[source["qc_column"]], source["qc_column"])].copy()
    required_contrasts = set(source["contrasts"])
    observed_contrasts = set(frame["contrast_id"].astype(str))
    if observed_contrasts != required_contrasts:
        raise RuntimeError(
            f"contrast registry drift for {source['study_id']}: "
            f"{sorted(observed_contrasts)} != {sorted(required_contrasts)}"
        )
    output = pd.DataFrame({
        "raw_entity_id": frame[source["entity_column"]].astype(str),
        "contrast_id": frame["contrast_id"].astype(str),
        "beta": pd.to_numeric(frame[source["beta_column"]], errors="raise"),
        "se": pd.to_numeric(frame[source["se_column"]], errors="raise"),
        "prevalence": (
            pd.to_numeric(frame[source["prevalence_column"]], errors="raise")
            if source.get("prevalence_column") else np.nan
        ),
        "reported_reliability": (
            pd.to_numeric(frame[source["reliability_column"]], errors="raise")
            if source.get("reliability_column") else np.nan
        ),
        "n_positive": (
            pd.to_numeric(frame["n_positive"], errors="raise")
            if "n_positive" in frame else np.nan
        ),
        "n_negative": (
            pd.to_numeric(frame["n_negative"], errors="raise")
            if "n_negative" in frame else np.nan
        ),
        "effect_model": (
            frame[source["effect_model_column"]].astype(str)
            if source.get("effect_model_column") else (
                frame["model"].astype(str)
                if "model" in frame else "reported_response_object"
            )
        ),
    })
    if output.duplicated(["contrast_id", "raw_entity_id"]).any():
        raise RuntimeError(f"duplicate contrast/entity effects in {source['study_id']}")
    finite = np.isfinite(output[["beta", "se"]].to_numpy(float)).all(axis=1)
    if not finite.all() or (output.se <= 0).any():
        raise RuntimeError(f"non-finite beta or non-positive SE in {source['study_id']}")
    return output, paths


def reverse_relations(relations: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for relation in relations:
        rows.append({"relation_direction": "registered", **relation})
        rows.append({
            "relation_direction": "reverse_registered",
            "left_study_id": relation["right_study_id"],
            "left_contrast_id": relation["right_contrast_id"],
            "right_study_id": relation["left_study_id"],
            "right_contrast_id": relation["left_contrast_id"],
            **{key: value for key, value in relation.items() if key not in {
                "left_study_id", "left_contrast_id", "right_study_id", "right_contrast_id"
            }},
        })
    return pd.DataFrame(rows)


def run(contract_path: Path, output_dir: Path, root: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    contract, contract_paths = load_contract(contract_path, root)

    response_blocks: list[pd.DataFrame] = []
    source_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    provenance: dict[str, str] = {
        str(path.relative_to(root)): sha256(path) for path in contract_paths
    }
    seen_mgf_ids: dict[str, set[str]] = {}
    for source in contract["sources"]:
        study_id = str(source["study_id"])
        effect_report_path = root / source["effect_report_path"]
        mgf_report_path = root / source["mgf_report_path"]
        read_report(effect_report_path, source["expected_effect_status"])
        read_report(mgf_report_path, source["expected_mgf_status"])
        effects, effect_paths = standardize_effects(source, root)
        identity, identity_path = load_identity(source, root)
        mgf_path = root / source["mgf_path"]
        mgf_ids = set(load_mgf_ids(mgf_path))
        seen_mgf_ids[study_id] = mgf_ids
        effect_ids = set(effects.raw_entity_id)
        if not effect_ids.issubset(mgf_ids):
            raise RuntimeError(f"effect entities lack spectra for {study_id}: {sorted(effect_ids - mgf_ids)[:3]}")
        identity_lookup = identity.set_index("raw_entity_id")
        if not effect_ids.issubset(set(identity_lookup.index)):
            raise RuntimeError(f"effect entities lack identity states for {study_id}")
        effects = effects.merge(identity, on="raw_entity_id", how="left", validate="many_to_one")
        effects.insert(0, "study_id", study_id)
        effects.insert(1, "source_family", str(source["source_family"]))
        effects.insert(3, "entity_id", study_id + "::" + effects.raw_entity_id)
        effects["species"] = source["species"]
        effects["biospecimen"] = source["biospecimen"]
        effects["data_role"] = contract["data_role"]
        response_blocks.append(effects)

        for contrast_id, metadata in source["contrasts"].items():
            block = effects.loc[effects.contrast_id.eq(contrast_id)]
            contrast_rows.append({
                "study_id": study_id,
                "source_family": source["source_family"],
                "contrast_id": contrast_id,
                "species": source["species"],
                "biospecimen": source["biospecimen"],
                "design": metadata["design"],
                "condition_family": metadata["condition_family"],
                "independent_units": int(metadata["independent_units"]),
                "eligible_response_entities": len(block),
                "library_supported_response_entities": int(block.identity_status.eq("library_supported").sum()),
                "unresolved_response_entities": int(block.identity_status.eq("unresolved_entity").sum()),
            })
        source_rows.append({
            "study_id": study_id,
            "source_family": source["source_family"],
            "species": source["species"],
            "biospecimen": source["biospecimen"],
            "biological_units": int(source["biological_units"]),
            "contrasts": len(source["contrasts"]),
            "ms2_spectra": int(source.get("nonempty_ms2_spectra", len(mgf_ids))),
            "identity_partition_entities": len(identity),
            "identity_partition_library_supported": int(identity.identity_status.eq("library_supported").sum()),
            "identity_partition_unresolved": int(identity.identity_status.eq("unresolved_entity").sum()),
            "response_eligible_entity_union": len(effect_ids),
            "response_eligible_library_supported_union": int(identity_lookup.loc[list(effect_ids)].identity_status.eq("library_supported").sum()),
            "response_eligible_unresolved_union": int(identity_lookup.loc[list(effect_ids)].identity_status.eq("unresolved_entity").sum()),
        })
        for path in [effect_report_path, mgf_report_path, identity_path, mgf_path, *effect_paths]:
            provenance[str(path.relative_to(root))] = sha256(path)

    responses = pd.concat(response_blocks, ignore_index=True)
    contrasts = pd.DataFrame(contrast_rows)
    sources = pd.DataFrame(source_rows)
    if responses.duplicated(["study_id", "contrast_id", "raw_entity_id"]).any():
        raise RuntimeError("duplicate global response rows")
    if sources.study_id.duplicated().any() or contrasts.duplicated(["study_id", "contrast_id"]).any():
        raise RuntimeError("duplicate source or contrast registry rows")
    candidate_counts = []
    for contrast in contrasts.itertuples(index=False):
        candidate_counts.append(int((contrasts.source_family != contrast.source_family).sum()))
    contrasts["source_disjoint_candidate_contrasts"] = candidate_counts

    relations = reverse_relations(contract.get("registered_relations", []))
    contrast_keys = set(zip(contrasts.study_id, contrasts.contrast_id))
    for row in relations.itertuples(index=False):
        if (row.left_study_id, row.left_contrast_id) not in contrast_keys or (
            row.right_study_id, row.right_contrast_id
        ) not in contrast_keys:
            raise RuntimeError("registered relation endpoint is absent")

    output_dir.mkdir(parents=True)
    response_path = output_dir / "response_entities.tsv.gz"
    source_path = output_dir / "source_manifest.tsv"
    contrast_path = output_dir / "contrast_manifest.tsv"
    relation_path = output_dir / "registered_relations.tsv"
    responses.to_csv(response_path, sep="\t", index=False, compression="gzip")
    sources.to_csv(source_path, sep="\t", index=False)
    contrasts.to_csv(contrast_path, sep="\t", index=False)
    relations.to_csv(relation_path, sep="\t", index=False)
    output_hashes = {
        path.name: sha256(path)
        for path in (response_path, source_path, contrast_path, relation_path)
    }
    report = {
        "schema": "contrast_atlas_open_response_collection_v1",
        "status": "DEVELOPMENT_CONSUMED_OPEN_RESPONSE_COLLECTION_FROZEN",
        "denominators": {
            "independent_studies": int(sources.study_id.nunique()),
            "source_families": int(sources.source_family.nunique()),
            "study_level_contrasts": len(contrasts),
            "biological_units_across_sources": int(sources.biological_units.sum()),
            "representative_ms2_spectra": int(sources.ms2_spectra.sum()),
            "identity_partitioned_feature_entities": int(sources.identity_partition_entities.sum()),
            "identity_partitioned_library_supported_entities": int(sources.identity_partition_library_supported.sum()),
            "identity_partitioned_unresolved_entities": int(sources.identity_partition_unresolved.sum()),
            "response_eligible_unique_feature_entities": int(sources.response_eligible_entity_union.sum()),
            "response_eligible_library_supported_entity_union": int(sources.response_eligible_library_supported_union.sum()),
            "response_eligible_unresolved_entity_union": int(sources.response_eligible_unresolved_union.sum()),
            "eligible_response_rows": len(responses),
            "library_supported_response_rows": int(responses.identity_status.eq("library_supported").sum()),
            "unresolved_response_rows": int(responses.identity_status.eq("unresolved_entity").sum()),
            "source_disjoint_candidates_per_query_min": min(candidate_counts),
            "source_disjoint_candidates_per_query_max": max(candidate_counts),
            "registered_directed_relevance_relations": len(relations),
            "registered_signed_direction_truth_relations": int(relations.expected_signed_direction.ne("unspecified").sum()),
        },
        "leakage_status": (
            "within-study phenotype labels were used only to estimate frozen beta/SE; "
            "spectral registries and identity partitions are phenotype blind; no cross-study "
            "score or result was used to construct the collection"
        ),
        "provenance": provenance,
        "outputs_sha256": output_hashes,
        "claim_boundary": [
            "All studies and prior pairwise results are development/consumed; this is not external validation.",
            "Unresolved means not library-supported under the frozen partition, not a new molecule.",
            "Feature entities may include adduct/isotope redundancy and are not asserted to be unique analytes.",
            "Only explicitly registered design-level relations are eligible for relevance auditing; signed response direction is not assumed unless separately registered.",
            "Every unregistered cross-study pair is unadjudicated, never a negative example.",
            "The collection freezes an open candidate universe but contains no retrieval-performance claim.",
        ],
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_dir / "checksums.sha256").write_text(
        "".join(f"{value}  {name}\n" for name, value in sorted({
            **output_hashes, "report.json": sha256(report_path),
        }.items())),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = run(args.contract.resolve(), args.output_dir.resolve(), root)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
