"""Compile public studies into fail-closed Figure-5 contrast assets.

The audit is intentionally prior to raw-data download and model scoring.  Study
titles and narrative summaries never substitute for row-level contrast labels,
an auditable MS1 quantification table, or MS/MS acquisition evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CONTROL_TOKENS = ("qc", "pool", "blank", "standard", "qstd", "reagent_mixture")
NON_CONTRAST_KEYS = {"sample source", "harvest"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_concatenated_json(path: Path) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    text = path.read_text(encoding="utf-8-sig")
    output: list[dict[str, Any]] = []
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            break
        value, position = decoder.raw_decode(text, position)
        if not isinstance(value, dict):
            raise TypeError(f"expected object in {path}")
        output.append(value)
    return output


def parse_factors(text: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for item in str(text).split("|"):
        if ":" not in item:
            continue
        key, value = item.split(":", 1)
        output[key.strip()] = value.strip()
    return output


def sample_role(row: dict[str, Any]) -> str:
    text = " ".join(
        str(row.get(field, "")) for field in ("sample_source", "factors", "raw_data", "local_sample_id")
    ).lower()
    if "blank" in text:
        return "blank"
    if "standard" in text or "qstd" in text or "reagent_mixture" in text:
        return "standard_or_reagent"
    if "qc" in text or "pool" in text:
        return "pooled_qc"
    return "study"


def archive_inventory(payload: dict[str, Any]) -> dict[str, Any]:
    top_files = [str(value) for value in payload.get("files", [])]
    archives = [value for value in top_files if value.lower().endswith(".zip")]
    expanded = payload.get("compressed_file_content", {})
    members = [row for rows in expanded.values() for row in rows]
    names = [str(row.get("name", "")) for row in members]
    basenames = {Path(name).name for name in names}
    return {
        "top_level_files": top_files,
        "raw_archives": archives,
        "expanded_archives": sorted(expanded),
        "archive_inventory_complete": set(archives) == set(expanded),
        "members_listed": len(names),
        "mzml_members_listed": sum(name.lower().endswith(".mzml") for name in names),
        "listed_uncompressed_bytes": sum(int(row.get("size", 0) or 0) for row in members),
        "explicit_dda_filename_members": sum("dda" in name.lower() for name in names),
        "result_tables": [value for value in top_files if value.lower().endswith("_results.txt")],
        "member_basenames": basenames,
    }


def audit_study(study: dict[str, Any], snapshot_dir: Path) -> dict[str, Any]:
    study_id = str(study["study_id"])
    paths = {
        endpoint: snapshot_dir / f"{study_id}__{endpoint}__json"
        for endpoint in ("summary", "analysis", "mwtab", "files", "factors")
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        return {
            **study,
            "status": "blocked_missing_snapshot",
            "missing": missing,
            "ready_for_targeted_raw_probe": False,
        }

    summary = read_json(paths["summary"])
    analyses = read_json(paths["analysis"])
    factors_payload = read_json(paths["factors"])
    factor_rows = list(factors_payload.values())
    inventory = archive_inventory(read_json(paths["files"]))
    mwtab_objects = read_concatenated_json(paths["mwtab"])
    mwtab_text = paths["mwtab"].read_text(encoding="utf-8-sig")

    role_counts = Counter(sample_role(row) for row in factor_rows)
    study_rows = [row for row in factor_rows if sample_role(row) == "study"]
    contrast_values: dict[str, set[str]] = defaultdict(set)
    for row in study_rows:
        for key, value in parse_factors(str(row.get("factors", ""))).items():
            if key.lower() not in NON_CONTRAST_KEYS and value not in {"", "-"}:
                contrast_values[key].add(value)
    contrast_values = {key: values for key, values in contrast_values.items() if len(values) >= 2}

    raw_names = {str(row.get("raw_data", "")) for row in factor_rows if row.get("raw_data")}
    listed = inventory.pop("member_basenames")
    joined = raw_names & listed
    join_fraction = len(joined) / len(raw_names) if raw_names else 0.0
    explicit_dda_text = bool(re.search(r"data[- ]?dependent|ddms2|\bdda\b", mwtab_text, re.I))
    explicit_ms2_text = bool(re.search(r"ms/ms|tandem|fragmentation", mwtab_text, re.I))
    project_titles = sorted({
        str(obj.get("PROJECT", {}).get("PROJECT_TITLE", "")).strip()
        for obj in mwtab_objects if obj.get("PROJECT", {}).get("PROJECT_TITLE")
    })

    gates = {
        "row_level_contrast_labels": bool(contrast_values),
        "study_injections_present": role_counts["study"] > 0,
        "pooled_qc_present": role_counts["pooled_qc"] > 0,
        "blank_present": role_counts["blank"] > 0,
        "quantitative_result_table_present": bool(inventory["result_tables"]),
        "raw_mzml_listed": inventory["mzml_members_listed"] > 0,
        "explicit_ms2_or_dda_evidence": bool(
            inventory["explicit_dda_filename_members"] or explicit_dda_text or explicit_ms2_text
        ),
        "archive_inventory_complete": inventory["archive_inventory_complete"],
        "factor_to_raw_inventory_join_ge_95pct": join_fraction >= 0.95,
    }
    ready = all(gates.values())
    blockers = [key for key, value in gates.items() if not value]
    if ready:
        status = "ready_for_targeted_raw_probe"
    elif gates["row_level_contrast_labels"] and gates["explicit_ms2_or_dda_evidence"]:
        status = "provisional_probe_after_inventory_resolution"
    else:
        status = "blocked_before_raw_download"

    return {
        **study,
        "title": summary.get("study_title"),
        "species": summary.get("species"),
        "registered_injections": int(summary.get("number_of_samples", 0) or 0),
        "analysis_panels": [
            {
                "analysis_id": value.get("analysis_id"),
                "chromatography": value.get("chromatography_type"),
                "ion_mode": value.get("ion_mode"),
                "instrument": value.get("ms_instrument_name"),
            }
            for value in analyses.values()
        ],
        "project_titles": project_titles,
        "role_counts_are_injections_not_biological_replicates": dict(role_counts),
        "row_level_contrast_values": {key: sorted(values) for key, values in contrast_values.items()},
        "unique_factor_raw_names": len(raw_names),
        "factor_raw_names_found_in_expanded_inventory": len(joined),
        "factor_to_raw_inventory_join_fraction": join_fraction,
        "explicit_dda_in_filename": inventory["explicit_dda_filename_members"] > 0,
        "explicit_dda_in_metadata_text": explicit_dda_text,
        "explicit_ms2_in_metadata_text": explicit_ms2_text,
        "inventory": inventory,
        "gates": gates,
        "blocking_reasons": blockers,
        "status": status,
        "ready_for_targeted_raw_probe": ready,
        "provenance": {endpoint: sha256(path) for endpoint, path in paths.items()},
        "claim_limit": (
            "Metadata and file-inventory readiness only. Counts are injections, not biological "
            "replicates; no raw scan, feature, spectrum, perturbation effect or algorithm result was inspected."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    registry = read_json(args.registry)
    reports = [audit_study(study, args.snapshot_dir) for study in registry["studies"]]
    source_groups: dict[str, list[str]] = defaultdict(list)
    for report in reports:
        source_groups[str(report["source_family"])].append(str(report["study_id"]))
    ready = [report["study_id"] for report in reports if report["ready_for_targeted_raw_probe"]]
    payload = {
        "status": "contrast_application_asset_audit_complete",
        "formal": True,
        "algorithm_bundle": registry["algorithm_bundle"],
        "studies": reports,
        "source_families": dict(source_groups),
        "ready_for_targeted_raw_probe": ready,
        "independent_source_families_ready": len({
            report["source_family"] for report in reports if report["ready_for_targeted_raw_probe"]
        }),
        "pass_to_multisource_figure5": len({
            report["source_family"] for report in reports if report["ready_for_targeted_raw_probe"]
        }) >= 2,
        "claim_limit": (
            "Development asset audit only. No sealed test, biological discovery, cross-study transfer, "
            "known-plus-unknown increment or module performance is established."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
