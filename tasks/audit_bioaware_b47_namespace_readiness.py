#!/usr/bin/env python
"""Truth-blind namespace and input-readiness audit for the B47 archives.

This stage may read observable-input headers and count spectra in an input MGF.
It never opens validation, confirmed-metabolite, or algorithm-output payloads.
The purpose is to decide whether a DreaMS/BioAware benchmark can be built, not
to compute an annotation result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = {
    "head_to_head": "67887553720068e18394fb6fb011f8365bc6714f5f3ce99c693fd13dc742dafa",
    "external_validation": "2d6ed0ae51b385b99025398e62939d8902096ba798e603dc5bd0acc8c9b28898",
}

HEAD_INPUTS = {
    "feature_table": "METDNA2/Input/CHDWB_HILICPOS_feature_table.csv",
    "sample_metadata": "METDNA2/Input/CHDWB_HILICPOS_sample_information.csv",
    "ms2": "METDNA2/Input/CHDWB_hilic_pos.mgf",
}

EXTERNAL_DATASETS = {
    "ST003356_HILIC_POS": {
        "feature_table": "CZ Biohub/ST003356_urine/asari/HILICPOS/preferred_Feature_table.tsv",
        "sample_metadata": None,
        "scope": "CZ Biohub/ST003356_urine/",
    },
    "ST001122_HILIC_POS": {
        "feature_table": "Oliver Fiehn/ST001122_urine/asari/preferred_Feature_table.tsv",
        "sample_metadata": None,
        "scope": "Oliver Fiehn/ST001122_urine/",
    },
    "ST001236_HILIC_POS": {
        "feature_table": "Broad_Institute/ST001236/asari/preferred_Feature_table.tsv",
        "sample_metadata": "Broad_Institute/ST001236/ST001236_metadata.csv",
        "scope": "Broad_Institute/ST001236/",
    },
    "ST002903_HILIC_POS": {
        "feature_table": "Broad_Institute/ST002903/asari/HILICPOS/preferred_Feature_table.tsv",
        "sample_metadata": None,
        "scope": "Broad_Institute/ST002903/",
    },
    "ST002576_C18_POS": {
        "feature_table": "Boston_University/C18POS/ST002576_AN004241_Results.txt",
        "sample_metadata": None,
        "scope": "Boston_University/C18POS/",
        "truth_tokens": ("st002576_c18pos", "an005610"),
    },
    "ST002576_C18_NEG": {
        "feature_table": "Boston_University/C18NEG/ST002576_AN004242_Results.txt",
        "sample_metadata": None,
        "scope": "Boston_University/C18NEG/",
        "truth_tokens": ("st002576_c18neg", "an005611"),
    },
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def normalized_members(archive: ZipFile) -> list[str]:
    return [
        item.filename.replace("\\", "/")
        for item in archive.infolist()
        if not item.is_dir() and "/__MACOSX/" not in f"/{item.filename.replace(chr(92), '/')}"
    ]


def suffix_matches(members: list[str], suffix: str | None) -> list[str]:
    if suffix is None:
        return []
    folded = suffix.casefold()
    return [name for name in members if name.casefold().endswith(folded)]


def unique_suffix(members: list[str], suffix: str) -> str:
    matches = suffix_matches(members, suffix)
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one member ending {suffix!r}; observed {matches}")
    return matches[0]


def read_header(archive: ZipFile, member: str) -> list[str]:
    """Read one observable-input header without retaining data rows."""
    with archive.open(member) as handle:
        first = handle.readline(1024 * 1024).decode("utf-8-sig", errors="replace").rstrip("\r\n")
    delimiter = "\t" if first.count("\t") > first.count(",") else ","
    return next(csv.reader([first], delimiter=delimiter))


def count_mgf_spectra(archive: ZipFile, member: str) -> int:
    count = 0
    with archive.open(member) as handle:
        for raw in handle:
            if raw.strip().upper() == b"BEGIN IONS":
                count += 1
    return count


def header_capabilities(columns: list[str]) -> dict[str, bool | int]:
    normalized = [column.strip().strip('"').casefold() for column in columns]
    mz_names = {"mz", "m/z", "row m/z", "rowmz", "precursormz", "precursor_mz"}
    rt_names = {"rt", "time", "rtime", "retention time", "retention_time"}
    metadata_names = {
        "id_number", "mz", "m/z", "rt", "time", "rtime", "rtime_left_base",
        "rtime_right_base", "parent_masstrack_id", "peak_area", "cselectivity",
        "goodness_fitting", "snr", "detection_counts", "mz.min", "mz.max", "name",
        "numpres.all.samples", "numpres.biological.samples", "median_cv",
        "peakscore", "qscore", "max.intensity", "feature", "feature_id", "id",
        "m/z_rt(sec)",
    }
    composite_mz_rt = any(item in {"m/z_rt(sec)", "mz_rt(sec)"} for item in normalized)
    sample_columns = sum(item not in metadata_names for item in normalized)
    return {
        "columns": len(columns),
        "has_mz": composite_mz_rt or any(item in mz_names for item in normalized),
        "has_rt": composite_mz_rt or any(item in rt_names for item in normalized),
        "mz_rt_combined_column": composite_mz_rt,
        "sample_abundance_columns": sample_columns,
        "has_sample_abundance": sample_columns > 0,
        "has_sample_identifiers": sample_columns > 0,
    }


def scope_counts(members: list[str], scope: str) -> dict[str, int]:
    scoped = [name for name in members if scope.casefold() in name.casefold()]
    truth_tokens = ("/validation/", "/confirmed_metabolite", "/confirmed_metabolites")
    algorithm_tokens = ("/msmica_result", "/msmica_results", "identified_metabolites")
    return {
        "members": len(scoped),
        "sealed_truth_or_evaluation_members": sum(
            any(token in name.casefold() for token in truth_tokens) for name in scoped
        ),
        "algorithm_output_members": sum(
            any(token in name.casefold() for token in algorithm_tokens) for name in scoped
        ),
    }


def token_match_count(members: list[str], tokens: tuple[str, ...]) -> int:
    return sum(
        any(token.casefold() in name.casefold() for token in tokens)
        and any(
            marker in name.casefold()
            for marker in ("/validation/", "/confirmed_metabolite", "/confirmed_metabolites")
        )
        for name in members
    )


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".json", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-to-head", type=Path, required=True)
    parser.add_argument("--external-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    paths = {
        "head_to_head": args.head_to_head.resolve(),
        "external_validation": args.external_validation.resolve(),
    }
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = digest(path)
        if observed != EXPECTED_SHA256[key]:
            raise RuntimeError(f"{key} SHA256 mismatch: {observed}")

    with ZipFile(paths["head_to_head"]) as head:
        head_members = normalized_members(head)
        head_feature = unique_suffix(head_members, HEAD_INPUTS["feature_table"])
        head_sample = unique_suffix(head_members, HEAD_INPUTS["sample_metadata"])
        head_mgf = unique_suffix(head_members, HEAD_INPUTS["ms2"])
        head_feature_caps = header_capabilities(read_header(head, head_feature))
        head_sample_columns = len(read_header(head, head_sample))
        head_mgf_spectra = count_mgf_spectra(head, head_mgf)
        head_truth_members = [
            name for name in head_members if "/validation/" in name.casefold()
        ]
        head_algorithm_members = [
            name
            for name in head_members
            if any(
                token in name.casefold()
                for token in ("/output/", "/msmica/", "/metdna3/", "/xmsannotator/")
            )
            and name not in {head_feature, head_sample, head_mgf}
        ]

    head_to_head = {
        "dataset": "CHDWB_HILIC_POS",
        "observable_inputs": {
            "feature_table": head_feature,
            "sample_metadata": head_sample,
            "ms2_mgf": head_mgf,
            "feature_table_capabilities": head_feature_caps,
            "sample_metadata_columns": head_sample_columns,
            "mgf_spectra": head_mgf_spectra,
        },
        "sealed_truth_or_evaluation_members": len(head_truth_members),
        "algorithm_output_members": len(head_algorithm_members),
        "gates": {
            "has_mz": bool(head_feature_caps["has_mz"]),
            "has_rt": bool(head_feature_caps["has_rt"]),
            "has_sample_abundance": bool(head_feature_caps["has_sample_abundance"]),
            "has_sample_metadata": head_sample_columns > 0,
            "has_ms2_input": head_mgf_spectra > 0,
            "truth_namespace_identifiable": len(head_truth_members) > 0,
            "algorithm_output_namespace_identifiable": len(head_algorithm_members) > 0,
        },
        "role": "development_candidate_after_physical_namespace_extraction",
        "claim_limit": (
            "Input completeness is established, but truth size, candidate-set reconstruction, "
            "seed leakage, and frozen-DreaMS error counts are not yet established."
        ),
    }
    head_to_head["pass_to_truth_separated_development_construction"] = all(
        head_to_head["gates"].values()
    )

    external_reports = {}
    with ZipFile(paths["external_validation"]) as external:
        members = normalized_members(external)
        all_ms2_inputs = [
            name
            for name in members
            if name.casefold().endswith((".mgf", ".msp", ".mzml", ".mzxml"))
        ]
        for dataset, specification in EXTERNAL_DATASETS.items():
            feature = unique_suffix(members, specification["feature_table"])
            columns = read_header(external, feature)
            caps = header_capabilities(columns)
            metadata_matches = suffix_matches(members, specification["sample_metadata"])
            counts = scope_counts(members, specification["scope"])
            if "truth_tokens" in specification:
                counts["sealed_truth_or_evaluation_members"] = token_match_count(
                    members, specification["truth_tokens"]
                )
            gates = {
                "has_mz": bool(caps["has_mz"]),
                "has_rt": bool(caps["has_rt"]),
                "has_sample_abundance": bool(caps["has_sample_abundance"]),
                "has_sample_identifiers": bool(caps["has_sample_identifiers"]),
                "has_explicit_sample_metadata": len(metadata_matches) == 1,
                "has_raw_or_standardized_ms2_input_in_archive": False,
                "truth_namespace_identifiable_by_path_only": (
                    counts["sealed_truth_or_evaluation_members"] > 0
                ),
            }
            external_reports[dataset] = {
                "observable_feature_table": feature,
                "feature_table_capabilities": caps,
                "explicit_sample_metadata": metadata_matches,
                "scope_inventory": counts,
                "gates": gates,
                "pass_to_dreams_external_retrieval_construction": all(
                    gates[key]
                    for key in (
                        "has_mz",
                        "has_rt",
                        "has_sample_abundance",
                        "has_sample_identifiers",
                        "has_raw_or_standardized_ms2_input_in_archive",
                        "truth_namespace_identifiable_by_path_only",
                    )
                ),
                "blocking_reason": (
                    "No raw or standardized MS/MS input is deposited in this archive; "
                    "author validation/MSMS-search tables are sealed evaluation products, not query spectra."
                ),
            }

    external_any_ms2 = len(all_ms2_inputs) > 0
    if external_any_ms2:
        raise RuntimeError(
            "external archive unexpectedly contains raw MS2-like files; update the frozen audit"
        )
    external_all_ready = all(
        item["pass_to_dreams_external_retrieval_construction"]
        for item in external_reports.values()
    )
    gates = {
        "archive_hashes_match_frozen_preflight": True,
        "head_to_head_input_complete_for_development": head_to_head[
            "pass_to_truth_separated_development_construction"
        ],
        "all_six_external_sources_have_ms1_abundance": all(
            item["gates"]["has_mz"]
            and item["gates"]["has_rt"]
            and item["gates"]["has_sample_abundance"]
            for item in external_reports.values()
        ),
        "all_six_external_sources_have_ms2_input": external_all_ready,
        "sealed_truth_payloads_never_opened": True,
        "algorithm_output_payloads_never_opened": True,
    }
    report = {
        "status": "bioaware_b47_namespace_readiness_complete",
        "formal": True,
        "observable_input_payloads_opened": True,
        "sealed_truth_payloads_opened": False,
        "algorithm_output_payloads_opened": False,
        "truth_values_read": False,
        "head_to_head": head_to_head,
        "external_validation": {
            "datasets": external_reports,
            "raw_or_standardized_ms2_members_in_archive": all_ms2_inputs,
            "pass_to_dreams_external_retrieval_construction": external_all_ready,
        },
        "gates": gates,
        "pass_to_model_fitting": all(gates.values()),
        "decision": (
            "Use CHDWB only as a truth-separated development candidate. Do not fit or report "
            "an external DreaMS/BioAware result until raw MS/MS for independent ST sources is "
            "acquired from their source repositories and joined without reading sealed truth."
        ),
        "next_stage": (
            "Create a physical extraction manifest for CHDWB observable inputs, then acquire raw "
            "MS/MS for external ST sources by accession. Re-run this gate before candidate construction."
        ),
        "claim_limit": (
            "This is an input-availability and namespace audit. It contains no BioAware score, "
            "DreaMS comparison, annotation metric, or performance claim."
        ),
        "provenance": {
            key: {"path": str(path), "sha256": digest(path)} for key, path in paths.items()
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "report.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "head_to_head": {
                    "mgf_spectra": head_mgf_spectra,
                    "pass_to_truth_separated_development_construction": head_to_head[
                        "pass_to_truth_separated_development_construction"
                    ],
                },
                "external_sources": len(external_reports),
                "external_sources_with_ms2": sum(
                    item["gates"]["has_raw_or_standardized_ms2_input_in_archive"]
                    for item in external_reports.values()
                ),
                "pass_to_model_fitting": report["pass_to_model_fitting"],
                "report": str(output / "report.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
