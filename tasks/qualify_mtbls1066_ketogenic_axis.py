#!/usr/bin/env python3
"""Freeze the public design boundary for the MTBLS1066 ketogenic-diet source.

This is a metadata-only qualification step.  It does not inspect abundance
effects and it does not declare a cross-study biological relation to be true.
The MetaboLights ISA tables contain one row per *group*, while the actual
biological replicates are encoded in the official FILES directory.  The script
therefore reconciles both sources before any response is estimated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


ACCESSION = "MTBLS1066"
BASE_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/"
    f"{ACCESSION}/"
)
FILES_URL = BASE_URL + "FILES/"
SAMPLE_URL = BASE_URL + f"s_{ACCESSION}.txt"
ASSAY_NAMES = (
    f"a_{ACCESSION}_NEG_HILIC_LCMS.txt",
    f"a_{ACCESSION}_NEG_RP_LCMS.txt",
    f"a_{ACCESSION}_POS_HILIC_LCMS.txt",
    f"a_{ACCESSION}_POS_RP_LCMS.txt",
)
BIOLOGICAL_GROUPS = {"H_C", "H_MCT", "H_LCT", "BC_C", "BC_MCT"}
GROUP_PATTERN = re.compile(
    r"_(H_LCT|H_MCT|H_C|BC_MCT|BC_C|Pool|Blank)_(R?\d+)", re.IGNORECASE
)
EXPECTED_REPLICATES = {
    "H_C": 4,
    "H_MCT": 4,
    "H_LCT": 5,
    "BC_C": 4,
    "BC_MCT": 3,
    "POOL": 10,
    "BLANK": 12,
}


def fetch(url: str, *, byte_range: tuple[int, int] | None = None) -> bytes:
    headers = {"User-Agent": "DreaMS-Contrast-Atlas-metadata-audit/1.0"}
    if byte_range is not None:
        headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_directory(html: str) -> list[str]:
    names = re.findall(r'href="([^"/]+\.mzML)"', html, flags=re.IGNORECASE)
    return sorted(set(names))


def modality(filename: str) -> tuple[str, str]:
    lower = filename.lower()
    chromatography = "HILIC" if "_hilic.mzml" in lower else "RP"
    if "_pos_" in lower:
        polarity = "POS"
    elif "_neg_" in lower:
        polarity = "NEG"
    else:
        raise ValueError(f"cannot infer polarity: {filename}")
    return chromatography, polarity


def parse_file_rows(names: Iterable[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name in names:
        match = GROUP_PATTERN.search(name)
        if match is None:
            raise ValueError(f"unrecognized MTBLS1066 mzML filename: {name}")
        group = match.group(1).upper()
        replicate = match.group(2).upper()
        chromatography, polarity = modality(name)
        role = "biological" if group in BIOLOGICAL_GROUPS else group.lower()
        rows.append(
            {
                "accession": ACCESSION,
                "filename": name,
                "official_url": FILES_URL + name,
                "group": group,
                "replicate": replicate,
                "role": role,
                "chromatography": chromatography,
                "polarity": polarity,
                "panel": f"{polarity}_{chromatography}",
                "primary_pos_rp": bool(
                    group in {"H_C", "H_MCT", "BC_C", "BC_MCT"}
                    and chromatography == "RP"
                    and polarity == "POS"
                ),
                "semantic_audit_only": bool(group == "H_LCT"),
            }
        )
    return rows


def parse_tsv(payload: bytes) -> list[dict[str, str]]:
    text = payload.decode("utf-8-sig", errors="strict")
    return list(csv.DictReader(io.StringIO(text), delimiter="\t"))


def group_sample_metadata(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        group = str(row.get("Sample Name", "")).strip().upper()
        if not group:
            continue
        result[group] = {
            "source_name_template": str(row.get("Source Name", "")),
            "organism": str(row.get("Characteristics[Organism]", "")),
            "variant": str(row.get("Characteristics[Variant]", "")),
            "organism_part": str(row.get("Characteristics[Organism part]", "")),
            "diet": str(row.get("Factor Value[Diet]", "")),
            "health_status": str(row.get("Factor Value[Health status]", "")),
        }
    return result


def assay_group_rows(payload: bytes) -> list[str]:
    rows = parse_tsv(payload)
    return [str(row.get("Sample Name", "")).strip().upper() for row in rows]


def spectrum_probe(url: str, maximum_bytes: int) -> dict[str, object]:
    if maximum_bytes <= 0:
        return {"status": "not_requested"}
    payload = fetch(url, byte_range=(0, maximum_bytes - 1))
    ms1 = payload.count(b'name="ms level" value="1"')
    ms2 = payload.count(b'name="ms level" value="2"')
    return {
        "status": "sample_level_ms2_observed" if ms2 > 0 else "ms2_not_seen_in_prefix",
        "url": url,
        "requested_prefix_bytes": maximum_bytes,
        "received_bytes": len(payload),
        "received_sha256": sha256_bytes(payload),
        "ms1_tags_in_received_bytes": ms1,
        "ms2_tags_in_received_bytes": ms2,
        "claim_boundary": (
            "A positive count proves that this biological sample contains DDA MS2. "
            "Prefix counts are not whole-file scan denominators."
        ),
    }


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/contrast_atlas_mtbls1066_kd_qualification_v1_20261022"),
    )
    parser.add_argument(
        "--spectrum-probe-bytes",
        type=int,
        default=2 * 1024 * 1024,
        help="Read only this prefix of one biological POS-RP mzML to prove sample-level DDA.",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")

    directory_payload = fetch(FILES_URL)
    sample_payload = fetch(SAMPLE_URL)
    assay_payloads = {name: fetch(BASE_URL + name) for name in ASSAY_NAMES}
    filenames = parse_directory(directory_payload.decode("utf-8", errors="replace"))
    rows = parse_file_rows(filenames)
    samples = group_sample_metadata(parse_tsv(sample_payload))

    replicate_sets: dict[str, set[str]] = defaultdict(set)
    file_counts = Counter()
    panel_counts = Counter()
    for row in rows:
        group = str(row["group"])
        replicate_sets[group].add(str(row["replicate"]))
        file_counts[group] += 1
        panel_counts[str(row["panel"])] += 1

    observed_replicates = {key: len(value) for key, value in replicate_sets.items()}
    if observed_replicates != EXPECTED_REPLICATES:
        raise RuntimeError(
            f"official file topology changed: {observed_replicates} != {EXPECTED_REPLICATES}"
        )
    if set(panel_counts.values()) != {42} or set(panel_counts) != {
        "NEG_HILIC", "NEG_RP", "POS_HILIC", "POS_RP"
    }:
        raise RuntimeError(f"unexpected panel topology: {dict(panel_counts)}")

    assay_groups = {name: assay_group_rows(payload) for name, payload in assay_payloads.items()}
    if any(set(groups) != set(samples) for groups in assay_groups.values()):
        raise RuntimeError("ISA assay group rows do not match the sample table")

    primary_files = [
        row for row in rows
        if bool(row["primary_pos_rp"])
    ]
    if len(primary_files) != 15:
        raise RuntimeError(f"expected 15 primary POS-RP biological files, got {len(primary_files)}")
    probe_file = next(
        row for row in rows
        if row["group"] == "H_C"
        and row["replicate"] == "R1"
        and row["panel"] == "POS_RP"
    )
    probe = spectrum_probe(str(probe_file["official_url"]), args.spectrum_probe_bytes)

    h_lct = samples.get("H_LCT", {})
    h_mct = samples.get("H_MCT", {})
    semantic_collision = bool(h_lct.get("diet") and h_lct.get("diet") == h_mct.get("diet"))

    report = {
        "schema": "contrast_atlas_public_intervention_source_qualification_v1",
        "status": "qualified_with_semantic_exclusion",
        "accession": ACCESSION,
        "evaluation_role": "development_consumed_after_use",
        "leakage_status": "metadata_and_acquisition_only_no_abundance_effects_opened",
        "official_sources": {
            "files_directory": {
                "url": FILES_URL,
                "sha256": sha256_bytes(directory_payload),
            },
            "sample_table": {"url": SAMPLE_URL, "sha256": sha256_bytes(sample_payload)},
            "assays": {
                name: {"url": BASE_URL + name, "sha256": sha256_bytes(payload)}
                for name, payload in assay_payloads.items()
            },
        },
        "denominators": {
            "mzml_files": len(rows),
            "biological_units": sum(observed_replicates[g] for g in BIOLOGICAL_GROUPS),
            "biological_units_by_group": {
                group: observed_replicates[group] for group in sorted(BIOLOGICAL_GROUPS)
            },
            "files_by_group": dict(sorted(file_counts.items())),
            "files_by_panel": dict(sorted(panel_counts.items())),
            "primary_pos_rp_biological_files": len(primary_files),
        },
        "design": {
            "species": "Mus musculus",
            "biospecimen": "blood plasma",
            "sex_and_model": "female CD-1 nu/nu mice",
            "primary_response": {
                "contrast_id": "healthy_lct_mct8_kd_minus_control",
                "positive_group": "H_MCT",
                "negative_group": "H_C",
                "units": {"H_MCT": 4, "H_C": 4},
                "interpretation": "ketogenic-diet response in healthy mice",
            },
            "secondary_response": {
                "contrast_id": "breast_cancer_chemo_lct_mct8_kd_minus_control",
                "positive_group": "BC_MCT",
                "negative_group": "BC_C",
                "units": {"BC_MCT": 3, "BC_C": 4},
                "interpretation": (
                    "ketogenic-diet response conditional on breast-cancer xenograft "
                    "and cyclophosphamide background"
                ),
            },
            "h_lct_status": {
                "status": "audit_only_excluded_from_claim_bearing_contrasts",
                "units": observed_replicates["H_LCT"],
                "reason": (
                    "The official filenames separate H_LCT and H_MCT, but the ISA sample "
                    "table assigns the same diet label to both groups.  Their biological "
                    "meaning cannot be inferred from the filenames."
                ),
                "isa_diet_label_collision": semantic_collision,
            },
        },
        "sample_level_ms2_probe": probe,
        "admission": {
            "within_source_directed_response": True,
            "sample_level_ms2_evidence": probe.get("status") == "sample_level_ms2_observed",
            "independent_cross_source_validation": False,
            "prespecified_metabolite_direction_truth": False,
            "relation_to_hfd_sources": (
                "intervention-family relevance only; not exact diet replication and not a "
                "predeclared same-direction metabolite truth"
            ),
        },
        "claim_boundary": [
            "This qualification establishes a real public intervention design and sample-level MS2 acquisition.",
            "It does not establish a cross-study retrieval success or an unknown-information increment.",
            "H_MCT and H_LCT must not be pooled until their conflicting public metadata are resolved independently.",
            "All later effects from this source are development/consumed, never unopened external validation.",
        ],
    }

    args.output_dir.mkdir(parents=True)
    write_tsv(args.output_dir / "official_file_manifest.tsv", rows)
    write_tsv(args.output_dir / "primary_pos_rp_manifest.tsv", primary_files)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
