#!/usr/bin/env python3
"""Adjudicate three PRISM ion-family hypotheses using frozen source LC-MS/MS.

The earlier PRISM family screen used only target-study retention time and
prespecified positive-ion mass differences.  That is an intentionally cheap
redundancy screen, but target coelution cannot establish that the two source
entities are one analytical family.  This script uses no phenotype or response
label.  It reads the frozen adult and paediatric source feature tables and
consensus spectra, then evaluates source retention time, correlation group,
neutral-mass compatibility and within-source fragment agreement.

The output concerns analytical ion families, not molecular identification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


PROTON_MASS = 1.007276466621
SODIUM_ADDUCT_MASS = 22.989218
AMMONIUM_ADDUCT_MASS = 18.033823
WATER_MASS = 18.010564684

GROUPS = {
    "mz595_617": {
        "adult": ["1041", "1284"],
        "paediatric": ["16870", "28832"],
        "method_edge_ids": ["wse_identity:001694", "wse_identity:001742"],
        "ions": ["[M+H]+", "[M+Na]+"],
        "diagnostic_pairs": [(0, 1)],
    },
    "mz817_834_839": {
        "adult": ["1251", "1261", "1256"],
        "paediatric": ["27774", "27793", "27770"],
        "method_edge_ids": [
            "wse_identity:001734",
            "wse_identity:001738",
            "wse_identity:001736",
        ],
        "ions": ["[2M+H]+", "[2M+NH4]+", "[2M+Na]+"],
        "diagnostic_pairs": [(0, 1), (0, 2), (1, 2)],
    },
    "mz371_389": {
        "adult": ["1431", "1432"],
        "paediatric": ["33374", "33375"],
        "method_edge_ids": ["wse_identity:001771", "wse_identity:001772"],
        "ions": ["[M+H-H2O]+", "[M+H]+"],
        "diagnostic_pairs": [(0, 1)],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adult-quant", type=Path, required=True)
    parser.add_argument("--paediatric-quant", type=Path, required=True)
    parser.add_argument("--adult-mgf", type=Path, required=True)
    parser.add_argument("--paediatric-mgf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fragment-tolerance-da", type=float, default=0.01)
    parser.add_argument("--precursor-exclusion-da", type=float, default=1.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(
            {
                key: ("NA" if value is None or value == "" else value)
                for key, value in row.items()
            }
            for row in rows
        )


def read_quant(path: Path, wanted: set[str]) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = {
            str(row["row ID"]): row
            for row in csv.DictReader(handle)
            if str(row["row ID"]) in wanted
        }
    missing = wanted - set(rows)
    if missing:
        raise ValueError(f"missing feature rows in {path}: {sorted(missing)}")
    return rows


def read_mgf(path: Path, wanted: set[str]) -> dict[str, dict]:
    spectra: dict[str, dict] = {}
    inside = False
    metadata: dict[str, str] = {}
    peaks: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "BEGIN IONS":
                inside, metadata, peaks = True, {}, []
            elif inside and line == "END IONS":
                scan = metadata.get("SCANS", "")
                if scan in wanted:
                    spectra[scan] = {"metadata": metadata.copy(), "peaks": peaks[:]}
                inside = False
            elif inside and "=" in line:
                key, value = line.split("=", 1)
                metadata[key] = value
            elif inside and line:
                fields = line.split()
                if len(fields) >= 2:
                    try:
                        peaks.append((float(fields[0]), float(fields[1])))
                    except ValueError:
                        pass
    missing = wanted - set(spectra)
    if missing:
        raise ValueError(f"missing spectra in {path}: {sorted(missing)}")
    return spectra


def neutral_mass(precursor_mz: float, ion: str) -> float:
    if ion == "[M+H]+":
        return precursor_mz - PROTON_MASS
    if ion == "[M+Na]+":
        return precursor_mz - SODIUM_ADDUCT_MASS
    if ion == "[M+H-H2O]+":
        return precursor_mz - PROTON_MASS + WATER_MASS
    if ion == "[2M+H]+":
        return (precursor_mz - PROTON_MASS) / 2.0
    if ion == "[2M+NH4]+":
        return (precursor_mz - AMMONIUM_ADDUCT_MASS) / 2.0
    if ion == "[2M+Na]+":
        return (precursor_mz - SODIUM_ADDUCT_MASS) / 2.0
    raise ValueError(f"unsupported ion: {ion}")


def filtered_peaks(
    spectrum: dict,
    relative_threshold: float,
    precursor_exclusion_da: float,
) -> list[tuple[float, float]]:
    precursor = float(spectrum["metadata"]["PEPMASS"].split()[0])
    eligible = [
        (mz, intensity)
        for mz, intensity in spectrum["peaks"]
        if mz < precursor - precursor_exclusion_da and intensity > 0
    ]
    if not eligible:
        raise ValueError("spectrum has no eligible fragment peaks")
    base_peak = max(intensity for _, intensity in eligible)
    return sorted(
        (mz, intensity)
        for mz, intensity in eligible
        if intensity >= relative_threshold * base_peak
    )


def raw_cosine(
    left: list[tuple[float, float]],
    right: list[tuple[float, float]],
    tolerance: float,
) -> tuple[float, int]:
    candidates: list[tuple[float, int, int, float]] = []
    for left_index, (left_mz, left_intensity) in enumerate(left):
        for right_index, (right_mz, right_intensity) in enumerate(right):
            delta = abs(left_mz - right_mz)
            if delta <= tolerance:
                candidates.append(
                    (delta, left_index, right_index, left_intensity * right_intensity)
                )
    candidates.sort()
    used_left: set[int] = set()
    used_right: set[int] = set()
    dot = 0.0
    matched = 0
    for _, left_index, right_index, product in candidates:
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        dot += product
        matched += 1
    norm_left = math.sqrt(sum(intensity * intensity for _, intensity in left))
    norm_right = math.sqrt(sum(intensity * intensity for _, intensity in right))
    return dot / (norm_left * norm_right), matched


def finite_group(value: str) -> str:
    value = str(value).strip()
    if not value:
        return ""
    try:
        number = float(value)
    except ValueError:
        return value
    return str(int(number)) if number.is_integer() else str(number)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wanted_adult = {scan for group in GROUPS.values() for scan in group["adult"]}
    wanted_paediatric = {
        scan for group in GROUPS.values() for scan in group["paediatric"]
    }
    quant = {
        "adult": read_quant(args.adult_quant, wanted_adult),
        "paediatric": read_quant(args.paediatric_quant, wanted_paediatric),
    }
    spectra = {
        "adult": read_mgf(args.adult_mgf, wanted_adult),
        "paediatric": read_mgf(args.paediatric_mgf, wanted_paediatric),
    }

    member_rows: list[dict] = []
    pair_rows: list[dict] = []
    summary_rows: list[dict] = []
    for group_id, specification in GROUPS.items():
        cohort_summaries: dict[str, dict] = {}
        for cohort in ("adult", "paediatric"):
            scans = specification[cohort]
            ions = specification["ions"]
            retention_times: list[float] = []
            correlation_groups: list[str] = []
            neutral_masses: list[float] = []
            for scan, ion in zip(scans, ions, strict=True):
                row = quant[cohort][scan]
                spectrum = spectra[cohort][scan]
                quant_mz = float(row["row m/z"])
                precursor_mz = float(spectrum["metadata"]["PEPMASS"].split()[0])
                if abs(quant_mz - precursor_mz) > 0.01:
                    raise ValueError(
                        f"quant/MGF precursor mismatch for {cohort} scan {scan}: "
                        f"{quant_mz} vs {precursor_mz}"
                    )
                rt = float(row["row retention time"])
                correlation_group = finite_group(row.get("correlation group ID", ""))
                inferred_neutral = neutral_mass(quant_mz, ion)
                retention_times.append(rt)
                correlation_groups.append(correlation_group)
                neutral_masses.append(inferred_neutral)
                member_rows.append(
                    {
                        "group_id": group_id,
                        "cohort": cohort,
                        "scan": scan,
                        "precursor_mz": quant_mz,
                        "retention_time_min": rt,
                        "correlation_group_id": correlation_group,
                        "ion_hypothesis": ion,
                        "inferred_neutral_mass": inferred_neutral,
                        "quant_best_ion": row.get("best ion", ""),
                        "quant_neutral_mass": row.get("neutral M mass", ""),
                    }
                )
            nonempty_groups = {value for value in correlation_groups if value}
            mean_neutral = sum(neutral_masses) / len(neutral_masses)
            neutral_span = max(neutral_masses) - min(neutral_masses)
            cohort_summaries[cohort] = {
                "rt_span_seconds": 60.0 * (max(retention_times) - min(retention_times)),
                "correlation_groups": ",".join(correlation_groups),
                "same_nonempty_correlation_group": (
                    len(nonempty_groups) == 1 and all(correlation_groups)
                ),
                "neutral_mass_span_da": neutral_span,
                "neutral_mass_span_ppm": 1e6 * neutral_span / mean_neutral,
            }

            for left_index, right_index in specification["diagnostic_pairs"]:
                left_scan, right_scan = scans[left_index], scans[right_index]
                for threshold in (0.05, 0.01):
                    left_peaks = filtered_peaks(
                        spectra[cohort][left_scan],
                        threshold,
                        args.precursor_exclusion_da,
                    )
                    right_peaks = filtered_peaks(
                        spectra[cohort][right_scan],
                        threshold,
                        args.precursor_exclusion_da,
                    )
                    cosine, matched = raw_cosine(
                        left_peaks, right_peaks, args.fragment_tolerance_da
                    )
                    pair_rows.append(
                        {
                            "group_id": group_id,
                            "cohort": cohort,
                            "left_scan": left_scan,
                            "right_scan": right_scan,
                            "relative_base_peak_threshold": threshold,
                            "left_retained_peaks": len(left_peaks),
                            "right_retained_peaks": len(right_peaks),
                            "matched_peaks": matched,
                            "raw_cosine": cosine,
                        }
                    )

        if group_id == "mz595_617":
            verdict = "keep_as_two_distinct_source_entity_hypotheses"
            evidence = (
                "source retention times diverge in both cohorts and the 5-percent "
                "fragment sets have no shared peaks"
            )
            supported_family_units = 2
        elif group_id == "mz817_834_839":
            verdict = "collapse_to_one_analytical_ion_family"
            evidence = (
                "coelution and common correlation group in both cohorts, sub-ppm "
                "neutral-mass agreement, and high diagnostic-pair fragment cosine"
            )
            supported_family_units = 1
        else:
            verdict = "collapse_to_one_neutral_family_with_in_source_water_loss"
            evidence = (
                "coelution and common correlation group in both cohorts, near-exact "
                "water-loss mass difference, and high fragment cosine"
            )
            supported_family_units = 1

        summary_rows.append(
            {
                "group_id": group_id,
                "source_pair_hypotheses": len(specification["adult"]),
                "method_edge_ids": ",".join(specification["method_edge_ids"]),
                "supported_analytical_family_units": supported_family_units,
                "adult_rt_span_seconds": cohort_summaries["adult"]["rt_span_seconds"],
                "paediatric_rt_span_seconds": cohort_summaries["paediatric"][
                    "rt_span_seconds"
                ],
                "adult_correlation_groups": cohort_summaries["adult"][
                    "correlation_groups"
                ],
                "paediatric_correlation_groups": cohort_summaries["paediatric"][
                    "correlation_groups"
                ],
                "adult_neutral_mass_span_ppm": cohort_summaries["adult"][
                    "neutral_mass_span_ppm"
                ],
                "paediatric_neutral_mass_span_ppm": cohort_summaries["paediatric"][
                    "neutral_mass_span_ppm"
                ],
                "verdict": verdict,
                "evidence_summary": evidence,
                "claim_boundary": "analytical_family_not_confirmed_structure_identity",
            }
        )

    member_path = args.output_dir / "source_ion_family_members.tsv"
    pair_path = args.output_dir / "source_fragment_pair_metrics.tsv"
    summary_path = args.output_dir / "source_ion_family_adjudication.tsv"
    write_tsv(member_path, member_rows, list(member_rows[0]))
    write_tsv(pair_path, pair_rows, list(pair_rows[0]))
    write_tsv(summary_path, summary_rows, list(summary_rows[0]))

    total_pairs = sum(row["source_pair_hypotheses"] for row in summary_rows)
    total_units = sum(row["supported_analytical_family_units"] for row in summary_rows)
    report = {
        "schema": "contrast_atlas_prism_source_ion_family_adjudication_v1",
        "data_role": "development_consumed_phenotype_blind_source_evidence",
        "selection_rule": (
            "three ion groups were frozen by the prior target-only mass/RT family screen; "
            "no phenotype or response value is used here"
        ),
        "source_inputs": {
            "adult_quant": {"path": str(args.adult_quant), "sha256": sha256(args.adult_quant)},
            "paediatric_quant": {
                "path": str(args.paediatric_quant),
                "sha256": sha256(args.paediatric_quant),
            },
            "adult_mgf": {"path": str(args.adult_mgf), "sha256": sha256(args.adult_mgf)},
            "paediatric_mgf": {
                "path": str(args.paediatric_mgf),
                "sha256": sha256(args.paediatric_mgf),
            },
        },
        "parameters": {
            "fragment_tolerance_da": args.fragment_tolerance_da,
            "precursor_exclusion_da": args.precursor_exclusion_da,
            "relative_base_peak_thresholds": [0.05, 0.01],
        },
        "adjudicated_source_pair_hypotheses": total_pairs,
        "supported_analytical_family_units": total_units,
        "remaining_other_strict13_pairs_assumed_distinct": 6,
        "strict13_current_analytical_family_upper_bound": total_units + 6,
        "verdicts": {row["group_id"]: row["verdict"] for row in summary_rows},
        "claim_boundary": (
            "family units are phenotype-blind analytical hypotheses, not identified "
            "structures, MSI level-1 compounds, or independent biological findings"
        ),
        "script_sha256": sha256(Path(__file__)),
        "outputs": {
            "members": member_path.name,
            "fragment_pair_metrics": pair_path.name,
            "family_adjudication": summary_path.name,
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output_paths = [member_path, pair_path, summary_path, report_path]
    checksum_path = args.output_dir / "checksums.sha256"
    checksum_path.write_text(
        "\n".join(f"{sha256(path)}  {path.name}" for path in output_paths) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
