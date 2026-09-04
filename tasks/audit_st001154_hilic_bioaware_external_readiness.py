#!/usr/bin/env python
"""Audit the ST001154 HILIC-negative channel for frozen BioAware evaluation.

The HILIC channel is assessed separately from the lipid-rich CSH channel.  No
DreaMS or BioAware score is computed.  Network reachability is evaluated with
the query truth removed from its own seed set.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
from pathlib import Path

import pandas as pd

try:
    from audit_st001154_bioaware_external_readiness import (
        align_targets,
        analysis_record,
        bool_series,
        checksum,
        load_ms2,
        read_concatenated_json,
    )
except ModuleNotFoundError:  # imported as tasks.* during tests
    from tasks.audit_st001154_bioaware_external_readiness import (
        align_targets,
        analysis_record,
        bool_series,
        checksum,
        load_ms2,
        read_concatenated_json,
    )


def undirected_edge_graph(edges: pd.DataFrame) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {}
    for left, right in edges[["ik14_a", "ik14_b"]].itertuples(index=False):
        left, right = str(left), str(right)
        if left == right:
            continue
        graph.setdefault(left, set()).add(right)
        graph.setdefault(right, set()).add(left)
    return graph


def rhea_graph(participants: pd.DataFrame, maximum_participants: int = 8) -> dict[str, set[str]]:
    local = participants.loc[~bool_series(participants["is_currency"])].copy()
    graph: dict[str, set[str]] = {}
    for _, group in local.groupby("reaction_id", sort=False):
        identities = sorted(set(group["compound_id"].astype(str)))
        if not (1 < len(identities) <= maximum_participants):
            continue
        for left, right in itertools.combinations(identities, 2):
            graph.setdefault(left, set()).add(right)
            graph.setdefault(right, set()).add(left)
    return graph


def reachable_to_other_seed(
    query: str, seeds: set[str], graph: dict[str, set[str]], maximum_depth: int
) -> bool:
    target_seeds = seeds - {query}
    frontier = {query}
    visited = {query}
    for _ in range(maximum_depth):
        following = {
            neighbor
            for node in frontier
            for neighbor in graph.get(node, set())
            if neighbor not in visited
        }
        if following & target_seeds:
            return True
        visited |= following
        frontier = following
        if not frontier:
            break
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mwtab",
        type=Path,
        default=Path(
            "data/reference/bioaware_public_cohort_probe_20260901/"
            "ST001154__mwtab__json"
        ),
    )
    parser.add_argument(
        "--mzml",
        type=Path,
        default=Path(
            "data/reference/ST001154_negative_pilot_20260901/mzml/"
            "KOMP_HILIC_NEG_345321_152.mzML"
        ),
    )
    parser.add_argument(
        "--sample-workbook",
        type=Path,
        default=Path(
            "data/reference/ST001154_negative_pilot_20260901/"
            "KOMP_All_AssaysSampleDetails.xlsx"
        ),
    )
    parser.add_argument(
        "--library-integrity",
        type=Path,
        default=Path(
            "data/validation/mona_negative_library_chemical_integrity_v1/"
            "library_row_integrity.csv.gz"
        ),
    )
    parser.add_argument(
        "--development-units",
        type=Path,
        default=Path("data/validation/bioaware_metdna3_external_negative_units_v2"),
    )
    parser.add_argument(
        "--emrn-network",
        type=Path,
        default=Path(
            "data/reference/metdna2_emrn_network_20260828/metdna2_emrn_edges.csv.gz"
        ),
    )
    parser.add_argument(
        "--rhea-participants",
        type=Path,
        default=Path(
            "data/reference/bioaware_rhea_offline_20260827/rhea_participants.csv.gz"
        ),
    )
    parser.add_argument("--analysis-id", default="AN001945")
    parser.add_argument("--sample-label", default="345321_152")
    parser.add_argument("--primary-ppm", type=float, default=10.0)
    parser.add_argument("--primary-rt-sec", type=float, default=6.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/bioaware_st001154_hilic_external_readiness_v1"),
    )
    args = parser.parse_args()
    required_paths = (
        args.mwtab,
        args.mzml,
        args.sample_workbook,
        args.library_integrity,
        args.emrn_network,
        args.rhea_participants,
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    record = analysis_record(read_concatenated_json(args.mwtab), args.analysis_id)
    author = pd.DataFrame(record["MS_METABOLITE_DATA"]["Metabolites"])
    required_columns = {"Metabolite", "Adducts", "retention times", "m/z", "InChiKey"}
    if not required_columns.issubset(author.columns):
        raise RuntimeError(f"HILIC author table lacks {sorted(required_columns-set(author.columns))}")
    author["ik14"] = author["InChiKey"].fillna("").astype(str).str[:14]
    author["target_mz"] = pd.to_numeric(author["m/z"], errors="coerce")
    author["target_rt_sec"] = pd.to_numeric(author["retention times"], errors="coerce") * 60.0
    structured = author.loc[
        author["Adducts"].eq("[M-H]-")
        & author["ik14"].str.len().eq(14)
        & author["target_mz"].notna()
        & author["target_rt_sec"].notna()
    ].copy()
    if structured["ik14"].duplicated().any():
        raise RuntimeError("HILIC author structure identities are not unique")

    abundance = pd.DataFrame(record["MS_METABOLITE_DATA"]["Data"])
    if args.sample_label not in abundance.columns:
        raise RuntimeError(f"sample abundance column is absent: {args.sample_label}")
    sample_abundance = dict(
        zip(
            abundance["Metabolite"].astype(str),
            pd.to_numeric(abundance[args.sample_label], errors="coerce"),
            strict=True,
        )
    )
    structured["sample_abundance"] = structured["Metabolite"].map(sample_abundance)
    if structured["sample_abundance"].isna().any() or (structured["sample_abundance"] <= 0).any():
        raise RuntimeError("structured HILIC targets are not all observed in the pilot sample")

    ms2, level_counts = load_ms2(args.mzml)
    if set(ms2["polarity_code"].unique()) != {2}:
        raise RuntimeError("HILIC pilot MS2 is not uniformly negative polarity")
    aligned = align_targets(structured, ms2, args.primary_ppm, args.primary_rt_sec)
    if aligned["ik14"].duplicated().any():
        raise RuntimeError("HILIC strict alignment produced duplicate truth identities")

    sheet = pd.read_excel(args.sample_workbook, sheet_name="HILICNEG")
    expected_raw = args.mzml.stem + ".raw"
    sample_rows = sheet.loc[sheet["FileName"].astype(str).eq(expected_raw)]
    if len(sample_rows) != 1 or str(sample_rows.iloc[0]["SAMPLETYPE"]) != "StudySample":
        raise RuntimeError("converted HILIC RAW is not a uniquely registered StudySample")

    library = pd.read_csv(args.library_integrity)
    approved = library.loc[bool_series(library["approved_m_h_reference"])].copy()
    approved["ik14"] = approved["inchikey"].fillna("").astype(str).str[:14]
    full_formula_rows = approved[["inchikey", "calculated_formula"]].drop_duplicates()
    if full_formula_rows.groupby("inchikey")["calculated_formula"].nunique().max() != 1:
        raise RuntimeError("approved MONA full InChIKeys map to multiple formulas")
    full_formula = dict(full_formula_rows.itertuples(index=False, name=None))
    ik14_formula_rows = approved[["ik14", "calculated_formula"]].drop_duplicates()
    grouped_ik14 = ik14_formula_rows.groupby("ik14")["calculated_formula"].agg(list)
    ik14_formula = {key: values[0] for key, values in grouped_ik14.items() if len(values) == 1}
    formula_candidate_count = approved.groupby("calculated_formula")["ik14"].nunique().to_dict()
    approved_ids = set(approved["ik14"])

    aligned["approved_mona_truth"] = aligned["ik14"].isin(approved_ids)
    aligned["truth_formula"] = aligned["InChiKey"].map(full_formula)
    fallback = aligned["truth_formula"].isna()
    aligned.loc[fallback, "truth_formula"] = aligned.loc[fallback, "ik14"].map(ik14_formula)
    aligned["formula_candidate_identities"] = (
        aligned["truth_formula"].map(formula_candidate_count).fillna(0).astype(int)
    )
    aligned["formula_ambiguous"] = aligned["formula_candidate_identities"] >= 2

    development_paths = glob.glob(str(args.development_units / "*" / "queries.csv.gz"))
    if not development_paths:
        raise RuntimeError("frozen BioAware development query ledgers are absent")
    development_ids = set(
        pd.concat(
            [pd.read_csv(path, usecols=["truth_ik14"]) for path in development_paths],
            ignore_index=True,
        )["truth_ik14"].astype(str)
    )
    aligned["development_identity_overlap"] = aligned["ik14"].isin(development_ids)
    aligned["primary_candidate_query"] = (
        aligned["approved_mona_truth"]
        & aligned["formula_ambiguous"]
        & ~aligned["development_identity_overlap"]
    )

    emrn = pd.read_csv(args.emrn_network)
    graph0 = undirected_edge_graph(emrn.loc[pd.to_numeric(emrn["minimum_step"]) == 0])
    rhea = rhea_graph(pd.read_csv(args.rhea_participants))
    seed_ids = set(aligned["ik14"])
    query_ids = set(aligned.loc[aligned["primary_candidate_query"], "ik14"])
    for name, graph in (("emrn", graph0), ("rhea", rhea)):
        aligned[f"{name}_in_graph"] = aligned["ik14"].isin(graph)
        aligned[f"{name}_seed_reachable_depth1"] = aligned["ik14"].map(
            lambda value: reachable_to_other_seed(value, seed_ids, graph, 1)
        )
        aligned[f"{name}_seed_reachable_depth2"] = aligned["ik14"].map(
            lambda value: reachable_to_other_seed(value, seed_ids, graph, 2)
        )

    primary = aligned.loc[aligned["primary_candidate_query"]].copy()
    gates = {
        "structured_author_m_h_identities_ge_100": int(structured["ik14"].nunique()) >= 100,
        "strict_raw_aligned_identities_ge_60": int(aligned["ik14"].nunique()) >= 60,
        "clean_formula_ambiguous_queries_ge_20": int(primary["ik14"].nunique()) >= 20,
        "emrn_depth2_reachable_queries_ge_5": int(primary["emrn_seed_reachable_depth2"].sum()) >= 5,
        "rhea_depth2_reachable_queries_ge_10": int(primary["rhea_seed_reachable_depth2"].sum()) >= 10,
        "uniform_negative_polarity_ms2": set(ms2["polarity_code"].unique()) == {2},
        "biological_study_sample_verified": True,
    }
    report = {
        "status": "bioaware_st001154_hilic_external_readiness_complete",
        "formal": True,
        "outcome_status": "unopened; no DreaMS or BioAware scores computed",
        "study": {
            "study_id": "ST001154",
            "analysis_id": args.analysis_id,
            "sample_file": expected_raw,
            "sample_label": args.sample_label,
            "sample_type": str(sample_rows.iloc[0]["SAMPLETYPE"]),
            "chromatography": "HILIC",
            "polarity": "negative",
        },
        "raw_scan_audit": {
            "spectra_by_ms_level": level_counts,
            "precursor_bearing_ms2": int(len(ms2)),
            "strict_alignment_ppm": args.primary_ppm,
            "strict_alignment_rt_seconds": args.primary_rt_sec,
        },
        "author_truth": {
            "rows": int(len(author)),
            "structured_exact_m_h_identities": int(structured["ik14"].nunique()),
            "strict_raw_aligned_identities": int(aligned["ik14"].nunique()),
            "strict_aligned_approved_mona": int(aligned["approved_mona_truth"].sum()),
            "strict_aligned_formula_ambiguous": int(aligned["formula_ambiguous"].sum()),
            "strict_aligned_development_overlap": int(aligned["development_identity_overlap"].sum()),
            "clean_primary_candidate_queries": int(primary["ik14"].nunique()),
        },
        "network_headroom": {
            "seed_identities": len(seed_ids),
            "primary_queries": len(query_ids),
            "emrn_primary_in_graph": int(primary["emrn_in_graph"].sum()),
            "emrn_primary_seed_reachable_depth1": int(primary["emrn_seed_reachable_depth1"].sum()),
            "emrn_primary_seed_reachable_depth2": int(primary["emrn_seed_reachable_depth2"].sum()),
            "rhea_primary_in_graph": int(primary["rhea_in_graph"].sum()),
            "rhea_primary_seed_reachable_depth1": int(primary["rhea_seed_reachable_depth1"].sum()),
            "rhea_primary_seed_reachable_depth2": int(primary["rhea_seed_reachable_depth2"].sum()),
            "query_truth_removed_from_own_seed_search": True,
        },
        "gates": gates,
        "pass_to_frozen_external_manifest": all(gates.values()),
        "provenance": {
            "mwtab_sha256": checksum(args.mwtab),
            "mzml_sha256": checksum(args.mzml),
            "sample_workbook_sha256": checksum(args.sample_workbook),
            "library_integrity_sha256": checksum(args.library_integrity),
            "emrn_network_sha256": checksum(args.emrn_network),
            "rhea_participants_sha256": checksum(args.rhea_participants),
            "script_sha256": checksum(Path(__file__)),
        },
        "claim_limit": (
            "Author structures with raw precursor/RT-aligned MS2 are an external author truth tier, "
            "not locally reinjected MSI Level 1 standards. Readiness and network reachability only; "
            "no algorithm outcome, phenotype, flux, enzyme, or SOTA claim is evaluated."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = args.output_dir / "strict_aligned_hilic_targets.csv.gz"
    report_path = args.output_dir / "report.json"
    aligned.sort_values(["target_rt_sec", "target_mz", "ik14"]).to_csv(
        aligned_path, index=False, compression="gzip"
    )
    report["provenance"]["aligned_targets_sha256"] = checksum(aligned_path)
    temporary = report_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(report_path)
    print(json.dumps(report, indent=2), flush=True)
    if not report["pass_to_frozen_external_manifest"]:
        raise RuntimeError("ST001154 HILIC readiness gates failed")


if __name__ == "__main__":
    main()
