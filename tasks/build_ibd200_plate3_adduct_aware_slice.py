#!/usr/bin/env python3
"""Rebuild the IBD Plate3 candidate axis from explicit neutral masses.

The v1 application slice matched query precursor m/z directly to a protonated
reference library.  This replacement keeps the same 662-query denominator,
sample folds, context and sealed effects, but constructs candidates from the
phenotype-blind FBMN neutral mass when a best-ion assignment exists.  Queries
without an explicit neutral mass receive no candidate and therefore abstain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def neutral_candidate_tables(
    queries: pd.DataFrame,
    library_manifest: Path,
    neutral_ppm: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = [
        "row", "inchikey", "smiles", "formula", "exact_mass", "precursor_mz",
        "quality_label", "n_peaks", "spectrum_id",
    ]
    library = pd.read_csv(library_manifest, usecols=columns)
    library = library.loc[
        library["inchikey"].notna()
        & library["smiles"].notna()
        & np.isfinite(pd.to_numeric(library["exact_mass"], errors="coerce"))
    ].copy()
    library["exact_mass"] = library["exact_mass"].astype(float)
    order = np.argsort(library["exact_mass"].to_numpy(), kind="mergesort")
    sorted_mass = library["exact_mass"].to_numpy()[order]
    spectrum_rows: list[pd.DataFrame] = []
    for query in queries.itertuples(index=False):
        neutral_mass = float(query.neutral_mass)
        if not np.isfinite(neutral_mass) or neutral_mass <= 0:
            continue
        delta = neutral_mass * neutral_ppm * 1e-6
        lo = np.searchsorted(sorted_mass, neutral_mass - delta, side="left")
        hi = np.searchsorted(sorted_mass, neutral_mass + delta, side="right")
        if lo == hi:
            continue
        selected = library.iloc[order[lo:hi]].copy()
        selected.insert(0, "entity_id", str(query.entity_id))
        selected.insert(1, "query_precursor_mz", float(query.precursor_mz))
        selected.insert(2, "query_best_ion", str(query.best_ion))
        selected.insert(3, "query_neutral_mass", neutral_mass)
        selected["neutral_mass_ppm"] = (
            1e6 * (selected["exact_mass"] - neutral_mass).abs() / neutral_mass
        )
        selected["candidate_id"] = selected["inchikey"].astype(str)
        spectrum_rows.append(selected)
    if spectrum_rows:
        spectra = pd.concat(spectrum_rows, ignore_index=True)
    else:
        spectra = pd.DataFrame(columns=[
            "entity_id", "candidate_id", "query_precursor_mz", "query_best_ion",
            "query_neutral_mass", "neutral_mass_ppm", *columns,
        ])
    structures = (
        spectra.sort_values(
            ["entity_id", "candidate_id", "neutral_mass_ppm", "quality_label", "row"],
            ascending=[True, True, True, True, True], kind="mergesort",
        )
        .groupby(["entity_id", "candidate_id"], sort=False, as_index=False)
        .agg(
            smiles=("smiles", "first"),
            formula=("formula", "first"),
            exact_mass=("exact_mass", "first"),
            query_precursor_mz=("query_precursor_mz", "first"),
            query_best_ion=("query_best_ion", "first"),
            query_neutral_mass=("query_neutral_mass", "first"),
            minimum_neutral_mass_ppm=("neutral_mass_ppm", "min"),
            library_spectra=("row", "size"),
        )
    )
    structures["cross_adduct_reference"] = ~structures["query_best_ion"].eq("[M+H]+")
    return spectra, structures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-slice", required=True, type=Path)
    parser.add_argument("--gnps-quant", required=True, type=Path)
    parser.add_argument("--library-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--neutral-ppm", type=float, default=10.0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    queries_path = args.parent_slice / "query_entities.tsv.gz"
    parent_report_path = args.parent_slice / "report.json"
    passthrough_names = [
        "phenotype_blind_context_folds.tsv",
        "phenotype_blind_sample_context.npz",
        "degree_preserving_wrong_context.tsv.gz",
        "sealed_heldout_effects.tsv.gz",
    ]
    inputs = [queries_path, parent_report_path, args.gnps_quant, args.library_manifest]
    inputs.extend(args.parent_slice / name for name in passthrough_names)
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    queries = pd.read_csv(queries_path, sep="\t", dtype={"raw_entity_id": str})
    quant = pd.read_csv(args.gnps_quant, low_memory=False)
    quant["raw_entity_id"] = quant["row ID"].astype(str)
    ion = quant[[
        "raw_entity_id", "row retention time", "best ion", "neutral M mass",
        "correlation group ID", "annotation network number",
    ]].rename(columns={
        "row retention time": "retention_time",
        "best ion": "best_ion",
        "neutral M mass": "neutral_mass",
        "correlation group ID": "correlation_group_id",
        "annotation network number": "annotation_network_number",
    })
    queries = queries.merge(ion, on="raw_entity_id", how="left", validate="one_to_one")
    queries["adduct_annotation_status"] = np.where(
        queries["best_ion"].notna() & queries["neutral_mass"].notna(),
        "explicit_fbm_ion_and_neutral_mass", "adduct_unresolved",
    )
    spectra, structures = neutral_candidate_tables(queries, args.library_manifest, args.neutral_ppm)
    structure_counts = structures.groupby("entity_id").size()
    spectrum_counts = spectra.groupby("entity_id").size()
    queries["candidate_structures"] = queries["entity_id"].map(structure_counts).fillna(0).astype(int)
    queries["candidate_library_spectra"] = queries["entity_id"].map(spectrum_counts).fillna(0).astype(int)
    queries["algorithm_answerable"] = (
        queries["spectral_eligible"].astype(bool) & queries["candidate_structures"].gt(0)
    )

    args.output_dir.mkdir(parents=True)
    compression = {"method": "gzip", "mtime": 0}
    output_queries = args.output_dir / "query_entities.tsv.gz"
    output_spectra = args.output_dir / "candidate_library_spectra.tsv.gz"
    output_structures = args.output_dir / "candidate_structures.tsv.gz"
    queries.to_csv(output_queries, sep="\t", index=False, compression=compression)
    spectra.to_csv(output_spectra, sep="\t", index=False, compression=compression)
    structures.to_csv(output_structures, sep="\t", index=False, compression=compression)
    passthrough_sha: dict[str, str] = {}
    for name in passthrough_names:
        source = args.parent_slice / name
        target = args.output_dir / name
        shutil.copy2(source, target)
        if sha256(source) != sha256(target):
            raise RuntimeError(f"passthrough copy drifted: {name}")
        passthrough_sha[name] = sha256(target)

    explicit = queries["adduct_annotation_status"].eq("explicit_fbm_ion_and_neutral_mass")
    cross_adduct_queries = set(
        structures.loc[structures["cross_adduct_reference"], "entity_id"].astype(str)
    )
    report = {
        "schema": "ibd200_plate3_adduct_aware_algorithm_application_slice_v1",
        "status": "PHENOTYPE_BLIND_ADDUCT_AWARE_APPLICATION_SLICE_COMPLETE",
        "data_role": "development_consumed",
        "denominators": {
            "queries": int(len(queries)),
            "queries_with_explicit_ion_and_neutral_mass": int(explicit.sum()),
            "queries_with_unresolved_adduct": int((~explicit).sum()),
            "queries_with_candidates": int(queries["candidate_structures"].gt(0).sum()),
            "spectral_and_candidate_answerable": int(queries["algorithm_answerable"].sum()),
            "candidate_structures": int(len(structures)),
            "candidate_library_spectrum_edges": int(len(spectra)),
            "candidate_structures_cross_adduct": int(structures["cross_adduct_reference"].sum()),
            "queries_with_cross_adduct_candidates": int(len(cross_adduct_queries)),
        },
        "candidate_contract": {
            "mass_axis": "candidate exact neutral mass within 10 ppm of explicit FBMN neutral mass",
            "reference_spectra": "frozen GNPS Gold/Silver positive protonated library",
            "cross_adduct_task": "queries whose best ion is not [M+H]+ are scored against the same protonated reference candidates",
            "unresolved_adduct": "zero candidates and explicit abstention",
            "query_denominator": "all 662 corrected Plate3 response entities",
        },
        "leakage_status": (
            "candidate construction read only feature identifiers, precursor/retention/ion annotations, "
            "representative spectral eligibility and the frozen library; phenotype and effect files were not read"
        ),
        "provenance": {
            "parent_query_entities_sha256": sha256(queries_path),
            "parent_report_sha256": sha256(parent_report_path),
            "GNPS_quant_sha256": sha256(args.gnps_quant),
            "library_manifest_sha256": sha256(args.library_manifest),
            "query_entities_sha256": sha256(output_queries),
            "candidate_spectra_sha256": sha256(output_spectra),
            "candidate_structures_sha256": sha256(output_structures),
            "passthrough_sha256": passthrough_sha,
            "implementation_sha256": sha256(Path(__file__)),
        },
        "claim_boundary": [
            "FBMN best-ion and neutral-mass assignments are phenotype-blind hypotheses, not definitive adduct identities.",
            "Cross-adduct spectrum scoring is a harder real-application task and must be reported separately from [M+H]+ scoring.",
            "This slice creates a fair candidate axis but does not establish any structure identification or biological gain.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
