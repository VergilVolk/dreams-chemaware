#!/usr/bin/env python
"""Build an outcome-unopened ST001154 HILIC-negative BioAware manifest."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from pyopenms import MSExperiment, MzMLFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from annotation._inference import preprocess_spectrum

try:
    from audit_st001154_bioaware_external_readiness import (
        analysis_record,
        bool_series,
        checksum,
        read_concatenated_json,
    )
    from audit_st001154_hilic_bioaware_external_readiness import (
        reachable_to_other_seed,
        undirected_edge_graph,
    )
except ModuleNotFoundError:  # imported as tasks.* during tests
    from tasks.audit_st001154_bioaware_external_readiness import (
        analysis_record,
        bool_series,
        checksum,
        read_concatenated_json,
    )
    from tasks.audit_st001154_hilic_bioaware_external_readiness import (
        reachable_to_other_seed,
        undirected_edge_graph,
    )


def exclusive_assignments(
    targets: pd.DataFrame, spectra: pd.DataFrame, ppm: float, rt_seconds: float
) -> pd.DataFrame:
    mz = spectra["observed_precursor_mz"].to_numpy(float)
    rt = spectra["observed_rt_sec"].to_numpy(float)
    options: dict[int, list[int]] = {}
    compatible: dict[int, set[str]] = {}
    for target_index, row in enumerate(targets.itertuples(index=False)):
        ppm_error = np.abs(mz - float(row.target_mz)) / float(row.target_mz) * 1e6
        rt_error = np.abs(rt - float(row.target_rt_sec))
        choices = np.flatnonzero((ppm_error <= ppm) & (rt_error <= rt_seconds)).tolist()
        options[target_index] = choices
        for choice in choices:
            compatible.setdefault(choice, set()).add(str(row.ik14))
    assigned = []
    for target_index, row in enumerate(targets.itertuples(index=False)):
        exclusive = [choice for choice in options[target_index] if compatible[choice] == {str(row.ik14)}]
        if not exclusive:
            continue
        best = min(
            exclusive,
            key=lambda choice: (
                abs(float(spectra.iloc[choice]["observed_precursor_mz"]) - float(row.target_mz))
                / float(row.target_mz)
                * 1e6,
                abs(float(spectra.iloc[choice]["observed_rt_sec"]) - float(row.target_rt_sec)),
                str(spectra.iloc[choice]["native_id"]),
            ),
        )
        record = row._asdict()
        record.update(spectra.iloc[best].to_dict())
        record["compatible_scans"] = len(options[target_index])
        record["exclusive_scans"] = len(exclusive)
        record["match_ppm"] = abs(float(record["observed_precursor_mz"]) - float(row.target_mz)) / float(row.target_mz) * 1e6
        record["match_rt_sec"] = abs(float(record["observed_rt_sec"]) - float(row.target_rt_sec))
        assigned.append(record)
    result = pd.DataFrame(assigned)
    if not result.empty and result["native_id"].duplicated().any():
        raise RuntimeError("exclusive assignment reused an MS2 native ID")
    return result


def load_experiment(path: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict]:
    experiment = MSExperiment()
    MzMLFile().load(str(path), experiment)
    rows = []
    tensors: dict[str, np.ndarray] = {}
    levels: dict[int, int] = {}
    for spectrum in experiment:
        level = int(spectrum.getMSLevel())
        levels[level] = levels.get(level, 0) + 1
        if level != 2 or not spectrum.getPrecursors() or spectrum.size() < 2:
            continue
        native_id = str(spectrum.getNativeID())
        precursor_mz = float(spectrum.getPrecursors()[0].getMZ())
        mz, intensity = spectrum.get_peaks()
        raw = np.vstack([np.asarray(mz, float), np.asarray(intensity, float)])
        tensors[native_id] = preprocess_spectrum(raw, precursor_mz, 100).numpy().astype(np.float32)
        rows.append(
            {
                "native_id": native_id,
                "observed_rt_sec": float(spectrum.getRT()),
                "observed_precursor_mz": precursor_mz,
                "polarity_code": int(spectrum.getInstrumentSettings().getPolarity()),
                "peak_count": int(spectrum.size()),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty or frame["native_id"].duplicated().any():
        raise RuntimeError(f"invalid precursor-bearing MS2 table: {path}")
    if set(frame["polarity_code"].unique()) != {2}:
        raise RuntimeError(f"non-negative MS2 found in {path}")
    return frame, tensors, {str(key): value for key, value in sorted(levels.items())}


def candidate_rows_for_mz(
    approved: pd.DataFrame, target_mz: float, ppm: float
) -> pd.DataFrame:
    tolerance = target_mz * ppm * 1e-6
    return approved.loc[
        approved["precursor_mz"].between(target_mz - tolerance, target_mz + tolerance)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection-dir",
        type=Path,
        default=Path("data/validation/bioaware_st001154_hilic_confirmation_selection_v1"),
    )
    parser.add_argument(
        "--raw-dir", type=Path,
        default=Path("data/reference/ST001154_negative_pilot_20260901"),
    )
    parser.add_argument(
        "--mwtab", type=Path,
        default=Path("data/reference/bioaware_public_cohort_probe_20260901/ST001154__mwtab__json"),
    )
    parser.add_argument(
        "--library-integrity", type=Path,
        default=Path("data/validation/mona_negative_library_chemical_integrity_v1/library_row_integrity.csv.gz"),
    )
    parser.add_argument(
        "--development-units", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_negative_units_v2"),
    )
    parser.add_argument(
        "--emrn-network", type=Path,
        default=Path("data/reference/metdna2_emrn_network_20260828/metdna2_emrn_edges.csv.gz"),
    )
    parser.add_argument("--analysis-id", default="AN001945")
    parser.add_argument("--match-ppm", type=float, default=10.0)
    parser.add_argument("--match-rt-sec", type=float, default=6.0)
    parser.add_argument("--candidate-ppm", type=float, default=10.0)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_st001154_hilic_external_manifest_v1"),
    )
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_dir}")
    selection_report_path = args.selection_dir / "report.json"
    selection_path = args.selection_dir / "samples.csv"
    for path in (
        selection_report_path, selection_path, args.mwtab, args.library_integrity,
        args.emrn_network,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    selection_report = json.loads(selection_report_path.read_text(encoding="utf-8"))
    if selection_report.get("status") != "bioaware_st001154_hilic_confirmation_samples_frozen":
        raise RuntimeError("invalid frozen sample selection")
    if checksum(selection_path) != selection_report["provenance"]["samples_sha256"]:
        raise RuntimeError("frozen sample selection hash mismatch")
    selection = pd.read_csv(selection_path)

    record = analysis_record(read_concatenated_json(args.mwtab), args.analysis_id)
    author = pd.DataFrame(record["MS_METABOLITE_DATA"]["Metabolites"])
    author["ik14"] = author["InChiKey"].fillna("").astype(str).str[:14]
    author["target_mz"] = pd.to_numeric(author["m/z"], errors="coerce")
    author["target_rt_sec"] = pd.to_numeric(author["retention times"], errors="coerce") * 60.0
    author = author.loc[
        author["Adducts"].eq("[M-H]-")
        & author["ik14"].str.len().eq(14)
        & author["target_mz"].notna()
        & author["target_rt_sec"].notna()
    ].copy()
    abundance = pd.DataFrame(record["MS_METABOLITE_DATA"]["Data"])

    library = pd.read_csv(args.library_integrity)
    approved = library.loc[bool_series(library["approved_m_h_reference"])].copy()
    approved["ik14"] = approved["inchikey"].fillna("").astype(str).str[:14]
    approved["precursor_mz"] = pd.to_numeric(approved["precursor_mz"], errors="raise")
    full_formula_rows = approved[["inchikey", "calculated_formula"]].drop_duplicates()
    if full_formula_rows.groupby("inchikey")["calculated_formula"].nunique().max() != 1:
        raise RuntimeError("approved MONA full InChIKeys map to multiple formulas")
    full_formula = dict(full_formula_rows.itertuples(index=False, name=None))
    ik14_formula_rows = approved[["ik14", "calculated_formula"]].drop_duplicates()
    grouped_ik14 = ik14_formula_rows.groupby("ik14")["calculated_formula"].agg(list)
    ik14_formula = {key: values[0] for key, values in grouped_ik14.items() if len(values) == 1}

    development_paths = glob.glob(str(args.development_units / "*" / "queries.csv.gz"))
    if not development_paths:
        raise RuntimeError("frozen BioAware development ledgers are absent")
    development_ids = set(
        pd.concat(
            [pd.read_csv(path, usecols=["truth_ik14"]) for path in development_paths],
            ignore_index=True,
        )["truth_ik14"].astype(str)
    )
    emrn = pd.read_csv(args.emrn_network)
    graph0 = undirected_edge_graph(emrn.loc[pd.to_numeric(emrn["minimum_step"]) == 0])

    query_rows: list[dict] = []
    query_tensors: list[np.ndarray] = []
    seed_rows: list[dict] = []
    seed_tensors: list[np.ndarray] = []
    candidate_rows: list[dict] = []
    scan_reports = []
    source_hashes = {}
    for sample in selection.itertuples(index=False):
        sample_id = str(sample.KOMPLABEL)
        raw_name = str(sample.FileName)
        mzml_path = args.raw_dir / "mzml" / (Path(raw_name).stem + ".mzML")
        raw_path = args.raw_dir / raw_name
        for path in (raw_path, mzml_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        if sample_id not in abundance.columns:
            raise RuntimeError(f"sample has no HILIC abundance column: {sample_id}")
        observed = author.copy()
        intensity = dict(
            zip(
                abundance["Metabolite"].astype(str),
                pd.to_numeric(abundance[sample_id], errors="coerce"),
                strict=True,
            )
        )
        observed["sample_abundance"] = observed["Metabolite"].map(intensity)
        observed = observed.loc[observed["sample_abundance"].notna() & (observed["sample_abundance"] > 0)]
        spectra, tensors, level_counts = load_experiment(mzml_path)
        aligned = exclusive_assignments(observed, spectra, args.match_ppm, args.match_rt_sec)
        seed_ids = set(aligned["ik14"])
        sample_query_count = 0
        sample_network_count = 0
        for seed in aligned.itertuples(index=False):
            seed_rows.append(
                {
                    "sample_id": sample_id,
                    "seed_id": f"{sample_id}|{seed.native_id}",
                    "ik14": str(seed.ik14),
                    "native_id": str(seed.native_id),
                    "target_mz": float(seed.target_mz),
                    "target_rt_sec": float(seed.target_rt_sec),
                    "match_ppm": float(seed.match_ppm),
                    "match_rt_sec": float(seed.match_rt_sec),
                    "tensor_index": len(seed_tensors),
                }
            )
            seed_tensors.append(tensors[str(seed.native_id)])
        for source in aligned.itertuples(index=False):
            truth = str(source.ik14)
            if truth in development_ids:
                continue
            candidates = candidate_rows_for_mz(approved, float(source.target_mz), args.candidate_ppm)
            identities = set(candidates["ik14"])
            if truth not in identities or len(identities) < 2:
                continue
            truth_formula = full_formula.get(str(source.InChiKey), ik14_formula.get(truth))
            if truth_formula is None:
                continue
            query_id = f"ST001154:HILICNEG:{sample_id}:{source.native_id}"
            network_reachable = reachable_to_other_seed(truth, seed_ids, graph0, 2)
            query_rows.append(
                {
                    "query_id": query_id,
                    "sample_id": sample_id,
                    "truth_candidate_id": truth,
                    "truth_full_inchikey": str(source.InChiKey),
                    "truth_formula": truth_formula,
                    "target_name": str(source.Metabolite),
                    "feature_mz": float(source.target_mz),
                    "feature_rt_sec": float(source.target_rt_sec),
                    "observed_precursor_mz": float(source.observed_precursor_mz),
                    "observed_rt_sec": float(source.observed_rt_sec),
                    "spectrum_id": str(source.native_id),
                    "candidate_identities": int(len(identities)),
                    "same_formula_candidate_identities": int(
                        candidates.loc[candidates["calculated_formula"].eq(truth_formula), "ik14"].nunique()
                    ),
                    "emrn_seed_reachable_depth2": bool(network_reachable),
                    "query_tensor_index": len(query_tensors),
                }
            )
            query_tensors.append(tensors[str(source.native_id)])
            for candidate in candidates.itertuples(index=False):
                candidate_rows.append(
                    {
                        "query_id": query_id,
                        "candidate_id": str(candidate.ik14),
                        "library_row": int(candidate.library_row),
                        "library_precursor_mz": float(candidate.precursor_mz),
                    }
                )
            sample_query_count += 1
            sample_network_count += int(network_reachable)
        scan_reports.append(
            {
                "sample_id": sample_id,
                "spectra_by_ms_level": level_counts,
                "exclusive_aligned_seed_identities": int(aligned["ik14"].nunique()),
                "evaluable_queries": sample_query_count,
                "emrn_depth2_reachable_queries": sample_network_count,
            }
        )
        source_hashes[sample_id] = {
            "raw_sha256": checksum(raw_path),
            "mzml_sha256": checksum(mzml_path),
        }
        print(
            f"[manifest] {sample_id}: seeds={len(seed_ids)} queries={sample_query_count} "
            f"network={sample_network_count}",
            flush=True,
        )

    queries = pd.DataFrame(query_rows)
    seeds = pd.DataFrame(seed_rows)
    candidates = pd.DataFrame(candidate_rows).drop_duplicates(
        ["query_id", "candidate_id", "library_row"]
    )
    if queries.empty or seeds.empty or candidates.empty:
        raise RuntimeError("external manifest is empty")
    if queries["query_id"].duplicated().any() or seeds["seed_id"].duplicated().any():
        raise RuntimeError("external manifest identifiers are not unique")
    candidate_identity_counts = candidates.groupby("query_id")["candidate_id"].nunique()
    if (candidate_identity_counts < 2).any():
        raise RuntimeError("external query has fewer than two candidate identities")
    if not all(
        truth in set(candidates.loc[candidates["query_id"].eq(query_id), "candidate_id"])
        for query_id, truth in queries[["query_id", "truth_candidate_id"]].itertuples(index=False)
    ):
        raise RuntimeError("external candidate group omitted its truth identity")

    args.output_dir.mkdir(parents=True)
    paths = {
        "queries": args.output_dir / "queries.csv.gz",
        "candidates": args.output_dir / "candidate_references.csv.gz",
        "seeds": args.output_dir / "seed_features.csv.gz",
        "query_tensors": args.output_dir / "query_tensors.npz",
        "seed_tensors": args.output_dir / "seed_tensors.npz",
    }
    queries.to_csv(paths["queries"], index=False, compression="gzip")
    candidates.to_csv(paths["candidates"], index=False, compression="gzip")
    seeds.to_csv(paths["seeds"], index=False, compression="gzip")
    np.savez_compressed(paths["query_tensors"], query_tensor=np.stack(query_tensors))
    np.savez_compressed(paths["seed_tensors"], seed_tensor=np.stack(seed_tensors))
    gates = {
        "samples_eq_8": queries["sample_id"].nunique() == 8,
        "queries_ge_100": len(queries) >= 100,
        "truth_identities_ge_20": queries["truth_candidate_id"].nunique() >= 20,
        "same_formula_hard_queries_ge_40": int((queries["same_formula_candidate_identities"] >= 2).sum()) >= 40,
        "emrn_depth2_reachable_queries_ge_25": int(queries["emrn_seed_reachable_depth2"].sum()) >= 25,
        "development_identity_overlap_eq_0": not queries["truth_candidate_id"].isin(development_ids).any(),
    }
    report = {
        "status": "bioaware_st001154_hilic_external_manifest_complete",
        "formal": True,
        "outcome_status": "unopened; no DreaMS or BioAware scores computed",
        "samples": int(queries["sample_id"].nunique()),
        "queries": int(len(queries)),
        "truth_identities": int(queries["truth_candidate_id"].nunique()),
        "truth_formulas": int(queries["truth_formula"].nunique()),
        "same_formula_hard_queries": int((queries["same_formula_candidate_identities"] >= 2).sum()),
        "emrn_depth2_reachable_queries": int(queries["emrn_seed_reachable_depth2"].sum()),
        "seed_features": int(len(seeds)),
        "candidate_reference_rows": int(len(candidates)),
        "sample_scan_reports": scan_reports,
        "gates": gates,
        "pass_to_frozen_evaluation": all(gates.values()),
        "contracts": {
            "sample_selection_phenotype_blind": True,
            "technical_reinjections_excluded": True,
            "one_ms2_one_truth_identity": True,
            "candidate_generation": f"approved [M-H]- MONA rows within {args.candidate_ppm:g} ppm",
            "query_truth_removed_from_network_seeds_at_evaluation": True,
            "development_truth_identity_overlap": 0,
            "P2b": "forbidden",
        },
        "provenance": {
            "selection_report_sha256": checksum(selection_report_path),
            "selection_samples_sha256": checksum(selection_path),
            "mwtab_sha256": checksum(args.mwtab),
            "library_integrity_sha256": checksum(args.library_integrity),
            "emrn_network_sha256": checksum(args.emrn_network),
            "source_files": source_hashes,
            **{f"{name}_sha256": checksum(path) for name, path in paths.items()},
            "script_sha256": checksum(Path(__file__)),
        },
        "claim_limit": "Frozen external execution manifest only; no embedding, rank, correction, or SOTA result.",
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not report["pass_to_frozen_evaluation"]:
        raise RuntimeError("ST001154 HILIC external manifest gates failed")


if __name__ == "__main__":
    main()
