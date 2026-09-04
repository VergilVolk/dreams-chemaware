#!/usr/bin/env python
"""Audit MetDNA-style MS1 pre-mapping before recursive BioAware scoring.

The audit uses only observed feature m/z, polarity, a frozen source mass index,
and the official MetDNA2 adduct table.  Identity is used solely to measure
candidate recall after enumeration, never to construct a candidate set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def enumerate_candidates(
    observed_mz: float,
    polarity: str,
    maximum_step: int,
    masses: pd.DataFrame,
    adducts: pd.DataFrame,
    ppm: float,
    reported_adduct: str | None,
) -> set[str]:
    compounds = masses[masses["min_reaction_step"].le(maximum_step)]
    if reported_adduct is None:
        choices = adducts[
            adducts["polarity"].eq(polarity) & adducts["default_annotation"]
        ]
    else:
        choices = adducts[
            adducts["polarity"].eq(polarity) & adducts["adduct"].eq(reported_adduct)
        ]
    if choices.empty:
        return set()
    tolerance = observed_mz * ppm * 1e-6
    result: set[str] = set()
    exact_mass = compounds["monoisotopic_mass"].to_numpy(float)
    identities = compounds["inchikey1"].to_numpy(str)
    for row in choices.itertuples(index=False):
        theoretical = (exact_mass * int(row.nmol) + float(row.delta_mz)) / int(row.charge)
        result.update(identities[np.abs(theoretical - observed_mz) <= tolerance])
    return result


class MassCandidateIndex:
    """Sorted mass lookup preserving the earliest eMRN step per entry."""

    def __init__(self, masses: pd.DataFrame, adducts: pd.DataFrame) -> None:
        self.adducts = adducts.copy()
        self.arrays: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        exact_mass = masses["monoisotopic_mass"].to_numpy(float)
        identities = masses["inchikey1"].to_numpy(str)
        steps = masses["min_reaction_step"].to_numpy(np.int8)
        for row in self.adducts.itertuples(index=False):
            theoretical = (
                exact_mass * int(row.nmol) + float(row.delta_mz)
            ) / int(row.charge)
            order = np.argsort(theoretical, kind="stable")
            self.arrays[(str(row.polarity), str(row.adduct))] = (
                theoretical[order], identities[order], steps[order]
            )

    def query(
        self,
        observed_mz: float,
        polarity: str,
        maximum_step: int,
        ppm: float,
        reported_adduct: str | None,
    ) -> set[str]:
        if reported_adduct is None:
            choices = self.adducts[
                self.adducts["polarity"].eq(polarity)
                & self.adducts["default_annotation"]
            ]["adduct"].astype(str)
        else:
            choices = pd.Series([reported_adduct], dtype=str)
        tolerance = observed_mz * ppm * 1e-6
        lower, upper = observed_mz - tolerance, observed_mz + tolerance
        result: set[str] = set()
        for adduct in choices:
            arrays = self.arrays.get((polarity, adduct))
            if arrays is None:
                continue
            theoretical, identities, steps = arrays
            left = int(np.searchsorted(theoretical, lower, side="left"))
            right = int(np.searchsorted(theoretical, upper, side="right"))
            if right <= left:
                continue
            keep = steps[left:right] <= maximum_step
            result.update(identities[left:right][keep])
        return result


def summarise(frame: pd.DataFrame, truth_column: str) -> dict:
    return {
        "queries": int(len(frame)),
        "truth_covered": int(frame[truth_column].sum()),
        "truth_recall": float(frame[truth_column].mean()) if len(frame) else float("nan"),
        "candidate_count_median": float(frame["candidate_count"].median()) if len(frame) else float("nan"),
        "candidate_count_p90": float(frame["candidate_count"].quantile(0.9)) if len(frame) else float("nan"),
        "candidate_count_mean": float(frame["candidate_count"].mean()) if len(frame) else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index-dir", type=Path,
        default=Path("data/reference/metdna2_emrn_mass_adduct_20260828"),
    )
    parser.add_argument(
        "--development-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_development_v1"),
    )
    parser.add_argument(
        "--query-cache", type=Path,
        default=Path("data/validation/bioaware_metdna3_dreams_cache_v2/queries.csv.gz"),
    )
    parser.add_argument(
        "--failure-table", type=Path,
        default=Path("data/validation/bioaware_metdna3_failure_decomposition_v1/per_error_query.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_ms1_premapping_v1"),
    )
    parser.add_argument("--ppm", type=float, default=15.0)
    args = parser.parse_args()
    mass_path = args.index_dir / "emrn_compound_mass.csv.gz"
    adduct_path = args.index_dir / "metdna2_adducts.csv"
    truth_path = args.development_dir / "development_level1.csv.gz"
    for path in (mass_path, adduct_path, truth_path, args.query_cache, args.failure_table):
        if not path.exists():
            raise FileNotFoundError(path)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {output}")

    masses = pd.read_csv(mass_path)
    adducts = pd.read_csv(adduct_path)
    if adducts["default_annotation"].dtype != bool:
        adducts["default_annotation"] = (
            adducts["default_annotation"].astype(str).str.lower().eq("true")
        )
    truth = pd.read_csv(truth_path)
    benchmark = pd.read_csv(args.query_cache)
    failures = pd.read_csv(args.failure_table)
    error_ids = set(failures["query_id"].astype(str))
    required_truth = {"ik14", "mz", "polarity", "adduct"}
    required_query = {"query_id", "truth_ik14", "feature_mz", "polarity", "adduct"}
    if not required_truth.issubset(truth.columns) or not required_query.issubset(benchmark.columns):
        raise RuntimeError("benchmark input schema changed")
    if len(benchmark) != 117 or len(error_ids) != 22:
        raise RuntimeError("expected frozen 117-query benchmark with 22 official errors")
    mass_index = MassCandidateIndex(masses, adducts)

    def audit_rows(frame: pd.DataFrame, source: str) -> pd.DataFrame:
        records: list[dict] = []
        for row in frame.itertuples(index=False):
            if source == "level1":
                query_id = f"level1:{int(row.Index) if hasattr(row, 'Index') else len(records)}"
                truth_ik14 = str(row.ik14)
                observed_mz = float(row.mz)
            else:
                query_id = str(row.query_id)
                truth_ik14 = str(row.truth_ik14)
                observed_mz = float(row.feature_mz)
            for maximum_step in range(9):
                for protocol, fixed_adduct in (
                    ("default_adducts", None), ("reported_adduct", str(row.adduct))
                ):
                    candidates = mass_index.query(
                        observed_mz=observed_mz,
                        polarity=str(row.polarity),
                        maximum_step=maximum_step,
                        ppm=args.ppm,
                        reported_adduct=fixed_adduct,
                    )
                    records.append({
                        "source": source,
                        "query_id": query_id,
                        "truth_ik14": truth_ik14,
                        "polarity": str(row.polarity),
                        "reported_adduct": str(row.adduct),
                        "observed_mz": observed_mz,
                        "maximum_step": maximum_step,
                        "protocol": protocol,
                        "candidate_count": len(candidates),
                        "truth_covered": truth_ik14 in candidates,
                        "official_error": query_id in error_ids,
                    })
        return pd.DataFrame(records)

    # Resetting the index makes the all-Level-1 audit deterministic without using
    # any identity-derived feature identifier.
    level1 = truth.reset_index(drop=True).reset_index(names="Index")
    per_query = pd.concat(
        [audit_rows(level1, "level1"), audit_rows(benchmark, "dreams_benchmark")],
        ignore_index=True,
    )
    per_query_path = output / "per_query.csv.gz"
    per_query.to_csv(per_query_path, index=False, compression="gzip")

    reports: dict[str, dict] = {}
    for (source, protocol, step), group in per_query.groupby(
        ["source", "protocol", "maximum_step"], sort=True
    ):
        key = f"{source}|{protocol}|step{int(step)}"
        item = summarise(group, "truth_covered")
        if source == "dreams_benchmark":
            errors = group[group["official_error"]]
            item["official_errors"] = summarise(errors, "truth_covered")
            item["new_error_truth_coverage_vs_step0"] = int(
                errors["truth_covered"].sum()
                - per_query[
                    per_query["source"].eq(source)
                    & per_query["protocol"].eq(protocol)
                    & per_query["maximum_step"].eq(0)
                    & per_query["official_error"]
                ]["truth_covered"].sum()
            )
        reports[key] = item

    step0 = per_query[
        per_query["source"].eq("dreams_benchmark")
        & per_query["protocol"].eq("default_adducts")
        & per_query["maximum_step"].eq(0)
        & per_query["official_error"]
    ]
    step1 = per_query[
        per_query["source"].eq("dreams_benchmark")
        & per_query["protocol"].eq("default_adducts")
        & per_query["maximum_step"].eq(1)
        & per_query["official_error"]
    ]
    joined = step0[["query_id", "truth_covered"]].merge(
        step1[["query_id", "truth_covered", "candidate_count"]], on="query_id",
        suffixes=("_step0", "_step1"), validate="one_to_one",
    )
    new_error_coverage = int((~joined["truth_covered_step0"] & joined["truth_covered_step1"]).sum())
    report = {
        "status": "bioaware_metdna3_ms1_premapping_audit_complete",
        "formal": True,
        "ppm": args.ppm,
        "level1_rows": int(len(truth)),
        "dreams_queries": int(len(benchmark)),
        "official_errors": int(len(error_ids)),
        "results": reports,
        "primary_gate": {
            "step1_adds_at_least_two_official_error_truths": new_error_coverage >= 2,
            "new_official_error_truths_step1_vs_step0": new_error_coverage,
            "pass_to_recursive_propagation": new_error_coverage >= 2,
        },
        "provenance": {
            "mass_index_sha256": sha256(mass_path),
            "adduct_index_sha256": sha256(adduct_path),
            "truth_sha256": sha256(truth_path),
            "query_cache_sha256": sha256(args.query_cache),
            "failure_table_sha256": sha256(args.failure_table),
            "per_query_sha256": sha256(per_query_path),
        },
        "contracts": {
            "candidate_construction_uses_identity": False,
            "identity_use": "candidate-recall evaluation only",
            "outcome_tuning": False,
            "reported_adduct_protocol": "same-adduct upper-bound audit",
            "default_adduct_protocol": "official MetDNA2 annotation == Yes",
        },
        "claim_limit": (
            "MS1 mass-candidate coverage only. Passing does not establish network ranking, "
            "annotation improvement, or biological validity."
        ),
    }
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
