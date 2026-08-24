"""Targeted classical spectral-library matching for biology candidates.

Only library spectra inside the precursor-mass window are read into memory.
The resulting cosine evidence is an independent, interpretable complement to
DreaMS retrieval; it is not a calibrated FDR or MSI Level 1 identification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyopenms as oms
from matchms import Spectrum
from matchms.similarity import CosineGreedy


def clean_spectrum(mz: np.ndarray, intensity: np.ndarray, precursor_mz: float, top_n: int) -> Spectrum | None:
    mz = np.asarray(mz, dtype=float)
    intensity = np.asarray(intensity, dtype=float)
    keep = np.isfinite(mz) & np.isfinite(intensity) & (mz >= 40.0) & (mz <= precursor_mz + 5.0) & (intensity > 0)
    mz, intensity = mz[keep], intensity[keep]
    if len(mz) < 3:
        return None
    if len(mz) > top_n:
        indices = np.argpartition(intensity, -top_n)[-top_n:]
        mz, intensity = mz[indices], intensity[indices]
    order = np.argsort(mz)
    mz, intensity = mz[order], intensity[order]
    norm = np.linalg.norm(intensity)
    if norm <= 0:
        return None
    return Spectrum(mz=mz, intensities=intensity / norm, metadata={"precursor_mz": precursor_mz})


def parse_targeted_library(path: Path, targets: pd.DataFrame, ppm: float, top_n: int) -> dict[int, list[dict[str, object]]]:
    target_mz = targets.mz.to_numpy(float)
    target_ids = targets.feature_id.to_numpy(int)
    library: dict[int, list[dict[str, object]]] = {int(feature_id): [] for feature_id in target_ids}
    fields: dict[str, str] | None = None
    peaks_mz: list[float] = []
    peaks_intensity: list[float] = []
    eligible: np.ndarray = np.asarray([], dtype=int)
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "BEGIN IONS":
                fields, peaks_mz, peaks_intensity = {}, [], []
                eligible = np.asarray([], dtype=int)
            elif line == "END IONS":
                if fields is None or not len(eligible):
                    fields = None
                    continue
                try:
                    precursor_mz = float(fields["PEPMASS"].split()[0])
                except (KeyError, ValueError, IndexError):
                    fields = None
                    continue
                spectrum = clean_spectrum(np.asarray(peaks_mz), np.asarray(peaks_intensity), precursor_mz, top_n)
                if spectrum is not None:
                    for index in eligible:
                        library[int(target_ids[index])].append(
                            {
                                "spectrum": spectrum,
                                "precursor_mz": precursor_mz,
                                "name": fields.get("NAME", ""),
                                "smiles": fields.get("SMILES", ""),
                                "inchikey": fields.get("INCHIKEY", ""),
                                "adduct": fields.get("ADDUCT", ""),
                                "source": fields.get("SOURCE", ""),
                            }
                        )
                fields = None
            elif fields is not None and "=" in line:
                key, value = line.split("=", 1)
                fields[key.strip().upper()] = value.strip()
                if key.strip().upper() == "PEPMASS":
                    try:
                        precursor_mz = float(value.split()[0])
                        error = np.abs(target_mz - precursor_mz) / target_mz * 1e6
                        eligible = np.flatnonzero(error <= ppm)
                    except (ValueError, IndexError):
                        eligible = np.asarray([], dtype=int)
            elif fields is not None and len(eligible):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        peaks_mz.append(float(parts[0]))
                        peaks_intensity.append(float(parts[1]))
                    except ValueError:
                        pass
    return library


def extract_queries(links: pd.DataFrame, mzml_root: Path, top_n: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (panel, sample), group in links.groupby(["panel", "sample_name"]):
        path = mzml_root / panel / f"{sample}.mzML"
        exp = oms.MSExperiment()
        loader = oms.MzMLFile()
        options = loader.getOptions()
        options.setMSLevels([2])
        loader.setOptions(options)
        loader.load(str(path), exp)
        by_native = {spectrum.getNativeID(): spectrum for spectrum in exp}
        for link in group.itertuples(index=False):
            spectrum = by_native.get(link.native_id)
            if spectrum is None:
                continue
            mz, intensity = spectrum.get_peaks()
            clean = clean_spectrum(mz, intensity, float(link.precursor_mz), top_n)
            if clean is None:
                continue
            rows.append(
                {
                    "feature_id": int(link.feature_id),
                    "sample_name": sample,
                    "native_id": link.native_id,
                    "precursor_mz": float(link.precursor_mz),
                    "spectrum": clean,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="pos_rp")
    parser.add_argument(
        "--coverage",
        type=Path,
        default=Path("data/mtbls13729/biology_candidates/candidate_ms2_coverage.csv"),
    )
    parser.add_argument(
        "--links",
        type=Path,
        default=Path("data/mtbls13729/biology_candidates/candidate_ms2_links.csv.gz"),
    )
    parser.add_argument("--mzml-root", type=Path, default=Path("data/mtbls13729/mzml"))
    parser.add_argument("--library-mgf", type=Path, default=Path("data/reference/unified_v2/unified_pos.mgf"))
    parser.add_argument("--ppm", type=float, default=20.0)
    parser.add_argument("--fragment-tolerance-da", type=float, default=0.02)
    parser.add_argument("--top-n-peaks", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/biology_candidates"))
    args = parser.parse_args()

    coverage = pd.read_csv(args.coverage)
    targets = coverage[(coverage.panel == args.panel) & (coverage.n_ms2_spectra > 0)][["feature_id", "mz"]]
    links = pd.read_csv(args.links)
    links = links[(links.panel == args.panel) & links.feature_id.isin(targets.feature_id)]
    if targets.empty:
        raise SystemExit(f"No MS2-covered candidates for {args.panel}")
    if not args.library_mgf.exists():
        raise FileNotFoundError(args.library_mgf)

    print(f"Reading mass-window library candidates from {args.library_mgf}", flush=True)
    library = parse_targeted_library(args.library_mgf, targets, args.ppm, args.top_n_peaks)
    print(f"Extracting {len(links)} linked query spectra", flush=True)
    queries = extract_queries(links, args.mzml_root, args.top_n_peaks)
    scorer = CosineGreedy(tolerance=args.fragment_tolerance_da)
    hit_rows: list[dict[str, object]] = []
    for query in queries:
        candidates = library.get(query["feature_id"], [])
        scores = []
        for candidate in candidates:
            result = scorer.pair(query["spectrum"], candidate["spectrum"])
            score = float(result["score"].item())
            matches = int(result["matches"].item())
            scores.append((score, matches, candidate))
        scores.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for rank, (score, matches, candidate) in enumerate(scores[: args.top_k], start=1):
            hit_rows.append(
                {
                    "feature_id": query["feature_id"],
                    "sample_name": query["sample_name"],
                    "native_id": query["native_id"],
                    "query_precursor_mz": query["precursor_mz"],
                    "rank": rank,
                    "cosine_greedy": score,
                    "matched_peaks": matches,
                    "library_name": candidate["name"],
                    "library_smiles": candidate["smiles"],
                    "library_inchikey": candidate["inchikey"],
                    "library_adduct": candidate["adduct"],
                    "library_source": candidate["source"],
                    "library_precursor_mz": candidate["precursor_mz"],
                    "ppm_error": abs(query["precursor_mz"] - candidate["precursor_mz"]) / candidate["precursor_mz"] * 1e6,
                    "strong_hit": bool(score >= 0.7 and matches >= 4),
                    "tentative_hit": bool(score >= 0.5 and matches >= 3),
                }
            )

    hits = pd.DataFrame(hit_rows)
    if len(hits):
        # Native spectrum IDs are only unique within an mzML file. Counting
        # them across samples undercounts evidence whenever different files
        # reuse the same native ID. Use the sample-qualified spectrum ID as
        # the experimental evidence unit.
        hits["query_spectrum_id"] = (
            hits["sample_name"].astype(str) + "::" + hits["native_id"].astype(str)
        )
        consensus = (
            hits.groupby(["feature_id", "library_inchikey", "library_name", "library_smiles", "library_adduct", "library_source"], as_index=False)
            .agg(
                n_query_spectra=("query_spectrum_id", "nunique"),
                n_support_samples=("sample_name", "nunique"),
                best_cosine=("cosine_greedy", "max"),
                median_cosine=("cosine_greedy", "median"),
                max_matched_peaks=("matched_peaks", "max"),
                median_matched_peaks=("matched_peaks", "median"),
                median_ppm_error=("ppm_error", "median"),
            )
        )
        strong_support = (
            hits.loc[hits["strong_hit"]]
            .groupby(["feature_id", "library_inchikey"], as_index=False)
            .agg(
                n_strong_query_spectra=("query_spectrum_id", "nunique"),
                n_strong_support_samples=("sample_name", "nunique"),
            )
        )
        tentative_support = (
            hits.loc[hits["tentative_hit"]]
            .groupby(["feature_id", "library_inchikey"], as_index=False)
            .agg(
                n_tentative_query_spectra=("query_spectrum_id", "nunique"),
                n_tentative_support_samples=("sample_name", "nunique"),
            )
        )
        consensus = consensus.merge(strong_support, on=["feature_id", "library_inchikey"], how="left")
        consensus = consensus.merge(tentative_support, on=["feature_id", "library_inchikey"], how="left")
        for column in [
            "n_strong_query_spectra", "n_strong_support_samples",
            "n_tentative_query_spectra", "n_tentative_support_samples",
        ]:
            consensus[column] = consensus[column].fillna(0).astype(int)
        consensus["evidence_tier"] = np.select(
            [
                consensus.n_strong_support_samples >= 2,
                consensus.n_tentative_support_samples >= 1,
            ],
            ["library_supported_level2_candidate", "tentative_level3_candidate"],
            default="mass_only_or_unsupported",
        )
        consensus = consensus.sort_values(
            ["feature_id", "evidence_tier", "n_support_samples", "best_cosine"],
            ascending=[True, True, False, False],
        )
    else:
        consensus = pd.DataFrame()

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    hits_path = out / f"{args.panel}__classical_library_hits.csv.gz"
    consensus_path = out / f"{args.panel}__classical_library_consensus.csv"
    hits.to_csv(hits_path, index=False)
    consensus.to_csv(consensus_path, index=False)
    report = {
        "status": "complete",
        "panel": args.panel,
        "n_target_features": int(len(targets)),
        "n_query_spectra": int(len(queries)),
        "n_features_with_mass_window_library_candidates": int(sum(bool(value) for value in library.values())),
        "n_features_with_library_supported_candidate": (
            int(consensus.loc[consensus.evidence_tier == "library_supported_level2_candidate", "feature_id"].nunique())
            if len(consensus)
            else 0
        ),
        "hits": str(hits_path),
        "consensus": str(consensus_path),
        "interpretation_limit": "CosineGreedy evidence is uncalibrated and cannot establish MSI Level 1 without an authentic standard and RT match.",
    }
    (out / f"{args.panel}__classical_library_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
