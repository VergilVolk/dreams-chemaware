"""Characterize peak-level mechanisms for large DreaMS residual pair cohorts.

Continuous fragment and neutral-loss evidence is computed for every residual
pair. Screening thresholds are fitted on discovery only and then frozen before
application to confirmation. The resulting mechanism labels are hypotheses for
targeted occlusion, not chemical ground truth and not training labels by
themselves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from audit_e0_observability_residual import greedy_matches, peaks


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ATLAS = ROOT / "data/validation/dreams_structure_residual_atlas_large"
DEFAULT_DATA = ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5"
DEFAULT_DISCOVERY_MANIFEST = (
    ROOT / "data/validation/large_observability_embeddings_discovery/manifest.csv"
)
DEFAULT_CONFIRMATION_MANIFEST = (
    ROOT / "data/validation/large_observability_embeddings_confirmation/manifest.csv"
)
DEFAULT_OUTPUT = ROOT / "data/validation/dreams_residual_peak_mechanisms_large"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dir", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--discovery-manifest", type=Path, default=DEFAULT_DISCOVERY_MANIFEST)
    parser.add_argument("--confirmation-manifest", type=Path, default=DEFAULT_CONFIRMATION_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--neutral-loss-tolerance", type=float, default=0.02)
    parser.add_argument("--top-peaks-to-store", type=int, default=20)
    return parser.parse_args()


def read_spectra(handle: h5py.File, hdf5_rows: np.ndarray) -> dict[int, np.ndarray]:
    unique = np.unique(hdf5_rows.astype(np.int64))
    order = np.argsort(unique)
    loaded = np.asarray(handle["spectrum"][unique[order]])
    return {int(row): loaded[pos] for pos, row in enumerate(unique[order])}


def normalized(spectrum: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mz, intensity = peaks(spectrum)
    total = max(float(intensity.sum()), 1e-12)
    base = max(float(intensity.max()), 1e-12) if len(intensity) else 1.0
    return mz, intensity / total, intensity / base


def match_space(
    mz_a: np.ndarray,
    int_a: np.ndarray,
    mz_b: np.ndarray,
    int_b: np.ndarray,
    tolerance: float,
) -> dict[str, object]:
    matches = greedy_matches(mz_a, mz_b, tolerance) if len(mz_a) and len(mz_b) else []
    idx_a = {i for i, _ in matches}
    idx_b = {j for _, j in matches}
    unique_a = np.asarray(sorted(set(range(len(mz_a))) - idx_a), dtype=np.int64)
    unique_b = np.asarray(sorted(set(range(len(mz_b))) - idx_b), dtype=np.int64)
    matched_a = np.asarray(sorted(idx_a), dtype=np.int64)
    matched_b = np.asarray(sorted(idx_b), dtype=np.int64)
    sqrt_cosine = float(sum(np.sqrt(int_a[i] * int_b[j]) for i, j in matches))
    linear_norm_a = max(float(np.linalg.norm(int_a)), 1e-12)
    linear_norm_b = max(float(np.linalg.norm(int_b)), 1e-12)
    linear_cosine = float(sum(int_a[i] * int_b[j] for i, j in matches) / (linear_norm_a * linear_norm_b))
    top_a = set(np.argsort(int_a)[-min(10, len(int_a)):])
    top_b = set(np.argsort(int_b)[-min(10, len(int_b)):])
    top_matches = sum(i in top_a and j in top_b for i, j in matches)
    return {
        "match_count": int(len(matches)),
        "match_fraction_min": float(len(matches) / max(1, min(len(mz_a), len(mz_b)))),
        "shared_intensity_a": float(int_a[matched_a].sum()) if len(matched_a) else 0.0,
        "shared_intensity_b": float(int_b[matched_b].sum()) if len(matched_b) else 0.0,
        "unique_intensity_a": float(int_a[unique_a].sum()) if len(unique_a) else 0.0,
        "unique_intensity_b": float(int_b[unique_b].sum()) if len(unique_b) else 0.0,
        "sqrt_cosine": sqrt_cosine,
        "linear_cosine": linear_cosine,
        "top10_match_fraction": float(top_matches / max(1, min(10, len(mz_a), len(mz_b)))),
        "matches": matches,
        "unique_a": unique_a,
        "unique_b": unique_b,
    }


def top_peak_text(mz: np.ndarray, intensity: np.ndarray, indices: np.ndarray, cap: int) -> tuple[str, str]:
    if not len(indices):
        return "", ""
    ranked = indices[np.argsort(intensity[indices])[::-1]][:cap]
    return (
        ";".join(f"{float(mz[idx]):.5f}" for idx in ranked),
        ";".join(f"{float(intensity[idx]):.6g}" for idx in ranked),
    )


def pair_features(
    spectrum_a: np.ndarray,
    precursor_a: float,
    spectrum_b: np.ndarray,
    precursor_b: float,
    fragment_tolerance: float,
    neutral_loss_tolerance: float,
    peak_cap: int,
) -> dict[str, object]:
    mz_a, int_a, rel_a = normalized(spectrum_a)
    mz_b, int_b, rel_b = normalized(spectrum_b)
    fragment = match_space(mz_a, int_a, mz_b, int_b, fragment_tolerance)

    keep_a = precursor_a - mz_a > 0
    keep_b = precursor_b - mz_b > 0
    loss_a = precursor_a - mz_a[keep_a]
    loss_b = precursor_b - mz_b[keep_b]
    loss_int_a = int_a[keep_a]
    loss_int_b = int_b[keep_b]
    order_a = np.argsort(loss_a)
    order_b = np.argsort(loss_b)
    neutral = match_space(
        loss_a[order_a], loss_int_a[order_a],
        loss_b[order_b], loss_int_b[order_b], neutral_loss_tolerance,
    )

    unique_a_mz, unique_a_int = top_peak_text(
        mz_a, int_a, np.asarray(fragment["unique_a"], dtype=np.int64), peak_cap
    )
    unique_b_mz, unique_b_int = top_peak_text(
        mz_b, int_b, np.asarray(fragment["unique_b"], dtype=np.int64), peak_cap
    )
    shared_pairs = fragment["matches"]
    if shared_pairs:
        shared_indices = np.asarray([i for i, _ in shared_pairs], dtype=np.int64)
        shared_mz, shared_int = top_peak_text(mz_a, int_a, shared_indices, peak_cap)
    else:
        shared_mz, shared_int = "", ""

    min_shared = min(float(fragment["shared_intensity_a"]), float(fragment["shared_intensity_b"]))
    min_neutral_shared = min(float(neutral["shared_intensity_a"]), float(neutral["shared_intensity_b"]))
    return {
        "peak_count_a": int(len(mz_a)),
        "peak_count_b": int(len(mz_b)),
        "strong_peak_count_a": int(np.sum(rel_a >= 0.10)),
        "strong_peak_count_b": int(np.sum(rel_b >= 0.10)),
        "low_information_either": bool(np.sum(rel_a >= 0.10) < 3 or np.sum(rel_b >= 0.10) < 3),
        "fragment_match_count": fragment["match_count"],
        "fragment_match_fraction_min": fragment["match_fraction_min"],
        "fragment_shared_intensity_a": fragment["shared_intensity_a"],
        "fragment_shared_intensity_b": fragment["shared_intensity_b"],
        "fragment_min_shared_intensity": min_shared,
        "fragment_unique_intensity_a": fragment["unique_intensity_a"],
        "fragment_unique_intensity_b": fragment["unique_intensity_b"],
        "fragment_sqrt_cosine": fragment["sqrt_cosine"],
        "fragment_linear_cosine": fragment["linear_cosine"],
        "fragment_top10_match_fraction": fragment["top10_match_fraction"],
        "neutral_loss_match_count": neutral["match_count"],
        "neutral_loss_min_shared_intensity": min_neutral_shared,
        "neutral_loss_sqrt_cosine": neutral["sqrt_cosine"],
        "neutral_loss_top10_match_fraction": neutral["top10_match_fraction"],
        "shared_peak_mz_top": shared_mz,
        "shared_peak_intensity_top": shared_int,
        "unique_peak_mz_a_top": unique_a_mz,
        "unique_peak_intensity_a_top": unique_a_int,
        "unique_peak_mz_b_top": unique_b_mz,
        "unique_peak_intensity_b_top": unique_b_int,
    }


def characterize(
    priority_path: Path,
    manifest_path: Path,
    handle: h5py.File,
    fragment_tolerance: float,
    neutral_loss_tolerance: float,
    peak_cap: int,
) -> pd.DataFrame:
    priority = pd.read_csv(priority_path)
    manifest = pd.read_csv(manifest_path)
    row_a = priority["row_a"].to_numpy(np.int64)
    row_b = priority["row_b"].to_numpy(np.int64)
    hdf_a = manifest.iloc[row_a]["hdf5_row"].to_numpy(np.int64)
    hdf_b = manifest.iloc[row_b]["hdf5_row"].to_numpy(np.int64)
    spectra = read_spectra(handle, np.concatenate([hdf_a, hdf_b]))
    output = []
    for position, row in enumerate(priority.itertuples(index=False)):
        features = pair_features(
            spectra[int(hdf_a[position])], float(manifest.iloc[int(row.row_a)]["precursor_mz"]),
            spectra[int(hdf_b[position])], float(manifest.iloc[int(row.row_b)]["precursor_mz"]),
            fragment_tolerance, neutral_loss_tolerance, peak_cap,
        )
        output.append(row._asdict() | {
            "hdf5_row_a": int(hdf_a[position]),
            "hdf5_row_b": int(hdf_b[position]),
        } | features)
    return pd.DataFrame(output)


def fit_thresholds(discovery: pd.DataFrame) -> dict[str, float]:
    different = discovery["pair_type"].eq("different_identity")
    identity = discovery["pair_type"].eq("same_identity")
    return {
        "fragment_shared_q25": float(discovery.loc[different, "fragment_min_shared_intensity"].quantile(0.25)),
        "fragment_shared_q75": float(discovery.loc[different, "fragment_min_shared_intensity"].quantile(0.75)),
        "top10_match_q75": float(discovery.loc[different, "fragment_top10_match_fraction"].quantile(0.75)),
        "neutral_shared_q75": float(discovery.loc[different, "neutral_loss_min_shared_intensity"].quantile(0.75)),
        "identity_fragment_shared_q25": float(discovery.loc[identity, "fragment_min_shared_intensity"].quantile(0.25)),
        "identity_ce_delta_q75": float(discovery.loc[identity, "ce_delta"].dropna().quantile(0.75)),
    }


def assign_mechanism(frame: pd.DataFrame, thresholds: dict[str, float]) -> pd.Series:
    labels = []
    for row in frame.itertuples(index=False):
        family = row.residual_family_official_finetuned
        low_info = bool(row.low_information_either)
        if family == "same_identity_instability":
            if low_info:
                label = "low_information_identity_instability"
            elif not bool(row.same_instrument):
                label = "cross_instrument_identity_instability"
            elif np.isfinite(row.ce_delta) and row.ce_delta >= thresholds["identity_ce_delta_q75"]:
                label = "large_ce_shift_identity_instability"
            elif row.fragment_min_shared_intensity <= thresholds["identity_fragment_shared_q25"]:
                label = "fragmentation_divergence_identity_instability"
            else:
                label = "unresolved_identity_instability"
        elif family == "higher_than_structure_expected":
            if low_info:
                label = "spectrum_limited_overaggregation"
            elif (
                row.fragment_min_shared_intensity >= thresholds["fragment_shared_q75"]
                and row.fragment_top10_match_fraction >= thresholds["top10_match_q75"]
            ):
                label = "shared_major_peak_overaggregation_candidate"
            elif row.neutral_loss_min_shared_intensity >= thresholds["neutral_shared_q75"]:
                label = "neutral_loss_convergence_candidate"
            else:
                label = "unresolved_overaggregation"
        elif family == "lower_than_structure_expected":
            if low_info:
                label = "spectrum_limited_separation"
            elif row.fragment_min_shared_intensity <= thresholds["fragment_shared_q25"]:
                label = "fragmentation_divergence_candidate"
            elif not bool(row.same_instrument):
                label = "cross_instrument_separation_candidate"
            else:
                label = "unresolved_separation"
        else:
            label = "outside_primary_official_residual_family"
        labels.append(label)
    return pd.Series(labels, index=frame.index, dtype="object")


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "pairs": int(len(frame)),
        "molecules": int(len(set(frame["ik_a"]) | set(frame["ik_b"]))),
        "same_formula_pairs": int(frame["same_formula"].fillna(False).sum()),
        "low_information_pairs": int(frame["low_information_either"].sum()),
        "mechanism_counts": {str(k): int(v) for k, v in frame["mechanism_screen"].value_counts().items()},
        "median_fragment_min_shared_intensity": float(frame["fragment_min_shared_intensity"].median()),
        "median_fragment_top10_match_fraction": float(frame["fragment_top10_match_fraction"].median()),
        "median_neutral_loss_min_shared_intensity": float(frame["neutral_loss_min_shared_intensity"].median()),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.data, "r") as handle:
        discovery = characterize(
            args.atlas_dir / "discovery_residual_priority_cases.csv",
            args.discovery_manifest, handle,
            args.fragment_tolerance, args.neutral_loss_tolerance,
            args.top_peaks_to_store,
        )
        confirmation = characterize(
            args.atlas_dir / "confirmation_residual_priority_cases.csv",
            args.confirmation_manifest, handle,
            args.fragment_tolerance, args.neutral_loss_tolerance,
            args.top_peaks_to_store,
        )
    thresholds = fit_thresholds(discovery)
    discovery["mechanism_screen"] = assign_mechanism(discovery, thresholds)
    confirmation["mechanism_screen"] = assign_mechanism(confirmation, thresholds)
    discovery["mechanism_claim_status"] = "screening_hypothesis_requires_targeted_occlusion"
    confirmation["mechanism_claim_status"] = "locked_confirmation_screening_only"
    discovery.to_csv(args.output_dir / "discovery_peak_mechanisms.csv", index=False)
    confirmation.to_csv(args.output_dir / "confirmation_peak_mechanisms.csv", index=False)
    report = {
        "status": "large_residual_peak_mechanism_screen",
        "fragment_tolerance_da": args.fragment_tolerance,
        "neutral_loss_tolerance_da": args.neutral_loss_tolerance,
        "thresholds_fitted_on_discovery_only": thresholds,
        "discovery": summarize(discovery),
        "confirmation": summarize(confirmation),
        "claim_boundary": (
            "Mechanism labels are descriptive screening hypotheses. Causal claims require "
            "targeted peak masking against count/intensity/mz-matched random controls."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
