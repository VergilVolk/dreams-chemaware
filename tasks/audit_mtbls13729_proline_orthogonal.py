#!/usr/bin/env python
"""Audit the orthogonal evidence for the MTBLS13729 proline ion family.

The positive-RP feature was selected from the phenotype-aware full-space screen,
so all abundance p-values in this report are descriptive.  Identity evidence is
kept separate: exact mass/adduct, coeluting ion family, recurrent raw MS/MS,
classical library matching, and same-cohort negative-HILIC Level-1 proline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr


FEATURES = {
    345: {"role": "protonated_parent", "adduct": "[M+H]+", "theoretical_mz": 116.070605},
    301: {"role": "putative_sodium_like_coeluting_feature", "adduct": "[M+Na]+ hypothesis", "theoretical_mz": 138.052549},
    719: {"role": "coeluting_product_ion", "adduct": "fragment_or_in_source", "theoretical_mz": 70.065126},
}


def tissue(sample: str) -> str:
    return sample.split("-", maxsplit=1)[1]


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 4 or np.unique(x[ok]).size < 2 or np.unique(y[ok]).size < 2:
        return float("nan")
    return float(spearmanr(x[ok], y[ok]).statistic)


def upper_p(observed: float, null: np.ndarray) -> float:
    if not np.isfinite(observed):
        return float("nan")
    return float((1 + np.sum(null >= observed)) / (1 + len(null)))


def paired_samples(columns: list[str]) -> list[tuple[str, str, str, str]]:
    available = set(columns)
    output: list[tuple[str, str, str, str]] = []
    for number in range(1, 31):
        patient = f"P{number:02d}"
        tumour_label = "Ltu" if number <= 10 else ("Rtu" if number <= 20 else "Rmu")
        normal_label = "LN" if number <= 10 else "RN"
        tumour = f"{patient}-{tumour_label}"
        normal = f"{patient}-{normal_label}"
        if tumour in available and normal in available:
            output.append((patient, tumour_label, tumour, normal))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument(
        "--eic", type=Path,
        default=Path("data/mtbls13729/full_space_eic_v1/pos_rp__eic_auc_matrix.csv.gz"),
    )
    parser.add_argument(
        "--coverage", type=Path,
        default=Path("data/mtbls13729/expanded_ms2_links_v1/candidate_ms2_coverage.csv"),
    )
    parser.add_argument(
        "--links", type=Path,
        default=Path("data/mtbls13729/expanded_ms2_links_v1/candidate_ms2_links.csv.gz"),
    )
    parser.add_argument(
        "--classical", type=Path,
        default=Path("data/mtbls13729/expanded_candidate_classical_v1/pos_rp__classical_library_consensus.csv"),
    )
    parser.add_argument(
        "--maf", type=Path,
        default=Path("dreams-chemaware/_mtbls13729_meta/m_MTBLS13729_LC-MS_negative_hilic_metabolite_profiling_v2_maf.tsv"),
    )
    parser.add_argument("--mzml-dir", type=Path, default=Path("data/mtbls13729/mzml/pos_rp"))
    parser.add_argument("--diagnostic-mz", type=float, default=70.065126)
    parser.add_argument("--fragment-tolerance-da", type=float, default=0.02)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/mtbls13729/proline_orthogonal_audit_v1"),
    )
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    eic = pd.read_csv(args.eic).set_index("feature_id")
    missing = set(FEATURES) - set(eic.index.astype(int))
    if missing:
        raise RuntimeError(f"missing proline-family EIC features: {sorted(missing)}")
    eic = np.log2(eic.loc[list(FEATURES)].astype(float) + 1.0)
    coverage = pd.read_csv(args.coverage).set_index("feature_id")
    classical = pd.read_csv(args.classical)

    maf = pd.read_csv(args.maf, sep="\t")
    source_hits = maf[maf.database_identifier.eq("HMDB0000162")]
    if len(source_hits) != 1:
        raise RuntimeError(f"expected exactly one HMDB0000162 source row; found {len(source_hits)}")
    source = source_hits.iloc[0]
    common = [c for c in eic.columns if c in maf.columns]
    if len(common) < 50:
        raise RuntimeError(f"only {len(common)} common samples")
    source_raw = pd.to_numeric(source[common], errors="coerce").to_numpy(float)
    source_offset = max(1e-9, float(-np.nanmin(source_raw) + 1e-9))
    source_log = np.log2(source_raw + source_offset)
    positive_log = eic.loc[345, common].to_numpy(float)
    labels = np.asarray([tissue(c) for c in common], dtype=object)
    positive_resid = positive_log.copy()
    source_resid = source_log.copy()
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        positive_resid[idx] -= np.nanmean(positive_log[idx])
        source_resid[idx] -= np.nanmean(source_log[idx])
    cross_rho = safe_spearman(positive_log, source_log)
    residual_rho = safe_spearman(positive_resid, source_resid)
    null = np.empty(args.permutations, dtype=float)
    for iteration in range(args.permutations):
        shuffled = source_log.copy()
        for label in np.unique(labels):
            idx = np.flatnonzero(labels == label)
            shuffled[idx] = rng.permutation(shuffled[idx])
        null[iteration] = safe_spearman(positive_log, shuffled)

    pair_rows: list[dict[str, object]] = []
    for patient, subtype, tumour, normal in paired_samples(common):
        pair_rows.append({
            "patient": patient,
            "subtype": subtype,
            "positive_rp_delta": float(eic.loc[345, tumour] - eic.loc[345, normal]),
            "source_neg_hilic_delta": float(
                np.log2(float(source[tumour]) + source_offset)
                - np.log2(float(source[normal]) + source_offset)
            ),
            **{
                f"feature_{fid}_delta": float(eic.loc[fid, tumour] - eic.loc[fid, normal])
                for fid in FEATURES
            },
        })
    pairs = pd.DataFrame(pair_rows)
    pair_rho = safe_spearman(
        pairs.positive_rp_delta.to_numpy(float),
        pairs.source_neg_hilic_delta.to_numpy(float),
    )
    pair_null = np.asarray([
        safe_spearman(
            pairs.positive_rp_delta.to_numpy(float),
            rng.permutation(pairs.source_neg_hilic_delta.to_numpy(float)),
        )
        for _ in range(args.permutations)
    ])

    family_correlations: list[dict[str, object]] = []
    for left, right in ((345, 301), (345, 719), (301, 719)):
        x = eic.loc[left, common].to_numpy(float)
        y = eic.loc[right, common].to_numpy(float)
        family_correlations.append({
            "left": left,
            "right": right,
            "sample_spearman": safe_spearman(x, y),
            "paired_delta_spearman": safe_spearman(
                pairs[f"feature_{left}_delta"].to_numpy(float),
                pairs[f"feature_{right}_delta"].to_numpy(float),
            ),
        })

    effects: list[dict[str, object]] = []
    rmu = pairs[pairs.subtype.eq("Rmu")]
    for fid in FEATURES:
        values = rmu[f"feature_{fid}_delta"].to_numpy(float)
        positive = int(np.sum(values > 0))
        nonzero = int(np.sum(values != 0))
        effects.append({
            "feature_id": fid,
            "role": FEATURES[fid]["role"],
            "n_pairs": int(len(values)),
            "mean_log2fc": float(np.mean(values)),
            "positive_pairs": positive,
            "two_sided_sign_p": float(binomtest(positive, nonzero, 0.5).pvalue) if nonzero else float("nan"),
        })

    # Rank the exact source proline row against every negative-HILIC row by
    # cross-sample correlation.  This detects generic tissue effects and checks
    # whether proline is the best orthogonal identity bridge.
    # Use iterrows because sample names contain dashes and are not valid tuple
    # attributes.
    rank_rows: list[dict[str, object]] = []
    for _, row in maf.iterrows():
        raw = pd.to_numeric(row[common], errors="coerce").to_numpy(float)
        if np.isfinite(raw).sum() < 50:
            continue
        offset = max(1e-9, float(-np.nanmin(raw) + 1e-9))
        rank_rows.append({
            "database_identifier": row.database_identifier,
            "metabolite_identification": row.metabolite_identification,
            "spearman_rho": safe_spearman(positive_log, np.log2(raw + offset)),
        })
    ranks = pd.DataFrame(rank_rows).sort_values("spearman_rho", ascending=False).reset_index(drop=True)
    ranks["rank"] = np.arange(1, len(ranks) + 1)
    proline_rank = int(ranks.loc[ranks.database_identifier.eq("HMDB0000162"), "rank"].iloc[0])

    # Direct diagnostic-fragment recurrence from the linked raw spectra.
    try:
        import pyopenms as oms
    except ImportError as exc:
        raise RuntimeError("pyopenms is required for the raw MS/MS audit") from exc
    links = pd.read_csv(args.links)
    links = links[links.feature_id.eq(345)].copy()
    requested = {
        sample: set(group.native_id.astype(str))
        for sample, group in links.groupby("sample_name")
    }
    fragment_rows: list[dict[str, object]] = []
    for number, (sample, native_ids) in enumerate(sorted(requested.items()), start=1):
        experiment = oms.MSExperiment()
        loader = oms.MzMLFile()
        options = loader.getOptions()
        options.setMSLevels([2])
        loader.setOptions(options)
        loader.load(str(args.mzml_dir / f"{sample}.mzML"), experiment)
        for spectrum in experiment:
            if spectrum.getNativeID() not in native_ids:
                continue
            mz, intensity = spectrum.get_peaks()
            mz = np.asarray(mz, dtype=float)
            intensity = np.asarray(intensity, dtype=float)
            total = float(np.sum(intensity))
            mask = np.abs(mz - args.diagnostic_mz) <= args.fragment_tolerance_da
            fragment_rows.append({
                "sample": sample,
                "native_id": spectrum.getNativeID(),
                "diagnostic_present": bool(mask.any()),
                "diagnostic_relative_intensity": float(np.max(intensity[mask]) / np.max(intensity)) if mask.any() and np.max(intensity) > 0 else 0.0,
                "diagnostic_tic_fraction": float(np.sum(intensity[mask]) / total) if mask.any() and total > 0 else 0.0,
            })
        if number % 10 == 0 or number == len(requested):
            print(f"[proline MS2] {number}/{len(requested)} samples", flush=True)
    fragments = pd.DataFrame(fragment_rows)

    proline_library = classical[
        classical.feature_id.eq(345)
        & classical.library_name.astype(str).str.contains("proline", case=False, na=False)
    ].copy()
    if proline_library.empty:
        raise RuntimeError("no proline library match for feature 345")
    best_library = proline_library.sort_values(
        ["n_strong_support_samples", "median_cosine"], ascending=False
    ).iloc[0]

    feature_rows: list[dict[str, object]] = []
    for fid, spec in FEATURES.items():
        row = coverage.loc[fid]
        feature_rows.append({
            "feature_id": fid,
            **spec,
            "observed_mz": float(row.mz),
            "mass_error_ppm": float((float(row.mz) - spec["theoretical_mz"]) / spec["theoretical_mz"] * 1e6),
            "rt_sec": float(row.rt_sec),
            "n_ms2_spectra": int(row.n_ms2_spectra),
            "n_samples_with_ms2": int(row.n_samples_with_ms2),
            "eic_detection_fraction": float(row.eic_detection_fraction),
        })

    pd.DataFrame(feature_rows).to_csv(output / "proline_ion_family.csv", index=False)
    pd.DataFrame(effects).to_csv(output / "rmu_paired_effects.csv", index=False)
    pd.DataFrame(family_correlations).to_csv(output / "ion_family_correlations.csv", index=False)
    pairs.to_csv(output / "paired_crosspanel_deltas.csv", index=False)
    ranks.to_csv(output / "negative_hilic_identity_ranks.csv", index=False)
    fragments.to_csv(output / "feature345_diagnostic_fragment.csv", index=False)

    payload = {
        "status": "mtbls13729_proline_orthogonal_audit_complete",
        "formal": True,
        "identity_hypothesis": "proline parent with a coeluting product ion; feature 301 remains an ambiguous sodium-like partner",
        "positive_rp_feature": 345,
        "source_identity": {
            "database_identifier": "HMDB0000162",
            "name": str(source.metabolite_identification),
            "source_panel": "negative HILIC",
            "published_level": "Level 1",
            "mz": float(source.mass_to_charge),
            "rt_min": float(source.retention_time),
        },
        "crosspanel": {
            "common_samples": len(common),
            "sample_spearman": cross_rho,
            "within_tissue_spearman": residual_rho,
            "tissue_stratified_permutation_p": upper_p(cross_rho, null),
            "paired_delta_spearman": pair_rho,
            "paired_delta_permutation_p": upper_p(pair_rho, pair_null),
            "source_feature_rank": proline_rank,
            "source_features_rankable": int(len(ranks)),
        },
        "library_ms2": {
            "name": str(best_library.library_name),
            "adduct": str(best_library.library_adduct),
            "query_spectra": int(best_library.n_query_spectra),
            "support_samples": int(best_library.n_support_samples),
            "median_cosine": float(best_library.median_cosine),
            "strong_query_spectra": int(best_library.n_strong_query_spectra),
            "strong_support_samples": int(best_library.n_strong_support_samples),
        },
        "diagnostic_fragment": {
            "mz": args.diagnostic_mz,
            "spectra_audited": int(len(fragments)),
            "spectra_present": int(fragments.diagnostic_present.sum()),
            "samples_present": int(fragments.loc[fragments.diagnostic_present, "sample"].nunique()),
            "median_relative_intensity_when_present": float(
                fragments.loc[fragments.diagnostic_present, "diagnostic_relative_intensity"].median()
            ),
        },
        "ion_family": feature_rows,
        "rmu_effects": effects,
        "family_correlations": family_correlations,
        "claim_limit": (
            "This is same-cohort orthogonal evidence. The source negative-HILIC Level-1 proline "
            "supports molecule identity, while the newly recovered positive-RP ion family is Level 2 "
            "unless retention time is confirmed with an authentic positive-RP standard. Abundance does "
            "not establish mucinous specificity, flux, enzyme activity, or causality."
        ),
    }
    (output / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
