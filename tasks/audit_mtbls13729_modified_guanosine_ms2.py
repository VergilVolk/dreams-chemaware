"""Audit raw MS2 support for the two modified-guanosine ion families.

This is deliberately an identity-boundary audit.  It links DDA spectra only
inside each sample's resolved EIC peak and reports aglycone/neutral-loss and
cross-adduct consistency without assigning a positional isomer.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyopenms as oms


RIBOSE_LOSS = 132.042259
NA_MINUS_H = 21.9819442498


def matched_peak(mz: np.ndarray, intensity: np.ndarray, target: float, tol: float) -> tuple[bool, float, float]:
    idx = np.flatnonzero(np.abs(mz - target) <= tol)
    if not len(idx) or not len(intensity) or float(np.max(intensity)) <= 0:
        return False, np.nan, 0.0
    best = int(idx[np.argmax(intensity[idx])])
    return True, float(mz[best]), float(intensity[best] / np.max(intensity))


def binned_consensus(values: list[tuple[float, float]], tol: float) -> pd.DataFrame:
    if not values:
        return pd.DataFrame(columns=["mz", "mean_relative_intensity", "support_spectra"])
    values = sorted(values)
    clusters: list[list[tuple[float, float]]] = []
    for item in values:
        if not clusters or abs(item[0] - np.mean([x[0] for x in clusters[-1]])) > tol:
            clusters.append([item])
        else:
            clusters[-1].append(item)
    return pd.DataFrame(
        [
            {
                "mz": float(np.average([x[0] for x in c], weights=np.maximum([x[1] for x in c], 1e-12))),
                "mean_relative_intensity": float(np.mean([x[1] for x in c])),
                "support_spectra": int(len(c)),
            }
            for c in clusters
        ]
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ledger", type=Path, default=Path("data/mtbls13729/biology_closure_family_targets_v1/biology_candidate_ledger.csv"))
    p.add_argument("--links", type=Path, default=Path("data/mtbls13729/modified_guanosine_ms2_audit_v1/candidate_ms2_links.csv.gz"))
    p.add_argument("--mzml-root", type=Path, default=Path("data/mtbls13729/mzml/pos_rp"))
    p.add_argument("--fragment-tolerance-da", type=float, default=0.02)
    p.add_argument("--minimum-relative-intensity", type=float, default=0.005)
    p.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/modified_guanosine_ms2_audit_v1"))
    args = p.parse_args()

    target_ids = [1597, 7489, 3019, 8481]
    ledger = pd.read_csv(args.ledger)
    targets = ledger[ledger.feature_id.isin(target_ids)].set_index("feature_id")
    if set(target_ids) != set(targets.index.astype(int)):
        raise RuntimeError("the four frozen target features are not all present")
    links = pd.read_csv(args.links)
    links = links[links.feature_id.isin(target_ids)].copy()
    if links.empty:
        raise RuntimeError("no peak-resolved MS2 links")

    rows: list[dict[str, object]] = []
    consensus_fragments: dict[int, list[tuple[float, float]]] = defaultdict(list)
    consensus_losses: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for sample, group in links.groupby("sample_name"):
        path = args.mzml_root / f"{sample}.mzML"
        if not path.exists():
            raise FileNotFoundError(path)
        exp = oms.MSExperiment()
        loader = oms.MzMLFile()
        options = loader.getOptions()
        options.setMSLevels([2])
        loader.setOptions(options)
        loader.load(str(path), exp)
        spectra = {s.getNativeID(): s for s in exp}
        for link in group.itertuples(index=False):
            spectrum = spectra.get(link.native_id)
            if spectrum is None:
                raise RuntimeError(f"native id missing: {sample} {link.native_id}")
            mz, intensity = spectrum.get_peaks()
            mz = np.asarray(mz, float)
            intensity = np.asarray(intensity, float)
            if not len(mz) or float(np.max(intensity)) <= 0:
                continue
            rel = intensity / np.max(intensity)
            keep = rel >= args.minimum_relative_intensity
            mz_keep, rel_keep = mz[keep], rel[keep]
            precursor = float(link.precursor_mz)
            fid = int(link.feature_id)
            expected_aglycone = precursor - RIBOSE_LOSS
            present, observed, relative = matched_peak(
                mz, intensity, expected_aglycone, args.fragment_tolerance_da
            )
            for value, weight in zip(mz_keep, rel_keep, strict=True):
                consensus_fragments[fid].append((float(value), float(weight)))
                loss = precursor - float(value)
                if loss > 0:
                    consensus_losses[fid].append((loss, float(weight)))
            rows.append(
                {
                    "feature_id": fid,
                    "family": str(targets.loc[fid, "biology_label"]),
                    "sample_name": sample,
                    "native_id": link.native_id,
                    "precursor_mz": precursor,
                    "ms2_rt_sec": float(link.ms2_rt_sec),
                    "rt_error_sec": float(link.rt_error_sec),
                    "ppm_error": float(link.ppm_error),
                    "fragment_peaks": int(len(mz)),
                    "aglycone_target_mz": expected_aglycone,
                    "ribose_loss_132_supported": bool(present and relative >= args.minimum_relative_intensity),
                    "aglycone_observed_mz": observed,
                    "aglycone_relative_intensity": relative,
                }
            )

    detail = pd.DataFrame(rows)
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "modified_guanosine_ms2_details.csv.gz", index=False)
    summaries: list[dict[str, object]] = []
    consensus_paths: dict[str, str] = {}
    for fid in target_ids:
        sub = detail[detail.feature_id == fid]
        frag = binned_consensus(consensus_fragments[fid], args.fragment_tolerance_da)
        loss = binned_consensus(consensus_losses[fid], args.fragment_tolerance_da)
        if len(sub):
            frag["support_fraction"] = frag.support_spectra / len(sub)
            loss["support_fraction"] = loss.support_spectra / len(sub)
        frag = frag.sort_values(["support_spectra", "mean_relative_intensity"], ascending=False)
        loss = loss.sort_values(["support_spectra", "mean_relative_intensity"], ascending=False)
        frag_path = out / f"feature_{fid}_fragment_consensus.csv"
        loss_path = out / f"feature_{fid}_neutral_loss_consensus.csv"
        frag.to_csv(frag_path, index=False)
        loss.to_csv(loss_path, index=False)
        consensus_paths[str(fid)] = str(frag_path)
        summaries.append(
            {
                "feature_id": fid,
                "family": str(targets.loc[fid, "biology_label"]),
                "n_ms2_spectra": int(len(sub)),
                "n_samples": int(sub.sample_name.nunique()) if len(sub) else 0,
                "median_abs_rt_error_sec": float(np.median(np.abs(sub.rt_error_sec))) if len(sub) else np.nan,
                "median_ppm_error": float(np.median(sub.ppm_error)) if len(sub) else np.nan,
                "ribose_loss_support_spectra": int(sub.ribose_loss_132_supported.sum()) if len(sub) else 0,
                "ribose_loss_support_fraction": float(sub.ribose_loss_132_supported.mean()) if len(sub) else np.nan,
                "median_aglycone_relative_intensity": float(np.median(sub.aglycone_relative_intensity)) if len(sub) else np.nan,
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "modified_guanosine_ms2_summary.csv", index=False)

    pair_rows = []
    for name, protonated, sodium in [("methylguanosine", 1597, 7489), ("dimethylguanosine", 3019, 8481)]:
        a = binned_consensus(consensus_losses[protonated], args.fragment_tolerance_da)
        b = binned_consensus(consensus_losses[sodium], args.fragment_tolerance_da)
        a = a[a.support_spectra >= max(2, int(0.10 * max(1, len(detail[detail.feature_id == protonated]))))]
        b = b[b.support_spectra >= max(2, int(0.10 * max(1, len(detail[detail.feature_id == sodium]))))]
        matches = 0
        for value in a.mz.to_numpy(float):
            if len(b) and float(np.min(np.abs(b.mz.to_numpy(float) - value))) <= args.fragment_tolerance_da:
                matches += 1
        pair_rows.append(
            {
                "family": name,
                "protonated_feature": protonated,
                "sodium_feature": sodium,
                "observed_precursor_difference": float(targets.loc[sodium, "mz"] - targets.loc[protonated, "mz"]),
                "na_minus_h_residual_da": float(targets.loc[sodium, "mz"] - targets.loc[protonated, "mz"] - NA_MINUS_H),
                "recurrent_neutral_losses_protonated": int(len(a)),
                "recurrent_neutral_losses_sodium": int(len(b)),
                "matched_recurrent_neutral_losses": int(matches),
            }
        )
    family = pd.DataFrame(pair_rows)
    family.to_csv(out / "modified_guanosine_cross_adduct_ms2_consistency.csv", index=False)
    report = {
        "status": "mtbls13729_modified_guanosine_ms2_audit_complete",
        "formal": False,
        "summary": summaries,
        "cross_adduct_consistency": pair_rows,
        "conflicting_legacy_hit": {
            "feature_id": 8481,
            "name": "ADRENALINE BITARTRATE",
            "cosine": 0.5355,
            "interpretation": "low-confidence conflicting library hit; not accepted as identity evidence",
        },
        "claim_limit": "Peak-resolved raw MS2 and cross-adduct neutral-loss consistency can support an ion family or nucleoside-like class, not a positional isomer or MSI Level 1 identity.",
        "outputs": {
            "detail": str(out / "modified_guanosine_ms2_details.csv.gz"),
            "summary": str(out / "modified_guanosine_ms2_summary.csv"),
            "cross_adduct": str(out / "modified_guanosine_cross_adduct_ms2_consistency.csv"),
            "consensus_fragments": consensus_paths,
        },
    }
    (out / "modified_guanosine_ms2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
