#!/usr/bin/env python
"""Build label-free recurrent-fragment summaries for frozen biology candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def cluster_fragments(rows: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    ordered = rows.sort_values("fragment_mz").reset_index(drop=True)
    clusters: list[list[int]] = []
    current: list[int] = []
    center = np.nan
    for index, row in ordered.iterrows():
        mz = float(row.fragment_mz)
        if not current or mz - center <= tolerance:
            current.append(index)
            center = float(ordered.loc[current, "fragment_mz"].median())
        else:
            clusters.append(current)
            current = [index]
            center = mz
    if current:
        clusters.append(current)

    spectra_total = int(rows["spectrum_key"].nunique())
    samples_total = int(rows["sample"].nunique())
    output: list[dict[str, object]] = []
    for cluster_id, indices in enumerate(clusters):
        cluster = ordered.loc[indices].copy()
        cluster = cluster.sort_values("relative_intensity", ascending=False).drop_duplicates(
            "spectrum_key"
        )
        support_spectra = int(cluster.spectrum_key.nunique())
        support_samples = int(cluster["sample"].nunique())
        output.append(
            {
                "cluster_id": cluster_id,
                "fragment_mz": float(cluster.fragment_mz.median()),
                "support_spectra": support_spectra,
                "support_fraction": float(support_spectra / spectra_total),
                "support_samples": support_samples,
                "sample_fraction": float(support_samples / samples_total),
                "median_relative_intensity": float(cluster.relative_intensity.median()),
                "max_relative_intensity": float(cluster.relative_intensity.max()),
            }
        )
    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matches",
        type=Path,
        default=Path(
            "data/mtbls13729/frozen_candidate_ms2_coverage_v1/"
            "candidate_ms2_matches.csv.gz"
        ),
    )
    parser.add_argument(
        "--mzml-dir", type=Path, default=Path("data/mtbls13729/mzml/pos_rp")
    )
    parser.add_argument("--minimum-relative-intensity", type=float, default=0.005)
    parser.add_argument("--cluster-tolerance-da", type=float, default=0.02)
    parser.add_argument("--minimum-support-fraction", type=float, default=0.10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/mtbls13729/frozen_candidate_ms2_consensus_v1"),
    )
    args = parser.parse_args()

    try:
        import pyopenms as oms
    except ImportError as exc:
        raise RuntimeError("pyopenms is required to read mzML") from exc

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    matches = pd.read_csv(args.matches)
    matches = matches[matches.peak_resolved_match.astype(bool)].copy()
    if matches.empty:
        raise RuntimeError("no peak-resolved candidate MS2 matches")
    requested: dict[str, list[dict[str, object]]] = {}
    for row in matches.itertuples(index=False):
        requested.setdefault(str(row.sample), []).append(
            {
                "feature_id": int(row.feature_id),
                "native_id": str(row.native_id),
                "precursor_mz": float(row.precursor_mz),
            }
        )

    fragments: list[dict[str, object]] = []
    failed: list[dict[str, str]] = []
    for number, (sample, targets) in enumerate(sorted(requested.items()), start=1):
        path = args.mzml_dir / f"{sample}.mzML"
        experiment = oms.MSExperiment()
        loader = oms.MzMLFile()
        options = loader.getOptions()
        options.setMSLevels([2])
        loader.setOptions(options)
        try:
            loader.load(str(path), experiment)
        except Exception as exc:
            failed.append({"sample": sample, "error": repr(exc)})
            continue
        by_native = {spectrum.getNativeID(): spectrum for spectrum in experiment}
        for target in targets:
            spectrum = by_native.get(str(target["native_id"]))
            if spectrum is None:
                failed.append(
                    {
                        "sample": sample,
                        "error": f"native id absent: {target['native_id']}",
                    }
                )
                continue
            mz, intensity = spectrum.get_peaks()
            mz = np.asarray(mz, dtype=float)
            intensity = np.asarray(intensity, dtype=float)
            if not len(mz) or not np.isfinite(intensity).any() or np.nanmax(intensity) <= 0:
                continue
            relative = intensity / np.nanmax(intensity)
            keep = (
                (relative >= args.minimum_relative_intensity)
                & (mz < float(target["precursor_mz"]) - 1.0)
            )
            spectrum_key = f"{sample}|{target['native_id']}"
            for fragment_mz, relative_intensity in zip(mz[keep], relative[keep]):
                fragments.append(
                    {
                        "feature_id": int(target["feature_id"]),
                        "sample": sample,
                        "spectrum_key": spectrum_key,
                        "fragment_mz": float(fragment_mz),
                        "relative_intensity": float(relative_intensity),
                    }
                )
        if number % 10 == 0 or number == len(requested):
            print(f"[consensus MS2] {number}/{len(requested)} samples", flush=True)

    fragment_table = pd.DataFrame(fragments)
    if fragment_table.empty:
        raise RuntimeError("no fragments survived the relative-intensity filter")
    outputs: list[pd.DataFrame] = []
    summary: list[dict[str, object]] = []
    for feature_id, group in fragment_table.groupby("feature_id"):
        consensus = cluster_fragments(group, args.cluster_tolerance_da)
        consensus.insert(0, "feature_id", int(feature_id))
        consensus = consensus[
            consensus.support_fraction >= args.minimum_support_fraction
        ].sort_values(
            ["support_fraction", "median_relative_intensity"], ascending=False
        )
        consensus["support_rank"] = np.arange(1, len(consensus) + 1)
        outputs.append(consensus)
        top = consensus.head(10)
        summary.append(
            {
                "feature_id": int(feature_id),
                "spectra": int(group.spectrum_key.nunique()),
                "samples": int(group["sample"].nunique()),
                "recurrent_clusters": int(len(consensus)),
                "top_fragments": [
                    {
                        "mz": float(row.fragment_mz),
                        "support_fraction": float(row.support_fraction),
                        "median_relative_intensity": float(
                            row.median_relative_intensity
                        ),
                    }
                    for row in top.itertuples(index=False)
                ],
            }
        )
    consensus_all = pd.concat(outputs, ignore_index=True)
    consensus_all.to_csv(output / "candidate_recurrent_fragments.csv", index=False)
    pd.DataFrame(failed).to_csv(output / "failed_spectra.csv", index=False)
    payload = {
        "status": "mtbls13729_frozen_candidate_ms2_consensus_complete",
        "formal": True,
        "features": int(fragment_table.feature_id.nunique()),
        "spectra": int(fragment_table.spectrum_key.nunique()),
        "samples": int(fragment_table["sample"].nunique()),
        "failed_items": int(len(failed)),
        "parameters": {
            "minimum_relative_intensity": args.minimum_relative_intensity,
            "cluster_tolerance_da": args.cluster_tolerance_da,
            "minimum_support_fraction": args.minimum_support_fraction,
        },
        "summary": summary,
        "claim_limit": (
            "Recurrent fragments describe reproducible ion evidence. Structural assignment "
            "requires comparison to an authentic experimental spectrum acquired under a "
            "compatible method; recurrence alone is not an identity label."
        ),
    }
    (output / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
