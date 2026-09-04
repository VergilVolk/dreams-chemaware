"""Phenotype-blind screen of all pos-RP MS2 spectra for acylcarnitines.

The scan first applies literature-anchored carnitine fragment motifs, then
matches precursor masses to an enumerated Cn:u acylcarnitine series.  Phenotype
labels are not used for discovery.  The resulting RT-resolved species are
mapped back to the all-feature MS1 matrix for downstream paired statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyopenms as oms


C = 12.0
H = 1.00782503223
N = 14.00307400443
O = 15.99491461957
PROTON = 1.007276466621
SODIUM = 22.989218
FRAGMENTS = (60.0808, 85.0284, 144.1019)
NEUTRAL_LOSS = 59.0735


def theoretical_series(min_carbon: int, max_carbon: int, max_unsaturation: int) -> pd.DataFrame:
    rows = []
    for carbon in range(min_carbon, max_carbon + 1):
        for unsaturation in range(0, min(max_unsaturation, carbon - 1) + 1):
            # Carnitine C7H15NO3 + fatty acid CnH(2n-2u)O2 - H2O.
            neutral = (carbon + 7) * C + (2 * carbon - 2 * unsaturation + 13) * H + N + 4 * O
            rows.extend([
                {
                    "acyl_chain": f"C{carbon}:{unsaturation}",
                    "carbon": carbon,
                    "unsaturation": unsaturation,
                    "adduct": "[M+H]+",
                    "theoretical_mz": neutral + PROTON,
                },
                {
                    "acyl_chain": f"C{carbon}:{unsaturation}",
                    "carbon": carbon,
                    "unsaturation": unsaturation,
                    "adduct": "[M+Na]+",
                    "theoretical_mz": neutral + SODIUM,
                },
            ])
    return pd.DataFrame(rows)


def has_peak(mz: np.ndarray, intensity: np.ndarray, target: float, tolerance: float, min_relative: float) -> bool:
    eligible = np.flatnonzero(np.abs(mz - target) <= tolerance)
    if not len(eligible) or np.max(intensity) <= 0:
        return False
    return bool(np.max(intensity[eligible]) / np.max(intensity) >= min_relative)


def motif_count(mz: np.ndarray, intensity: np.ndarray, precursor_mz: float, tolerance: float, min_relative: float) -> tuple[int, bool]:
    present = [has_peak(mz, intensity, target, tolerance, min_relative) for target in FRAGMENTS]
    present.append(has_peak(mz, intensity, precursor_mz - NEUTRAL_LOSS, tolerance, min_relative))
    count = int(sum(present))
    supported = bool(count >= 3 and present[1] and (present[0] or present[2]))
    return count, supported


def cluster_rt(frame: pd.DataFrame, gap_sec: float) -> pd.DataFrame:
    output = []
    for (chain, adduct), group in frame.groupby(["acyl_chain", "adduct"]):
        group = group.sort_values("rt_sec").copy()
        cluster_id = (group.rt_sec.diff().fillna(np.inf) > gap_sec).cumsum()
        group["rt_cluster"] = cluster_id.to_numpy(int)
        output.append(group)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mzml-dir", type=Path, default=Path("data/mtbls13729/mzml/pos_rp"))
    p.add_argument("--consensus", type=Path, default=Path("data/mtbls13729/ms1_consensus/pos_rp__consensus_metadata.csv.gz"))
    p.add_argument("--paired-dir", type=Path, default=Path("data/mtbls13729/ms1_paired_analysis_peakresolved"))
    p.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/acylcarnitine_panel"))
    p.add_argument("--ppm", type=float, default=10.0)
    p.add_argument("--fragment-tolerance-da", type=float, default=0.02)
    p.add_argument("--min-relative-intensity", type=float, default=0.005)
    p.add_argument("--rt-cluster-gap-sec", type=float, default=12.0)
    p.add_argument("--map-rt-sec", type=float, default=12.0)
    p.add_argument("--min-carbon", type=int, default=2)
    p.add_argument("--max-carbon", type=int, default=26)
    p.add_argument("--max-unsaturation", type=int, default=8)
    args = p.parse_args()

    theory = theoretical_series(args.min_carbon, args.max_carbon, args.max_unsaturation)
    theory_mz = theory.theoretical_mz.to_numpy(float)
    scan_rows = []
    for path in sorted(args.mzml_dir.glob("*.mzML")):
        exp = oms.MSExperiment()
        loader = oms.MzMLFile()
        options = loader.getOptions()
        options.setMSLevels([2])
        loader.setOptions(options)
        try:
            loader.load(str(path), exp)
        except Exception as exc:
            print(f"{path.stem}: failed {exc!r}", flush=True)
            continue
        n_hits = 0
        for spectrum in exp:
            precursors = spectrum.getPrecursors()
            if not precursors:
                continue
            precursor_mz = float(precursors[0].getMZ())
            mz, intensity = spectrum.get_peaks()
            mz = np.asarray(mz, dtype=float)
            intensity = np.asarray(intensity, dtype=float)
            count, supported = motif_count(
                mz, intensity, precursor_mz,
                args.fragment_tolerance_da, args.min_relative_intensity,
            )
            if not supported:
                continue
            ppm = np.abs(theory_mz - precursor_mz) / theory_mz * 1e6
            best = int(np.argmin(ppm))
            if ppm[best] > args.ppm:
                continue
            assignment = theory.iloc[best]
            scan_rows.append({
                "sample_name": path.stem,
                "native_id": spectrum.getNativeID(),
                "rt_sec": float(spectrum.getRT()),
                "precursor_mz": precursor_mz,
                "motif_count": count,
                "acyl_chain": assignment.acyl_chain,
                "carbon": int(assignment.carbon),
                "unsaturation": int(assignment.unsaturation),
                "adduct": assignment.adduct,
                "theoretical_mz": float(assignment.theoretical_mz),
                "ppm_error": float((precursor_mz - assignment.theoretical_mz) / assignment.theoretical_mz * 1e6),
            })
            n_hits += 1
        print(f"{path.stem}: {n_hits} acylcarnitine-motif scans", flush=True)

    scans = pd.DataFrame(scan_rows)
    clustered = cluster_rt(scans, args.rt_cluster_gap_sec)
    if clustered.empty:
        raise SystemExit("No acylcarnitine motif spectra found")
    species = (
        clustered.groupby(["acyl_chain", "carbon", "unsaturation", "adduct", "rt_cluster"], as_index=False)
        .agg(
            median_precursor_mz=("precursor_mz", "median"),
            median_rt_sec=("rt_sec", "median"),
            n_ms2_spectra=("native_id", "nunique"),
            n_samples_with_ms2=("sample_name", "nunique"),
            best_motif_count=("motif_count", "max"),
            median_ppm_error=("ppm_error", "median"),
        )
    )
    species["species_id"] = np.arange(1, len(species) + 1)

    consensus = pd.read_csv(args.consensus)
    mappings = []
    for row in species.itertuples(index=False):
        ppm = np.abs(consensus.mz - row.median_precursor_mz) / row.median_precursor_mz * 1e6
        rt_delta = np.abs(consensus.rt_sec - row.median_rt_sec)
        eligible = (ppm <= args.ppm) & (rt_delta <= args.map_rt_sec)
        if not eligible.any():
            mappings.append({"species_id": row.species_id, "feature_id": np.nan, "map_ppm": np.nan, "map_rt_delta_sec": np.nan})
            continue
        score = np.hypot(ppm / args.ppm, rt_delta / args.map_rt_sec).where(eligible, np.inf)
        index = score.idxmin()
        mappings.append({
            "species_id": row.species_id,
            "feature_id": int(consensus.loc[index, "feature_id"]),
            "map_ppm": float(ppm.loc[index]),
            "map_rt_delta_sec": float(rt_delta.loc[index]),
        })
    species = species.merge(pd.DataFrame(mappings), on="species_id", how="left")

    for variant in ("log_raw", "pqn", "pqn_pair_drift"):
        stats_path = args.paired_dir / f"pos_rp__{variant}__paired_stats.csv.gz"
        if not stats_path.exists():
            print(f"[acylcarnitine] optional paired statistics absent: {stats_path}", flush=True)
            continue
        stats = pd.read_csv(stats_path)
        keep = [
            "feature_id", "rmu_vs_rn_n", "rmu_vs_rn_mean_log2fc", "rmu_vs_rn_ttest_p",
            "rmu_vs_rn_wilcoxon_p", "rmu_vs_rn_loo_sign_stability",
            "rtu_vs_rn_mean_log2fc", "interaction_log2fc", "interaction_p",
        ]
        stats = stats[keep].rename(columns={col: f"{variant}__{col}" for col in keep if col != "feature_id"})
        species = species.merge(stats, on="feature_id", how="left")

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    scans_path = out / "acylcarnitine_motif_scans.csv.gz"
    species_path = out / "acylcarnitine_rt_species.csv"
    scans.to_csv(scans_path, index=False)
    species.to_csv(species_path, index=False)
    report = {
        "status": "complete",
        "n_motif_scans": int(len(scans)),
        "n_samples_with_motif": int(scans.sample_name.nunique()),
        "n_rt_resolved_species": int(len(species)),
        "n_species_in_2plus_samples": int((species.n_samples_with_ms2 >= 2).sum()),
        "n_species_mapped_to_ms1": int(species.feature_id.notna().sum()),
        "n_mapped_species_in_2plus_samples": int(((species.n_samples_with_ms2 >= 2) & species.feature_id.notna()).sum()),
        "scans": str(scans_path),
        "species": str(species_path),
        "interpretation_limit": "Cn:u and adduct assignments remain class-level; RT isomers and double-bond positions are unresolved.",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
