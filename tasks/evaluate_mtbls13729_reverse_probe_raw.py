"""Build a phenotype-blind reverse spectral-search tensor for MTBLS13729.

Reference spectra are the probes.  Each probe is searched against every real
MTBLS13729 RPLC sample with a precursor-mass prefilter and auditable raw MS/MS
similarities.  No phenotype is used for matching or threshold selection.

The output is evidence, not an identity claim: a DDA MS2 miss is not absence,
and a library-spectrum match is not MSI Level 1 without same-platform standard
retention time and MS/MS confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks"))

from audit_large_observability_residual import symmetric_features  # noqa: E402


RAW_FEATURES = (
    "sqrt_cosine", "linear_cosine", "entropy_similarity",
    "intensity_coverage_min", "intensity_coverage_mean",
    "matched_peak_fraction_min", "top10_match_fraction",
    "neutral_loss_sqrt_cosine", "neutral_loss_coverage_min",
    "neutral_loss_coverage_mean", "peak_count_ratio",
)


def sha256(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def load_selected_mgf(path: Path, selected_records: set[int]) -> dict[int, np.ndarray]:
    output: dict[int, np.ndarray] = {}
    record = -1
    mz: list[float] | None = None
    intensity: list[float] | None = None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "BEGIN IONS":
                record += 1
                mz, intensity = ([] if record in selected_records else None), ([] if record in selected_records else None)
            elif line == "END IONS":
                if mz is not None and intensity is not None:
                    output[record] = np.asarray([mz, intensity], dtype=np.float32)
                mz = intensity = None
            elif mz is not None and "=" not in line:
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    mass, height = float(parts[0]), float(parts[1])
                except ValueError:
                    continue
                if mass > 0 and height > 0 and np.isfinite(mass) and np.isfinite(height):
                    mz.append(mass)
                    intensity.append(height)
    missing = selected_records - set(output)
    if missing:
        raise RuntimeError(f"selected MGF records not found: {sorted(missing)[:10]}")
    return output


def nearest_target(
    precursor: float, rt_sec: float, target_mz: np.ndarray, target_rt: np.ndarray,
    target_id: np.ndarray, ppm: float, rt_window: float,
) -> tuple[int | None, float, float]:
    if not (np.isfinite(precursor) and np.isfinite(rt_sec) and precursor > 0):
        return None, math.nan, math.nan
    delta = precursor * ppm * 1e-6
    left = int(np.searchsorted(target_mz, precursor - delta, side="left"))
    right = int(np.searchsorted(target_mz, precursor + delta, side="right"))
    if left == right:
        return None, math.nan, math.nan
    dppm = np.abs(target_mz[left:right] - precursor) / precursor * 1e6
    drt = np.abs(target_rt[left:right] - rt_sec)
    eligible = np.flatnonzero(drt <= rt_window)
    if not len(eligible):
        return None, math.nan, math.nan
    cost = (dppm[eligible] / ppm) ** 2 + (drt[eligible] / rt_window) ** 2
    local = int(eligible[np.lexsort((target_id[left:right][eligible], cost))[0]])
    return int(target_id[left + local]), float(dppm[local]), float(drt[local])


def parse_feature_id_set(value: object) -> set[int]:
    if pd.isna(value) or not str(value).strip():
        return set()
    output = set()
    for token in str(value).split(";"):
        try:
            output.add(int(float(token)))
        except ValueError as error:
            raise RuntimeError(f"invalid author feature id token: {token!r}") from error
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-dir", type=Path,
        default=ROOT / "data/mtbls13729/reverse_probe_manifest_v4",
    )
    parser.add_argument("--reference-neg", type=Path, default=ROOT / "data/reference/unified_v2/unified_neg.mgf")
    parser.add_argument("--reference-pos", type=Path, default=ROOT / "data/reference/unified_v2/unified_pos.mgf")
    parser.add_argument("--target-root", type=Path, default=ROOT / "data/mtbls13729/ms1_consensus")
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/mtbls13729/reverse_probe_raw_v3",
    )
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--rt-seconds", type=float, default=20.0)
    parser.add_argument("--peak-tolerance", type=float, default=0.02)
    parser.add_argument("--max-samples-per-panel", type=int, default=0)
    parser.add_argument("--max-reference-spectra-per-panel", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_report_path = args.manifest_dir / "report.json"
    spectra_path = args.manifest_dir / "reference_spectra.csv.gz"
    identities_path = args.manifest_dir / "reference_identities.csv"
    samples_path = args.manifest_dir / "sample_manifest.csv"
    for path in (manifest_report_path, spectra_path, identities_path, samples_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest_report = json.loads(manifest_report_path.read_text(encoding="utf-8"))
    if not manifest_report.get("pass_to_reverse_scan"):
        raise RuntimeError("reference probe manifest did not pass its gates")

    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)

    references = pd.read_csv(spectra_path)
    identities = pd.read_csv(identities_path)
    samples = pd.read_csv(samples_path)
    mgf_paths = {"neg_rp": args.reference_neg, "pos_rp": args.reference_pos}
    selected_spectra: dict[str, dict[int, np.ndarray]] = {}
    for panel, path in mgf_paths.items():
        records = set(references.loc[references.panel.eq(panel), "mgf_record"].astype(int))
        selected_spectra[panel] = load_selected_mgf(path, records)

    all_hits: list[dict[str, object]] = []
    for panel in ("neg_rp", "pos_rp"):
        panel_refs = references[references.panel.eq(panel)].copy()
        panel_samples = samples[samples.panel.eq(panel)].copy()
        if args.max_reference_spectra_per_panel:
            panel_refs = panel_refs.head(args.max_reference_spectra_per_panel)
        if args.max_samples_per_panel:
            panel_samples = panel_samples.head(args.max_samples_per_panel)
        target_path = args.target_root / f"{panel}__requantification_targets.csv.gz"
        if not target_path.is_file():
            raise FileNotFoundError(target_path)
        targets = pd.read_csv(target_path).sort_values(["mz", "rt_sec", "feature_id"], kind="stable")
        tmz = targets.mz.to_numpy(np.float64)
        trt = targets.rt_sec.to_numpy(np.float64)
        tid = targets.feature_id.to_numpy(np.int64)
        for sample_position, sample in enumerate(panel_samples.itertuples(index=False), start=1):
            hdf5_path = Path(str(sample.hdf5_path))
            if not hdf5_path.is_absolute():
                hdf5_path = ROOT / hdf5_path
            if not hdf5_path.is_file():
                raise FileNotFoundError(hdf5_path)
            with h5py.File(hdf5_path, "r") as handle:
                precursor = np.asarray(handle["precursor_mz"], dtype=np.float64)
                rt = np.asarray(handle["RT"], dtype=np.float64)
                scan = np.asarray(handle["scan_number"], dtype=np.int64)
                spectra = handle["spectrum"]
                order = np.argsort(precursor, kind="stable")
                sorted_mass = precursor[order]
                for ref in panel_refs.itertuples(index=False):
                    mass = float(ref.precursor_mz)
                    delta = mass * args.ppm * 1e-6
                    left = int(np.searchsorted(sorted_mass, mass - delta, side="left"))
                    right = int(np.searchsorted(sorted_mass, mass + delta, side="right"))
                    candidate_rows = order[left:right]
                    if not len(candidate_rows):
                        continue
                    ref_spectrum = selected_spectra[panel][int(ref.mgf_record)]
                    best: dict[str, object] | None = None
                    for row in candidate_rows:
                        row = int(row)
                        raw = symmetric_features(
                            ref_spectrum, mass,
                            np.asarray(spectra[row], dtype=np.float32), float(precursor[row]),
                            args.peak_tolerance,
                        )
                        feature_id, dppm, drt = nearest_target(
                            float(precursor[row]), float(rt[row]), tmz, trt, tid,
                            args.ppm, args.rt_seconds,
                        )
                        candidate = {
                            "reference_spectrum_id": ref.reference_spectrum_id,
                            "panel": panel,
                            "ik14": ref.ik14,
                            "reference_name": ref.name,
                            "hypothesis_family": ref.hypothesis_family,
                            "calibration_panel": bool(ref.calibration_panel),
                            "identity_claim_scope": ref.identity_claim_scope,
                            "sample_id": sample.sample_id,
                            "patient_id": sample.patient_id,
                            "side": sample.side,
                            "tissue": sample.tissue,
                            "histology": sample.histology,
                            "query_row": row,
                            "query_scan": int(scan[row]),
                            "query_precursor_mz": float(precursor[row]),
                            "query_rt_sec": float(rt[row]),
                            "precursor_delta_ppm": abs(float(precursor[row]) - mass) / mass * 1e6,
                            "linked_feature_id": feature_id,
                            "feature_delta_ppm": dppm,
                            "feature_delta_rt_sec": drt,
                        } | {name: float(raw[name]) for name in RAW_FEATURES}
                        score_key = (
                            candidate["sqrt_cosine"], candidate["entropy_similarity"],
                            candidate["neutral_loss_sqrt_cosine"], -candidate["precursor_delta_ppm"],
                            -candidate["query_row"],
                        )
                        if best is None or score_key > best["_score_key"]:
                            candidate["_score_key"] = score_key
                            best = candidate
                    assert best is not None
                    best.pop("_score_key")
                    all_hits.append(best)
            print(
                f"[{panel}] samples {sample_position}/{len(panel_samples)}; best spectrum hits={len(all_hits):,}",
                flush=True,
            )

    hits = pd.DataFrame(all_hits)
    if hits.empty:
        raise RuntimeError("no precursor-matched reference/sample spectrum pairs")
    hits_path = out / "reference_spectrum_sample_best_hits.csv.gz"
    hits.to_csv(hits_path, index=False, compression="gzip")

    identity_sample = (
        hits.sort_values(
            ["ik14", "sample_id", "sqrt_cosine", "entropy_similarity", "reference_spectrum_id"],
            ascending=[True, True, False, False, True], kind="stable",
        )
        .groupby(["panel", "ik14", "sample_id"], sort=False, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    expected = identities[[
        "panel", "ik14", "author_level1_feature_ids", "frozen_hypothesis_feature_id"
    ]].copy()
    expected["author_level1_feature_ids"] = expected.author_level1_feature_ids.fillna("").astype(str)
    identity_sample = identity_sample.merge(expected, on=["panel", "ik14"], how="left", validate="many_to_one")
    identity_sample["calibration_feature_match"] = [
        bool(pd.notna(feature) and int(feature) in parse_feature_id_set(ids))
        for feature, ids in zip(identity_sample.linked_feature_id, identity_sample.author_level1_feature_ids)
    ]
    identity_sample["hypothesis_feature_match"] = (
        identity_sample.linked_feature_id.notna()
        & identity_sample.frozen_hypothesis_feature_id.notna()
        & identity_sample.linked_feature_id.eq(identity_sample.frozen_hypothesis_feature_id)
    )
    identity_sample_path = out / "identity_sample_evidence.csv.gz"
    identity_sample.to_csv(identity_sample_path, index=False, compression="gzip")

    calibration = identity_sample[
        identity_sample.calibration_panel & identity_sample.linked_feature_id.notna()
    ]
    hypothesis = identity_sample[identity_sample.hypothesis_family.fillna("").ne("")]
    frozen_hypothesis = hypothesis[hypothesis.frozen_hypothesis_feature_id.notna()]
    report = {
        "status": "mtbls13729_reverse_probe_raw_v1_complete",
        "formal": False,
        "reference_spectra_attempted": int(len(references)),
        "reference_spectra_with_precursor_candidate": int(hits.reference_spectrum_id.nunique()),
        "reference_identities_attempted": int(len(identities)),
        "reference_identities_with_precursor_candidate": int(identity_sample.ik14.nunique()),
        "panel_sample_runs_attempted": int(len(samples)),
        "panel_sample_runs_with_precursor_candidate": int(
            identity_sample[["panel", "sample_id"]].drop_duplicates().shape[0]
        ),
        "precursor_matched_reference_sample_pairs": int(len(identity_sample)),
        "hypothesis_reference_sample_pairs": int(len(hypothesis)),
        "calibration_pairs_with_ms1_feature_link": int(len(calibration)),
        "calibration_feature_matches": int(calibration.calibration_feature_match.sum()),
        "calibration_feature_match_fraction": (
            float(calibration.calibration_feature_match.mean()) if len(calibration) else math.nan
        ),
        "frozen_hypothesis_pairs_with_ms1_feature_link": int(
            frozen_hypothesis.linked_feature_id.notna().sum()
        ),
        "frozen_hypothesis_feature_matches": int(frozen_hypothesis.hypothesis_feature_match.sum()),
        "protocol": {
            "direction": "reference spectrum -> every biological sample",
            "precursor_prefilter_ppm": args.ppm,
            "fragment_tolerance_da": args.peak_tolerance,
            "phenotype_used_for_matching_or_selection": False,
            "sample_aggregation": "maximum raw spectrum match per identity and sample",
            "ms1_bridge": f"nearest frozen requantification target within {args.ppm:g} ppm and {args.rt_seconds:g} seconds",
        },
        "provenance": {
            "manifest_report_sha256": sha256(manifest_report_path),
            "reference_spectra_sha256": sha256(spectra_path),
            "identity_sample_evidence_sha256": sha256(identity_sample_path),
            "spectrum_hits_sha256": sha256(hits_path),
        },
        "claim_limit": (
            "Raw reverse-search engineering pilot only. A DDA miss is not absence; a match is not MSI Level 1; "
            "phenotype association and official-DreaMS/E6/P2b comparison are not evaluated here."
        ),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
