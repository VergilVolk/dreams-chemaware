#!/usr/bin/env python
"""Outcome-blind OpenMS pilot for one MetDNA3 sample/separation unit.

The detector is selected only by technical-replicate reproducibility.  Level-1
identities and DreaMS outcomes are neither loaded nor used in this stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from extract_mtbls13729_ms1_features import detect_features, load_ms1
from pilot_mtbls13729_openms_features import greedy_match


FILE_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<polarity>pos|neg)_"
    r"(?P<window>70_300|70_1200|290_600|590_1200)_"
    r"(?P<replicate>[12])\.mzML$"
)


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


def extract_one_file(
    path_text: str,
    feature_dir_text: str,
    thresholds: list[float],
    mass_error_ppm: float,
    min_trace_sec: float,
    max_trace_sec: float,
) -> list[dict]:
    """Process-isolated extraction so OpenMS can use multiple files concurrently."""
    path = Path(path_text)
    feature_dir = Path(feature_dir_text)
    match = FILE_RE.match(path.name)
    if match is None:
        raise RuntimeError(f"unexpected development file: {path.name}")
    polarity = match.group("polarity")
    window = match.group("window")
    replicate = match.group("replicate")
    experiment, load_summary = load_ms1(path)
    rows: list[dict] = []
    for threshold in thresholds:
        features, detection = detect_features(
            experiment,
            noise_threshold=threshold,
            mass_error_ppm=mass_error_ppm,
            min_trace_length=min_trace_sec,
            max_trace_length=max_trace_sec,
        )
        features.insert(0, "source_file", path.name)
        features.insert(1, "polarity", "positive" if polarity == "pos" else "negative")
        features.insert(2, "mass_window", window)
        features.insert(3, "replicate", int(replicate))
        target = feature_dir / f"{path.stem}__noise_{threshold:g}.csv.gz"
        features.to_csv(target, index=False, compression="gzip")
        rows.append({
            "source_file": path.name,
            "polarity": "positive" if polarity == "pos" else "negative",
            "mass_window": window,
            "replicate": int(replicate),
            "noise_threshold": threshold,
            **load_summary,
            **detection,
            "feature_sha256": sha256(target),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mzml-dir", type=Path,
        default=Path("data/external/metdna3_2025/mzml/development"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_ms1_feature_pilot_v1"),
    )
    parser.add_argument("--noise-thresholds", nargs="+", type=float, default=[1e3, 1e4, 1e5])
    parser.add_argument("--mass-error-ppm", type=float, default=5.0)
    parser.add_argument("--min-trace-sec", type=float, default=5.0)
    parser.add_argument("--max-trace-sec", type=float, default=120.0)
    parser.add_argument("--match-ppm", type=float, default=15.0)
    parser.add_argument("--match-rt-sec", type=float, default=25.0)
    parser.add_argument("--minimum-median-features", type=int, default=500)
    parser.add_argument("--maximum-median-features", type=int, default=20000)
    parser.add_argument("--minimum-median-feature-jaccard", type=float, default=0.50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--expected-files", type=int, choices=(15, 16), default=16)
    parser.add_argument(
        "--frozen-noise-threshold", type=float, default=None,
        help="Use one threshold frozen before this panel; no within-panel threshold selection.",
    )
    args = parser.parse_args()
    files = sorted(path for path in args.mzml_dir.glob("*.mzML") if FILE_RE.match(path.name))
    if len(files) != args.expected_files:
        raise RuntimeError(f"expected {args.expected_files} frozen targeted mzML files, got {len(files)}")
    by_role: dict[tuple[str, str, int], Path] = {}
    for path in files:
        match = FILE_RE.match(path.name)
        assert match is not None
        key = (match.group("polarity"), match.group("window"), int(match.group("replicate")))
        if key in by_role:
            raise RuntimeError(f"duplicate acquisition role {key}: {path.name}")
        by_role[key] = path
    thresholds = (
        [float(args.frozen_noise_threshold)] if args.frozen_noise_threshold is not None
        else sorted(set(float(value) for value in args.noise_thresholds))
    )
    if any(value <= 0 for value in thresholds):
        raise ValueError("noise thresholds must be positive")
    if args.frozen_noise_threshold is None and len(thresholds) < 2:
        raise ValueError("threshold-selection mode requires at least two frozen thresholds")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    feature_dir = output / "features"
    existing = list(output.iterdir())
    if existing:
        # A worker may fail before writing its first feature table.  Permit only
        # that exact, content-free state; every material partial result remains
        # fail-closed and must be audited explicitly.
        empty_worker_scaffold = (
            existing == [feature_dir]
            and feature_dir.is_dir()
            and not any(feature_dir.iterdir())
        )
        if not empty_worker_scaffold:
            raise RuntimeError(f"fail-closed: output is non-empty: {output}")
    else:
        feature_dir.mkdir()

    extraction_rows: list[dict] = []
    workers = min(max(int(args.workers), 1), len(files))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                extract_one_file,
                str(path), str(feature_dir), thresholds,
                args.mass_error_ppm, args.min_trace_sec, args.max_trace_sec,
            ): path
            for path in files
        }
        for position, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            rows = future.result()
            extraction_rows.extend(rows)
            pd.DataFrame(extraction_rows).sort_values(
                ["source_file", "noise_threshold"]
            ).to_csv(output / "extraction.partial.csv", index=False)
            counts = ", ".join(
                f"{row['noise_threshold']:g}:{int(row['n_features']):,}" for row in rows
            )
            print(f"[MS1 {position}/{len(files)} complete] {path.name} features={counts}", flush=True)

    pair_rows: list[dict] = []
    for polarity in ("pos", "neg"):
        for window in ("70_300", "70_1200", "290_600", "590_1200"):
            first_path_raw = by_role.get((polarity, window, 1))
            second_path_raw = by_role.get((polarity, window, 2))
            if first_path_raw is None or second_path_raw is None:
                continue
            first = first_path_raw.name
            second = second_path_raw.name
            for threshold in thresholds:
                first_path = feature_dir / f"{Path(first).stem}__noise_{threshold:g}.csv.gz"
                second_path = feature_dir / f"{Path(second).stem}__noise_{threshold:g}.csv.gz"
                first_table = pd.read_csv(first_path)
                second_table = pd.read_csv(second_path)
                metrics = greedy_match(
                    first_table, second_table,
                    ppm=args.match_ppm, rt_sec=args.match_rt_sec,
                )
                pair_rows.append({
                    "polarity": "positive" if polarity == "pos" else "negative",
                    "mass_window": window,
                    "noise_threshold": threshold,
                    "replicate_1": first,
                    "replicate_2": second,
                    "n_features_1": len(first_table),
                    "n_features_2": len(second_table),
                    **metrics,
                })
    extraction = pd.DataFrame(extraction_rows)
    pairs = pd.DataFrame(pair_rows)
    selection_rows: list[dict] = []
    for threshold in thresholds:
        subset = pairs[pairs["noise_threshold"].eq(threshold)]
        median_features = float(
            extraction.loc[extraction["noise_threshold"].eq(threshold), "n_features"].median()
        )
        eligible = args.minimum_median_features <= median_features <= args.maximum_median_features
        selection_rows.append({
            "noise_threshold": threshold,
            "median_features_per_file": median_features,
            "median_feature_jaccard": float(subset["feature_jaccard"].median()),
            "median_match_fraction_min": float(subset["match_fraction_min"].median()),
            "minimum_pair_jaccard": float(subset["feature_jaccard"].min()),
            "eligible_feature_yield": bool(eligible),
        })
    selection = pd.DataFrame(selection_rows)
    eligible = selection[selection["eligible_feature_yield"]]
    if eligible.empty:
        raise RuntimeError("no threshold satisfies the preregistered feature-yield bounds")
    chosen = (
        eligible.iloc[0] if args.frozen_noise_threshold is not None else
        eligible.sort_values(
            ["median_feature_jaccard", "median_match_fraction_min", "noise_threshold"],
            ascending=[False, False, False],
        ).iloc[0]
    )
    if float(chosen["median_feature_jaccard"]) < args.minimum_median_feature_jaccard:
        raise RuntimeError(
            "technical-replicate feature stability below frozen minimum: "
            f"{float(chosen['median_feature_jaccard']):.4f} < {args.minimum_median_feature_jaccard:.4f}"
        )
    extraction_path = output / "extraction.csv"
    pairs_path = output / "replicate_pairs.csv"
    selection_path = output / "threshold_selection.csv"
    extraction.to_csv(extraction_path, index=False)
    pairs.to_csv(pairs_path, index=False)
    selection.to_csv(selection_path, index=False)
    report = {
        "status": "bioaware_metdna3_ms1_feature_pilot_complete",
        "formal": True,
        "files": len(files),
        "technical_pairs": int(len(pairs) / len(thresholds)),
        "thresholds": thresholds,
        "selected_noise_threshold": float(chosen["noise_threshold"]),
        "selection_metric": (
            "externally frozen threshold; technical-replicate stability audit only"
            if args.frozen_noise_threshold is not None else
            "maximum median technical-replicate feature Jaccard, then match fraction, "
            "within frozen feature-yield bounds"
        ),
        "selection": selection_rows,
        "parameters": {
            "mass_error_ppm": args.mass_error_ppm,
            "min_trace_sec": args.min_trace_sec,
            "max_trace_sec": args.max_trace_sec,
            "match_ppm": args.match_ppm,
            "match_rt_sec": args.match_rt_sec,
            "minimum_median_features": args.minimum_median_features,
            "maximum_median_features": args.maximum_median_features,
            "minimum_median_feature_jaccard": args.minimum_median_feature_jaccard,
            "frozen_noise_threshold": args.frozen_noise_threshold,
        },
        "provenance": {
            "mzml_sha256": {path.name: sha256(path) for path in files},
            "extraction_sha256": sha256(extraction_path),
            "replicate_pairs_sha256": sha256(pairs_path),
            "threshold_selection_sha256": sha256(selection_path),
        },
        "contracts": {
            "level1_identity_loaded": False,
            "dreams_outcomes_loaded": False,
            "phenotype_loaded": False,
            "selection_is_outcome_blind": True,
            "within_panel_threshold_selection": args.frozen_noise_threshold is None,
        },
        "claim_limit": "Preprocessing selection only; no candidate or annotation result.",
    }
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
