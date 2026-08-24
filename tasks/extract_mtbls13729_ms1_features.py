#!/usr/bin/env python
"""Checkpointed OpenMS MS1 feature extraction for MTBLS13729.

The same detector parameters are used for every sample within a panel. Missing
features are *not* interpreted as zero here; consensus linking and targeted EIC
gap filling are separate downstream stages.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd

from pilot_mtbls13729_openms_features import detect_features, load_ms1


SAMPLE_RE = re.compile(r"^P\d{2}-(?:Ltu|Rtu|Rmu|LN|RN)$")


def load_exclusions(path: Path | None) -> set[tuple[str, str]]:
    if path is None or not path.exists():
        return set()
    frame = pd.read_csv(path, sep="\t")
    return {(str(row.panel), str(row.sample_name)) for row in frame.itertuples(index=False)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mzml-root", type=Path, default=Path("data/mtbls13729/mzml"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/ms1_features_full"))
    parser.add_argument("--panels", nargs="+", default=["neg_rp", "pos_rp"])
    parser.add_argument("--noise-threshold", type=float, default=10000.0)
    parser.add_argument("--mass-error-ppm", type=float, default=5.0)
    parser.add_argument("--min-trace-sec", type=float, default=5.0)
    parser.add_argument("--max-trace-sec", type=float, default=120.0)
    parser.add_argument("--exclusions", type=Path, default=Path("data/mtbls13729/ms1_acquisition_audit/exclusions.tsv"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = args.mzml_root.resolve()
    out = args.output_dir.resolve()
    feature_dir = out / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    exclusions = load_exclusions(args.exclusions)
    rows: list[dict[str, object]] = []

    for panel in args.panels:
        paths = sorted((root / panel).glob("*.mzML"))
        paths = [p for p in paths if SAMPLE_RE.match(p.stem)]
        for i, path in enumerate(paths, start=1):
            sample = path.stem
            target = feature_dir / f"{panel}__{sample}__noise_{args.noise_threshold:g}.csv.gz"
            summary_path = feature_dir / f"{panel}__{sample}__summary.json"
            base = {"panel": panel, "sample_name": sample, "source": str(path), "output": str(target)}
            if (panel, sample) in exclusions:
                record = {**base, "status": "excluded", "reason": "listed in exclusions.tsv"}
                rows.append(record)
                print(json.dumps(record), flush=True)
                continue
            if target.exists() and summary_path.exists() and not args.force:
                record = json.loads(summary_path.read_text(encoding="utf-8"))
                record["status"] = "reused"
                rows.append(record)
                print(json.dumps({"panel": panel, "sample": sample, "status": "reused"}), flush=True)
                continue

            print(f"[{panel} {i}/{len(paths)}] Loading {sample}", flush=True)
            started = time.time()
            try:
                experiment, load_summary = load_ms1(path)
                features, detection = detect_features(
                    experiment,
                    noise_threshold=args.noise_threshold,
                    mass_error_ppm=args.mass_error_ppm,
                    min_trace_length=args.min_trace_sec,
                    max_trace_length=args.max_trace_sec,
                )
                features.insert(0, "sample_name", sample)
                features.insert(0, "panel", panel)
                features.to_csv(target, index=False)
                record = {
                    **base,
                    "status": "complete",
                    "elapsed_seconds": time.time() - started,
                    **load_summary,
                    **detection,
                }
                summary_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
                rows.append(record)
                print(json.dumps({"panel": panel, "sample": sample, "n_features": len(features), "seconds": record["elapsed_seconds"]}), flush=True)
            except Exception as exc:  # preserve cohort progress and surface failures
                record = {**base, "status": "failed", "error": repr(exc), "elapsed_seconds": time.time() - started}
                rows.append(record)
                print(json.dumps(record), flush=True)

            pd.DataFrame(rows).to_csv(out / "extraction_manifest.partial.csv", index=False)

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out / "extraction_manifest.csv", index=False)
    report = {
        "status": "complete_with_failures" if (manifest["status"] == "failed").any() else "complete",
        "parameters": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "counts": manifest["status"].value_counts().to_dict(),
        "manifest": str(out / "extraction_manifest.csv"),
        "interpretation": "Detection absence is not zero abundance; targeted EIC gap filling is required after consensus construction.",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
