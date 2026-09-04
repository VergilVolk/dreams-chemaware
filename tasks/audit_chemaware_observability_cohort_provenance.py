"""Audit artifact-level provenance of the frozen observability cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cohort-dir", type=Path,
        default=ROOT / "data/validation/large_observability_cohort",
    )
    parser.add_argument(
        "--discovery-manifest", type=Path,
        default=ROOT / "data/validation/large_observability_embeddings_discovery/manifest.csv",
    )
    parser.add_argument(
        "--confirmation-manifest", type=Path,
        default=ROOT / "data/validation/large_observability_embeddings_confirmation/manifest.csv",
    )
    parser.add_argument(
        "--test-manifest", type=Path,
        default=ROOT / "data/validation/large_observability_embeddings_test_frozen_gate_20260902/manifest.csv",
    )
    parser.add_argument(
        "--builder", type=Path,
        default=ROOT / "tasks/build_large_observability_cohort.py",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/chemaware_observability_cohort_provenance_audit_20260902/report.json",
    )
    args = parser.parse_args()
    selected_path = args.cohort_dir / "selected_spectra.csv"
    cohort_report_path = args.cohort_dir / "report.json"
    manifests = {
        "discovery": args.discovery_manifest,
        "confirmation": args.confirmation_manifest,
        "test": args.test_manifest,
    }
    required = [selected_path, cohort_report_path, args.builder, *manifests.values()]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    selected = pd.read_csv(selected_path)
    cohort_report = json.loads(cohort_report_path.read_text(encoding="utf-8"))
    frames = {name: pd.read_csv(path) for name, path in manifests.items()}
    for name, frame in frames.items():
        if set(frame["audit_split"].astype(str)) != {name}:
            raise RuntimeError(f"manifest split mismatch: {name}")
    selected_rows = set(selected["hdf5_row"].astype(int))
    manifest_rows = {name: set(frame["hdf5_row"].astype(int)) for name, frame in frames.items()}
    union_rows = set().union(*manifest_rows.values())
    pairwise = {}
    names = list(frames)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            pairwise[f"{left}__{right}"] = {
                "row_overlap": len(manifest_rows[left] & manifest_rows[right]),
                "identity_overlap": len(
                    set(frames[left]["ik14"].astype(str)) & set(frames[right]["ik14"].astype(str))
                ),
                "formula_overlap": len(
                    set(frames[left]["formula"].astype(str)) & set(frames[right]["formula"].astype(str))
                ),
            }
    actual_counts = {
        name: {
            "spectra": len(frame),
            "identities": int(frame["ik14"].nunique()),
            "formulas": int(frame["formula"].nunique()),
        }
        for name, frame in frames.items()
    }
    report_counts_match = all(
        int(cohort_report["counts"][name]["spectra"]) == values["spectra"]
        and int(cohort_report["counts"][name]["molecules"]) == values["identities"]
        and int(cohort_report["counts"][name]["formulas"]) == values["formulas"]
        for name, values in actual_counts.items()
    )
    builder_text = args.builder.read_text(encoding="utf-8")
    recorded_rule = str(cohort_report.get("split_rule", ""))
    generator_drift = (
        "stable hash" in recorded_rule
        and "stratification by dominant ring class" in builder_text
    )
    output = {
        "status": "chemaware_observability_cohort_artifact_provenance_audited",
        "artifact_rows": {
            "selected_spectra": len(selected),
            "manifest_union": len(union_rows),
            "union_equals_selected": union_rows == selected_rows,
            "missing_from_manifests": len(selected_rows - union_rows),
            "unexpected_in_manifests": len(union_rows - selected_rows),
        },
        "actual_counts": actual_counts,
        "cohort_report_counts_match_actual_manifests": report_counts_match,
        "pairwise_isolation": pairwise,
        "generator_provenance": {
            "recorded_split_rule": recorded_rule,
            "current_builder_mentions_stratified_greedy_rule": (
                "stratification by dominant ring class" in builder_text
            ),
            "generator_contract_drift_detected": generator_drift,
            "current_builder_can_be_claimed_byte_reproducer": False,
            "artifact_manifests_remain_frozen_and_auditable": True,
        },
        "provenance": {
            "selected_spectra_sha256": sha256(selected_path),
            "cohort_report_sha256": sha256(cohort_report_path),
            "current_builder_sha256": sha256(args.builder),
            "manifest_sha256": {name: sha256(path) for name, path in manifests.items()},
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "Results are reproducible from the frozen manifests and pair tables, but the current "
            "cohort builder is not claimed to reproduce the historical selected_spectra artifact."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
