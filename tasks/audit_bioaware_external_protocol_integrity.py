#!/usr/bin/env python3
"""Audit whether the frozen BioAware external benchmark is actually evaluable.

The external manifest describes polarity-specific panels, while the candidate
library may support only a subset of adducts.  This audit fails closed when a
nominal panel has no candidate-library support or no evaluated queries.  It
also reports the result on the *actually evaluated* panels as descriptive
evidence, never as a replacement for the preregistered full benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import binomtest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=object,
    )


def formula_bootstrap(frame: pd.DataFrame, repeats: int, seed: int) -> dict[str, float | int]:
    effects = (
        frame.assign(delta=frame.final_correct.astype(int) - frame.baseline_correct.astype(int))
        .groupby(frame.truth_formula.astype(str), sort=True)
        .delta.mean()
        .to_numpy(dtype=float)
    )
    if len(effects) == 0:
        raise RuntimeError("no formula clusters in evaluated external results")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(effects), size=(repeats, len(effects)))
    values = effects[draws].mean(axis=1)
    return {
        "mean": float(effects.mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "clusters": int(len(effects)),
        "resamples": int(repeats),
    }


def actual_summary(frame: pd.DataFrame, repeats: int, seed: int) -> dict[str, object]:
    corrected = int(frame.corrected.sum())
    introduced = int(frame.introduced.sum())
    discordant = corrected + introduced
    panel_effects = {
        str(panel): float((group.final_correct.astype(int) - group.baseline_correct.astype(int)).mean())
        for panel, group in frame.groupby("panel_id", sort=True)
    }
    return {
        "queries": int(len(frame)),
        "identities": int(frame.truth_candidate_id.astype(str).nunique()),
        "formulas": int(frame.truth_formula.astype(str).nunique()),
        "baseline_recall1": float(frame.baseline_correct.mean()),
        "recall1": float(frame.final_correct.mean()),
        "delta_recall1": float(
            (frame.final_correct.astype(int) - frame.baseline_correct.astype(int)).mean()
        ),
        "corrected": corrected,
        "introduced": introduced,
        "risk_weighted_net_lambda2": corrected - 2 * introduced,
        "mcnemar_exact_p": (
            float(binomtest(min(corrected, introduced), discordant, 0.5).pvalue)
            if discordant
            else 1.0
        ),
        "panel_effects": panel_effects,
        "nonnegative_panels": int(sum(effect >= 0 for effect in panel_effects.values())),
        "formula_cluster_bootstrap": formula_bootstrap(frame, repeats, seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path("data/validation/bioaware_metdna3_external_manifest_v1"),
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("data/validation/bioaware_metdna3_external_v3_v1"),
    )
    parser.add_argument(
        "--reference-hdf5",
        type=Path,
        default=Path("data/models/MassSpecGym_MurckoHist_split.hdf5"),
    )
    parser.add_argument("--v4-summary", type=Path)
    parser.add_argument("--v6-summary", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/bioaware_external_protocol_integrity_20260901"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()

    manifest_path = args.manifest_root / "report.json"
    for path in (manifest_path, args.reference_hdf5):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "bioaware_metdna3_external_manifest_frozen":
        raise RuntimeError("external manifest status mismatch")

    with h5py.File(args.reference_hdf5, "r") as handle:
        if "adduct" not in handle:
            raise RuntimeError("candidate library lacks adduct metadata")
        supported_adducts = sorted(set(decode_strings(np.asarray(handle["adduct"]))))

    intended_panels = sorted(manifest["panels"])
    intended_units = sorted(manifest["units"])
    parts: list[pd.DataFrame] = []
    unit_reports: dict[str, object] = {}
    result_provenance: dict[str, object] = {}
    for unit in intended_units:
        truth_path = args.manifest_root / unit / "external_level1.csv.gz"
        report_path = args.results_root / unit / "result" / "report.json"
        transitions_path = args.results_root / unit / "result" / "query_transitions.csv.gz"
        for path in (truth_path, report_path, transitions_path):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
        truth = pd.read_csv(truth_path)
        result_report = json.loads(report_path.read_text(encoding="utf-8"))
        if result_report.get("status") != "bioaware_v3_frozen_external_panel_evaluation_complete":
            raise RuntimeError(f"unexpected frozen result status: {unit}")
        if result_report.get("provenance", {}).get("transitions_sha256") != sha256(transitions_path):
            raise RuntimeError(f"transition provenance mismatch: {unit}")
        transitions = pd.read_csv(transitions_path)
        if transitions.empty:
            raise RuntimeError(f"empty frozen external transitions: {unit}")
        parts.append(transitions)
        truth_counts = truth.groupby("polarity", sort=True).size().to_dict()
        truth_adducts = {
            str(polarity): sorted(set(group.adduct.astype(str)))
            for polarity, group in truth.groupby("polarity", sort=True)
        }
        evaluated_counts = transitions.groupby("polarity", sort=True).size().to_dict()
        unit_reports[unit] = {
            "truth_rows_by_polarity": {str(k): int(v) for k, v in truth_counts.items()},
            "truth_adducts_by_polarity": truth_adducts,
            "evaluated_queries_by_polarity": {str(k): int(v) for k, v in evaluated_counts.items()},
            "unsupported_truth_adducts": sorted(set(truth.adduct.astype(str)) - set(supported_adducts)),
        }
        result_provenance[unit] = {
            "truth_sha256": sha256(truth_path),
            "report_sha256": sha256(report_path),
            "transitions_sha256": sha256(transitions_path),
        }

    combined = pd.concat(parts, ignore_index=True)
    if combined.query_id.duplicated().any():
        raise RuntimeError("query IDs overlap across external units")
    actual_panels = sorted(set(combined.panel_id.astype(str)))
    missing_panels = sorted(set(intended_panels) - set(actual_panels))
    unexpected_panels = sorted(set(actual_panels) - set(intended_panels))
    if unexpected_panels:
        raise RuntimeError(f"evaluated panels are absent from manifest: {unexpected_panels}")

    optional_summaries: dict[str, object] = {}
    for name, path in (("v4", args.v4_summary), ("v6", args.v6_summary)):
        if path is None:
            continue
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        optional_summaries[name] = {
            "status": payload.get("status"),
            "pass": payload.get("pass"),
            "delta_recall1": payload.get("delta_recall1"),
            "corrected": payload.get("corrected"),
            "introduced": payload.get("introduced"),
            "formula_cluster_ci": payload.get("formula_cluster_bootstrap"),
            "sha256": sha256(path),
        }

    protocol_complete = len(actual_panels) == len(intended_panels) and not missing_panels
    report = {
        "status": "bioaware_external_protocol_integrity_complete",
        "formal": True,
        "intended_panels": len(intended_panels),
        "actual_evaluated_panels": len(actual_panels),
        "actual_panel_ids": actual_panels,
        "missing_panel_ids": missing_panels,
        "candidate_library_supported_adducts": supported_adducts,
        "unit_audit": unit_reports,
        "actually_evaluated_descriptive_result": actual_summary(
            combined, args.bootstrap_resamples, args.seed
        ),
        "later_router_summaries": optional_summaries,
        "gates": {
            "all_manifest_panels_evaluated": protocol_complete,
            "negative_mode_candidate_library_supported": any(adduct.endswith("-") for adduct in supported_adducts),
            "full_external_claim_allowed": protocol_complete,
        },
        "decision": (
            "full_external_protocol_complete"
            if protocol_complete
            else "protocol_incomplete_do_not_claim_16_panel_or_sota"
        ),
        "contracts": {
            "actually_evaluated_result_is_descriptive_only": not protocol_complete,
            "no_threshold_refit": True,
            "P2b": "forbidden",
            "phenotype": "forbidden",
        },
        "provenance": {
            "manifest_sha256": sha256(manifest_path),
            "reference_hdf5_sha256": sha256(args.reference_hdf5),
            "unit_results": result_provenance,
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "The local candidate library supports only the listed adducts. Missing polarity panels are not "
            "zero-effect panels; they are unevaluable. The actually evaluated subset cannot establish the "
            "preregistered 16-panel claim or SOTA."
        ),
    }
    args.output_dir.mkdir(parents=True)
    combined_path = args.output_dir / "actually_evaluated_query_transitions.csv.gz"
    combined.to_csv(combined_path, index=False, compression="gzip")
    report["provenance"]["combined_sha256"] = sha256(combined_path)
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
