#!/usr/bin/env python
"""Pool the five untouched panels for the frozen BioAware V6 router."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


UNITS = ("Mouse_brain__rplc", "Mouse_liver__hilic", "Mouse_liver__rplc",
         "NIST_plasma__hilic", "NIST_plasma__rplc")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap(frame: pd.DataFrame, repeats: int, seed: int) -> dict:
    grouped = frame.groupby("global_formula", sort=True).delta.agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    n_clusters = len(grouped)
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, float)
    for index in range(repeats):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        values[index] = float(sums[draw].sum() / counts[draw].sum())
    return {"mean": float(frame.delta.mean()), "ci_low": float(np.quantile(values, .025)),
            "ci_high": float(np.quantile(values, .975)), "clusters": n_clusters,
            "resamples": repeats}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/validation/bioaware_metdna3_external_v3_v1"))
    parser.add_argument("--artifact", type=Path, default=Path("data/validation/bioaware_v6_identifiable_router_frozen_v2_20260830/artifact.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/bioaware_v6_external_5panel_summary_v1"))
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    expected_artifact_sha256 = sha256(args.artifact)
    if artifact.get("status") != "bioaware_v6_identifiable_router_artifact_frozen":
        raise RuntimeError("unexpected V6 artifact")
    if tuple(artifact["confirmatory_external_panels"]["required"]) != UNITS:
        raise RuntimeError("V6 confirmatory panel list changed")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output: {args.output_dir}")
    parts, panel_reports, hashes = [], {}, {}
    for unit in UNITS:
        report_path = args.root / unit / "result_v6" / "report.json"
        transition_path = args.root / unit / "result_v6" / "query_transitions.csv.gz"
        if not report_path.exists() or not transition_path.exists():
            raise FileNotFoundError(f"missing V6 result: {unit}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("panel") != unit or not report.get("formal"):
            raise RuntimeError(f"invalid V6 report: {unit}")
        if report.get("provenance", {}).get("artifact_sha256") != expected_artifact_sha256:
            raise RuntimeError(f"V6 artifact provenance mismatch: {unit}")
        frame = pd.read_csv(transition_path)
        frame["unit_id"] = unit
        frame["global_query_id"] = unit + "|" + frame.query_id.astype(str)
        # The same chemical formula observed in multiple panels/studies is one
        # dependence cluster, not one independent cluster per LC unit.
        frame["global_formula"] = frame.truth_formula.astype(str)
        parts.append(frame); panel_reports[unit] = report
        hashes[unit] = {"report": sha256(report_path), "transitions": sha256(transition_path)}
    pooled = pd.concat(parts, ignore_index=True)
    if pooled.global_query_id.duplicated().any():
        raise RuntimeError("query overlap across V6 panels")
    corrected, introduced = int(pooled.corrected.sum()), int(pooled.introduced.sum())
    discordant = corrected + introduced
    ci = bootstrap(pooled, args.bootstrap_resamples, args.seed)
    gates = {
        "five_panels_present": len(panel_reports) == 5,
        "formula_cluster_ci_positive": ci["ci_low"] > 0,
        "corrected_gt_introduced": corrected > introduced,
        "risk_weighted_net_lambda2_positive": corrected - 2 * introduced > 0,
        "no_panel_degrades": all(row["delta_recall1"] >= 0 for row in panel_reports.values()),
    }
    report = {
        "status": "bioaware_v6_external_5panel_summary_complete", "formal": True,
        "panels": 5, "queries": int(len(pooled)),
        "baseline_recall1": float(pooled.baseline_correct.mean()),
        "recall1": float(pooled.final_correct.mean()), "delta_recall1": float(pooled.delta.mean()),
        "corrected": corrected, "introduced": introduced,
        "risk_weighted_net_lambda2": corrected - 2 * introduced,
        "interventions": int(pooled.intervene.sum()),
        "mcnemar_exact_p": float(binomtest(min(corrected, introduced), discordant, .5).pvalue) if discordant else 1.0,
        "formula_cluster_bootstrap": ci, "panel_results": panel_reports,
        "gates": gates, "pass": bool(all(gates.values())),
        "contracts": {"fit_performed": False, "three_opened_panels_excluded": True,
                      "P2b": "forbidden", "phenotype": "forbidden"},
        "provenance": {"artifact_sha256": sha256(args.artifact), "panels": hashes},
        "claim_limit": "Five-panel V6 confirmation; SOTA additionally requires degree-decoy and matched-method comparison gates.",
    }
    args.output_dir.mkdir(parents=True)
    pooled_path = args.output_dir / "pooled_query_transitions.csv.gz"
    pooled.to_csv(pooled_path, index=False, compression="gzip")
    report["provenance"]["pooled_transitions_sha256"] = sha256(pooled_path)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
