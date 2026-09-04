#!/usr/bin/env python
"""Pool eight frozen external units into the preregistered 16-panel endpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary(frame: pd.DataFrame) -> dict:
    corrected = int(frame.corrected.sum())
    introduced = int(frame.introduced.sum())
    discordant = corrected + introduced
    return {
        "queries": int(len(frame)),
        "identities": int(frame.truth_candidate_id.astype(str).nunique()),
        "formulas": int(frame.truth_formula.astype(str).nunique()),
        "baseline_recall1": float(frame.baseline_correct.mean()),
        "router_recall1": float(frame.final_correct.mean()),
        "delta_recall1": float((frame.final_correct.astype(int)-frame.baseline_correct.astype(int)).mean()),
        "baseline_mrr": float(frame.baseline_rr.mean()),
        "router_mrr": float(frame.final_rr.mean()),
        "delta_mrr": float((frame.final_rr-frame.baseline_rr).mean()),
        "corrected": corrected, "introduced": introduced,
        "risk_weighted_net": corrected - 2 * introduced,
        "mcnemar_exact_p": float(binomtest(min(corrected, introduced), discordant, 0.5).pvalue) if discordant else 1.0,
    }


def formula_bootstrap(frame: pd.DataFrame, repeats: int, seed: int) -> dict:
    effects = frame.assign(delta=frame.final_correct.astype(int)-frame.baseline_correct.astype(int)).groupby(
        frame.truth_formula.astype(str), sort=True
    ).delta.mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    values = np.asarray([rng.choice(effects, len(effects), replace=True).mean() for _ in range(repeats)])
    return {"mean": float(effects.mean()), "ci_low": float(np.quantile(values, .025)),
            "ci_high": float(np.quantile(values, .975)), "formulas": int(len(effects)),
            "resamples": repeats}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, default=Path(
        "data/validation/bioaware_metdna3_external_manifest_v1/report.json"))
    parser.add_argument("--artifact", type=Path, default=Path(
        "data/validation/bioaware_v3_consensus_router_frozen_v2_20260830/artifact.json"))
    parser.add_argument("--decoy-report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    for path in (args.manifest_report, args.artifact):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    manifest = json.loads(args.manifest_report.read_text(encoding="utf-8"))
    if manifest.get("status") != "bioaware_metdna3_external_manifest_frozen":
        raise RuntimeError("external manifest is not frozen")
    units = sorted(manifest["units"])
    frames = []
    provenance = {}
    for unit in units:
        report_path = args.results_root / unit / "result" / "report.json"
        transitions_path = args.results_root / unit / "result" / "query_transitions.csv.gz"
        if not report_path.exists() or not transitions_path.exists():
            raise FileNotFoundError(f"missing frozen external unit result: {unit}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "bioaware_v3_frozen_external_panel_evaluation_complete":
            raise RuntimeError(f"unexpected external unit result: {unit}")
        if report.get("provenance", {}).get("artifact_sha256") != sha256(args.artifact):
            raise RuntimeError(f"artifact mismatch: {unit}")
        frame = pd.read_csv(transitions_path)
        frames.append(frame)
        provenance[unit] = {"report_sha256": sha256(report_path),
                            "transitions_sha256": sha256(transitions_path)}
    combined = pd.concat(frames, ignore_index=True)
    if combined.query_id.duplicated().any():
        raise RuntimeError("external query IDs collide across units")
    if combined.panel_id.nunique() != 16:
        raise RuntimeError(f"expected 16 external panels, got {combined.panel_id.nunique()}")
    pooled = summary(combined)
    panels = {str(name): summary(group) for name, group in combined.groupby("panel_id", sort=True)}
    nonnegative = sum(row["delta_recall1"] >= 0 for row in panels.values())
    bootstrap = formula_bootstrap(combined, args.bootstrap_resamples, args.seed)
    decoy = None
    decoy_gate = False
    if args.decoy_report is not None and args.decoy_report.exists():
        decoy = json.loads(args.decoy_report.read_text(encoding="utf-8"))
        if decoy.get("status") != "bioaware_v3_external_degree_decoy_summary_complete":
            raise RuntimeError("unexpected degree-preserving decoy report status")
        if not decoy.get("formal") or int(decoy.get("decoy_repeats", 0)) < 10:
            raise RuntimeError("degree-preserving decoy report is not formal")
        if int(decoy.get("real", {}).get("queries", -1)) != len(combined):
            raise RuntimeError("real and degree-decoy external query coverage differs")
        decoy_gate = bool(decoy.get("real_network_beats_degree_preserving_decoy"))
    gates = {
        "pooled_formula_cluster_ci_low_positive": bootstrap["ci_low"] > 0,
        "pooled_mrr_nonnegative": pooled["delta_mrr"] >= 0,
        "corrected_gt_introduced": pooled["corrected"] > pooled["introduced"],
        "risk_weighted_net_positive": pooled["risk_weighted_net"] > 0,
        "at_least_12_of_16_panels_nonnegative": nonnegative >= 12,
        "beats_degree_preserving_decoys": decoy_gate,
    }
    report = {
        "status": "bioaware_v3_external_16panel_summary_complete",
        "formal": True, "pooled": pooled, "formula_cluster_bootstrap": bootstrap,
        "panels": panels, "nonnegative_panels": nonnegative,
        "degree_preserving_decoy": decoy, "gates": gates,
        "sota_gate_pass": bool(all(gates.values())),
        "contracts": {"fit_performed": False, "threshold_tuning": False,
                      "P2b": "forbidden", "phenotype": "forbidden"},
        "provenance": provenance | {
            "manifest_sha256": sha256(args.manifest_report), "artifact_sha256": sha256(args.artifact),
            "decoy_report_sha256": sha256(args.decoy_report) if args.decoy_report is not None else None},
        "claim_limit": "SOTA is supported only when every preregistered gate, including graph decoys, passes.",
    }
    args.output_dir.mkdir(parents=True)
    combined_path = args.output_dir / "query_transitions.csv.gz"
    combined.to_csv(combined_path, index=False, compression="gzip")
    report["provenance"]["combined_transitions_sha256"] = sha256(combined_path)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
