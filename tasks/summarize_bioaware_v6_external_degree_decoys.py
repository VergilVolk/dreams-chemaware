#!/usr/bin/env python
"""Compare frozen V6 real-network effect with degree-preserving graph decoys."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


UNITS = ("Mouse_brain__rplc", "Mouse_liver__hilic", "Mouse_liver__rplc",
         "NIST_plasma__hilic", "NIST_plasma__rplc")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path, unit: str, expected_artifact_sha256: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    report_path = path.parent / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("provenance", {}).get("artifact_sha256") != expected_artifact_sha256:
        raise RuntimeError(f"V6 artifact provenance mismatch: {report_path}")
    frame = pd.read_csv(path)
    frame["global_query_id"] = unit + "|" + frame.query_id.astype(str)
    return frame


def metrics(frame: pd.DataFrame) -> dict:
    corrected, introduced = int(frame.corrected.sum()), int(frame.introduced.sum())
    return {"queries": int(len(frame)), "delta_recall1": float(frame.delta.mean()),
            "corrected": corrected, "introduced": introduced,
            "risk_weighted_net_lambda2": corrected - 2 * introduced}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/validation/bioaware_metdna3_external_v3_v1"))
    parser.add_argument("--artifact", type=Path, default=Path("data/validation/bioaware_v6_identifiable_router_frozen_v2_20260830/artifact.json"))
    parser.add_argument("--network-decoys", type=Path, default=Path("data/reference/bioaware_degree_preserving_decoys_v1/report.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/bioaware_v6_external_degree_decoy_summary_v1"))
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 20:
        raise RuntimeError("formal V6 summary requires 20 decoys")
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    expected_artifact_sha256 = sha256(args.artifact)
    network = json.loads(args.network_decoys.read_text(encoding="utf-8"))
    if tuple(artifact["confirmatory_external_panels"]["required"]) != UNITS:
        raise RuntimeError("V6 confirmation set changed")
    if not network.get("formal") or int(network.get("repeats", 0)) < args.repeats:
        raise RuntimeError("invalid degree-preserving decoy set")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output: {args.output_dir}")
    real = pd.concat([
        load(args.root / unit / "result_v6" / "query_transitions.csv.gz", unit,
             expected_artifact_sha256)
        for unit in UNITS
    ], ignore_index=True)
    real_ids = set(real.global_query_id); real_metrics = metrics(real)
    rows = []
    for repeat in range(args.repeats):
        tag = f"repeat_{repeat:02d}"
        decoy = pd.concat([
            load(args.root / unit / "degree_decoys" / tag / "result_v6" /
                 "query_transitions.csv.gz", unit, expected_artifact_sha256)
            for unit in UNITS
        ], ignore_index=True)
        if set(decoy.global_query_id) != real_ids:
            raise RuntimeError(f"query coverage differs for {tag}")
        rows.append({"repeat": repeat, **metrics(decoy)})
    decoys = pd.DataFrame(rows)
    delta_p95 = float(decoys.delta_recall1.quantile(.95))
    risk_p95 = float(decoys.risk_weighted_net_lambda2.quantile(.95))
    delta_p = float((1 + np.sum(decoys.delta_recall1 >= real_metrics["delta_recall1"])) / (1 + len(decoys)))
    risk_p = float((1 + np.sum(decoys.risk_weighted_net_lambda2 >= real_metrics["risk_weighted_net_lambda2"])) / (1 + len(decoys)))
    gates = {"real_delta_positive": real_metrics["delta_recall1"] > 0,
             "real_risk_positive": real_metrics["risk_weighted_net_lambda2"] > 0,
             "delta_beats_decoy_p95": real_metrics["delta_recall1"] > delta_p95,
             "risk_beats_decoy_p95": real_metrics["risk_weighted_net_lambda2"] > risk_p95,
             "delta_empirical_p_le_0_05": delta_p <= .05,
             "risk_empirical_p_le_0_05": risk_p <= .05}
    args.output_dir.mkdir(parents=True)
    decoy_path = args.output_dir / "degree_preserving_decoys.csv"
    decoys.to_csv(decoy_path, index=False)
    report = {"status": "bioaware_v6_external_degree_decoy_summary_complete", "formal": True,
              "external_panels": 5, "decoy_repeats": args.repeats, "real": real_metrics,
              "degree_preserving_decoys": {"delta_mean": float(decoys.delta_recall1.mean()),
                  "delta_p95": delta_p95, "risk_net_mean": float(decoys.risk_weighted_net_lambda2.mean()),
                  "risk_net_p95": risk_p95, "delta_empirical_p": delta_p, "risk_empirical_p": risk_p},
              "gates": gates, "pass": bool(all(gates.values())),
              "contracts": {"fit_performed": False, "degree_sequence_preserved": True,
                            "query_coverage_exact": True, "P2b": "forbidden", "phenotype": "forbidden"},
              "provenance": {"artifact_sha256": sha256(args.artifact),
                             "network_decoy_report_sha256": sha256(args.network_decoys),
                             "decoy_results_sha256": sha256(decoy_path)},
              "claim_limit": "V6 graph-specificity test; performance confirmation is a separate five-panel gate."}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
