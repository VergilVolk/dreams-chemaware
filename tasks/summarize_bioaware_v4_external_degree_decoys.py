#!/usr/bin/env python
"""Compare frozen BioAware V4 on seven external panels with degree-preserving decoys."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


UNITS = (
    "BV2cell__rplc", "Mouse_brain__hilic", "Mouse_brain__rplc",
    "Mouse_liver__hilic", "Mouse_liver__rplc", "NIST_plasma__hilic",
    "NIST_plasma__rplc",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_transitions(path: Path, unit: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "query_id", "truth_formula", "baseline_correct", "final_correct",
        "corrected", "introduced",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"{path} lacks {sorted(missing)}")
    frame = frame.copy()
    frame["unit_id"] = unit
    frame["global_query_id"] = unit + "|" + frame.query_id.astype(str)
    if frame.global_query_id.duplicated().any():
        raise RuntimeError(f"duplicate query IDs in {path}")
    return frame


def metrics(frame: pd.DataFrame) -> dict:
    baseline = frame.baseline_correct.astype(bool).to_numpy()
    final = frame.final_correct.astype(bool).to_numpy()
    corrected = int(np.sum((~baseline) & final))
    introduced = int(np.sum(baseline & (~final)))
    return {
        "queries": int(len(frame)),
        "baseline_recall1": float(np.mean(baseline)),
        "recall1": float(np.mean(final)),
        "delta_recall1": float(np.mean(final.astype(int) - baseline.astype(int))),
        "corrected": corrected,
        "introduced": introduced,
        "risk_weighted_net_lambda2": int(corrected - 2 * introduced),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_v3_v1"),
    )
    parser.add_argument(
        "--network-decoys", type=Path,
        default=Path("data/reference/bioaware_degree_preserving_decoys_v1/report.json"),
    )
    parser.add_argument(
        "--artifact", type=Path,
        default=Path("data/validation/bioaware_v4_high_precision_router_frozen_v1_20260830/artifact.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_v4_external_degree_decoy_summary_v1"),
    )
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()

    if args.repeats < 20:
        raise RuntimeError("formal V4 summary requires all twenty frozen decoys")
    network = json.loads(args.network_decoys.read_text(encoding="utf-8"))
    artifact = json.loads(args.artifact.read_text(encoding="utf-8"))
    if not network.get("formal") or int(network.get("repeats", 0)) < args.repeats:
        raise RuntimeError("degree-preserving network report is not formal")
    if artifact.get("status") != "bioaware_v4_high_precision_router_artifact_frozen":
        raise RuntimeError("unexpected V4 artifact")
    if tuple(artifact["confirmatory_external_panels"]["excluded"]) != ("BV2cell__hilic",):
        raise RuntimeError("consumed V4 panel declaration changed")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {output}")

    real = pd.concat([
        load_transitions(args.root / unit / "result_v4" / "query_transitions.csv.gz", unit)
        for unit in UNITS
    ], ignore_index=True)
    if real.global_query_id.duplicated().any():
        raise RuntimeError("query identities overlap across real V4 panels")
    real_ids = set(real.global_query_id)
    real_metrics = metrics(real)

    rows: list[dict] = []
    for repeat in range(args.repeats):
        tag = f"repeat_{repeat:02d}"
        decoy = pd.concat([
            load_transitions(
                args.root / unit / "degree_decoys" / tag / "result_v4" /
                "query_transitions.csv.gz", unit,
            ) for unit in UNITS
        ], ignore_index=True)
        if set(decoy.global_query_id) != real_ids:
            raise RuntimeError(f"query coverage differs for {tag}")
        row = metrics(decoy)
        row["repeat"] = repeat
        rows.append(row)

    decoys = pd.DataFrame(rows)
    decoy_path = output / "degree_preserving_decoys.csv"
    decoys.to_csv(decoy_path, index=False)
    delta_p95 = float(decoys.delta_recall1.quantile(.95))
    risk_p95 = float(decoys.risk_weighted_net_lambda2.quantile(.95))
    empirical_delta_p = float(
        (1 + np.sum(decoys.delta_recall1.to_numpy() >= real_metrics["delta_recall1"]))
        / (1 + len(decoys))
    )
    empirical_risk_p = float(
        (1 + np.sum(
            decoys.risk_weighted_net_lambda2.to_numpy()
            >= real_metrics["risk_weighted_net_lambda2"]
        )) / (1 + len(decoys))
    )
    gates = {
        "real_delta_positive": real_metrics["delta_recall1"] > 0,
        "real_risk_net_positive": real_metrics["risk_weighted_net_lambda2"] > 0,
        "real_delta_beats_decoy_p95": real_metrics["delta_recall1"] > delta_p95,
        "real_risk_net_beats_decoy_p95": (
            real_metrics["risk_weighted_net_lambda2"] > risk_p95
        ),
        "delta_empirical_p_le_0_05": empirical_delta_p <= .05,
        "risk_empirical_p_le_0_05": empirical_risk_p <= .05,
    }
    report = {
        "status": "bioaware_v4_external_degree_decoy_summary_complete",
        "formal": True,
        "external_panels": len(UNITS),
        "consumed_panel_excluded": "BV2cell__hilic",
        "decoy_repeats": args.repeats,
        "real": real_metrics,
        "degree_preserving_decoys": {
            "delta_mean": float(decoys.delta_recall1.mean()),
            "delta_p95": delta_p95,
            "risk_net_mean": float(decoys.risk_weighted_net_lambda2.mean()),
            "risk_net_p95": risk_p95,
            "delta_empirical_p": empirical_delta_p,
            "risk_empirical_p": empirical_risk_p,
        },
        "gates": gates,
        "pass": bool(all(gates.values())),
        "contracts": {
            "fit_performed": False,
            "threshold_tuning": False,
            "degree_sequence_preserved": True,
            "query_coverage_exact": True,
            "P2b": "forbidden",
            "phenotype": "forbidden",
        },
        "provenance": {
            "artifact_sha256": sha256(args.artifact),
            "network_decoy_report_sha256": sha256(args.network_decoys),
            "decoy_results_sha256": sha256(decoy_path),
        },
        "claim_limit": (
            "This tests whether the frozen seven-panel V4 effect exceeds degree-preserving "
            "network rewiring. A broad SOTA claim also requires a positive independent "
            "seven-panel performance gate under the same candidate protocol."
        ),
    }
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
