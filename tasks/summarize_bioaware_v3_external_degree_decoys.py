#!/usr/bin/env python
"""Pool the frozen external V3 result and degree-preserving network decoys."""
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
    "BV2cell__hilic", "BV2cell__rplc", "Mouse_brain__hilic", "Mouse_brain__rplc",
    "Mouse_liver__hilic", "Mouse_liver__rplc", "NIST_plasma__hilic", "NIST_plasma__rplc",
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


def metrics(frame: pd.DataFrame) -> dict:
    baseline = frame.baseline_correct.astype(bool).to_numpy()
    final = frame.final_correct.astype(bool).to_numpy()
    corrected = int(np.sum((~baseline) & final))
    introduced = int(np.sum(baseline & (~final)))
    return {
        "queries": int(len(frame)),
        "formulas": int(frame.truth_formula.astype(str).nunique()),
        "baseline_recall1": float(np.mean(baseline)),
        "recall1": float(np.mean(final)),
        "delta_recall1": float(np.mean(final.astype(int) - baseline.astype(int))),
        "corrected": corrected,
        "introduced": introduced,
        "risk_weighted_net_lambda2": int(corrected - 2 * introduced),
    }


def load_transitions(path: Path, unit: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {"query_id", "truth_formula", "baseline_correct", "final_correct"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"{path} lacks {sorted(required - set(frame.columns))}")
    frame = frame.copy()
    frame["unit_id"] = unit
    frame["global_query_id"] = unit + "|" + frame.query_id.astype(str)
    if frame.global_query_id.duplicated().any():
        raise RuntimeError(f"duplicate query IDs in {path}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_v3_v1"),
    )
    parser.add_argument(
        "--network-decoys", type=Path,
        default=Path("data/reference/bioaware_degree_preserving_decoys_v1/report.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_v3_degree_decoy_summary_v1"),
    )
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.repeats < 10:
        raise RuntimeError("formal summary requires at least ten decoys")
    if not args.network_decoys.exists():
        raise FileNotFoundError(args.network_decoys)
    network_report = json.loads(args.network_decoys.read_text(encoding="utf-8"))
    if not network_report.get("formal") or int(network_report.get("repeats", 0)) < args.repeats:
        raise RuntimeError("degree-preserving decoy network report is not formal")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {output}")

    real_parts = [
        load_transitions(args.root / unit / "result" / "query_transitions.csv.gz", unit)
        for unit in UNITS
    ]
    real = pd.concat(real_parts, ignore_index=True)
    if real.global_query_id.duplicated().any():
        raise RuntimeError("query identities overlap across external units")
    real_metrics = metrics(real)

    decoy_rows: list[dict] = []
    real_ids = set(real.global_query_id)
    for repeat in range(args.repeats):
        tag = f"repeat_{repeat:02d}"
        parts = [
            load_transitions(
                args.root / unit / "degree_decoys" / tag / "result" / "query_transitions.csv.gz",
                unit,
            ) for unit in UNITS
        ]
        frame = pd.concat(parts, ignore_index=True)
        if set(frame.global_query_id) != real_ids:
            raise RuntimeError(f"query coverage differs for {tag}")
        item = metrics(frame)
        item["repeat"] = repeat
        decoy_rows.append(item)
    decoys = pd.DataFrame(decoy_rows)
    decoy_path = output / "degree_preserving_decoys.csv"
    decoys.to_csv(decoy_path, index=False)
    empirical_p = float(
        (1 + np.sum(decoys.delta_recall1.to_numpy() >= real_metrics["delta_recall1"]))
        / (1 + len(decoys))
    )
    decoy_p95 = float(decoys.delta_recall1.quantile(0.95))
    risk_p95 = float(decoys.risk_weighted_net_lambda2.quantile(0.95))
    gates = {
        "real_delta_beats_decoy_p95": bool(real_metrics["delta_recall1"] > decoy_p95),
        "real_risk_net_beats_decoy_p95": bool(
            real_metrics["risk_weighted_net_lambda2"] > risk_p95
        ),
        "empirical_p_le_0_05": bool(empirical_p <= 0.05),
    }
    report = {
        "status": "bioaware_v3_external_degree_decoy_summary_complete",
        "formal": True,
        "external_units": len(UNITS),
        "decoy_repeats": args.repeats,
        "real": real_metrics,
        "degree_preserving_decoys": {
            "delta_mean": float(decoys.delta_recall1.mean()),
            "delta_p95": decoy_p95,
            "risk_net_mean": float(decoys.risk_weighted_net_lambda2.mean()),
            "risk_net_p95": risk_p95,
            "empirical_p": empirical_p,
        },
        "gates": gates,
        "real_network_beats_degree_preserving_decoy": bool(all(gates.values())),
        "pass": bool(all(gates.values())),
        "provenance": {
            "network_decoy_report_sha256": sha256(args.network_decoys),
            "decoy_results_sha256": sha256(decoy_path),
        },
        "claim_limit": (
            "This tests whether the frozen V3 external gain exceeds degree-preserving graph "
            "rewiring. It is necessary but not sufficient for a broad SOTA claim."
        ),
    }
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
