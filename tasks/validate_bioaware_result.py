#!/usr/bin/env python
"""Fail-closed consistency validation for a BioAware evaluation directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


FORBIDDEN_PHENOTYPE = re.compile(
    r"(^|_)(rmu|rtu|rn|ln|tumou?r|normal|case|control|phenotype|disease|histology|tissue|group_?label|fold_?change|log2fc)(_|$)",
    re.I,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(a: float, b: float, atol: float = 1e-12) -> bool:
    return bool(np.isclose(float(a), float(b), atol=atol, rtol=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    required = [
        out / "report.json",
        out / "candidate_scores.csv.gz",
        out / "query_decisions.csv",
        out / "per_query_transitions.csv",
        out / "evidence_paths.csv.gz",
        out / "degree_preserving_decoys.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing BioAware artifacts: {missing}")

    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "bioaware_v1_evaluation_complete":
        raise RuntimeError("unexpected BioAware report status")
    scored = pd.read_csv(out / "candidate_scores.csv.gz")
    decisions = pd.read_csv(out / "query_decisions.csv")
    transitions = pd.read_csv(out / "per_query_transitions.csv")
    try:
        decoys = pd.read_csv(out / "degree_preserving_decoys.csv")
    except pd.errors.EmptyDataError:
        decoys = pd.DataFrame()
    if scored.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("duplicate query/candidate rows")
    if decisions.query_id.duplicated().any() or transitions.query_id.duplicated().any():
        raise RuntimeError("query decisions/transitions must be one row per query")
    if set(decisions.query_id.astype(str)) != set(transitions.query_id.astype(str)):
        raise RuntimeError("query sets differ between decisions and transitions")

    suspicious = sorted(column for column in scored.columns if FORBIDDEN_PHENOTYPE.search(str(column)))
    if suspicious:
        raise RuntimeError(f"phenotype-like columns leaked into BioAware scoring table: {suspicious}")

    corrected = int((~transitions.baseline_correct.astype(bool) & transitions.final_correct.astype(bool)).sum())
    introduced = int((transitions.baseline_correct.astype(bool) & ~transitions.final_correct.astype(bool)).sum())
    baseline = float(transitions.baseline_correct.astype(bool).mean())
    final = float(transitions.final_correct.astype(bool).mean())
    real = report["real_network"]
    if int(real["n_queries"]) != len(transitions):
        raise RuntimeError("n_queries mismatch")
    if corrected != int(real["corrected"]) or introduced != int(real["introduced"]):
        raise RuntimeError("corrected/introduced mismatch")
    if not close(baseline, real["baseline_recall1"]) or not close(final, real["bioaware_recall1"]):
        raise RuntimeError("Recall@1 mismatch")
    if not close(final - baseline, real["delta_recall1"]):
        raise RuntimeError("delta Recall@1 mismatch")
    intervention = float(decisions.bioaware_applied.astype(bool).mean())
    if not close(intervention, real["intervention_rate"]):
        raise RuntimeError("intervention rate mismatch")

    config = report["configuration"]
    applied = decisions[decisions.bioaware_applied.astype(bool)]
    if (applied.spectral_margin > float(config["maximum_spectral_margin_for_override"]) + 1e-12).any():
        raise RuntimeError("BioAware override crossed the frozen spectral-margin safety gate")
    if (applied.network_advantage + 1e-12 < float(config["minimum_network_advantage"])).any():
        raise RuntimeError("BioAware override crossed the frozen network-advantage gate")

    formal = bool(report.get("formal"))
    if formal and len(decoys) < 10:
        raise RuntimeError("formal evaluation has fewer than 10 network decoys")
    expected_pass = all(
        bool(value) for key, value in report["gates"].items() if key != "pass"
    )
    if bool(report["gates"]["pass"]) != expected_pass:
        raise RuntimeError("aggregate gate does not match component gates")

    for name in ("candidates", "participants", "seeds"):
        path = Path(report["provenance"][name])
        if not path.exists() or sha256(path) != report["provenance"][f"{name}_sha256"]:
            raise RuntimeError(f"provenance mismatch: {name}")

    print(
        json.dumps(
            {
                "status": "bioaware_v1_result_validation_passed",
                "formal": formal,
                "queries": int(len(transitions)),
                "corrected": corrected,
                "introduced": introduced,
                "decoys": int(len(decoys)),
                "gate_pass": bool(report["gates"]["pass"]),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
