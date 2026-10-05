#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.input_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "bioaware_catalogue_v1_frozen" or report.get("formal") is not True:
        raise RuntimeError("invalid frozen BioAware catalogue report")
    contracts = report.get("contracts", {})
    if not (
        contracts.get("candidate_static_prior") is True
        and contracts.get("B44_read") is False
        and contracts.get("P2b_used") is False
        and contracts.get("shared_embedding_changed") is False
    ):
        raise RuntimeError("frozen BioAware catalogue contract failed")
    if report["deployment_gate"]["gate_name"] == "no_op":
        raise RuntimeError("frozen deployment gate is no-op")
    model = joblib.load(args.input_dir / "model.joblib")
    if not hasattr(model, "predict_proba"):
        raise RuntimeError("frozen model cannot score candidates")
    lookup = pd.read_csv(args.input_dir / "catalogue_lookup.csv.gz")
    if lookup["candidate_id"].duplicated().any() or len(lookup) < 10000:
        raise RuntimeError("catalogue lookup is incomplete or duplicated")
    print("[validate_bioaware_b44_catalogue_v1] PASS", {
        "catalogue_nodes": len(lookup),
        "gate": report["deployment_gate"]["gate_name"],
        "features": report["features"],
    }, flush=True)


if __name__ == "__main__":
    main()
