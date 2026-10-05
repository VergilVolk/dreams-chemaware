#!/usr/bin/env python
"""Fail-closed validator for B5 graph-prior external transfer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = {
    "st001154_author_candidates": 161,
    "st001154_same_formula_10ppm": 150,
    "kgmn200std_hidden_seed": 162,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    report_path = args.output_dir / "report.json"
    artifact_path = args.output_dir / "graph_prior_artifact.json"
    for path in (report_path, artifact_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "bioaware_b5_graph_prior_external_transfer_complete":
        raise RuntimeError("wrong B5 report status")
    if not report.get("model_fitted_once_before_external_scoring"):
        raise RuntimeError("B5 model freeze boundary missing")
    train = report["training_after_union_purge"]
    if train["external_identity_overlap"] != 0 or train["external_formula_overlap"] != 0:
        raise RuntimeError("external truth leakage")
    if set(report["panels"]) != set(EXPECTED):
        raise RuntimeError("B5 panel set changed")
    for name, expected in EXPECTED.items():
        panel = report["panels"][name]
        if panel["queries"] != expected:
            raise RuntimeError(f"{name}: query count changed")
        transition = args.output_dir / f"{name}__transitions.csv.gz"
        if not transition.is_file() or transition.stat().st_size == 0:
            raise FileNotFoundError(transition)
    contracts = report["contracts"]
    if contracts["P2b_used"] or contracts["phenotype_used"] or contracts["shared_embedding_changed"]:
        raise RuntimeError("B5 contract violation")
    print(
        "[validate_bioaware_b5_graph_prior_external_transfer] PASS "
        f"calibration_research={report['pass_to_calibration_research']} "
        f"gate_validated={report['frozen_gate_is_validated']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
