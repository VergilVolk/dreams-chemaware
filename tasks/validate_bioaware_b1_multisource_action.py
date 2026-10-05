#!/usr/bin/env python
"""Independent fail-closed validator for BioAware B1 outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report_path = args.output_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != "bioaware_b1_multisource_action_v1":
        raise RuntimeError("B1 schema mismatch")
    candidates_path = args.output_dir / "harmonized_candidates.csv.gz"
    if sha256(candidates_path) != report["provenance"]["harmonized_candidates_sha256"]:
        raise RuntimeError("candidate artifact hash mismatch")
    candidates = pd.read_csv(candidates_path)
    queries = candidates[["query_id", "truth_candidate_id", "truth_formula"]].drop_duplicates()
    protocol = report["candidate_protocol"]
    if len(candidates) != protocol["candidate_rows"] or len(queries) != protocol["queries"]:
        raise RuntimeError("candidate protocol counts do not replay")
    if len(queries) != 1426:
        raise RuntimeError("formal B1 expects 1426 queries")
    recipes = report["recipes"]
    for recipe, payload in recipes.items():
        path = args.output_dir / f"{recipe}__transitions.csv.gz"
        if sha256(path) != payload["transitions_sha256"]:
            raise RuntimeError(f"{recipe}: transition hash mismatch")
        frame = pd.read_csv(path)
        if len(frame) != 1426 or frame.query_id.nunique() != 1426:
            raise RuntimeError(f"{recipe}: query coverage mismatch")
        if frame.groupby("query_id").size().ne(1).any():
            raise RuntimeError(f"{recipe}: duplicate transition rows")
        summary = payload["pooled"]
        corrected = int(frame.corrected.astype(bool).sum())
        introduced = int(frame.introduced.astype(bool).sum())
        if corrected != summary["corrected"] or introduced != summary["introduced"]:
            raise RuntimeError(f"{recipe}: transition counts do not replay")
        observed = float(frame.gated_correct.astype(bool).mean())
        if abs(observed - float(summary["gated_recall1"])) > 1e-12:
            raise RuntimeError(f"{recipe}: recall does not replay")
        for fold in payload["folds"]:
            if fold["identity_overlap"] != 0 or fold["formula_overlap"] != 0:
                raise RuntimeError(f"{recipe}: leakage flag is nonzero")
    if report["scientific_pass"] != all(report["gates"].values()):
        raise RuntimeError("scientific pass is inconsistent with gates")
    print(
        "[validate_bioaware_b1_multisource_action] PASS "
        f"queries={len(queries)} scientific_pass={report['scientific_pass']}"
    )


if __name__ == "__main__":
    main()
