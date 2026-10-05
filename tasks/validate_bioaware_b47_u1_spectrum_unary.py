#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from bioaware_b47_u1_core import sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    directory = args.input.resolve()
    required = (
        "report.json", "frozen_recipe.json", "recipe_scan.csv.gz",
        "per_query_oof.csv.gz",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(missing)
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    frozen = json.loads((directory / "frozen_recipe.json").read_text(encoding="utf-8"))
    if (
        report.get("status") != "bioaware_b47_u1_spectrum_unary_development_complete"
        or report.get("formal") is not True
        or report.get("development_graph_only") is not True
        or report.get("candidate_recipes") != 33
        or len(report.get("outer_folds", [])) != 5
    ):
        raise RuntimeError("U1 report header/denominator is invalid")
    contract = report.get("contracts", {})
    required_true = (
        "only_official_spectrum_cosines_used", "formula_outer_oof",
        "candidate_set_unchanged", "ties_count_against_positive",
    )
    required_false = (
        "B47_truth_opened", "reaction_network_used", "phenotype_used", "P2b_used",
        "shared_embedding_changed", "adduct_pooling_validated",
    )
    if any(contract.get(name) is not True for name in required_true) or any(
        contract.get(name) is not False for name in required_false
    ):
        raise RuntimeError("U1 scientific contract changed")
    if report.get("frozen_recipe") != frozen:
        raise RuntimeError("U1 frozen recipe/report mismatch")
    if bool(report.get("pass_to_b47_truthblind_apply")) != all(
        bool(value) for value in report.get("gates", {}).values()
    ):
        raise RuntimeError("U1 gate conjunction is inconsistent")
    provenance = report.get("provenance", {})
    files = {
        "recipe_scan_sha256": "recipe_scan.csv.gz",
        "per_query_oof_sha256": "per_query_oof.csv.gz",
        "frozen_recipe_sha256": "frozen_recipe.json",
    }
    for key, name in files.items():
        if provenance.get(key) != sha256_file(directory / name):
            raise RuntimeError(f"U1 artifact hash mismatch: {name}")
    query = pd.read_csv(directory / "per_query_oof.csv.gz")
    recipe = pd.read_csv(directory / "recipe_scan.csv.gz")
    if len(query) != 83619 or query["query_index"].nunique() != 83619:
        raise RuntimeError("U1 query ledger denominator/uniqueness changed")
    if len(recipe) != 33 or recipe["recipe"].nunique() != 33:
        raise RuntimeError("U1 recipe ledger denominator/uniqueness changed")
    if any(token in column.casefold() for column in query.columns for token in (
        "phenotype", "disease", "b47_truth", "reaction_score", "p2b",
    )):
        raise RuntimeError("U1 query ledger contains a forbidden field")
    print(
        "[validate_bioaware_b47_u1_spectrum_unary] PASS "
        f"queries={len(query):,} pass_to_b47={bool(report.get('pass_to_b47_truthblind_apply'))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
