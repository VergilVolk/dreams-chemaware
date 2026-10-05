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
    for name in ("report.json", "frozen_policy.json", "per_query_safe_oof.csv.gz"):
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    policy = json.loads((directory / "frozen_policy.json").read_text(encoding="utf-8"))
    if (
        report.get("status") != "bioaware_b47_u1b_safe_veto_complete"
        or report.get("formal") is not True
        or report.get("development_graph_only") is not True
        or report.get("frozen_policy") != policy
        or bool(report.get("pass_to_b47_truthblind_apply"))
        != all(bool(value) for value in report.get("gates", {}).values())
    ):
        raise RuntimeError("U1b report/gate schema is invalid")
    if (
        report.get("contracts", {}).get("top1_identity_change_forbidden_by_construction") is not True
        or report.get("contracts", {}).get("baseline_tie_resolution_forbidden") is not True
        or report.get("contracts", {}).get("candidate_order_used_for_tie_breaking") is not False
        or report.get("oof", {}).get("top1_identity_switches") != 0
        or report.get("oof", {}).get("introduced") != 0
        or report.get("oof", {}).get("corrected") != 0
    ):
        raise RuntimeError("U1b structural safety contract failed")
    provenance = report.get("provenance", {})
    if provenance.get("per_query_sha256") != sha256_file(directory / "per_query_safe_oof.csv.gz"):
        raise RuntimeError("U1b per-query hash mismatch")
    if provenance.get("frozen_policy_sha256") != sha256_file(directory / "frozen_policy.json"):
        raise RuntimeError("U1b policy hash mismatch")
    query = pd.read_csv(directory / "per_query_safe_oof.csv.gz")
    if len(query) != 83619 or query["query_index"].nunique() != 83619:
        raise RuntimeError("U1b query ledger denominator changed")
    if query["introduced"].astype(bool).any():
        raise RuntimeError("U1b ledger contains an introduced Top-1 error")
    print(
        "[validate_bioaware_b47_u1b_safe_veto] PASS "
        f"corrected={int(query['corrected'].sum())} introduced=0 "
        f"pass_to_b47={bool(report.get('pass_to_b47_truthblind_apply'))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
