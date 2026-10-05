"""Compute-node preflight for the frozen V11 historical-best action supplier."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from noise_historical_best_action_bank_v1 import (
    HISTORICAL_BEST_ACTION_SOURCES,
    REGISTERED_FULL_LEDGER_COUNTS,
    HistoricalBestActionBankV1,
    HistoricalBestActionBankV1Config,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def audit(ledger: Path) -> dict[str, object]:
    actions = pd.read_csv(ledger)
    result = HistoricalBestActionBankV1(HistoricalBestActionBankV1Config(
        margin_floor=5e-6,
        require_all_registered_sources=True,
    )).select(actions)
    report = dict(result.report)
    observed = {
        key: int(report[key])
        for key in (
            "strict_top1_rows_before_margin_floor",
            "strict_top1_champions_before_margin_floor",
            "selected_rows",
            "selected_queries",
        )
    }
    expected = {
        key: int(REGISTERED_FULL_LEDGER_COUNTS[key]) for key in observed
    }
    if observed != expected:
        raise RuntimeError(
            "historical-best registered count drift: "
            + json.dumps({"observed": observed, "expected": expected}, sort_keys=True)
        )
    if set(report["selected_sources"]) != HISTORICAL_BEST_ACTION_SOURCES:
        raise RuntimeError("historical-best champion union lost a mature source")
    denominator = int(REGISTERED_FULL_LEDGER_COUNTS["outer_train_queries"])
    headroom_pp = 100.0 * int(report["selected_queries"]) / denominator
    report.update({
        "status": "noise_historical_best_action_bank_v1_preflight_pass",
        "registered_counts_exact": True,
        "action_headroom_denominator": denominator,
        "strict_champion_action_headroom_pp": headroom_pp,
        "injector_executed": False,
        "model_loaded": False,
        "claim_limit": (
            "Action-space headroom only; shared-encoder gain is decided by V11."
        ),
    })
    return report


def main() -> None:
    args = arguments()
    report = audit(args.ledger)
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
