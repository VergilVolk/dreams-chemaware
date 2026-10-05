"""Fold-1 arm comparison with the predeclared selection rule.

Arms:
  clean-only       -- v2 trainer report (fold1_diagnostics inside)
  action-rotation  -- v2 trainer report (fold1_diagnostics inside)
  averaged         -- standalone scorer JSON for run 2347055's checkpoint

Predeclared rule (fixed before any arm trained): the winner is the arm with
the largest fold-1 strict recall@1 improvement over the Stage-1 warm start;
ties break on mean-margin improvement.  The held fold is spent only when the
winner's recall improvement is strictly positive; otherwise the run reports
NO-GO and the outer held fold stays untouched.  Arm selection happens on
train-side material only (fold 1 lies inside Stage-1's training
distribution and is diagnostic, never a performance claim).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = ("clean-only", "averaged", "action-rotation")


def arm_block(source: dict) -> dict:
    diagnostics = source["fold1_diagnostics"]
    before = diagnostics["before"]
    after = diagnostics["after"]
    return {
        "before": before,
        "after": after,
        "recall1_delta": after["recall_at_1_strict"] - before["recall_at_1_strict"],
        "margin_delta": after["mean_margin"] - before["mean_margin"],
        "mrr_delta": after["mean_reciprocal_rank"] - before["mean_reciprocal_rank"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-only-report", type=Path, required=True)
    parser.add_argument("--action-rotation-report", type=Path, required=True)
    parser.add_argument("--averaged-scorer-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    blocks = {
        "clean-only": arm_block(json.loads(
            args.clean_only_report.read_text(encoding="utf-8"),
        )),
        "averaged": arm_block(json.loads(
            args.averaged_scorer_json.read_text(encoding="utf-8"),
        )),
        "action-rotation": arm_block(json.loads(
            args.action_rotation_report.read_text(encoding="utf-8"),
        )),
    }
    for arm in ARMS:
        block = blocks[arm]
        if block["before"]["queries"] != blocks["clean-only"]["before"]["queries"]:
            raise RuntimeError(f"arm {arm} scored a different fold-1 query set")

    winner = max(
        ARMS, key=lambda arm: (
            blocks[arm]["recall1_delta"], blocks[arm]["margin_delta"],
        ),
    )
    proceed = blocks[winner]["recall1_delta"] > 0.0
    report = {
        "status": "noise_relation_t1_t3_v2_fold1_selection_complete",
        "arms": blocks,
        "selection_rule": (
            "argmax fold-1 strict recall@1 delta over the Stage-1 warm start; "
            "ties break on mean-margin delta; held evaluation is spent only "
            "when the winner's recall delta is strictly positive"
        ),
        "selected_arm": winner if proceed else None,
        "proceed_to_held": proceed,
        "no_go_reason": None if proceed else (
            "no arm improved fold-1 strict recall over the Stage-1 warm start"
        ),
        "claim_limit": (
            "Fold 1 lies inside Stage-1's outer-train distribution; these "
            "deltas select an arm and justify (or refuse) spending the held "
            "fold.  They are not performance claims."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
