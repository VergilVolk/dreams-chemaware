#!/usr/bin/env python
"""Fail-closed adjudication of frozen GNPS grand-fusion comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PANELS = ("identity_disjoint", "formula_disjoint")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("router", "reranker", "encoder"), required=True)
    parser.add_argument("--wse", type=Path, required=True)
    parser.add_argument("--p2b", type=Path, required=True)
    parser.add_argument("--noise", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def recursively_find(node: Any, predicate, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], dict]]:
    found: list[tuple[tuple[str, ...], dict]] = []
    if isinstance(node, dict):
        if predicate(node, path):
            found.append((path, node))
        for key, value in node.items():
            found.extend(recursively_find(value, predicate, path + (str(key),)))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            found.extend(recursively_find(value, predicate, path + (str(index),)))
    return found


def primary_recall_block(panel: dict) -> tuple[tuple[str, ...], dict]:
    candidates = recursively_find(
        panel,
        lambda row, path: (
            ("delta_pp" in row or "delta" in row)
            and ("ci_low_pp" in row or "ci_low" in row)
            and any("recall" in part.lower() and ("1" in part or "one" in part.lower()) for part in path)
            and not any("near" in part.lower() for part in path)
        ),
    )
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one primary Recall@1 paired block, found {len(candidates)}")
    return candidates[0]


def transition_block(panel: dict) -> tuple[tuple[str, ...], dict]:
    candidates = recursively_find(
        panel,
        lambda row, path: (
            "corrected" in row and "introduced" in row
            and not any("near" in part.lower() for part in path)
        ),
    )
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one overall transition block, found {len(candidates)}")
    return candidates[0]


def pp(row: dict, name: str) -> float:
    if f"{name}_pp" in row:
        return float(row[f"{name}_pp"])
    return 100.0 * float(row[name])


def load_comparison(path: Path) -> dict:
    body = json.loads(path.read_text(encoding="utf-8"))
    if body.get("status") != "gnps_gold_silver_10ppm_pair_score_evaluation_complete":
        raise RuntimeError(f"unexpected evaluation status: {path}")
    output = {}
    for panel in PANELS:
        panel_body = body.get("panels", {}).get(panel)
        if not isinstance(panel_body, dict):
            raise RuntimeError(f"missing panel {panel}: {path}")
        recall_path, recall = primary_recall_block(panel_body)
        transition_path, transition = transition_block(panel_body)
        corrected = int(transition["corrected"])
        introduced = int(transition["introduced"])
        output[panel] = {
            "delta_pp": pp(recall, "delta"),
            "ci_low_pp": pp(recall, "ci_low"),
            "ci_high_pp": pp(recall, "ci_high"),
            "corrected": corrected,
            "introduced": introduced,
            "risk_net_lambda2": corrected - 2 * introduced,
            "recall_block_path": "/".join(recall_path),
            "transition_block_path": "/".join(transition_path),
        }
    return output


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    comparisons = {
        "wse": load_comparison(args.wse),
        "p2b": load_comparison(args.p2b),
        "noise": load_comparison(args.noise),
    }
    wse = comparisons["wse"]
    deltas = [wse[panel]["delta_pp"] for panel in PANELS]
    common = {
        "all_wse_ci_low_positive": all(wse[panel]["ci_low_pp"] > 0 for panel in PANELS),
        "all_wse_risk_net_positive": all(wse[panel]["risk_net_lambda2"] > 0 for panel in PANELS),
        "all_wse_exchange_gt_2": all(
            wse[panel]["corrected"] > 2 * wse[panel]["introduced"] for panel in PANELS
        ),
        "beats_p2b_pointwise": all(comparisons["p2b"][panel]["delta_pp"] > 0 for panel in PANELS),
        "beats_noise_pointwise": all(comparisons["noise"][panel]["delta_pp"] > 0 for panel in PANELS),
    }
    if args.mode in {"router", "reranker"}:
        magnitude = max(deltas) >= 1.0 and min(deltas) >= 0.5
    else:
        magnitude = max(deltas) >= 0.5 and min(deltas) >= 0.0
    gates = {**common, "magnitude_gate": magnitude}
    passed = all(gates.values())
    report = {
        "status": "GRAND_FUSION_EXTERNAL_ADJUDICATION_COMPLETE",
        "mode": args.mode,
        "comparisons": comparisons,
        "gates": gates,
        "promotion_authorized": passed,
        "verdict": "PROMOTE" if passed else "STOP_NO_EXTERNAL_CLAIM",
        "claim_limit": (
            "This adjudicates the preregistered frozen comparisons. Failure of any gate disables "
            "promotion; the same GNPS outcomes may not be reused to tune another arm."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
