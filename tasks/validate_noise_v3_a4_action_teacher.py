"""Fail-closed validation for formal A4 nonlinear action-teacher artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_v3_a4_action_teacher",
    )
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    for name in ("decision.json", "oof_selected_actions.csv.gz", "oof_action_scores.npz"):
        if not (args.output_dir / name).is_file():
            raise FileNotFoundError(args.output_dir / name)
    decision = json.loads((args.output_dir / "decision.json").read_text(encoding="utf-8"))
    selected = pd.read_csv(args.output_dir / "oof_selected_actions.csv.gz")
    with np.load(args.output_dir / "oof_action_scores.npz") as body:
        required = {
            "action_index", "query_position", "dose", "p_benefit", "p_harm",
            "predicted_margin_change", "utility", "corrected", "introduced",
            "exact_margin_change", "fold",
        }
        missing = required - set(body.files)
        if missing:
            raise RuntimeError(f"OOF score artifact misses {sorted(missing)}")
        n = len(body["utility"])
        if any(len(body[name]) != n for name in required):
            raise RuntimeError("OOF arrays are misaligned")
        if not np.all(np.isfinite(body["utility"])):
            raise RuntimeError("non-finite OOF utility")
    if decision["status"] != "noise_v3_a4_nonlinear_action_teacher_complete":
        raise RuntimeError("wrong decision status")
    if not args.allow_smoke:
        if not decision["formal"]:
            raise RuntimeError("formal validator received smoke output")
        integrity = decision["integrity"]
        if integrity["source_scan_queries"] != 4998:
            raise RuntimeError("formal A4 source must contain all 4,998 scan queries")
        if integrity["source_errors"] != 1805 or integrity["source_controls"] != 3193:
            raise RuntimeError("formal A4 source error/control counts changed")
        if integrity["policy_eligible_queries"] != 4916:
            raise RuntimeError("formal action table must contain 4,916 eligible queries")
        if integrity["policy_eligible_error_queries"] != 1784:
            raise RuntimeError("formal eligible-error count changed")
        if integrity["policy_eligible_control_queries"] != 3132:
            raise RuntimeError("formal eligible-control count changed")
        if integrity["zero_policy_action_errors"] != 21:
            raise RuntimeError("formal zero-action error count changed")
        if integrity["zero_policy_action_controls"] != 61:
            raise RuntimeError("formal zero-action control count changed")
        if decision["integrity"]["formula_fold_overlap"] != 0:
            raise RuntimeError("formula leakage")
        if len(selected) != 4916:
            raise RuntimeError("selected action table is not eligible-query complete")
    print("[validate_noise_v3_a4_action_teacher] PASS", flush=True)


if __name__ == "__main__":
    main()
