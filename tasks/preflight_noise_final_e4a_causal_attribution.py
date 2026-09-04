"""Read-only preflight of the real R0 ledger for E4-A causal attribution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from noise_final_core import sha256_file
from train_noise_final_e4a_direct_augmentation import FIXED_POLICY, materialize_causal_arm


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--r0-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_r0_faithful_s3a",
    )
    args = parser.parse_args()
    report_path = args.r0_dir / "report.json"
    actions_path = args.r0_dir / "training_actions.csv.gz"
    if not report_path.is_file() or not actions_path.is_file():
        raise FileNotFoundError("formal R0 report/training action ledger is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    contracts = report.get("contracts", {})
    if (
        not report.get("formal")
        or contracts.get("P2b") != "forbidden"
        or contracts.get("action_outcomes_absent_from_training_manifest") is not True
        or int(contracts.get("matched_controls_preserved", -1)) != 2
    ):
        raise RuntimeError("R0 does not satisfy the frozen outcome-free control contract")
    actions = pd.read_csv(actions_path, low_memory=False)
    forbidden = {"corrected", "introduced", "target_rank", "target_margin", "random_margin"}
    leaked = forbidden.intersection(actions.columns)
    if leaked:
        raise RuntimeError(f"outcome columns leaked into R0: {sorted(leaked)}")
    selected = []
    for selector, attenuation, step in FIXED_POLICY["curriculum"]:
        block = actions.loc[
            actions["selector"].astype(str).eq(selector)
            & np.isclose(actions["attenuation"].astype(float), attenuation)
            & actions["step"].astype(int).eq(step)
        ].copy()
        if block.empty:
            raise RuntimeError(f"missing causal curriculum cell {selector}|{attenuation}|{step}")
        selected.append(block)
    selected_actions = pd.concat(selected, ignore_index=True)
    if selected_actions.duplicated(["query_index", "selector", "attenuation", "step"]).any():
        raise RuntimeError("R0 causal curriculum contains duplicate query-action rows")
    if len(selected_actions) < 10000 or selected_actions["query_index"].nunique() < 1000:
        raise RuntimeError("R0 causal curriculum coverage is unexpectedly small")

    invariant = [
        "query_index", "query_row", "query_ik14", "query_formula", "selector",
        "attenuation", "step", "hard_negative_row", "formula_fold",
    ]
    audits = {}
    materialized = {}
    for arm in ("clean_duplicate", "matched_random", "targeted"):
        frame, audit = materialize_causal_arm(selected_actions, arm)
        materialized[arm] = frame
        audits[arm] = audit
        if not frame[invariant].equals(selected_actions[invariant]):
            raise RuntimeError(f"{arm} changed a sampler/candidate invariant")
    if not (
        len(materialized["clean_duplicate"])
        == len(materialized["matched_random"])
        == len(materialized["targeted"])
    ):
        raise RuntimeError("causal arms have unequal action-row coverage")
    output = {
        "status": "noise_final_e4a_causal_attribution_preflight_passed",
        "formal": True,
        "r0_actions_sha256": sha256_file(actions_path),
        "curriculum_rows": int(len(selected_actions)),
        "curriculum_queries": int(selected_actions["query_index"].nunique()),
        "curriculum_identities": int(selected_actions["query_ik14"].astype(str).nunique()),
        "curriculum_formulas": int(selected_actions["query_formula"].astype(str).nunique()),
        "arms": audits,
        "contracts": {
            "only_action_path_differs": True,
            "same_rows_queries_policies_hard_negatives_and_folds": True,
            "control_assignment_outcome_free": True,
            "P2b": "forbidden",
            "P3_consumed": False,
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
