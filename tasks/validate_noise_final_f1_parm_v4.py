"""Fail-closed validator for one F1-v4 P-arm pilot run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    required = [
        args.run_dir / "decision.json",
        args.run_dir / "adapter.pt",
        args.run_dir / "outer_full_predictions.npz",
        args.run_dir / "outer_challenge_predictions.npz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    decision = json.loads(required[0].read_text(encoding="utf-8"))
    if decision.get("status") != "noise_final_f1_parm_fold_complete":
        raise RuntimeError("unexpected F1 decision status")
    if decision.get("P2b_used") is not False:
        raise RuntimeError("P2b contamination in F1-v4")
    if decision.get("query_reference_encoder_shared") is not True:
        raise RuntimeError("F1-v4 is not a shared query/reference encoder")
    if decision.get("teacher_policy") != "corrected_only":
        raise RuntimeError("F1-v4 must use corrected-only P-arm supervision")
    if decision.get("objective") != "teacher_margin":
        raise RuntimeError("F1-v4 must use the fixed teacher-margin objective")
    if int(decision.get("rescue_train_examples", 0)) < 1:
        raise RuntimeError("F1-v4 has no rescue examples")
    if int(decision.get("safety_train_examples", 0)) < 1:
        raise RuntimeError("F1-v4 has no safety examples")
    if not decision.get("history"):
        raise RuntimeError("F1-v4 history is empty")

    zero = decision["history"][0]
    zero_full = zero["inner_full_graph"]
    if int(zero_full.get("corrected", -1)) != 0 or int(zero_full.get("introduced", -1)) != 0:
        raise RuntimeError("zero-init adapter does not reproduce the frozen baseline")
    if abs(float(zero_full.get("delta_recall1", 1.0))) > 1e-12:
        raise RuntimeError("zero-init Recall@1 differs from the frozen baseline")

    checkpoint = torch.load(required[1], map_location="cpu", weights_only=False)
    if checkpoint.get("P2b_used") is not False:
        raise RuntimeError("checkpoint contains a P2b contract violation")
    if checkpoint.get("query_reference_encoder_shared") is not True:
        raise RuntimeError("checkpoint is not a shared encoder adapter")
    if int(checkpoint.get("seed", -1)) != int(decision["seed"]):
        raise RuntimeError("checkpoint/decision seed mismatch")
    if int(checkpoint.get("outer_fold", -1)) != int(decision["outer_fold"]):
        raise RuntimeError("checkpoint/decision fold mismatch")

    for path in required[2:]:
        with np.load(path) as body:
            old_rank = np.asarray(body["old_rank"])
            new_rank = np.asarray(body["new_rank"])
        if old_rank.shape != new_rank.shape or old_rank.size == 0:
            raise RuntimeError(f"malformed prediction artifact: {path}")
        if np.any(old_rank < 1) or np.any(new_rank < 1):
            raise RuntimeError(f"invalid rank in prediction artifact: {path}")

    outer = decision["outer_full_graph"]
    challenge = decision["outer_c1_challenge"]
    print(json.dumps({
        "status": "noise_final_f1_parm_v4_validation_passed",
        "seed": int(decision["seed"]),
        "outer_fold": int(decision["outer_fold"]),
        "best_epoch": int(decision["best_epoch"]),
        "outer_full_delta_recall1": float(outer["delta_recall1"]),
        "outer_full_corrected": int(outer["corrected"]),
        "outer_full_introduced": int(outer["introduced"]),
        "outer_challenge_identity_equal_delta_recall1": float(
            challenge["identity_equal_delta_recall1"]
        ),
        "outer_challenge_corrected": int(challenge["corrected"]),
        "outer_challenge_introduced": int(challenge["introduced"]),
        "scientific_pass_not_asserted": True,
    }, indent=2))


if __name__ == "__main__":
    main()
