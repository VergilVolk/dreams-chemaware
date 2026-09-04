"""Fail-closed validator for an E4-A direct shared-embedding run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    decision_path = args.output_dir / "decision.json"
    checkpoint_path = args.output_dir / "final_shared_encoder.pt"
    if not decision_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("E4-A output lacks decision or shared encoder checkpoint")
    report = json.loads(decision_path.read_text(encoding="utf-8"))
    if report.get("status") != "noise_final_e4a_direct_augmentation_complete":
        raise RuntimeError("unexpected E4-A status")
    contracts = report.get("contracts", {})
    expected = {
        "shared_query_reference_encoder": True,
        "model_weights_changed": True,
        "clean_and_augmented_raw_spectra_train_same_encoder": True,
        "inference_clean_spectrum_only": True,
        "P2b": "forbidden",
        "P3_consumed": False,
    }
    for key, value in expected.items():
        if contracts.get(key) != value:
            raise RuntimeError(f"E4-A contract failed: {key}={contracts.get(key)!r}")
    configuration = report.get("configuration", {})
    causal_arm = configuration.get("causal_arm", "legacy")
    if causal_arm != "legacy":
        if causal_arm not in {"clean_duplicate", "matched_random", "targeted"}:
            raise RuntimeError(f"unknown E4-A causal arm: {causal_arm}")
        causal_audit = report.get("causal_action_audit", {})
        if causal_audit.get("arm") != causal_arm or int(causal_audit.get("rows", 0)) < 1000:
            raise RuntimeError("E4-A causal action audit is missing or too small")
        causal_expected = {
            "causal_attribution_arm": causal_arm,
            "causal_arm_changes_only_action_view": True,
            "matched_control_selection_uses_outcome": False,
            "causal_sampler_keys_arm_invariant": True,
            "causal_candidate_references_arm_invariant": True,
        }
        for key, value in causal_expected.items():
            if contracts.get(key) != value:
                raise RuntimeError(f"E4-A causal contract failed: {key}")
        if causal_arm == "matched_random" and set(
            causal_audit.get("matched_control_index_counts", {})
        ) != {"0", "1"}:
            raise RuntimeError("E4-A matched-random arm did not use both frozen controls")
    guided_policy = configuration.get("guided_noise_policy", "none")
    expected_outcome_loss = guided_policy == "selected"
    if contracts.get("action_outcomes_used_in_loss_or_sample_weight") is not expected_outcome_loss:
        raise RuntimeError(
            "E4-A outcome-loss contract disagrees with selected guided teacher mode"
        )
    transfer_mode = configuration.get("direct_transfer_mode", "symmetric")
    reference_mode = configuration.get("rank_reference_mode", "shared")
    if transfer_mode == "official_action":
        if contracts.get("teacher") != "frozen_official_raw_action_embedding":
            raise RuntimeError("official-action transfer did not declare its frozen raw-action teacher")
        if contracts.get("official_action_targets_frozen_before_optimizer") is not True:
            raise RuntimeError("official-action targets were not frozen before optimization")
        if contracts.get("action_selection") != "fixed":
            raise RuntimeError("E8 may not combine official-action transfer with outcome-mined actions")
    elif guided_policy == "selected":
        if contracts.get("teacher") != "outer_fold_isolated_privileged_action_margin":
            raise RuntimeError("selected E14 mode did not declare its outer-fold teacher")
        if report.get("guided_teacher_replay", {}).get(
            "action_margin_max_abs_error", 1.0
        ) > 2e-4:
            raise RuntimeError("selected E14 action teacher did not replay exactly")
        if not configuration.get("initial_student_checkpoint"):
            raise RuntimeError("selected E14 mode did not declare mature initialization")
    elif contracts.get("teacher") != "forbidden":
        raise RuntimeError(f"unexpected direct-transfer teacher: {contracts.get('teacher')!r}")
    if reference_mode == "official":
        if contracts.get("official_reference_anchors_training_only") is not True:
            raise RuntimeError("official reference mode lacks its training-only anchor contract")
    elif contracts.get("official_reference_anchors_training_only") is not False:
        raise RuntimeError("shared reference mode unexpectedly declares official anchors")
    held = report.get("held_clean", {})
    for key in ("baseline_recall1", "recall1", "delta_recall1", "corrected", "introduced"):
        if key not in held:
            raise RuntimeError(f"E4-A held-clean metric missing: {key}")
    positive_weight = float(configuration.get("positive_stream_weight", 0.0))
    if positive_weight > 0:
        for key in (
            "real_cross_condition_positive_pairs_train_same_encoder",
            "positive_pair_selection_uses_model_outcome",
        ):
            if key not in contracts:
                raise RuntimeError(f"P/N/S contract missing: {key}")
        if contracts["real_cross_condition_positive_pairs_train_same_encoder"] is not True:
            raise RuntimeError("P-arm did not train the shared encoder")
        if contracts["positive_pair_selection_uses_model_outcome"] is not False:
            raise RuntimeError("P-arm pair selection used an outcome")
        for key in ("held_cross_condition_positive", "held_positive_clean"):
            if key not in held:
                raise RuntimeError(f"P/N/S held metric missing: {key}")
    print(
        "[validate_noise_final_e4a_direct_augmentation] PASS "
        f"policy={report['configuration']['policy']} "
        f"dR1={held['delta_recall1']:+.4f} C/I={held['corrected']}/{held['introduced']}"
    )


if __name__ == "__main__":
    main()
