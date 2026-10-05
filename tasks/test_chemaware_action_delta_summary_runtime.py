"""Exercise action-delta Stage-1 summary before any GPU model is constructed."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import summarize_chemaware_action_delta_transfer as summary


def arm_report(arm: str, delta: float) -> dict:
    optimization = {
        "training_objective": "direct_action_delta_transfer",
        "seed": 20260904,
        "fold_seed": 20260935,
        "inner_fold": 3,
        "outer_fold": 4,
        "epochs": 2,
        "batch_queries": 4,
        "max_action_identities": 512,
        "max_safety_identities": 512,
        "max_eval_identities": 0,
        "unfreeze_blocks": 1,
        "backbone_lr": 2e-6,
        "head_lr": 1e-5,
        "weight_decay": 1e-4,
        "temperature": 0.1,
        "lambda_clean_rank": 1.0,
        "lambda_action_rank": 1.0,
        "safety_stream_weight": 1.0,
        "lambda_margin_floor": 2.0,
        "lambda_preserve": 20.0,
        "margin_floor_slack": 0.005,
        "grad_clip": 5.0,
        "action_delta_alpha": 0.5,
        "action_delta_huber": 0.02,
        "base_chemical_operator_split": True,
        "separate_optimizer_moments": True,
        "chemical_weight_decay": 0.0,
    }
    action_bank = {
        "bank_sha256": "a" * 64,
        "selected_setting": 6,
        "mode": "conflict_attenuate",
        "strength": 0.75,
        "top_k": 3,
        "action_generation_seed": 20260935,
        "eligible_training_actions": 20,
        "corrective_rank_actions": 10,
        "corrective_margin_actions": 10,
        "margin_role_dose_calibration": "median positive corrective-rank margin gain",
        "formula_equal_training_mass": True,
    }
    return {
        "status": "PASS",
        "preflight": {
            "arm": arm,
            "provenance": {
                "graph_sha256": "b" * 64,
                "evaluation_manifest_sha256": "c" * 64,
                "official_checkpoint_sha256": "d" * 64,
                "architecture_checkpoint_sha256": "0" * 64,
                "teacher_report_sha256": "e" * 64,
                "teacher_predictions_sha256": "f" * 64,
            },
            "contracts": {
                "qualified_action_effect_target": True,
                "trainable_action_view_forward": False,
                "chemical_reference_gradient": False,
                "candidate_centred_action_effect": True,
                "formula_equal_action_mass": True,
                "unsupported_action_role_weight_zero": True,
                "outer_fold_evaluated": False,
            },
            "action_bank": action_bank | {
                "training_arm": arm,
                "training_arm_is_matched_control": arm != "correct_synthetic",
            },
        },
        "optimization": optimization,
        "final": {
            "broad_inner": {
                "delta_recall1": delta,
                "delta_recall5": 0.0,
                "delta_recall10": 0.0,
                "delta_recall20": 0.0,
                "delta_recall50": 0.0,
                "delta_mrr": delta / 2,
                "delta_macro_auc": delta / 3,
                "delta_micro_auc": delta / 4,
                "preservation_mean": 0.999,
            }
        },
        "history": [{"train": {"clip_fraction": 0.0}}],
        "gradient_clipping": {"maximum_stream_fraction": 0.0},
        "chemical_optimizer_signal": {
            "gate_passed": True,
            "all_losses_nonincreasing": True,
            "all_group_geometries_valid": True,
            "observations": 4,
            "expected_observations": 4,
        },
        "chemical_target_fit": {
            "initial": {"formula_weighted_huber": 0.0 if arm == "clean_duplicate" else 0.2},
            "final": {"formula_weighted_huber": 0.0 if arm == "clean_duplicate" else 0.1},
            "formula_weighted_huber_reduction": 0.0 if arm == "clean_duplicate" else 0.1,
        },
    }


def write_arm(root: Path, arm: str, corrected: bool) -> None:
    directory = root / arm
    directory.mkdir()
    n = 20
    initial = np.full(n, 2, dtype=np.int64)
    final = np.ones(n, dtype=np.int64) if corrected else initial.copy()
    delta = float(np.mean(final == 1) - np.mean(initial == 1))
    pd.DataFrame({
        "query_index": np.arange(n),
        "formula": [f"C{i + 1}H{i + 2}" for i in range(n)],
        "identity": [f"identity_{i}" for i in range(n)],
        "initial_rank": initial,
        "final_rank": final,
        "initial_margin": np.full(n, -0.1),
        "final_margin": np.full(n, 0.1 if corrected else -0.1),
        "candidate_count": np.full(n, 3),
    }).to_csv(directory / "broad_inner_per_query.csv.gz", index=False, compression="gzip")
    (directory / "final_shared_encoder.pt").write_bytes(b"runtime-contract-placeholder")
    (directory / "COMPLETE.json").write_text(json.dumps({
        "status": "CHEMAWARE_DIRECT_ARM_COMPLETE", "arm": arm,
        "report_status": "PASS",
    }))
    (directory / "report.json").write_text(json.dumps(arm_report(arm, delta)))
    (directory / "action_target_audit.json").write_text(json.dumps({
        "qualified_actions": n,
        "rank_replay_mismatches": 0,
        "maximum_margin_replay_error": 0.0,
        "zero_target_actions": n if arm == "clean_duplicate" else 0,
        "frozen_action_corrected": n if arm == "correct_synthetic" else 0,
        "frozen_action_introduced": 0,
        "frozen_action_mean_margin_gain": 0.2 if arm == "correct_synthetic" else 0.0,
        "frozen_action_positive_margin_fraction": 1.0 if arm == "correct_synthetic" else 0.0,
        "action_view_trainable_forward": False,
        "chemical_reference_gradient": False,
    }))


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        write_arm(root, "clean_duplicate", False)
        write_arm(root, "correct_synthetic", True)
        write_arm(root, "candidate_swapped", False)
        write_arm(root, "peak_permuted", False)

        def run(stage: str) -> dict:
            output = root / f"{stage}.json"
            original = sys.argv
            try:
                sys.argv = [
                    "summarize_chemaware_action_delta_transfer.py",
                    "--root", str(root), "--output", str(output),
                    "--stage", stage, "--bootstrap-draws", "10000",
                ]
                with contextlib.redirect_stdout(io.StringIO()):
                    summary.main()
            finally:
                sys.argv = original
            return json.loads(output.read_text())

        primary = run("primary")
        assert primary["status"] == "CHEMAWARE_ACTION_DELTA_STAGE1_PASS"
        assert primary["pass_to_matched_controls"] is True
        assert primary["correct_vs_controls"]["clean_duplicate"][
            "formula_bootstrap_recall1"
        ]["formula_cluster_bootstrap_95ci"][0] > 0
        full = run("full")
        assert full["status"] == "CHEMAWARE_ACTION_DELTA_CAUSAL_PASS"
        assert full["pass_to_additional_seed"] is True
        assert set(full["correct_vs_controls"]) == {
            "clean_duplicate", "candidate_swapped", "peak_permuted",
        }
        broken_path = root / "candidate_swapped" / "report.json"
        broken = json.loads(broken_path.read_text())
        broken["chemical_optimizer_signal"]["gate_passed"] = False
        broken_path.write_text(json.dumps(broken))
        original = sys.argv
        try:
            sys.argv = [
                "summarize_chemaware_action_delta_transfer.py",
                "--root", str(root), "--output", str(root / "must_fail.json"),
                "--stage", "full", "--bootstrap-draws", "10000",
            ]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    summary.main()
            except RuntimeError as error:
                assert "realized chemical-optimizer audit" in str(error)
            else:
                raise AssertionError("a failed optimizer audit reached causal summary")
        finally:
            sys.argv = original
    print("PASS: ChemAware action-delta Stage-1/full summary runtime contracts")


if __name__ == "__main__":
    main()
