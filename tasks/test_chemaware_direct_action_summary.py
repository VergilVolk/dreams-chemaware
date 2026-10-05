"""End-to-end synthetic contract for four-arm causal summarization."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

from summarize_chemaware_direct_action_views import ARMS, main


def main_test() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        rows = pd.DataFrame({
            "query_index": list(range(8)),
            "formula": [f"F{i}" for i in range(8)],
            "identity": [f"I{i}" for i in range(8)],
            "initial_rank": [2] * 8,
            "final_rank": [2] * 8,
            "initial_margin": [-0.1] * 8,
            "final_margin": [-0.1] * 8,
        })
        optimization = {
            "epochs": 2, "batch_queries": 4, "max_action_identities": 512,
            "max_safety_identities": 512,
            "backbone_lr": 2e-6, "head_lr": 1e-5, "temperature": 0.1,
            "training_objective": "direct_projected_guarded",
            "lambda_margin_floor": 2.0,
            "lambda_preserve": 20.0,
            "grad_clip": 5.0,
            "maximum_action_gradient_ratio": 0.25,
        }
        contracts = {
            "teacher_score_loss": False,
            "teacher_embedding_loss": False,
            "clean_action_embedding_consistency": False,
            "official_embedding_preservation_loss": True,
            "official_margin_floor_loss": True,
            "action_gradient_nonconflicting_with_primary": True,
            "action_gradient_norm_cap": 0.25,
        }
        for arm in ARMS:
            directory = root / arm
            directory.mkdir()
            arm_rows = rows.copy()
            if arm == "correct_synthetic":
                arm_rows["final_rank"] = 1
            arm_rows.to_csv(directory / "inner_per_query.csv.gz", index=False, compression="gzip")
            report = {
                "status": "PASS" if arm == "correct_synthetic" else "FAIL",
                "preflight": {"arm": arm, "contracts": contracts},
                "optimization": optimization,
                "scope": {"outer_fold_evaluated": False},
                "final": {"inner_all": {"delta_recall1": 1.0 if arm == "correct_synthetic" else 0.0}},
                "gates": {"synthetic_gate": arm == "correct_synthetic"},
            }
            (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
            (directory / "final_shared_encoder.pt").write_bytes(b"synthetic checkpoint")
            (directory / "COMPLETE.json").write_text(json.dumps({
                "status": "CHEMAWARE_DIRECT_ARM_COMPLETE",
                "arm": arm,
                "report_status": report["status"],
            }), encoding="utf-8")
        stage1_output = root / "stage1.json"
        previous = sys.argv
        try:
            sys.argv = [
                "summarize", "--root", str(root), "--output", str(stage1_output),
                "--stage", "primary",
            ]
            main()
        finally:
            sys.argv = previous
        stage1 = json.loads(stage1_output.read_text(encoding="utf-8"))
        assert stage1["status"] == "CHEMAWARE_DIRECT_ACTION_STAGE1_PASS"
        assert stage1["pass_to_matched_controls"] is True

        output = root / "summary.json"
        try:
            sys.argv = ["summarize", "--root", str(root), "--output", str(output)]
            main()
        finally:
            sys.argv = previous
        result = json.loads(output.read_text(encoding="utf-8"))
        assert result["status"] == "CHEMAWARE_DIRECT_ACTION_CAUSAL_PASS"
        assert all(result["gates"].values())
    print("direct action causal summary contracts passed")


if __name__ == "__main__":
    main_test()
