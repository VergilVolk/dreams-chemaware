"""Synthetic paired-contract test for the eight-arm ChemAware summary."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

import summarize_chemaware_mass_kernel_direct_arms as summary  # noqa: E402


def metrics(old_rank: np.ndarray, new_rank: np.ndarray) -> dict:
    old_hit = old_rank == 1
    new_hit = new_rank == 1
    return {
        "queries": int(len(old_rank)),
        "baseline_recall1": float(np.mean(old_hit)),
        "recall1": float(np.mean(new_hit)),
        "delta_recall1": float(np.mean(new_hit) - np.mean(old_hit)),
        "baseline_mrr": float(np.mean(1.0 / old_rank)),
        "mrr": float(np.mean(1.0 / new_rank)),
        "delta_mrr": float(np.mean(1.0 / new_rank) - np.mean(1.0 / old_rank)),
        "corrected": int(np.sum(~old_hit & new_hit)),
        "introduced": int(np.sum(old_hit & ~new_hit)),
        "delta_mean_margin": 0.01,
    }


def main() -> None:
    count = 240
    query = np.arange(count, dtype=np.int64)
    formula = np.asarray([f"F{i}" for i in range(count)])
    old_rank = np.ones(count, dtype=np.int16)
    old_rank[:60] = 2
    corrected = {
        "none": 0,
        "mass": 10,
        "rule_response": 12,
        "rule_mass": 35,
        "rule_mass_shifted": 9,
        "mass_error": 10,
        "rule_mass_error": 45,
        "rule_mass_shifted_error": 9,
    }
    with tempfile.TemporaryDirectory(prefix="chemaware-summary-") as temporary:
        root = Path(temporary) / "runs"
        for arm in summary.ARMS:
            arm_dir = root / arm
            arm_dir.mkdir(parents=True)
            new_rank = old_rank.copy()
            new_rank[:corrected[arm]] = 1
            teacher, beta, scope = summary.EXPECTED[arm]
            report = {
                "status": "ARM_COMPLETE",
                "optimization": {
                    "teacher_arm": teacher,
                    "teacher_beta": beta,
                    "teacher_scope": scope,
                },
                "selected_step": 100,
                "final_inner": metrics(old_rank, new_rank),
                "preservation": 0.999,
                "mean_clip_fraction": 0.1,
            }
            (arm_dir / "report.json").write_text(
                json.dumps(report), encoding="utf-8",
            )
            np.savez_compressed(
                arm_dir / "inner_per_query.npz", query=query, formula=formula,
                old_rank=old_rank, new_rank=new_rank,
            )
        output = Path(temporary) / "summary"
        original = sys.argv
        try:
            sys.argv = [
                "summary", "--root", str(root), "--output-dir", str(output),
                "--bootstrap-draws", "500", "--seed", "20260905",
            ]
            summary.main()
        finally:
            sys.argv = original
        result = json.loads((output / "report.json").read_text(encoding="utf-8"))
        assert result["status"] == "CAUSAL_CHEMISTRY_DEVELOPMENT_PASS"
        assert result["selected_target_for_outer_confirmation"] == "rule_mass_error"
        assert all(result["target_gates"]["rule_mass_error"].values())
        assert result["contracts"]["routed_mass_and_shifted_controls_are_scope_matched"]
        assert result["release_eligible"] is False

        # Absolute improvement or a tie with continuation must never be called
        # a causal chemistry pass.  Force both target arms to reproduce none on
        # every query while leaving their single-arm reports nominally good.
        with np.load(root / "none" / "inner_per_query.npz") as loaded:
            none_arrays = {key: loaded[key].copy() for key in loaded.files}
        for arm in summary.TARGETS:
            np.savez_compressed(
                root / arm / "inner_per_query.npz",
                query=none_arrays["query"], formula=none_arrays["formula"],
                old_rank=none_arrays["old_rank"], new_rank=none_arrays["new_rank"],
            )
        failed_output = Path(temporary) / "summary_tied_targets"
        original = sys.argv
        try:
            sys.argv = [
                "summary", "--root", str(root), "--output-dir", str(failed_output),
                "--bootstrap-draws", "500", "--seed", "20260905",
            ]
            summary.main()
        finally:
            sys.argv = original
        failed = json.loads((failed_output / "report.json").read_text(encoding="utf-8"))
        assert failed["status"] == "CAUSAL_CHEMISTRY_DEVELOPMENT_FAIL"
        assert failed["selected_target_for_outer_confirmation"] is None
        assert failed["release_eligible"] is False
        assert not failed["target_gates"]["rule_mass"]["continuation_point_estimate_positive"]
    print("PASS: chemical-rule paired summary contracts")


if __name__ == "__main__":
    main()
