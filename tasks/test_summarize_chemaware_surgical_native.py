"""CPU contracts for the frozen ChemAware surgical release decision."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def row(name: str, recall1: float, micro: float = 0.95, macro: float = 0.96) -> dict:
    return {
        "name": name,
        "checkpoint": f"/{name}.ckpt",
        "metrics": {
            "recall1": recall1,
            "recall3": 0.99,
            "mrr": 0.96,
            "micro_auc": micro,
            "macro_auc": macro,
        },
    }


def main() -> None:
    official = row("official", 0.90, 0.94, 0.95)
    stage1 = row("stage1_base", 0.92)
    correct = row("surgical_correct", 0.951)
    correct["paired_vs_official"] = {
        "delta_recall1": 0.051,
        "delta_mrr": 0.02,
        "corrected_at_1": 110,
        "introduced_at_1": 10,
        "formula_cluster_bootstrap_delta_recall1_ci95": [0.04, 0.06],
    }
    correct["paired_vs_stage1_base"] = {
        "delta_recall1": 0.031,
        "delta_mrr": 0.01,
        "corrected_at_1": 70,
        "introduced_at_1": 10,
        "formula_cluster_bootstrap_delta_recall1_ci95": [0.02, 0.04],
    }
    report = {
        "formula_role": 3,
        "results": [
            official, stage1, correct,
            row("null_a", 0.925), row("null_b", 0.93), row("null_c", 0.92),
        ],
    }
    script = Path(__file__).resolve().parent / "summarize_chemaware_surgical_native.py"
    with tempfile.TemporaryDirectory(prefix="chem_surgical_summary_") as raw:
        directory = Path(raw)
        source = directory / "evaluation.json"
        output = directory / "decision.json"
        source.write_text(json.dumps(report), encoding="utf-8")
        subprocess.run([
            sys.executable, str(script), "--evaluation", str(source),
            "--output", str(output), "--target-delta-recall1", "0.05",
        ], check=True, capture_output=True, text=True)
        decision = json.loads(output.read_text(encoding="utf-8"))
        assert decision["status"] == "CHEMAWARE_SURGICAL_FIVE_PP_TARGET_REACHED"
        assert decision["method_valid"] is True
        assert decision["five_pp_target_reached"] is True

        correct["metrics"]["micro_auc"] = 0.94
        source.write_text(json.dumps(report), encoding="utf-8")
        subprocess.run([
            sys.executable, str(script), "--evaluation", str(source),
            "--output", str(output), "--target-delta-recall1", "0.05",
        ], check=True, capture_output=True, text=True)
        decision = json.loads(output.read_text(encoding="utf-8"))
        assert decision["status"] == "CHEMAWARE_SURGICAL_RELEASE_FAIL"
        assert decision["five_pp_target_reached"] is False
    print("PASS: ChemAware surgical release-decision contracts")


if __name__ == "__main__":
    main()
