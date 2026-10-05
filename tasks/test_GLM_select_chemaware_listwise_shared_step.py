"""Contracts for Arm-1-only shared-step selection."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parent / "GLM_select_chemaware_listwise_shared_step.py"


def row(name: str, checkpoint: Path, corrected: int, introduced: int,
        delta: float, delta_mrr: float) -> dict[str, object]:
    return {
        "name": name,
        "checkpoint": str(checkpoint),
        "metrics": {},
        "paired_vs_phaseA_2pp": {
            "corrected_at_1": corrected,
            "introduced_at_1": introduced,
            "delta_recall1": delta,
            "delta_mrr": delta_mrr,
        },
    }


def test_arm1_alone_selects_one_shared_step(tmp_path: Path) -> None:
    report = {
        "formula_role": 2,
        "formula_role_contract_passed": True,
        "results": [
            {"name": "official", "checkpoint": "official.pt", "metrics": {}},
            {"name": "phaseA_2pp", "checkpoint": "phasea.ckpt", "metrics": {}},
            row("arm1_step900", tmp_path / "a1-900.ckpt", 5, 1, .01, .005),
            row("arm1_step1800", tmp_path / "a1-1800.ckpt", 9, 2, .012, .006),
            row("arm2_step900", tmp_path / "a2-900.ckpt", 99, 0, .5, .5),
            row("arm2_step1800", tmp_path / "a2-1800.ckpt", 0, 99, -.5, -.5),
        ],
    }
    source = tmp_path / "eval.json"
    source.write_text(json.dumps(report), encoding="utf-8")
    output = tmp_path / "selected.json"
    subprocess.run([
        sys.executable, str(SCRIPT), "--evaluation", str(source),
        "--expected-step", "900", "--expected-step", "1800",
        "--output", str(output),
    ], check=True, capture_output=True, text=True)
    selected = json.loads(output.read_text(encoding="utf-8"))
    assert selected["selected"]["step"] == 1800
    assert selected["chemical_arm_used_for_selection"] is False
    assert output.with_suffix(".arm1_checkpoint.txt").read_text().strip().endswith(
        "a1-1800.ckpt"
    )
    assert output.with_suffix(".arm2_checkpoint.txt").read_text().strip().endswith(
        "a2-1800.ckpt"
    )


def test_missing_paired_arm2_step_fails_closed(tmp_path: Path) -> None:
    report = {
        "formula_role": 2,
        "formula_role_contract_passed": True,
        "results": [
            {"name": "phaseA_2pp", "checkpoint": "phasea.ckpt", "metrics": {}},
            row("arm1_step900", tmp_path / "a1.ckpt", 1, 0, .01, .01),
        ],
    }
    source = tmp_path / "eval.json"
    source.write_text(json.dumps(report), encoding="utf-8")
    result = subprocess.run([
        sys.executable, str(SCRIPT), "--evaluation", str(source),
        "--expected-step", "900", "--output", str(tmp_path / "out.json"),
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "paired Arm-2 checkpoint is absent" in result.stderr
