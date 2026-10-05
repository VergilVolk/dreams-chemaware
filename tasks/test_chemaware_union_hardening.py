"""CPU semantic contracts for the hardened ChemAware union continuation."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from audit_chemaware_role3_confirmation import decision
from chemaware_native_adam import adam_state_steps, restore_native_adam_state


def adam_restore_contract() -> None:
    source_model = torch.nn.Linear(2, 1)
    source = torch.optim.Adam(source_model.parameters(), lr=2e-6, weight_decay=0.0)
    source_model(torch.ones(1, 2)).sum().backward()
    source.step()
    state = source.state_dict()

    target_model = torch.nn.Linear(2, 1)
    target = torch.optim.Adam(target_model.parameters(), lr=2e-6, weight_decay=0.0)
    initial = restore_native_adam_state(target, state, 2e-6, 0.0)
    target.zero_grad(set_to_none=True)
    target_model(torch.ones(1, 2)).sum().backward()
    target.step()
    final = adam_state_steps(target.state_dict())
    assert all(final[key] == value + 1 for key, value in initial.items())


def role3_protected_baseline_contract() -> None:
    common = {
        "recall3": 0.99, "micro_auc": 0.95, "macro_auc": 0.96,
        "mrr": 0.95,
    }
    report = {
        "formula_role": 3,
        "formula_role_contract_passed": True,
        "results": [
            {"name": "seed", "metrics": {**common, "recall1": 0.92}},
            {"name": "phaseA_2pp", "metrics": {**common, "recall1": 0.93}},
            {
                "name": "candidate",
                "metrics": {**common, "recall1": 0.925},
                "paired_vs_seed": {
                    "delta_recall1": 0.005, "delta_mrr": 0.0,
                    "corrected_at_1": 5, "introduced_at_1": 1,
                    "formula_cluster_bootstrap_delta_recall1_ci95": [0.001, 0.01],
                },
            },
        ],
    }
    result = decision(report, "seed", "candidate", "phaseA_2pp")
    assert not result["confirmed"]
    assert not result["gates"]["protected_recall1_nonnegative"]


def report_role_gate_contract() -> None:
    # A downstream selector/auditor must reject a report that merely echoes a
    # role label without the exact-policy proof.
    report = {"formula_role": 3, "results": []}
    try:
        decision(report, "seed", "candidate", "phaseA_2pp")
    except RuntimeError as error:
        assert "exact frozen panel" in str(error)
    else:
        raise AssertionError("unproven formula-role report was accepted")


def role2_protected_fallback_contract() -> None:
    metrics = {
        "recall3": 0.99, "mrr": 0.95, "micro_auc": 0.95, "macro_auc": 0.96,
    }
    report = {
        "formula_role": 2, "formula_role_contract_passed": True,
        "results": [
            {"name": "phaseA_2pp", "checkpoint": "phaseA.ckpt",
             "metrics": {**metrics, "recall1": 0.93}},
            {"name": "seed", "checkpoint": "step-000750.ckpt",
             "metrics": {**metrics, "recall1": 0.92}},
            {"name": "candidate", "checkpoint": "step-001000.ckpt",
             "metrics": {**metrics, "recall1": 0.925},
             "paired_vs_seed": {
                 "delta_recall1": 0.005, "delta_mrr": 0.0,
                 "corrected_at_1": 5, "introduced_at_1": 1,
                 "formula_cluster_bootstrap_delta_recall1_ci95": [0.001, 0.01],
             }},
        ],
    }
    with tempfile.TemporaryDirectory(prefix="chem_selector_") as raw:
        root = Path(raw)
        evaluation = root / "evaluation.json"
        output = root / "selection.json"
        evaluation.write_text(json.dumps(report), encoding="utf-8")
        subprocess.run([
            sys.executable, str(Path(__file__).with_name(
                "select_chemaware_residual_checkpoint.py"
            )),
            "--evaluation", str(evaluation), "--base-name", "seed",
            "--protected-baseline-name", "phaseA_2pp",
            "--exclude-name", "phaseA_2pp", "--require-positive-formula-ci",
            "--output", str(output),
        ], check=True, capture_output=True, text=True)
        selected = json.loads(output.read_text(encoding="utf-8"))
        assert selected["selected"]["name"] == "phaseA_2pp"
        assert not selected["advanced_beyond_base"]


def main() -> None:
    adam_restore_contract()
    role3_protected_baseline_contract()
    report_role_gate_contract()
    role2_protected_fallback_contract()
    print("PASS: ChemAware union hardening semantic contracts", flush=True)


if __name__ == "__main__":
    main()
