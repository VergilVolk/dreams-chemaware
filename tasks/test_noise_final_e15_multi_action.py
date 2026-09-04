"""Behavioral tests for E15 multi-action, risk and sampling contracts."""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

import build_noise_final_e15_multi_action_ledger as ledger  # noqa: E402
import noise_final_e15_core as core  # noqa: E402


def fake_items(count: int = 160) -> list[core.ExposureItem]:
    return [core.ExposureItem(
        query_index=index,
        identity=f"IK{index // 2:04d}",
        formula=f"F{index % 41:03d}",
        source=("A4", "C1", "E14", "R0")[index % 4],
        family=("candidate", "prototype", "transfer")[index % 3],
        action_id=f"action-{index}",
        supervision_kind=("corrective", "harmful")[index % 2],
    ) for index in range(count)]


def main() -> None:
    for path in (
        ROOT / "tasks/noise_final_e15_core.py",
        ROOT / "tasks/build_noise_final_e15_multi_action_ledger.py",
        ROOT / "tasks/validate_noise_final_e15_multi_action_ledger.py",
    ):
        ast.parse(path.read_text(encoding="utf-8"))

    items = fake_items()
    epoch, report = core.bounded_stratified_epoch(
        items, np.random.default_rng(7), maximum_exposure=1,
    )
    keys = [(item.query_index, item.action_id, item.supervision_kind) for item in epoch]
    if len(epoch) != len(items) or len(keys) != len(set(keys)):
        raise RuntimeError("E15 bounded sampler dropped or recycled an action")
    if report["maximum_exposure"] != 1:
        raise RuntimeError("E15 bounded sampler exceeded one exposure")

    batches = core.stratified_calibration_indices(
        items, np.random.default_rng(9), microbatches=32, batch_size=4,
    )
    flattened = [index for batch in batches for index in batch]
    if len(batches) != 32 or len(flattened) != 128 or len(set(flattened)) != 128:
        raise RuntimeError("E15 calibration is not 32 x 4 unique observations")

    clean = torch.tensor([-0.10, 0.02], requires_grad=True)
    baseline = torch.tensor([-0.08, 0.03])
    preservation = torch.tensor([0.990, 0.999], requires_grad=True)
    risk = core.risk_protection_loss(clean, baseline, preservation)
    risk.backward()
    if risk.item() <= 0 or clean.grad is None or preservation.grad is None:
        raise RuntimeError("E15 risk branch does not protect clean geometry")
    risk_source = inspect.getsource(core.risk_protection_loss)
    forbidden_risk = ("action_margin", "self_transfer", "consistency")
    found_risk = [token for token in forbidden_risk if token in risk_source]
    if found_risk:
        raise RuntimeError(f"harmful action imitation leaked into E15 risk loss: {found_risk}")

    corrective = core.corrective_margin_loss(
        torch.tensor([-0.10]), torch.tensor([0.08]), torch.tensor([-0.10]),
        torch.tensor([0.18]),
    )
    weaker = core.corrective_margin_loss(
        torch.tensor([-0.10]), torch.tensor([-0.10]), torch.tensor([-0.10]),
        torch.tensor([0.0]),
    )
    if not corrective < weaker:
        raise RuntimeError("E15 corrective loss does not reward a better action margin")

    projected, diagnostic = core.project_corrective_against_risk(
        [torch.tensor([1.0, -1.0])], [torch.tensor([-1.0, 0.0])],
    )
    if not diagnostic["conflict"] or float(torch.dot(projected[0], torch.tensor([-1.0, 0.0]))) < -1e-7:
        raise RuntimeError("E15 risk gradient projection failed")
    tiny, tiny_diagnostic = core.project_corrective_against_risk(
        [torch.tensor([1.0])], [torch.tensor([-1e-9])], minimum_risk_norm=1e-6,
    )
    if tiny_diagnostic["risk_projection_active"] or tiny_diagnostic["conflict"]:
        raise RuntimeError("numerical risk gradient activated E15 projection")
    if not torch.equal(tiny[0], torch.tensor([1.0])):
        raise RuntimeError("inactive E15 risk projection changed corrective gradient")

    frame = pd.DataFrame([
        {
            "query_index": query, "source": source, "action_family": family,
            "action_id": f"{query}-{source}-{offset}", "replicated_formula_folds": 3,
            "conditional_identities": 20 + offset, "margin_delta": 0.01 * (offset + 1),
        }
        for query in range(3)
        for offset, (source, family) in enumerate((
            ("A4", "delete"), ("C1", "prototype"), ("E14", "transfer"),
            ("R0", "candidate"), ("A4", "delete"), ("C1", "prototype"),
        ))
    ])
    selected = ledger.select_diverse(frame, maximum=4, harmful=False)
    counts = selected.groupby("query_index").size()
    if not counts.eq(4).all() or selected.groupby("query_index")["source"].nunique().min() < 3:
        raise RuntimeError("E15 diverse top-K collapsed back to one action/family")

    builder_source = inspect.getsource(ledger)
    forbidden_builder = (
        "np.argmax(result_margin",
        'selected["query_index"].duplicated().any()',
        '"one_selected_action_per_query": True',
    )
    found_builder = [token for token in forbidden_builder if token in builder_source]
    if found_builder:
        raise RuntimeError(f"legacy single-action behavior entered E15: {found_builder}")

    # Regression for the server failure on 2026-08-31: the historical A4
    # decision stores formality at integrity.formal, not at the top level.
    a4_decision = ROOT / "data/validation/g8r_noise_v3_a4_exact_peak_scan/decision.json"
    if a4_decision.is_file():
        a4_report = ledger.check_report(
            a4_decision, "noise_v3_a4_exact_peak_scan_decision",
            formal_path=("integrity", "formal"),
        )
        integrity = a4_report.get("integrity", {})
        expected = {
            "official_errors_scanned": 1805,
            "safety_controls": 3193,
            "fragment_actions": 206288,
            "exact_variants": 825152,
        }
        if any(int(integrity.get(key, -1)) != value for key, value in expected.items()):
            raise RuntimeError("local A4 decision does not reproduce frozen integrity counts")

    sbatch = (ROOT / "tasks/run_noise_final_e15_multi_action_ledger.sbatch").read_text(
        encoding="utf-8"
    )
    required_sbatch = (
        "#SBATCH --partition=gpu",
        "#SBATCH --gpus=1",
        "set -euo pipefail",
        '[[ ! -e "$OUTPUT" ]]',
        "python -u tasks/test_noise_final_e15_preflight.py",
        "python -u tasks/test_noise_final_e15_multi_action.py",
        "python -u tasks/audit_noise_final_e15_preflight.py",
        "python -u tasks/build_noise_final_e15_multi_action_ledger.py",
        "python -u tasks/validate_noise_final_e15_multi_action_ledger.py",
    )
    missing_sbatch = [token for token in required_sbatch if token not in sbatch]
    if missing_sbatch:
        raise RuntimeError(f"E15-M0 sbatch contract is incomplete: {missing_sbatch}")
    execution = required_sbatch[4:]
    positions = [sbatch.index(token) for token in execution]
    if positions != sorted(positions):
        raise RuntimeError("E15-M0 tests/audit/build/validation order drifted")
    forbidden_sbatch = ("python -u tasks/train_noise", "--array=", "--mem=", "python - <<")
    found_sbatch = [token for token in forbidden_sbatch if token in sbatch]
    if found_sbatch:
        raise RuntimeError(f"E15-M0 unexpectedly launches training: {found_sbatch}")
    print("[test_noise_final_e15_multi_action] PASS", flush=True)


if __name__ == "__main__":
    main()
