from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from grand_unified_evidence_core import AllModuleEvidenceModel


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_external_chain_binds_artifacts_and_opening_ledger_is_one_time(tmp_path):
    modules = ("official_dreams", "weighted_spectral_entropy")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({
        "schema": "dreams_grand_unified_component_registry_v3",
        "modules": [{"name": name, "status": "enabled"} for name in modules],
    }), encoding="utf-8")
    model_path = tmp_path / "model.pt"
    model = AllModuleEvidenceModel(modules)
    torch.save({"state_dict": model.state_dict(), "module_names": modules, "seed": 7}, model_path)
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({
        "freeze_contract_status": "DEVELOPMENT_MODEL_FROZEN_EXTERNAL_UNOPENED",
        "external_test_opened": False,
        "frozen_model_sha256": sha256(model_path),
        "module_registry_sha256": sha256(registry),
        "module_names": list(modules),
    }), encoding="utf-8")

    ptr = np.asarray([0, 2, 4], dtype=np.int64)
    labels = np.asarray([1, 0, 1, 0], dtype=np.int8)
    query_ids = np.asarray(["QUERY00000001", "QUERY00000002"])
    candidate_ids = np.asarray(["CANDIDATE00001", "CANDIDATE00002", "CANDIDATE00003", "CANDIDATE00004"])
    groups = np.asarray(["C2H6O", "C3H8O"])
    panel = tmp_path / "panel_identity_disjoint.npz"
    np.savez_compressed(
        panel,
        query_ptr=ptr,
        molecule_label=labels,
        query_ik14=query_ids,
        molecule_ik14=candidate_ids,
        query_formula=groups,
        near_query=np.asarray([False, True]),
    )
    checksums = tmp_path / "checksums.sha256"
    checksums.write_text(f"{sha256(panel)}  {panel.name}\n", encoding="utf-8")
    benchmark_report = tmp_path / "benchmark_report.json"
    benchmark_report.write_text(json.dumps({
        "dataset_id": "enveda_180_filtered_20260713",
        "evaluation_role": "sealed_external_test",
        "truth_status": "frozen_unscored",
        "performance_scores_opened": False,
        "checksums": {panel.name: sha256(panel)},
        "leakage_guard": {
            "exclusion_policy_complete": True,
            "selected_consumed_identity_overlap": 0,
            "selected_consumed_spectrum_overlap": 0,
        },
    }), encoding="utf-8")
    component = tmp_path / "component_scores.npz"
    np.savez_compressed(
        component,
        module_names=np.asarray(modules),
        scores=np.asarray([[0.9, 0.1, 0.8, 0.2], [0.8, 0.2, 0.9, 0.1]], dtype=np.float32),
        availability=np.ones((2, 4), dtype=np.float32),
        query_ptr=ptr,
        labels=labels,
        group_ids=groups,
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        near_query=np.asarray([False, True]),
    )
    sealed = tmp_path / "sealed.npz"
    subprocess.run([
        sys.executable, str(ROOT / "tasks/seal_grand_unified_external_bundle.py"),
        "--component-score-bundle", str(component),
        "--panel-npz", str(panel), "--panel", "identity_disjoint",
        "--benchmark-report", str(benchmark_report),
        "--benchmark-checksums", str(checksums),
        "--model", str(model_path), "--freeze-report", str(freeze),
        "--module-registry", str(registry), "--out", str(sealed),
    ], check=True)
    ledger = tmp_path / "opening_ledger.json"
    result = tmp_path / "external_result.json"
    command = [
        sys.executable, str(ROOT / "tasks/evaluate_frozen_grand_unified_external.py"),
        "--model", str(model_path), "--freeze-report", str(freeze),
        "--test-bundle", str(sealed), "--component-score-bundle", str(component),
        "--module-registry", str(registry), "--benchmark-report", str(benchmark_report),
        "--benchmark-checksums", str(checksums), "--benchmark-panel", str(panel),
        "--comparison-baseline", "weighted_spectral_entropy",
        "--opening-ledger", str(ledger), "--out-report", str(result), "--device", "cpu",
    ]
    subprocess.run(command, check=True)
    assert json.loads(ledger.read_text(encoding="utf-8"))["status"] == "OPENED_ONCE_COMPLETE_NO_RETRY"
    assert json.loads(result.read_text(encoding="utf-8"))["dataset_id"] == "enveda_180_filtered_20260713"

    retry = command.copy()
    retry[retry.index(str(result))] = str(tmp_path / "retry_result.json")
    second = subprocess.run(retry, text=True, capture_output=True)
    assert second.returncode != 0
    assert "FileExistsError" in second.stderr
