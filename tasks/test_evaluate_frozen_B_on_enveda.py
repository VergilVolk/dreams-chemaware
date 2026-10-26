#!/usr/bin/env python
"""Lightweight synthetic contracts for frozen-B Enveda replay."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

from conditional_null_energy_core import AnchoredEvidenceEnergy
from evaluate_frozen_B_on_enveda import load_checkpoint, main as evaluator_main
from train_wse_conditional_residual import CHEMICAL_FEATURE_NAMES, SPECTRAL_FEATURE_NAMES


PANELS = ("identity_disjoint", "formula_disjoint")


def _checkpoint(path: Path, *, selected_arm: str = "B_joint_residual") -> None:
    feature_count = len(SPECTRAL_FEATURE_NAMES) + len(CHEMICAL_FEATURE_NAMES)
    torch.manual_seed(11)
    model = AnchoredEvidenceEnergy(
        feature_count,
        mode="joint",
        spectral_feature_count=len(SPECTRAL_FEATURE_NAMES),
        hidden=4,
        residual_bound=0.25,
    )
    # AnchoredEvidenceEnergy initializes both output layers at zero, so this
    # fixture's frozen B score must exactly preserve WSE.
    torch.save({
        "state_dict": model.state_dict(),
        "model_class": "AnchoredEvidenceEnergy",
        "selected_arm": selected_arm,
        "mode": "joint",
        "spectral_feature_names": SPECTRAL_FEATURE_NAMES,
        "chemical_feature_names": CHEMICAL_FEATURE_NAMES,
        "spectral_feature_count": len(SPECTRAL_FEATURE_NAMES),
        "hidden": 4,
        "residual_bound_raw_wse_units": 0.25,
        "scaler_mean": np.zeros(feature_count, dtype=np.float32),
        "scaler_scale": np.ones(feature_count, dtype=np.float32),
        "training_seed": 17,
        "training_queries": 4,
        "enveda_accessed": False,
    }, path)


def _panel(panel: str) -> dict[str, np.ndarray]:
    if panel == "identity_disjoint":
        ptr = np.asarray([0, 2, 5], dtype=np.int64)
        labels = np.asarray([1, 0, 0, 1, 0], dtype=np.int8)
        wse = np.asarray([0.9, 0.1, 0.2, 0.8, 0.3], dtype=np.float32)
        formulas = np.asarray(["C2H4", "C3H6"])
    else:
        ptr = np.asarray([0, 2, 4], dtype=np.int64)
        labels = np.asarray([0, 1, 1, 0], dtype=np.int8)
        wse = np.asarray([0.1, 0.9, 0.8, 0.2], dtype=np.float32)
        formulas = np.asarray(["C4H8", "C5H10"])
    count = len(labels)
    query_count = len(ptr) - 1
    return {
        "query_ptr": ptr,
        "labels": labels,
        "query_keys": np.asarray(
            [f"{panel}:row={index}" for index in range(query_count)], dtype="U32",
        ),
        "formula": formulas.astype("U16"),
        "near": np.arange(query_count) % 2 == 0,
        "weighted_spectral_entropy": wse,
        "noise_v1": (0.8 * wse + 0.05).astype(np.float32),
        "p2b_noise_v1_frozen": (0.9 * wse + 0.02).astype(np.float32),
        "query_ids": np.asarray(
            [f"QUERY{panel[0]}{index}" for index in range(query_count)], dtype="U16",
        ),
        "candidate_ids": np.asarray(
            [f"CAND{panel[0]}{index}" for index in range(count)], dtype="U16",
        ),
    }


def _write_fixture(root: Path, *, drift_formula: bool = False) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / "candidate_predictions_oof.npz"
    chem_dir = root / ("chem_drifted" if drift_formula else "chem")
    chem_dir.mkdir()
    ledger: dict[str, np.ndarray] = {}
    for panel_index, panel_name in enumerate(PANELS):
        panel = _panel(panel_name)
        prefix = f"{panel_name}__"
        for field in (
            "query_ptr", "labels", "query_keys", "formula", "near",
            "weighted_spectral_entropy", "noise_v1", "p2b_noise_v1_frozen",
        ):
            ledger[prefix + field] = panel[field]
        # These intentionally invalid values prove that the B-only evaluator
        # neither loads nor validates the other Enveda-trained arm scores.
        ledger[prefix + "additive_linear_control"] = np.full(
            len(panel["labels"]), np.nan, dtype=np.float32,
        )
        ledger[prefix + "nonlinear_reliability"] = np.full(
            len(panel["labels"]), np.nan, dtype=np.float32,
        )
        formulas = panel["formula"].copy()
        if drift_formula and panel_index == 0:
            formulas[0] = "DRIFT"
        count = len(panel["labels"])
        available = np.arange(count) % 2 == 1
        zero = np.zeros(count, dtype=np.float32)
        correct = zero.copy()
        correct[available] = 0.2
        np.savez_compressed(
            chem_dir / f"continuous_chem_{panel_name}.npz",
            schema=np.asarray("gnps_chemaware_continuous_null_evidence_v1"),
            panel=np.asarray(panel_name),
            query_ptr=panel["query_ptr"],
            query_ids=panel["query_ids"],
            candidate_ids=panel["candidate_ids"],
            query_formula=formulas,
            near_query=panel["near"],
            availability=available,
            utility_correct=correct,
            utility_zero=zero,
            utility_reversed=zero,
            utility_candidate_mismatch=zero,
            policy_sha256=np.asarray("a" * 64),
            panel_sha256=np.asarray(("b" if panel_index == 0 else "c") * 64),
        )
    np.savez_compressed(ledger_path, **ledger)
    return ledger_path, chem_dir


def _run(
    checkpoint: Path, ledger: Path, chem_dir: Path, output: Path,
) -> None:
    previous = sys.argv
    sys.argv = [
        "evaluate_frozen_B_on_enveda.py",
        "--checkpoint", str(checkpoint),
        "--prediction-ledger", str(ledger),
        "--chem-dir", str(chem_dir),
        "--output", str(output),
        "--bootstrap-resamples", "40",
        "--batch-candidates", "3",
        "--device", "cpu",
    ]
    try:
        evaluator_main()
    finally:
        sys.argv = previous


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="frozen_B_enveda_contract_") as raw:
        root = Path(raw)
        checkpoint = root / "frozen_B.pt"
        _checkpoint(checkpoint)
        ledger, chem_dir = _write_fixture(root)
        output = root / "evaluation"
        _run(checkpoint, ledger, chem_dir, output)

        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        assert report["training_or_selection_on_enveda"] is False
        assert report["model_selection_results_read"] is False
        assert report["checkpoint"]["selected_arm"] == "B_joint_residual"
        for panel_name in PANELS:
            panel = report["panels"][panel_name]
            assert panel["methods"]["B_frozen_GNPS_fullfit"] == panel["methods"][
                "weighted_spectral_entropy"
            ]
            comparison = panel["B_vs_weighted_spectral_entropy"]
            assert comparison["corrected"] == 0
            assert comparison["introduced"] == 0
            assert comparison["formula_cluster_bootstrap_95ci"]["ci_low"] == 0
            assert comparison["formula_cluster_bootstrap_95ci"]["ci_high"] == 0

        with np.load(output / "candidate_scores.npz", allow_pickle=False) as scores:
            assert str(scores["schema"].item()) == "frozen_B_on_enveda_candidate_scores_v1"
            assert not any(
                forbidden in key
                for key in scores.files
                for forbidden in (
                    "__labels", "__formula", "__near",
                    "additive_linear_control", "nonlinear_reliability",
                )
            )
            for panel_name in PANELS:
                assert np.array_equal(
                    scores[f"{panel_name}__B_frozen_GNPS_fullfit"],
                    scores[f"{panel_name}__weighted_spectral_entropy"],
                )
                assert np.array_equal(
                    scores[f"{panel_name}__candidate_ids"],
                    _panel(panel_name)["candidate_ids"],
                )

        _, drifted_chem = _write_fixture(root / "drift", drift_formula=True)
        try:
            _run(checkpoint, root / "drift" / "candidate_predictions_oof.npz",
                 drifted_chem, root / "drifted_evaluation")
        except RuntimeError as exc:
            assert "candidate axis drifted" in str(exc)
        else:
            raise AssertionError("same-length formula-axis drift must fail closed")

        bad_checkpoint = root / "not_B.pt"
        _checkpoint(bad_checkpoint, selected_arm="A_spectral_residual")
        try:
            load_checkpoint(bad_checkpoint, "cpu")
        except RuntimeError as exc:
            assert "not the frozen B contract" in str(exc)
        else:
            raise AssertionError("non-B checkpoint must fail closed")

    print("PASS: frozen B Enveda replay contracts")


if __name__ == "__main__":
    main()
