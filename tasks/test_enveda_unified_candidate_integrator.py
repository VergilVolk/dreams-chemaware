#!/usr/bin/env python
"""CPU contracts for the Enveda unified candidate evidence integrator."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from enveda_unified_candidate_integrator_core import (
    CHECKPOINT_SCHEMA,
    CONTROL_ARM,
    PRIMARY_ARM,
    CandidateEvidenceIntegrator,
    IntegratorSettings,
    fit_weighted_normalization,
    formula_fold,
    occurrence_weights,
    predict_from_payload,
    transform_features,
    unavailable_chemaware,
    validate_global_query_formula,
    weighted_listwise_loss,
    weighted_query_residual_penalty,
)
from gnps_pair_score_cache import write_pair_score_cache
from evaluate_enveda_integrator_on_gnps import main as evaluator_main
from train_enveda_unified_candidate_integrator import (
    PANELS, SCORE_METHODS, _group_occurrences, _make_panels, main as trainer_main,
)


def _panel(rows: np.ndarray, formulas: np.ndarray) -> dict[str, np.ndarray]:
    rows = np.asarray(rows, dtype=np.int64)
    count = len(rows)
    query_ptr = np.arange(0, 2 * count + 1, 2, dtype=np.int64)
    molecule_ptr = np.arange(0, 2 * count + 1, dtype=np.int64)
    candidate_rows = np.column_stack((rows, (rows + 5) % 10)).reshape(-1)
    query_ids = np.asarray([f"IK{row}" for row in rows], dtype="U16")
    candidate_ids = np.column_stack((
        query_ids, np.asarray([f"NEG{row}" for row in rows], dtype="U16"),
    )).reshape(-1)
    return {
        "query_row": rows,
        "query_ik14": query_ids,
        "query_formula": np.asarray(formulas, dtype="U16"),
        "near_query": np.arange(count) % 2 == 0,
        "query_ptr": query_ptr,
        "molecule_ptr": molecule_ptr,
        "molecule_label": np.tile(np.asarray([1, 0], dtype=np.int8), count),
        "molecule_ik14": candidate_ids,
        "molecule_formula": np.repeat(np.asarray(formulas, dtype="U16"), 2),
        "candidate_row": candidate_rows,
    }


def _write_loader_fixture(root: Path) -> SimpleNamespace:
    benchmark = root / "benchmark"
    benchmark.mkdir()
    pd.DataFrame({"row": np.arange(10)}).to_csv(
        benchmark / "manifest.csv.gz", index=False, compression="gzip",
    )
    (benchmark / "report.json").write_text("{}", encoding="utf-8")
    panels = {
        "identity_disjoint": _panel(
            np.asarray([0, 1, 2, 3, 4, 5]),
            np.asarray(["C1H2", "C2H4", "C3H6", "C4H8", "C5H10", "C6H12"]),
        ),
        "formula_disjoint": _panel(
            np.asarray([0, 1, 6, 7]),
            np.asarray(["C1H2", "C2H4", "C7H14", "C8H16"]),
        ),
    }
    for name, panel in panels.items():
        np.savez_compressed(benchmark / f"panel_{name}.npz", **panel)

    score_bundle = root / "scores.npz"
    arrays: dict[str, np.ndarray] = {"method_names": np.asarray(SCORE_METHODS)}
    msg_pairs: dict[str, np.ndarray] = {}
    for name, panel in panels.items():
        labels = panel["molecule_label"].astype(np.float32)
        arrays[f"scores_{name}"] = np.stack((
            0.2 + 0.6 * labels,
            0.1 + 0.7 * labels,
            0.3 + 0.5 * labels,
            0.15 + 0.65 * labels,
        )).astype(np.float32)
        msg_pairs[name] = (0.25 + 0.5 * labels).astype(np.float32)
    np.savez_compressed(score_bundle, **arrays)
    msg_cache = root / "msg"
    write_pair_score_cache(
        msg_cache, benchmark, {"name": "noise_msg_fusion_stack_v1"}, msg_pairs,
    )

    rng = np.random.default_rng(4)
    embedding_paths = []
    for index in range(3):
        values = rng.normal(size=(10, 8)).astype(np.float32)
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        path = root / f"embedding_{index}.npz"
        np.savez_compressed(path, rows=np.arange(10), embeddings=values)
        embedding_paths.append(path)

    chem_dir = root / "chem"
    chem_dir.mkdir()
    (chem_dir / "report.json").write_text("{}", encoding="utf-8")
    for name, panel in panels.items():
        count = len(panel["molecule_label"])
        available = panel["molecule_label"] == 0
        correct = np.zeros(count, dtype=np.float32)
        correct[available] = 0.4
        np.savez_compressed(
            chem_dir / f"continuous_chem_{name}.npz",
            schema=np.asarray("gnps_chemaware_continuous_null_evidence_v1"),
            query_ptr=panel["query_ptr"], query_ids=panel["query_ik14"],
            candidate_ids=panel["molecule_ik14"], query_formula=panel["query_formula"],
            near_query=panel["near_query"], availability=available,
            utility_correct=correct, utility_zero=np.zeros(count, dtype=np.float32),
            utility_reversed=np.zeros(count, dtype=np.float32),
            utility_candidate_mismatch=np.zeros(count, dtype=np.float32),
        )
    return SimpleNamespace(
        benchmark=benchmark, score_bundle=score_bundle, msg_cache=msg_cache,
        stage1_embeddings=embedding_paths[0], phasea_embeddings=embedding_paths[1],
        fullmsg_embeddings=embedding_paths[2], chem_dir=chem_dir,
        comparator_embedding=[], comparator_score=[],
    )


def _write_comparator_archive(
    path: Path, benchmark: Path, *, drift_candidate_axis: bool = False,
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for panel in PANELS:
        prefix = panel.replace("_disjoint", "")
        with np.load(benchmark / f"panel_{panel}.npz", allow_pickle=False) as body:
            candidate_ids = np.asarray(body["molecule_ik14"]).astype("U")
            if drift_candidate_axis and panel == PANELS[0]:
                candidate_ids = candidate_ids.copy()
                candidate_ids[[0, 1]] = candidate_ids[[1, 0]]
            arrays.update({
                f"{prefix}__query_ptr": np.asarray(body["query_ptr"]),
                f"{prefix}__labels": np.asarray(body["molecule_label"]),
                f"{prefix}__query_ids": np.asarray(body["query_ik14"]).astype("U"),
                f"{prefix}__candidate_ids": candidate_ids,
                f"{prefix}__formula": np.asarray(body["query_formula"]).astype("U"),
                f"{prefix}__near": np.asarray(body["near_query"], dtype=bool),
                f"{prefix}__B": np.linspace(
                    0.0, 1.0, len(body["molecule_label"]), dtype=np.float32,
                ),
            })
    np.savez_compressed(path, **arrays)


def main() -> None:
    keys = {
        "identity_disjoint": np.asarray(["q1", "q2"]),
        "formula_disjoint": np.asarray(["q1", "q3"]),
    }
    formulas = {
        "identity_disjoint": np.asarray(["C2H4", "C3H6"]),
        "formula_disjoint": np.asarray(["C2H4", "C4H8"]),
    }
    validate_global_query_formula(keys, formulas)
    weights = occurrence_weights(keys)
    assert np.allclose(weights["identity_disjoint"], [0.5, 1.0])
    assert np.allclose(weights["formula_disjoint"], [0.5, 1.0])
    assert formula_fold("C2H4", 5, 20261026) == formula_fold("C2H4", 5, 20261026)
    absent = unavailable_chemaware(3)
    assert not absent["chem_valid"].any()
    assert all(not np.asarray(value).any() for name, value in absent.items() if name != "chem_valid")

    names = (
        "weighted_spectral_entropy",
        "representation_signal",
        "chemaware_v2_available",
        "chemaware_v2_correct_minus_zero",
    )
    block_a = np.asarray([
        [0.0, -1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 2.0],
    ], dtype=np.float32)
    block_b = np.asarray([
        [0.5, 0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0, -1.0],
    ], dtype=np.float32)
    mean, scale = fit_weighted_normalization([
        (block_a, np.asarray([0.25, 0.25])),
        (block_b, np.asarray([0.25, 0.25])),
    ], names)
    assert mean[2] == 0 and scale[2] == 1
    assert mean[3] == 0 and scale[3] > 0
    transformed_zero = transform_features(np.zeros((1, 4), dtype=np.float32), mean, scale)
    assert transformed_zero[0, 2] == 0 and transformed_zero[0, 3] == 0

    settings = IntegratorSettings(
        scalar_features=len(names), anchor_index=0, hidden=4,
    )
    values = transform_features(np.vstack((block_a, block_b)), mean, scale)
    tensor = torch.as_tensor(values)
    for arm in (PRIMARY_ARM, CONTROL_ARM):
        model = CandidateEvidenceIntegrator(arm, settings)
        observed = model(tensor).detach().numpy()
        assert np.allclose(observed, values[:, 0])

    model = CandidateEvidenceIntegrator(PRIMARY_ARM, settings)
    logits = model(tensor)
    loss = weighted_listwise_loss(
        logits,
        np.asarray([0, 2, 4]),
        np.asarray([1, 0, 1, 0]),
        np.asarray([0.5, 1.0]),
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    penalty = weighted_query_residual_penalty(
        torch.tensor([1.0, 3.0, 2.0, 2.0]),
        np.asarray([0, 2, 4]), np.asarray([0.5, 1.0]),
    )
    # Query MSEs are 5 and 4; query-balanced result is (0.5*5+1*4)/1.5.
    assert np.isclose(float(penalty), 13.0 / 3.0)

    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "scalar_feature_names": names,
        "scalar_mean": mean,
        "scalar_scale": scale,
        "anchor_index": 0,
        "hidden": 4,
        "arms": {
            PRIMARY_ARM: model.state_dict(),
            CONTROL_ARM: CandidateEvidenceIntegrator(CONTROL_ARM, settings).state_dict(),
        },
    }
    deployed = predict_from_payload(payload, names, np.vstack((block_a, block_b)))
    direct = model(tensor).detach().numpy()
    assert np.allclose(deployed, direct)
    try:
        predict_from_payload(payload, names[::-1], np.vstack((block_a, block_b)))
    except RuntimeError:
        pass
    else:
        raise AssertionError("feature-order drift must fail closed")

    with tempfile.TemporaryDirectory(prefix="enveda_integrator_contract_") as raw:
        fixture = _write_loader_fixture(Path(raw))
        comparator = Path(raw) / "candidate_predictions.npz"
        _write_comparator_archive(comparator, fixture.benchmark)
        fixture.comparator_score = [("B_gnps_nested_oof", comparator)]
        panels, _ = _make_panels(fixture)
        assert set(panels) == set(PANELS)
        assert panels["identity_disjoint"].feature_names == panels["formula_disjoint"].feature_names
        assert panels["identity_disjoint"].feature_names[0] == "noise_v1"
        # Rows 0 and 1 occur in both panels and therefore receive 1/2 in each.
        assert np.allclose(panels["identity_disjoint"].query_weights[:2], 0.5)
        assert np.allclose(panels["formula_disjoint"].query_weights[:2], 0.5)
        assert np.allclose(panels["identity_disjoint"].query_weights[2:], 1.0)
        groups = _group_occurrences(
            panels,
            [(name, query) for name in PANELS for query in range(len(panels[name].query_keys))],
        )
        repeated = [group for group in groups if len(group) == 2]
        assert len(repeated) == 2
        assert all(np.isclose(sum(panels[name].query_weights[q] for name, q in group), 1.0)
                   for group in groups)
        assert "weighted_spectral_entropy" in panels["identity_disjoint"].feature_names
        assert "chemaware_v2_available" in panels["identity_disjoint"].feature_names
        assert "B_gnps_nested_oof" in panels["identity_disjoint"].scores
        drifted = Path(raw) / "candidate_predictions_drifted.npz"
        _write_comparator_archive(
            drifted, fixture.benchmark, drift_candidate_axis=True,
        )
        fixture.comparator_score = [("B_gnps_nested_oof", drifted)]
        try:
            _make_panels(fixture)
        except RuntimeError as exc:
            assert "candidate axis drifted" in str(exc)
        else:
            raise AssertionError("same-length comparator axis drift must fail closed")
        fixture.comparator_score = []
        output = Path(raw) / "trained"
        previous_argv = sys.argv
        sys.argv = [
            "train_enveda_unified_candidate_integrator.py",
            "--benchmark", str(fixture.benchmark),
            "--score-bundle", str(fixture.score_bundle),
            "--msg-cache", str(fixture.msg_cache),
            "--stage1-embeddings", str(fixture.stage1_embeddings),
            "--phasea-embeddings", str(fixture.phasea_embeddings),
            "--fullmsg-embeddings", str(fixture.fullmsg_embeddings),
            "--chem-dir", str(fixture.chem_dir),
            "--output", str(output),
            "--folds", "2", "--epochs", "1", "--batch-queries", "8",
            "--hidden", "4", "--bootstrap-resamples", "100", "--device", "cpu",
        ]
        try:
            trainer_main()
        finally:
            sys.argv = previous_argv
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        assert report["evaluation_role"] == "fusion_meta_development_consumed"
        assert report["allow_final_external_claim"] is False
        assert report["primary_model"] == PRIMARY_ARM
        try:
            saved = torch.load(
                output / "unified_candidate_integrator.pt",
                map_location="cpu", weights_only=False,
            )
        except TypeError:
            saved = torch.load(output / "unified_candidate_integrator.pt", map_location="cpu")
        assert saved["anchor_feature"] == "weighted_spectral_entropy"
        assert saved["arm_selection_from_oof"] is False
        assert saved["primary_residual_bound"] is None
        assert saved["optional_families"] == ("chemaware_v2_continuous_null",)
        assert "noise_msg_fusion_stack_v1" in saved["required_families"]

        fixed_output = Path(raw) / "fixed_evaluation"
        sys.argv = [
            "evaluate_enveda_integrator_on_gnps.py",
            "--benchmark", str(fixture.benchmark),
            "--checkpoint", str(output / "unified_candidate_integrator.pt"),
            "--score-bundle", str(fixture.score_bundle),
            "--msg-cache", str(fixture.msg_cache),
            "--stage1-embeddings", str(fixture.stage1_embeddings),
            "--phasea-embeddings", str(fixture.phasea_embeddings),
            "--fullmsg-embeddings", str(fixture.fullmsg_embeddings),
            "--chem-dir", str(fixture.chem_dir),
            "--output", str(fixed_output),
            "--bootstrap-resamples", "100", "--device", "cpu",
        ]
        try:
            evaluator_main()
        finally:
            sys.argv = previous_argv
        fixed_report = json.loads(
            (fixed_output / "report.json").read_text(encoding="utf-8")
        )
        assert fixed_report["training_or_selection_on_gnps"] is False
        assert fixed_report["checkpoint"]["deployed_arm"] == PRIMARY_ARM

    print("PASS: Enveda unified candidate integrator contracts")


if __name__ == "__main__":
    main()
