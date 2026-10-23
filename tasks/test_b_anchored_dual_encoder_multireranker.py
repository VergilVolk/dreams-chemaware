#!/usr/bin/env python
"""Fast CPU contracts for the B-anchored integration primitives."""
from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from tempfile import TemporaryDirectory

from apply_b_anchored_dual_encoder_multireranker import (
    build_b_inference_panel,
    load_inference_bundle,
    query_deployment_status,
)
from train_b_anchored_dual_encoder_multireranker import (
    AnchoredResidual,
    ARM_EXPERTS,
    EXPERTS,
    build_features,
    feature_columns,
    fit_arm,
    fit_scale,
    query_score_features_inference,
)


def synthetic_bundle() -> dict:
    names = np.asarray((
        "weighted_spectral_entropy", "noise_v1", "p2b_noise_v1_frozen",
        "chemaware_stage1_encoder", "chemaware_phase_a_encoder",
        "noise_msg_fusion_stack_v1", "noise_rrf_1_7",
    ))
    scores = np.asarray([
        [0.1, 0.2, 0.3, 0.4], [0.2, 0.4, 0.3, 0.1], [0.4, 0.1, 0.2, 0.3],
        [0.3, 0.5, 0.2, 0.0], [0.1, 0.6, 0.5, 0.2],
        [0.5, 0.2, 0.3, 0.2], [0.4, 0.2, 0.1, 0.5],
    ], dtype=np.float32)
    availability = np.ones_like(scores)
    availability[4, 2:] = 0
    return {"module_names": names, "scores": scores, "availability": availability,
            "query_ptr": np.asarray([0, 2, 4], dtype=np.int64)}


def main() -> None:
    features, names = build_features(synthetic_bundle())
    assert features.shape == (4, 4 * len(EXPERTS))
    assert len(names) == features.shape[1]
    # Missing Phase-A evidence is exactly zero, including after scaling.
    phase = feature_columns((1,))
    assert np.all(features[2:, phase] == 0)
    scale = fit_scale(features, np.arange(4))
    assert np.all(features[2:, phase] / scale[phase] == 0)
    assert np.all(scale > 0)
    production = synthetic_bundle()
    production["module_names"] = production["module_names"][:-1]
    production["scores"] = production["scores"][:-1]
    production["availability"] = production["availability"][:-1]
    production_features, _ = build_features(production, experts=(0, 1, 2))
    assert production_features.shape == (4, 12)  # retired RRF is not required at deployment

    torch.manual_seed(1)
    for linear in (False, True):
        model = AnchoredResidual(features.shape[1], hidden=4, bound=0.5, linear=linear)
        x = torch.as_tensor(features)
        base = torch.arange(4, dtype=torch.float32)
        assert torch.equal(model(base, torch.zeros_like(x)), base)
        assert torch.all(torch.abs(model.residual(x)) <= 0.5 + 1e-7)

    # Exercise every fixed arm through the real listwise optimizer.
    repeated = np.tile(features, (3, 1))
    labels = np.tile(np.asarray([1, 0, 1, 0], dtype=np.int8), 3)
    base = np.tile(np.asarray([0.2, 0.1, 0.1, 0.2], dtype=np.float32), 3)
    panel = {
        "features": repeated,
        "query_ptr": np.arange(0, 13, 2, dtype=np.int64),
        "labels": labels,
        "parent": {"B": base},
    }
    recipe = {"hidden": 4, "learning_rate": 0.01, "epochs": 2, "trust_weight": 0.05}
    queries = np.arange(6, dtype=np.int64)
    for arm, experts in ARM_EXPERTS.items():
        fitted = fit_arm(panel, queries, experts, recipe, 7, "cpu",
                         base,
                         linear=(arm == "complete_spectral_additive"))
        assert np.isfinite(fitted[-1])

    # Real inference is truth-free and retains zero/single/multi-candidate
    # queries without manufacturing a ranking decision for the first two.
    ptr = np.asarray([0, 0, 1, 3], dtype=np.int64)
    values = np.asarray([0.4, 0.1, 0.8], dtype=np.float32)
    summarized = query_score_features_inference(values, ptr)
    assert np.array_equal(summarized[0], np.asarray([0.4, 0.0, 0.0], dtype=np.float32))
    status, multi = query_deployment_status(ptr)
    assert status.tolist() == ["no_candidate", "single_candidate_no_reranking", "ranked_multi_candidate"]
    assert multi.tolist() == [False, True, True]
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "truth_free.npz"
        np.savez_compressed(
            path, module_names=np.asarray(["m"]), scores=values[None, :],
            availability=np.ones((1, len(values)), dtype=np.int8), query_ptr=ptr,
            query_ids=np.asarray(["q0", "q1", "q2"]),
            candidate_ids=np.asarray(["c0", "c1", "c2"]),
        )
        loaded = load_inference_bundle(path)
        assert "labels" not in loaded and "group_ids" not in loaded
        core_path = Path(temporary) / "core.npz"
        module_names = np.asarray([
            "weighted_spectral_entropy", "noise_v1", "p2b_noise_v1_frozen",
        ])
        np.savez_compressed(
            core_path, module_names=module_names,
            scores=np.vstack((values, values + 0.1, values + 0.2)),
            availability=np.ones((3, len(values)), dtype=np.int8), query_ptr=ptr,
            query_ids=np.asarray(["q0", "q1", "q2"]),
            candidate_ids=np.asarray(["c0", "c1", "c2"]),
            # These truth-like fields must not be loaded by inference.
            labels=np.asarray([1, 1, 0]), group_ids=np.asarray(["x", "y", "z"]),
        )
        chem_path = Path(temporary) / "chem.npz"
        np.savez_compressed(
            chem_path, schema=np.asarray("gnps_chemaware_continuous_null_evidence_v1"),
            query_ptr=ptr, query_ids=np.asarray(["q0", "q1", "q2"]),
            candidate_ids=np.asarray(["c0", "c1", "c2"]),
            availability=np.asarray([1, 0, 1], dtype=np.int8),
            utility_correct=np.asarray([0.4, 0.0, 0.7], dtype=np.float32),
            utility_zero=np.zeros(3, dtype=np.float32),
            utility_reversed=np.asarray([0.1, 0.0, 0.2], dtype=np.float32),
            utility_candidate_mismatch=np.asarray([0.2, 0.0, 0.3], dtype=np.float32),
            query_formula=np.asarray(["SHOULD", "NOT", "LOAD"]),
            near_query=np.asarray([1, 1, 1], dtype=np.int8),
        )
        truth_free_core = load_inference_bundle(core_path)
        assert set(truth_free_core) == {
            "module_names", "scores", "availability", "query_ptr", "query_ids", "candidate_ids"
        }
        inference_panel = build_b_inference_panel(truth_free_core, chem_path)
        assert inference_panel["features"].shape == (3, 13)
    sbatch = (Path(__file__).resolve().parent / "run_b_anchored_dual_encoder_multireranker_gnps.sbatch").read_text()
    assert "#SBATCH --gpus=1" in sbatch
    assert "#SBATCH --cpus-per-task=10" in sbatch
    assert "enveda" not in sbatch.lower()
    assert "massspecgym" not in sbatch.lower()
    for module in ("chemaware_stage1_encoder", "chemaware_phase_a_encoder",
                   "noise_msg_fusion_stack_v1", "noise_rrf_1_7"):
        assert f'--module {module}' in sbatch
    refit = (Path(__file__).resolve().parent / "run_b_dual_multireranker_nested_refit.sbatch").read_text()
    assert "#SBATCH --gpus=1" in refit and "#SBATCH --cpus-per-task=10" in refit
    assert "modeling_nested_oof" in refit
    assert "encode_gnps_gold_silver_10ppm_checkpoint.py" not in refit
    assert "--core-identity-bundle" in refit and "--identity-chem" in refit
    print("PASS: B-anchored dual-encoder/multi-reranker contracts")


if __name__ == "__main__":
    main()
