#!/usr/bin/env python3
"""CPU scientific contracts for the Full23 multi-reference increment."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import tempfile

import numpy as np
import torch

from full23_multireference_readout_core import (
    ARMS,
    ENCODER_NAMES,
    SCHEMA,
    TOP1_ARM,
    TOPK_ARM,
    Full23AnchoredMultiReferenceReadout,
    MultiReferenceSettings,
    arm_mask,
    model_from_payload,
    stable_topk_references,
)
from train_enveda_unified_candidate_integrator import PANELS
import train_full23_multireference_readout as trainer
from evaluate_full23_multireference_on_gnps import _increment_gate
from unified_additive_release import CHECKPOINT_SHA256, sha256_file


def _unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def _production_width(values: np.ndarray) -> np.ndarray:
    output = np.zeros((len(values), 1024), dtype=np.float32)
    output[:, :values.shape[1]] = values
    return output


def _graph() -> SimpleNamespace:
    return SimpleNamespace(
        n_queries=1,
        query_row=np.asarray([0], dtype=np.int64),
        query_ptr=np.asarray([0, 2], dtype=np.int64),
        molecule_ptr=np.asarray([0, 3, 5], dtype=np.int64),
        molecule_label=np.asarray([1, 0], dtype=np.int8),
        molecule_ik14=np.asarray(["POS", "NEG"]),
        molecule_formula=np.asarray(["C2H4", "C2H4"]),
        pair_candidate_row=np.asarray([1, 2, 3, 4, 5], dtype=np.int64),
    )


def main() -> None:
    graph = _graph()
    rows = np.asarray([5, 0, 3, 1, 4, 2], dtype=np.int64)
    canonical = _production_width(_unit(np.asarray([
        [1.00, 0.00],
        [0.95, 0.05],
        [0.80, 0.20],
        [0.60, 0.40],
        [0.30, 0.70],
        [0.10, 0.90],
    ])))
    embeddings = canonical[rows]
    index = stable_topk_references(graph, rows, embeddings)
    assert index.reference_rows.tolist() == [[1, 2, 3], [4, 5, -1]]
    assert index.reference_mask.tolist() == [[True, True, True], [True, True, False]]
    assert index.query_positions.shape == (2,)
    assert np.allclose(
        index.molecule_scores,
        [canonical[0] @ canonical[1], canonical[0] @ canonical[4]],
    )
    # The first retained row is exactly the maximum raw-cosine reference.
    assert index.reference_rows[0, 0] == 1 and index.reference_rows[1, 0] == 4

    # K=3 is lossless only while the frozen graph has no fourth unique reference.
    overflow = _graph()
    overflow.molecule_ptr = np.asarray([0, 4, 5], dtype=np.int64)
    try:
        stable_topk_references(overflow, rows, embeddings)
    except RuntimeError as error:
        assert "more than the registered 3 unique references" in str(error)
    else:
        raise AssertionError("reference multiplicity above K must fail closed")

    anchor = np.asarray([0.2, 0.8], dtype=np.float32)
    settings = MultiReferenceSettings()
    model = Full23AnchoredMultiReferenceReadout(settings)
    query = np.stack([index.query_positions] * len(ENCODER_NAMES), axis=1)
    references = np.stack([index.reference_positions] * len(ENCODER_NAMES), axis=1)
    masks = np.stack([index.reference_mask] * len(ENCODER_NAMES), axis=1)
    banks = {name: torch.as_tensor(embeddings.copy()) for name in ENCODER_NAMES}
    tensors = (
        torch.as_tensor(anchor), torch.as_tensor(query),
        torch.as_tensor(references), torch.as_tensor(masks), banks,
    )
    # Zero-initialized residual means the new model begins exactly at Full23 v1.
    assert torch.equal(model(*tensors), torch.as_tensor(anchor))
    top1 = arm_mask(masks, TOP1_ARM)
    assert np.all(top1[:, :, 0]) and not np.any(top1[:, :, 1:])
    assert np.array_equal(arm_mask(masks, TOPK_ARM), masks)

    # Pooling is invariant to valid-reference order and ignores padded slots.
    torch.manual_seed(7)
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, mean=0.0, std=0.1)
    original = model(*tensors)
    permuted_references = references.copy()
    permuted_masks = masks.copy()
    permuted_references[:, :, [0, 1]] = permuted_references[:, :, [1, 0]]
    permuted_masks[:, :, [0, 1]] = permuted_masks[:, :, [1, 0]]
    permuted = model(
        tensors[0], tensors[1], torch.as_tensor(permuted_references),
        torch.as_tensor(permuted_masks), banks,
    )
    assert torch.allclose(original, permuted, atol=1e-7, rtol=0.0)
    padded_references = references.copy()
    padded_references[1, :, 2] = index.query_positions[1]
    padded = model(
        tensors[0], tensors[1], torch.as_tensor(padded_references), tensors[3], banks,
    )
    assert torch.allclose(original, padded, atol=1e-7, rtol=0.0)

    payload = {
        "schema": SCHEMA,
        "encoder_names": ENCODER_NAMES,
        "official_embedding_direct_input": False,
        "full23_anchor_has_indirect_official_dependency": True,
        "top_k": 3,
        "settings": settings.__dict__,
        "arms": {arm: model.state_dict() for arm in ARMS},
    }
    restored = model_from_payload(payload, TOPK_ARM)
    assert torch.allclose(restored(*tensors), original, atol=1e-7, rtol=0.0)

    # The fixed fitting loop consumes the same candidate/reference axes end to end.
    base = SimpleNamespace(
        query_ptr=graph.query_ptr,
        labels=graph.molecule_label,
        query_keys=np.asarray(["row=0"]),
        query_weights=np.asarray([0.5], dtype=np.float32),
    )
    representation = trainer.RepresentationPanel(
        base=base,
        anchor=anchor,
        topk={name: index for name in ENCODER_NAMES},
    )
    fitted, history = trainer._fit_arm(
        TOP1_ARM,
        {panel: representation for panel in PANELS},
        [(panel, 0) for panel in PANELS],
        banks,
        SimpleNamespace(device="cpu"),
    )
    assert len(history) == trainer.EPOCHS and np.all(np.isfinite(history))
    assert fitted.trainable_parameters == model.trainable_parameters

    try:
        Full23AnchoredMultiReferenceReadout(MultiReferenceSettings(
            encoder_names=ENCODER_NAMES + ("official_dreams",),
        ))
    except ValueError:
        pass
    else:
        raise AssertionError("official DreaMS must not enter the representation head")

    score_names = (
        "official_dreams", "weighted_spectral_entropy", "noise_v1",
        "p2b_noise_v1_frozen", "noise_msg_fusion_stack_v1",
        "chemaware_stage1", "chemaware_phasea", "noise_chemaware_fullmsg",
    )
    panels, archive = {}, {}
    for panel_name in PANELS:
        scores = {name: np.asarray([0.8, 0.2], dtype=np.float32) for name in score_names}
        panels[panel_name] = SimpleNamespace(
            query_ptr=np.asarray([0, 2], dtype=np.int64),
            labels=np.asarray([1, 0], dtype=np.int8),
            query_keys=np.asarray(["row=0"]),
            formulas=np.asarray(["C2H4"]),
            query_weights=np.asarray([1.0], dtype=np.float32),
            scores=scores,
        )
        prefix = f"{panel_name}__"
        archive.update({
            prefix + "query_ptr": panels[panel_name].query_ptr,
            prefix + "labels": panels[panel_name].labels,
            prefix + "query_keys": panels[panel_name].query_keys,
            prefix + "formula": panels[panel_name].formulas,
            prefix + "query_weight": panels[panel_name].query_weights,
            prefix + "additive_linear_control": np.asarray([1.5, -0.5], dtype=np.float32),
            **{prefix + name: values for name, values in scores.items()},
        })
    with tempfile.TemporaryDirectory(prefix="full23_multiref_contract_") as raw:
        path = Path(raw) / "parent_oof.npz"
        np.savez_compressed(path, **archive)
        observed_sha = sha256_file(path)
        (path.parent / "report.json").write_text(json.dumps({
            "control_model": "additive_linear_control",
            "checkpoint": {"sha256": CHECKPOINT_SHA256},
            "prediction_ledger": {"sha256": observed_sha},
            "features": {"feature_selection_policy": "exclude_query_constant_context_v1"},
        }), encoding="utf-8")
        original_sha = trainer.FULL23_OOF_SHA256
        try:
            trainer.FULL23_OOF_SHA256 = observed_sha
            loaded = trainer._load_full23_oof(path, panels)
        finally:
            trainer.FULL23_OOF_SHA256 = original_sha
        assert set(loaded) == set(PANELS)
        assert all(np.array_equal(values, [1.5, -0.5]) for values in loaded.values())

    # Promotion is conjunctive across both consumed GNPS panels.
    metrics = {
        panel: {
            TOPK_ARM: {"near": {"queries": 4, "recall@1": 0.75}},
            "full23_v1": {"near": {"queries": 4, "recall@1": 0.75}},
        }
        for panel in PANELS
    }
    comparisons = {
        panel: {
            f"{TOPK_ARM}_vs_full23_v1": {
                "corrected": 4,
                "introduced": 1,
                "recall1_formula_cluster_ci": {"ci_low": 0.001},
            },
        }
        for panel in PANELS
    }
    assert _increment_gate(TOPK_ARM, "full23_v1", metrics, comparisons)["pass"]
    comparisons[PANELS[-1]][f"{TOPK_ARM}_vs_full23_v1"][
        "recall1_formula_cluster_ci"
    ]["ci_low"] = 0.0
    assert not _increment_gate(TOPK_ARM, "full23_v1", metrics, comparisons)["pass"]
    print("PASS: Full23 multi-reference representation contracts")


if __name__ == "__main__":
    main()
