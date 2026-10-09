from __future__ import annotations

import numpy as np
import pytest
import torch
import json
from pathlib import Path

from grand_unified_evidence_core import (
    AllModuleEvidenceModel,
    listwise_loss,
    rank_logit_evidence,
    strict_ranks,
)
from train_grand_unified_evidence_model import assert_sealed_external_test
from build_grand_unified_evidence_bundle import DEFAULT_MODULES, validate_candidate_keys


MODULES = (
    "official_dreams", "noise_v1", "chemaware_stage1_encoder", "wse",
    "noise_p2b", "noise_rrf_1_7", "p2b_fragment", "p2b_neutral_loss",
    "chemaware_candidate", "bioaware_event",
)


def _fixture():
    query_ptr = np.asarray([0, 3, 6], dtype=np.int64)
    labels = np.asarray([1, 0, 0, 1, 0, 0], dtype=np.int8)
    base = np.asarray([0.9, 0.3, 0.1, 0.8, 0.4, 0.2])
    scores = np.stack([base + 0.01 * i for i in range(len(MODULES))])
    availability = np.ones_like(scores, dtype=np.float32)
    availability[-1, :3] = 0.0  # BioAware absent for one query, not negative.
    return query_ptr, labels, scores, availability


def test_candidate_level_availability_gates_weight_and_contribution():
    ptr, _, scores, availability = _fixture()
    evidence, summaries = rank_logit_evidence(scores, ptr, availability)
    model = AllModuleEvidenceModel(MODULES)
    out = model(
        torch.tensor(evidence), torch.tensor(ptr), torch.tensor(summaries),
        torch.tensor(availability),
    )
    weights = out.module_weights.detach().numpy()
    assert np.all(weights[availability == 0] == 0)
    assert np.allclose(weights.sum(axis=0), 1.0)
    # Each initially present module makes a non-zero candidate contribution.
    contribution = out.module_contributions.detach().numpy()
    for module in range(len(MODULES)):
        if availability[module].any():
            assert np.any(np.abs(contribution[module]) > 0)


def test_missing_context_is_neutral_and_loss_is_finite():
    ptr, labels, scores, availability = _fixture()
    scores[-1, :3] = 1e6  # Must not leak through candidate availability=0.
    evidence, summaries = rank_logit_evidence(scores, ptr, availability)
    model = AllModuleEvidenceModel(MODULES)
    out = model(
        torch.tensor(evidence), torch.tensor(ptr), torch.tensor(summaries),
        torch.tensor(availability),
    )
    loss = listwise_loss(
        out.logits, torch.tensor(ptr), torch.tensor(labels),
        out.module_weights, torch.tensor(availability),
    )
    assert torch.isfinite(loss)
    assert np.all(strict_ranks(out.logits.detach().numpy(), ptr, labels) == 1)


def test_perturbing_each_available_module_changes_joint_output():
    ptr, _, scores, availability = _fixture()
    evidence, summaries = rank_logit_evidence(scores, ptr, availability)
    model = AllModuleEvidenceModel(MODULES)
    args = (torch.tensor(ptr), torch.tensor(summaries), torch.tensor(availability))
    baseline = model(torch.tensor(evidence), *args).logits.detach().numpy()
    for module in range(len(MODULES)):
        if not availability[module].any():
            continue
        changed = evidence.copy()
        changed[module, 3:6] *= -1
        got = model(torch.tensor(changed), *args).logits.detach().numpy()
        assert not np.allclose(got, baseline), MODULES[module]


def test_reliability_can_assign_exact_zero_weight():
    ptr, _, scores, availability = _fixture()
    evidence, summaries = rank_logit_evidence(scores, ptr, availability)
    model = AllModuleEvidenceModel(MODULES)
    with torch.no_grad():
        model.reliability[-1].weight.zero_()
        model.reliability[-1].bias.zero_()
        model.reliability[-1].bias[2] = -2.0
    out = model(
        torch.tensor(evidence), torch.tensor(ptr), torch.tensor(summaries),
        torch.tensor(availability),
    )
    assert np.all(out.module_weights.detach().numpy()[2] == 0)


def test_sparse_candidate_evidence_does_not_broadcast():
    ptr, _, scores, availability = _fixture()
    availability[-1] = 0
    availability[-1, 4] = 1
    evidence, summaries = rank_logit_evidence(scores, ptr, availability)
    model = AllModuleEvidenceModel(MODULES)
    out = model(
        torch.tensor(evidence), torch.tensor(ptr), torch.tensor(summaries),
        torch.tensor(availability),
    )
    weights = out.module_weights.detach().numpy()
    assert weights[-1, 4] > 0
    assert np.all(weights[-1, np.arange(weights.shape[1]) != 4] == 0)


def test_unavailable_fill_values_cannot_change_reliability_or_output():
    ptr, _, scores, availability = _fixture()
    availability[-1] = 0
    availability[-1, 4] = 1
    altered = scores.copy()
    altered[-1, availability[-1] == 0] = 1e12
    evidence_a, summaries_a = rank_logit_evidence(scores, ptr, availability)
    evidence_b, summaries_b = rank_logit_evidence(altered, ptr, availability)
    assert np.array_equal(evidence_a, evidence_b)
    assert np.array_equal(summaries_a, summaries_b)
    model = AllModuleEvidenceModel(MODULES)
    output_a = model(
        torch.tensor(evidence_a), torch.tensor(ptr), torch.tensor(summaries_a),
        torch.tensor(availability),
    ).logits.detach().numpy()
    output_b = model(
        torch.tensor(evidence_b), torch.tensor(ptr), torch.tensor(summaries_b),
        torch.tensor(availability),
    ).logits.detach().numpy()
    assert np.array_equal(output_a, output_b)


def test_extra_candidate_axis_mismatch_fails_closed():
    query_ids = np.asarray(["q1", "q2"])
    candidate_ids = np.asarray(["a", "b", "c"])
    payload = {"query_ids": query_ids, "candidate_ids": candidate_ids[::-1]}
    with pytest.raises(ValueError, match="candidate_ids"):
        validate_candidate_keys(payload, query_ids, candidate_ids, "example")


def test_registry_preserves_asset_identity_and_retired_status():
    registry = json.loads((Path(__file__).with_name("grand_unified_components_v2.json")).read_text())
    status = {item["name"]: item["status"] for item in registry["modules"]}
    assert status["chemaware_stage1_encoder"] == "enabled"
    assert status["chemaware_phase_a_encoder"] == "provisional"
    assert status["chemaware_v2_reranker"] == "development_only"
    assert status["noise_rrf_1_7"] == "retired_audit_only"
    assert status["p2b_noise_v1_frozen"] == "enabled"
    for raw_p2b in ("p2b_sqrt_cosine", "p2b_unweighted_entropy", "neutral_loss_sqrt_cosine"):
        assert status[raw_p2b] == "audit_only_derivative"
        assert raw_p2b not in DEFAULT_MODULES
    assert "p2b_noise_v1_frozen" in DEFAULT_MODULES


def test_consumed_development_collection_cannot_be_external_test(tmp_path):
    bundle = {
        "dataset_id": np.asarray("gnps_gold_silver_10ppm"),
        "evaluation_role": np.asarray("sealed_external_test"),
        "truth_status": np.asarray("unopened"),
        "allow_final_claim": np.asarray(1, dtype=np.int8),
    }
    with pytest.raises(ValueError, match="DATA_LEAKAGE_GUARD"):
        assert_sealed_external_test(bundle, tmp_path / "fake.npz")


def test_declared_unopened_external_collection_is_not_yet_scoreable(tmp_path):
    bundle = {
        "dataset_id": np.asarray("enveda_180_filtered_20260713"),
        "evaluation_role": np.asarray("sealed_external_test"),
        "truth_status": np.asarray("unopened"),
        "allow_final_claim": np.asarray(1, dtype=np.int8),
    }
    with pytest.raises(ValueError, match="DATA_LEAKAGE_GUARD"):
        assert_sealed_external_test(bundle, tmp_path / "sealed.npz")


def test_frozen_unscored_external_collection_passes_guard(tmp_path):
    bundle = {
        "dataset_id": np.asarray("enveda_180_filtered_20260713"),
        "evaluation_role": np.asarray("sealed_external_test"),
        "truth_status": np.asarray("frozen_unscored"),
        "allow_final_claim": np.asarray(1, dtype=np.int8),
        "leakage_guard_complete": np.asarray(1, dtype=np.int8),
        "model_frozen_before_scoring": np.asarray(1, dtype=np.int8),
    }
    for key in (
        "benchmark_report_sha256", "benchmark_checksums_sha256",
        "benchmark_panel_sha256", "component_score_bundle_sha256",
        "freeze_contract_sha256", "module_registry_sha256", "model_sha256",
    ):
        bundle[key] = np.asarray("a" * 64)
    assert_sealed_external_test(bundle, tmp_path / "sealed.npz")


def test_arbitrary_self_declared_external_collection_is_rejected(tmp_path):
    bundle = {
        "dataset_id": np.asarray("convenient_new_external_v1"),
        "evaluation_role": np.asarray("sealed_external_test"),
        "truth_status": np.asarray("frozen_unscored"),
        "allow_final_claim": np.asarray(1, dtype=np.int8),
        "leakage_guard_complete": np.asarray(1, dtype=np.int8),
        "model_frozen_before_scoring": np.asarray(1, dtype=np.int8),
    }
    for key in (
        "benchmark_report_sha256", "benchmark_checksums_sha256",
        "benchmark_panel_sha256", "component_score_bundle_sha256",
        "freeze_contract_sha256", "module_registry_sha256", "model_sha256",
    ):
        bundle[key] = np.asarray("a" * 64)
    with pytest.raises(ValueError, match="only the frozen Enveda-180"):
        assert_sealed_external_test(bundle, tmp_path / "sealed.npz")
