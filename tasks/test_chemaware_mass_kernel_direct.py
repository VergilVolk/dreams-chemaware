"""Small deterministic contracts for mass-kernel direct distillation."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from audit_chemaware_mass_kernel_embedding import fused_ranks
from audit_chemaware_peak_late_interaction import fused_ranks as late_fused_ranks
from audit_chemaware_peak_overlap_teacher import (
    fused_ranks as overlap_fused_ranks,
    weighted_bidirectional_coverage,
)
from train_chemaware_full_candidate_direct import (
    chemical_teacher_query_weights,
    kernel_teacher_loss,
    validate_clean_observability_report,
    validate_data_semantics_report,
)
from noise_final_core import sha256_file


class FakeCache:
    def get(self, row: int) -> dict[str, np.ndarray]:
        base = np.asarray([
            1.0 + (row % 2), 0.5 + (row % 3), 0.25 + (row % 5),
        ], dtype=np.float32)
        base /= np.linalg.norm(base)
        return {
            "mass": base,
            "intensity_permuted": np.roll(base, 1),
            "mass_shifted": np.roll(base, 2),
            "rule_response": np.roll(base, 1),
            "rule_mass": np.concatenate((base, np.roll(base, 1))) / np.sqrt(2.0),
            "rule_mass_shifted": np.concatenate((np.roll(base, 2), base)) / np.sqrt(2.0),
            "rule_response_shifted": np.roll(base, 2),
            "rule_response_row_permuted": np.roll(base, row % 3),
            "rule_mass_row_permuted": np.concatenate((base, np.roll(base, row % 3))) / np.sqrt(2.0),
        }


def main() -> None:
    # A rule teacher cannot reach a formal GPU run through a retrospective or
    # provenance-mismatched observability report.
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest = root / "manifest.npz"; manifest.write_bytes(b"manifest")
        token_dir = root / "tokens"; token_dir.mkdir()
        (token_dir / "report.json").write_text("{}", encoding="utf-8")
        rules = root / "rules.json"; rules.write_text("{}", encoding="utf-8")
        blocked = root / "blocked.json"
        blocked.write_text(json.dumps({
            "status": "RETROSPECTIVE_DIAGNOSTIC_ONLY", "formal": False,
            "pass_to_gpu_training": False,
        }), encoding="utf-8")
        try:
            validate_clean_observability_report(blocked, manifest, token_dir, rules)
        except RuntimeError:
            pass
        else:
            raise AssertionError("retrospective observability report was accepted")
        semantics = root / "semantics.json"
        semantics.write_text(json.dumps({
            "status": "CHEMAWARE_DATA_SEMANTICS_PASS",
            "development_training_admissible": True,
            "release_eligible": False,
            "schema_contract": {
                "membership_used_to_select_rows": False,
                "membership_blind_row_reconstruction_exact": True,
            },
            "provenance": {"manifest_sha256": sha256_file(manifest)},
        }), encoding="utf-8")
        accepted = validate_data_semantics_report(semantics, manifest, token_dir)
        assert accepted["release_eligible"] is False
        payload = json.loads(semantics.read_text(encoding="utf-8"))
        payload["schema_contract"]["membership_used_to_select_rows"] = True
        semantics.write_text(json.dumps(payload), encoding="utf-8")
        try:
            validate_data_semantics_report(semantics, manifest, token_dir)
        except RuntimeError:
            pass
        else:
            raise AssertionError("membership-filtered data semantics report was accepted")

    # Fusion must happen per spectrum pair before the molecule-level max.  The
    # old, invalid max(global)+max(kernel) order would rank the true molecule
    # first in this counterexample even though no one reference supports both.
    aggregation_counterexample = {
        "query": np.asarray([0]),
        "global": np.asarray([np.asarray([0.9, 0.0, 0.8])], dtype=object),
        "mass": np.asarray([np.asarray([0.0, 0.9, 0.8])], dtype=object),
        "reference_ptr": np.asarray([np.asarray([0, 2, 3])], dtype=object),
        "labels": np.asarray([np.asarray([True, False])], dtype=object),
    }
    assert int(fused_ranks(aggregation_counterexample, "mass", 1.0)[0]) == 2
    late_counterexample = {
        "query": np.asarray([0]),
        "official_scores": aggregation_counterexample["global"],
        "mass_token_scores": aggregation_counterexample["mass"],
        "reference_ptr": aggregation_counterexample["reference_ptr"],
        "labels": aggregation_counterexample["labels"],
    }
    assert int(late_fused_ranks(
        late_counterexample, "mass_token", 1.0, normalize_within_query=False,
    )[0]) == 2
    overlap_counterexample = {
        "query": np.asarray([0]),
        "global": aggregation_counterexample["global"],
        "direct_or_loss": aggregation_counterexample["mass"],
        "reference_ptr": aggregation_counterexample["reference_ptr"],
        "labels": aggregation_counterexample["labels"],
    }
    assert int(overlap_fused_ranks(
        overlap_counterexample, "direct_or_loss", 1.0,
    )[0]) == 2
    overlap = weighted_bidirectional_coverage(
        np.asarray([[[True, False], [False, False]]]),
        np.asarray([0.75, 0.25]), np.asarray([[0.4, 0.6]]),
    )
    assert np.allclose(overlap, [0.575])

    base = torch.tensor([0.5, 1.0, 1.5, 1.0])
    error = np.asarray([False, True, False, False])
    margin = np.asarray([0.5, -0.1, 0.01, 0.2])
    all_weight = chemical_teacher_query_weights(base, error, margin, "all", 0.02)
    assert torch.equal(all_weight, base)
    error_weight = chemical_teacher_query_weights(
        base, error, margin, "official_error", 0.02,
    )
    assert torch.equal(error_weight > 0, torch.tensor([False, True, False, False]))
    assert torch.allclose(error_weight.sum(), base.sum())
    boundary_weight = chemical_teacher_query_weights(
        base, error, margin, "official_error_or_boundary", 0.02,
    )
    assert torch.equal(boundary_weight > 0, torch.tensor([False, True, True, False]))
    assert torch.allclose(boundary_weight.sum(), base.sum())

    query = F.normalize(torch.tensor([[1.0, 0.2, 0.1], [0.1, 1.0, 0.2]]), dim=1)
    reference = F.normalize(torch.tensor([
        [1.0, 0.1, 0.0], [0.2, 1.0, 0.0],
        [0.0, 1.0, 0.1], [1.0, 0.0, 0.2],
    ]), dim=1)
    query.requires_grad_(True); reference.requires_grad_(True)
    candidate_ptr = np.asarray([0, 2, 4], dtype=np.int64)
    reference_ptr = np.arange(5, dtype=np.int64)
    reference_edge = np.arange(4, dtype=np.int64)
    rows_q = np.asarray([10, 11]); rows_r = np.asarray([20, 21, 22, 23])
    weight = torch.tensor([0.5, 1.5])
    zero = kernel_teacher_loss(
        query, reference, query.detach(), reference.detach(), rows_q, rows_r,
        candidate_ptr, reference_ptr, reference_edge, FakeCache(), "none",
        0.1, 0.07, weight,
    )
    assert float(zero) == 0.0
    loss = kernel_teacher_loss(
        query, reference, query.detach(), reference.detach(), rows_q, rows_r,
        candidate_ptr, reference_ptr, reference_edge, FakeCache(), "mass",
        0.1, 0.07, weight,
    )
    assert torch.isfinite(loss) and float(loss) > 0
    loss.backward()
    assert query.grad is not None and float(torch.linalg.vector_norm(query.grad)) > 0
    assert reference.grad is not None and float(torch.linalg.vector_norm(reference.grad)) > 0

    rank_equivalent_loss = kernel_teacher_loss(
        query, reference, query.detach(), reference.detach(), rows_q, rows_r,
        candidate_ptr, reference_ptr, reference_edge, FakeCache(), "mass",
        0.1, 0.07, weight, "rank_equivalent_kl",
    )
    assert torch.isfinite(rank_equivalent_loss)
    assert not torch.allclose(loss, rank_equivalent_loss)
    zero_margin_transfer = kernel_teacher_loss(
        query, reference, query.detach(), reference.detach(), rows_q, rows_r,
        candidate_ptr, reference_ptr, reference_edge, FakeCache(), "mass",
        0.0, 0.07, weight, "positive_margin_transfer",
    )
    assert float(zero_margin_transfer) == 0.0
    margin_transfer = kernel_teacher_loss(
        query, reference, query.detach(), reference.detach(), rows_q, rows_r,
        candidate_ptr, reference_ptr, reference_edge, FakeCache(), "mass",
        0.4, 0.07, weight, "positive_margin_transfer", 0.5, 0.05, 0.01,
    )
    assert torch.isfinite(margin_transfer) and float(margin_transfer) >= 0

    # The replacement objective supervises every query-reference chemistry
    # residual, including a non-hardest negative that scalar PMT cannot see.
    q = F.normalize(torch.tensor([[1.0, 0.0]], requires_grad=True), dim=1)
    r = F.normalize(torch.tensor([
        [1.0, 0.0], [0.8, 0.6], [0.0, 1.0],
    ], requires_grad=True), dim=1)
    q.retain_grad(); r.retain_grad()
    residual_loss = kernel_teacher_loss(
        q, r, q.detach(), r.detach(), np.asarray([10]), np.asarray([20, 21, 22]),
        np.asarray([0, 3]), np.asarray([0, 1, 2, 3]), np.arange(3),
        FakeCache(), "rule_response", 0.4, 0.07, torch.ones(1),
        "candidate_residual_huber", 0.5, 0.05, 0.01, 0.5, 0.0, 0.02,
    )
    assert torch.isfinite(residual_loss) and float(residual_loss) > 0
    residual_loss.backward()
    assert r.grad is not None and float(torch.linalg.vector_norm(r.grad[2])) > 0

    for variant in (
        "rule_response", "rule_mass", "rule_mass_shifted",
        "rule_response_shifted", "rule_response_row_permuted", "rule_mass_row_permuted",
    ):
        rule_loss = kernel_teacher_loss(
            query, reference, query.detach(), reference.detach(), rows_q, rows_r,
            candidate_ptr, reference_ptr, reference_edge, FakeCache(), variant,
            0.4, 0.07, weight,
        )
        assert torch.isfinite(rule_loss) and float(rule_loss) > 0

    beta = 0.1
    e1 = F.normalize(torch.tensor([1.0, 2.0, 3.0]), dim=0)
    e2 = F.normalize(torch.tensor([2.0, 0.5, 1.0]), dim=0)
    k1 = F.normalize(torch.tensor([0.5, 1.0, 0.2]), dim=0)
    k2 = F.normalize(torch.tensor([1.0, 0.1, 0.7]), dim=0)
    joined1 = F.normalize(torch.cat((e1, np.sqrt(beta) * k1)), dim=0)
    joined2 = F.normalize(torch.cat((e2, np.sqrt(beta) * k2)), dim=0)
    expected = (torch.dot(e1, e2) + beta * torch.dot(k1, k2)) / (1 + beta)
    assert torch.allclose(torch.dot(joined1, joined2), expected, atol=1e-6)
    print("PASS: mass-kernel direct distillation contracts")


if __name__ == "__main__":
    main()
