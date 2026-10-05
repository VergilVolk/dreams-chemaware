#!/usr/bin/env python
"""Direct gradient and dose tests for the T1/T3 algorithmic core."""
import torch
import numpy as np
from types import SimpleNamespace

from build_noise_relation_t1_t3_corpus import build_split, cap_action_anchors
from noise_relation_t1_t3_core import batch_relation_complete_loss, relation_complete_loss


def test_all_anchor_positive_negative_roles_receive_gradient() -> None:
    torch.manual_seed(7)
    anchors = torch.randn(3, 8, requires_grad=True)
    references = torch.randn(7, 8, requires_grad=True)
    pointer = torch.tensor([0, 3, 5, 7])
    loss, detail = relation_complete_loss(
        anchors, references, pointer, triplet_margin=2.0,
    )
    loss.backward()
    assert anchors.grad is not None and torch.all(anchors.grad.norm(dim=1) > 0)
    assert references.grad is not None and torch.all(references.grad.norm(dim=1) > 0)
    assert detail["anchor_count"].item() == 3


def test_duplicate_anchor_does_not_change_query_loss() -> None:
    anchor = torch.randn(1, 8)
    references = torch.randn(5, 8)
    pointer = torch.tensor([0, 2, 4, 5])
    single, _ = relation_complete_loss(anchor, references, pointer)
    repeated, _ = relation_complete_loss(anchor.repeat(5, 1), references, pointer)
    assert torch.allclose(single, repeated, atol=1e-6, rtol=0)


def test_queries_not_relation_counts_define_batch_dose() -> None:
    a = torch.randn(1, 8)
    r = torch.randn(4, 8)
    p = torch.tensor([0, 2, 3, 4])
    first, _ = relation_complete_loss(a, r, p)
    second, _ = relation_complete_loss(a + 0.1, r, p)
    weights = torch.ones(1)
    batch, _ = batch_relation_complete_loss([
        (a, r, p, weights), (a + 0.1, r, p, weights),
    ])
    assert torch.allclose(batch, 0.5 * (first + second), atol=1e-7, rtol=0)


def test_duplicate_negative_reference_does_not_reweight_its_molecule() -> None:
    torch.manual_seed(9)
    anchor = torch.randn(1, 8)
    positive = torch.randn(2, 8)
    negative_a = torch.randn(1, 8)
    negative_b = torch.randn(1, 8)
    original = torch.cat((positive, negative_a, negative_b), dim=0)
    duplicated = torch.cat((positive, negative_a, negative_a, negative_b), dim=0)
    loss_a, _ = relation_complete_loss(
        anchor, original, torch.tensor([0, 2, 3, 4]), triplet_margin=2.0,
    )
    loss_b, _ = relation_complete_loss(
        anchor, duplicated, torch.tensor([0, 2, 4, 5]), triplet_margin=2.0,
    )
    assert torch.allclose(loss_a, loss_b, atol=1e-6, rtol=0)


def test_corpus_keeps_positive_first_and_all_query_actions() -> None:
    graph = SimpleNamespace(
        query_ptr=np.asarray([0, 3, 5]),
        molecule_ptr=np.asarray([0, 2, 4, 5, 7, 9]),
        pair_candidate_row=np.asarray([10, 11, 20, 21, 22, 30, 31, 40, 41]),
        query_row=np.asarray([10, 30]),
        query_formula=np.asarray(["F", "G"]),
        molecule_formula=np.asarray(["F", "F", "X", "G", "G"]),
        molecule_mces_grade=np.asarray([-1, 1, 3, -1, 2]),
    )
    pair = np.asarray([1.0, .5, .8, .7, .9, 1.0, .4, .6, .5])
    molecule = np.asarray([1.0, .8, .9, 1.0, .6])
    arrays, report = build_split(
        graph, pair, molecule, np.asarray([0, 1]), {0: [3, 1, 3]},
        max_positive_refs=4, max_negative_molecules=4,
        additional_boundary_molecules=2, max_negative_refs=3,
    )
    assert report["eligible_queries"] == 2
    assert np.array_equal(arrays["action_index"], np.asarray([1, 3]))
    assert np.array_equal(arrays["action_ptr"], np.asarray([0, 2, 2]))
    assert np.array_equal(np.diff(arrays["molecule_ptr"]), np.asarray([3, 2]))
    assert arrays["reference_row"][0] == 11 and arrays["reference_row"][4] == 31


def test_action_anchor_cap_is_deterministic_and_budget_safe() -> None:
    raw = {5: [9, 3, 3, 7, 1], 6: list(range(200)), 7: []}
    capped, count = cap_action_anchors(raw, 40)
    assert count == 1
    assert capped[5] == [1, 3, 7, 9]
    assert len(capped[6]) == 40 and capped[6][0] == 0 and capped[6][-1] == 199
    assert capped[6] == sorted(set(capped[6]))
    assert capped[7] == []
    assert all(len(actions) <= 40 for actions in capped.values())
    again, again_count = cap_action_anchors(raw, 40)
    assert again == capped and again_count == count


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"[test_noise_relation_t1_t3_core] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
