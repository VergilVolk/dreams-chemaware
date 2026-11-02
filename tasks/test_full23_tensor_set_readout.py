from __future__ import annotations

import numpy as np
import torch

from full23_tensor_set_readout_core import (
    ENCODER_NAMES,
    Full23TensorSetReadout,
    TensorSetSettings,
    weighted_anchor_safe_ranking_loss,
    structural_frontier_query_weights,
)


def main() -> None:
    torch.manual_seed(17)
    settings = TensorSetSettings(
        embedding_dimension=12,
        pair_dimension=8,
        encoder_dimension=8,
        attention_heads=2,
        fusion_hidden=16,
    )
    model = Full23TensorSetReadout(settings, residual_bound=0.5)
    banks = {
        name: torch.randn(24, settings.embedding_dimension, requires_grad=True)
        for name in ENCODER_NAMES
    }
    count = 5
    query = torch.tensor([[0, 1, 2, 3]] * count)
    refs = torch.tensor([
        [[4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
    ] * count)
    mask = torch.ones(count, len(ENCODER_NAMES), 3, dtype=torch.bool)
    mask[1, :, 2] = False
    anchor = torch.linspace(-0.2, 0.2, count)

    # Zero-initialized residual is exactly the parent, not merely close.
    initial = model(anchor, query, refs, mask, banks)
    assert torch.equal(initial, anchor)

    labels = np.asarray([1, 0, 0, 1, 0], dtype=np.int8)
    ptr = np.asarray([0, 3, 5], dtype=np.int64)
    weights = np.asarray([1.0, 1.0], dtype=np.float32)
    loss, parts = weighted_anchor_safe_ranking_loss(
        initial, anchor, ptr, labels, weights,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())

    # The zero final layer initially blocks upstream gradients by design; one
    # optimizer step opens the path, after which every encoder receives signal.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    for bank in banks.values():
        bank.grad = None
    logits = model(anchor, query, refs, mask, banks)
    loss, _ = weighted_anchor_safe_ranking_loss(logits, anchor, ptr, labels, weights)
    loss.backward()
    reference_rows = {
        ENCODER_NAMES[0]: (4, 5, 6),
        ENCODER_NAMES[1]: (7, 8, 9),
        ENCODER_NAMES[2]: (10, 11, 12),
        ENCODER_NAMES[3]: (13, 14, 15),
    }
    for name, bank in banks.items():
        assert bank.grad is not None and torch.count_nonzero(bank.grad) > 0, name
        # All three reference slots are valid for candidate 0 and must be
        # reachable; this explicitly guards against an accidental Top-1-only
        # implementation.
        for row in reference_rows[name]:
            assert torch.count_nonzero(bank.grad[row]) > 0, (name, row)

    # Changing a valid non-top reference changes a trained readout, while a
    # masked reference cannot affect it.
    model.eval()
    with torch.no_grad():
        base = model(anchor, query, refs, mask, banks)
        changed_banks = {name: value.detach().clone() for name, value in banks.items()}
        changed_banks[ENCODER_NAMES[0]][6] += 3.0
        changed = model(anchor, query, refs, mask, changed_banks)
        assert not torch.equal(base, changed)
        masked_banks = {name: value.detach().clone() for name, value in banks.items()}
        masked_banks[ENCODER_NAMES[0]][6] += 3.0
        single_mask = mask.clone()
        single_mask[:, 0, 2] = False
        before = model(anchor, query, refs, single_mask, banks)
        after = model(anchor, query, refs, single_mask, masked_banks)
        assert torch.allclose(before, after, atol=1e-6, rtol=0.0)

    # Transfer-safe variants remove reference-count shortcuts and may center
    # raw residuals within each query without changing the frozen parent at
    # initialization.
    safe_settings = TensorSetSettings(
        embedding_dimension=12, pair_dimension=8, encoder_dimension=8,
        attention_heads=2, fusion_hidden=16, include_reference_count=False,
        query_center_raw_residual=True,
    )
    safe = Full23TensorSetReadout(safe_settings, residual_bound=0.5)
    safe_initial = safe(anchor, query, refs, mask, banks, query_ptr=ptr)
    assert torch.equal(safe_initial, anchor)
    safe_logits, safe_raw, safe_bounded = safe.forward_components(
        anchor, query, refs, mask, banks, query_ptr=ptr,
    )
    assert torch.equal(safe_logits, anchor)
    assert torch.equal(safe_raw, torch.zeros_like(safe_raw))
    assert torch.equal(safe_bounded, torch.zeros_like(safe_bounded))
    try:
        safe(anchor, query, refs, mask, banks)
    except ValueError as error:
        assert "query_ptr" in str(error)
    else:
        raise AssertionError("query-centered model accepted missing query_ptr")

    # Later cumulative v1.3 repairs may expose anchor competition and retain
    # signed q-r information without changing the exact parent initialization.
    richer_settings = TensorSetSettings(
        embedding_dimension=12, pair_dimension=16, encoder_dimension=8,
        attention_heads=2, fusion_hidden=16, include_reference_count=False,
        query_center_raw_residual=True, include_anchor_context=True,
        include_signed_difference=True,
    )
    richer = Full23TensorSetReadout(richer_settings, residual_bound=0.8)
    richer_initial = richer(anchor, query, refs, mask, banks, query_ptr=ptr)
    assert torch.equal(richer_initial, anchor)
    context = richer._anchor_context(anchor, ptr)
    assert context.shape == (count, 3)
    assert torch.allclose(context[:3, 2], context[0, 2].expand(3))

    print("PASS: Full23 information-preserving tensor-set readout contracts")


if __name__ == "__main__":
    main()
