"""Static contract tests for E10."""
from __future__ import annotations
import torch
from audit_noise_final_e10_positive_residual_matrix import action_cells, cell_variant


def main() -> None:
    cells = action_cells()
    if len(cells) != 14 or len(set(cells)) != 14:
        raise AssertionError("E10 must contain 7 positive cells and 7 controls")
    clean = torch.zeros((6, 2), dtype=torch.float32)
    clean[0] = torch.tensor([500.0, 1.0])
    clean[1] = torch.tensor([100.0, 1.0])
    clean[2] = torch.tensor([150.0, 0.5])
    prevalence = torch.tensor([0, 1, 1, 0, 0, 0], dtype=torch.float32).numpy()
    target = clean[:, 1].numpy().copy()
    target[2] = 0.2
    intensity = cell_variant(clean, (prevalence, target), torch.empty((0, 2)).numpy(),
                             "consensus_projection", 0.5)
    if torch.equal(intensity, clean):
        raise AssertionError("E10 intensity action did not change the spectrum")
    expanded = action_cells("expanded")
    if len(expanded) != 38 or len(set(expanded)) != 38:
        raise AssertionError("E10-B must contain 19 positive cells and 19 controls")
    combined = cell_variant(
        clean, (prevalence, target),
        torch.tensor([[175.0, 0.4, 1.0]], dtype=torch.float32).numpy(),
        "transport_then_union", 0.5, 0.5,
    )
    if torch.equal(combined, clean):
        raise AssertionError("E10-B sequential action did not change the spectrum")
    print("[test_noise_final_e10_positive_residual_matrix] PASS", flush=True)


if __name__ == "__main__":
    main()
