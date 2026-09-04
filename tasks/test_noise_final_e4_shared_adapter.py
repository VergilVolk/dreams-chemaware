import torch

from train_noise_final_e4_shared_adapter import balanced_pcgrad, pcgrad, safe_target


def main() -> None:
    clean = torch.tensor([[1.0, 0.0]])
    raw = torch.tensor([[0.0, 1.0]])
    target = safe_target(clean, raw, 0.1)
    assert torch.allclose(target.norm(dim=1), torch.ones(1), atol=1e-6)
    assert float(target[0, 1]) > 0
    gradients = [[torch.tensor([1.0, 0.0])], [torch.tensor([-1.0, 1.0])], [torch.tensor([0.0, 1.0])]]
    merged, conflicts = pcgrad(gradients)
    assert conflicts > 0 and torch.isfinite(merged[0]).all()
    balanced, _, norms = balanced_pcgrad([
        [torch.tensor([100.0, 0.0])], [torch.tensor([0.0, 1.0])],
    ])
    assert norms == [100.0, 1.0]
    # The 100x objective may set the restored global scale, but it may not
    # erase the orthogonal objective direction.
    assert float(balanced[0][1]) > 0
    print("[test_noise_final_e4_shared_adapter] PASS")


if __name__ == "__main__":
    main()
