"""Deterministic identity-initialization tests for C2-A experts."""
from __future__ import annotations

import torch

from train_noise_v3_c2a_token_direction_expert import DirectionExpert


def main() -> None:
    torch.manual_seed(5)
    clean = torch.nn.functional.normalize(torch.randn(4, 16), dim=1)
    token = torch.randn(4, 7, 8)
    mz = torch.rand(4, 7) * 1000
    intensity = torch.rand(4, 7)
    valid = torch.ones(4, 7, dtype=torch.bool)
    for use_tokens in (False, True):
        model = DirectionExpert(16, 8, 12, use_tokens)
        kwargs = {"token": token, "mz": mz, "intensity": intensity, "valid": valid} if use_tokens else {}
        with torch.inference_mode():
            output = model(clean, **kwargs)
        assert torch.allclose(output, clean, atol=1e-6)
    print("[test_noise_v3_c2a_token_direction] PASS", flush=True)


if __name__ == "__main__": main()
