import numpy as np
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from train_noise_final_r2_shared_encoder import (
    Example, SpectrumStore, correction_loss, identity_epoch_sample, margins,
    parse_controls, parse_path, safety_loss,
)


class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 4)

    def forward(self, spectra):
        return F.normalize(self.linear(spectra.reshape(len(spectra), -1)), dim=-1)


def main() -> None:
    assert parse_path("3,7,9") == (3, 7, 9)
    assert parse_controls("3,7;4,8") == ((3, 7), (4, 8))
    examples = [
        Example(0, 1, "A", "F1", (2,), (3,)),
        Example(1, 4, "A", "F1", (5,), (6,)),
        Example(2, 7, "B", "F2", (8,), (9,)),
    ]
    sampled = identity_epoch_sample(examples, np.random.default_rng(3))
    assert len(sampled) == 2 and {value.identity for value in sampled} == {"A", "B"}
    encoded = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]])
    layout = [{"clean": 0, "positive": [1], "negative": [2]}]
    observed = margins(encoded, layout, "clean")
    assert torch.allclose(observed, torch.tensor([0.8]))
    store = SpectrumStore.__new__(SpectrumStore)
    store.rows = np.asarray([10, 11, 12], dtype=np.int64)
    store.position = {10: 0, 11: 1, 12: 2}
    store.tensor = torch.tensor([
        [[100.0, 1.1], [20.0, 1.0], [40.0, 0.7], [60.0, 0.4], [0.0, 0.0]],
        [[100.0, 1.1], [20.0, 0.9], [41.0, 0.7], [60.0, 0.3], [0.0, 0.0]],
        [[100.0, 1.1], [25.0, 1.0], [45.0, 0.6], [65.0, 0.2], [0.0, 0.0]],
    ])
    model = Dummy()
    official = {
        row: model(store.one(row).unsqueeze(0)).detach().numpy()[0]
        for row in (10, 11, 12)
    }
    action = Example(
        0, 10, "A", "F1", (11,), (12,), target_path=(1,),
        control_paths=((2,), (3,)), attenuation=0.5, official_margin=0.1,
    )
    args = SimpleNamespace(
        amp=False, rank_margin=0.05, temperature=0.1,
        specificity_margin=0.01, lambda_action_rank=0.75,
        lambda_transfer=0.5, lambda_specificity=0.25,
        lambda_preserve=5.0, protected_margin_slack=0.01,
        lambda_robust=0.5, lambda_protected=1.0,
    )
    loss, report = correction_loss(model, store, [action], torch.device("cpu"), official, args)
    assert torch.isfinite(loss) and "corr_specificity" in report
    loss.backward()
    model.zero_grad(set_to_none=True)
    protected = Example(0, 10, "A", "F1", (11,), (12,), official_margin=0.1)
    loss, report = safety_loss(
        model, store, [protected], torch.device("cpu"), official, args, False,
    )
    assert torch.isfinite(loss) and "protected_floor" in report
    loss.backward()
    print("[test_noise_final_r2_shared_encoder] PASS")


if __name__ == "__main__":
    main()
