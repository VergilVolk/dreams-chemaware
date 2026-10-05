"""Small checks for target-first A4 control construction."""
from __future__ import annotations

import numpy as np
import torch

from audit_noise_corrected_a4_router import action_identifier, choose_control_tokens


def main() -> None:
    clean = torch.zeros((8, 2), dtype=torch.float32)
    clean[0] = torch.tensor([500.0, 1.0])
    clean[1:6, 0] = torch.tensor([50., 51., 80., 81., 120.])
    clean[1:6, 1] = torch.tensor([1., .95, .6, .55, .2])
    roles = np.asarray([-1, 1, 1, 2, 2, 3, -1, -1], dtype=np.int8)
    tokens, kind = choose_control_tokens(clean, 1, roles, 1, 7)
    assert kind == "strict_same_role" and tokens.tolist() == [2]
    tokens, kind = choose_control_tokens(clean, 5, roles, 2, 7)
    assert len(tokens) == 2 and kind == "relaxed_peak_matched"
    assert action_identifier(9, 2, .5) == action_identifier(9, 2, .5)
    assert action_identifier(9, 2, .5) != action_identifier(9, 3, .5)
    print("[noise corrected A4 router tests] PASS=4")


if __name__ == "__main__":
    main()
