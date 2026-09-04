"""Unit tests for E9-B no-op-aware query collapse."""
from __future__ import annotations
import pandas as pd
from audit_noise_final_e9b_residual_action_headroom import choose


def main() -> None:
    frame = pd.DataFrame({
        "query_index": [1, 1], "selector": ["candidate_gradient", "role_confounder"],
        "step": [3, 1], "clean_rank": [2, 2], "clean_margin": [-0.1, -0.1],
        "frozen_rank": [1, 2], "frozen_margin": [0.2, -0.2],
        "online_rank": [2, 2], "online_margin": [-0.05, -0.2],
        "frozen_path": ["1,2,3", "4"], "online_path": ["1,2,5", "4"],
    })
    frozen = choose(frame, "frozen")
    online = choose(frame, "online")
    if frozen["frozen_oracle_selector"] != "candidate_gradient" or frozen["frozen_oracle_rank"] != 1:
        raise AssertionError("E9-B failed to select the beneficial action")
    if online["online_oracle_selector"] != "candidate_gradient":
        raise AssertionError("E9-B should select the less-negative candidate action")
    tie = frame.iloc[[0]].copy()
    tie["clean_rank"] = 1
    tie["clean_margin"] = 0.2
    tie["frozen_rank"] = 1
    tie["frozen_margin"] = 0.2
    if choose(tie, "frozen")["frozen_oracle_selector"] != "no_op":
        raise AssertionError("E9-B must prefer no-op on an exact margin tie")
    print("[test_noise_final_e9b_residual_action_headroom] PASS", flush=True)


if __name__ == "__main__":
    main()
