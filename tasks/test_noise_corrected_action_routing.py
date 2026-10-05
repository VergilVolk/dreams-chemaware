"""Deterministic checks for corrected action routing and diversity."""
from __future__ import annotations

import pandas as pd

from noise_corrected_action_routing import (
    RoutingThresholds, route_action, select_diverse_routed_actions,
)


def main() -> None:
    t = RoutingThresholds()
    assert route_action(
        clean_rank=2, clean_margin=-0.1, action_rank=1, action_margin=0.04,
        control_margin=-0.02, thresholds=t,
    ) == "corrective"
    assert route_action(
        clean_rank=1, clean_margin=0.1, action_rank=2, action_margin=-0.01,
        control_margin=0.0, thresholds=t,
    ) == "harmful"
    assert route_action(
        clean_rank=1, clean_margin=0.1, action_rank=1, action_margin=0.098,
        control_margin=0.09, thresholds=t,
    ) == "robustness_only"
    assert route_action(
        clean_rank=2, clean_margin=-0.1, action_rank=2, action_margin=-0.095,
        control_margin=-0.1, thresholds=t,
    ) == "uncertain"

    records = []
    for index, (source, family, gain) in enumerate([
        ("P", "union", .08), ("P", "union", .07),
        ("P", "transport", .06), ("N", "gradient", .05),
    ]):
        records.append({
            "query_index": 1, "action_id": f"c{index}", "source": source,
            "family": family, "route": "corrective", "margin_change": gain,
            "paired_advantage": gain,
        })
    for index, harm in enumerate((.09, .08, .07)):
        records.append({
            "query_index": 1, "action_id": f"h{index}", "source": "P",
            "family": "risk", "route": "harmful", "margin_change": -harm,
            "paired_advantage": -harm,
        })
    selected = select_diverse_routed_actions(
        pd.DataFrame(records), maximum_corrective_per_query=3,
        maximum_risk_per_query=2,
    )
    corrective = set(selected.loc[selected.selected_corrective, "action_id"])
    assert corrective == {"c3", "c2", "c0"}, corrective
    assert int(selected.selected_risk.sum()) == 2
    assert not selected.loc[selected.action_id.eq("c1"), "selected_corrective"].item()
    print("[noise corrected action routing tests] PASS=5")


if __name__ == "__main__":
    main()
