from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    decision = json.loads((args.output_dir / "decision.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(args.output_dir / "adapter.pt", map_location="cpu", weights_only=False)
    if decision.get("status") != "noise_final_e4_shared_adapter_pilot_complete":
        raise RuntimeError("wrong E4-M1 decision status")
    if checkpoint.get("status") != "noise_final_e4_shared_clean_embedding_adapter":
        raise RuntimeError("wrong E4-M1 checkpoint status")
    if decision.get("algorithm_version") != "e4_m1b_gradient_balanced_pcgrad":
        raise RuntimeError("E4-M1 uses the obsolete raw-loss-scale algorithm")
    active_families = decision.get("active_families")
    if not isinstance(active_families, list) or not active_families:
        raise RuntimeError("E4-M1 does not declare active mechanism families")
    if active_families != checkpoint.get("active_families"):
        raise RuntimeError("decision/checkpoint family mismatch")
    if decision["contracts"].get("P2b") != "forbidden" or decision["contracts"].get("P3_consumed") is not False:
        raise RuntimeError("E4-M1 contract violation")
    if decision["contracts"].get("raw_loss_scale_mixing") != "forbidden":
        raise RuntimeError("E4-M1 permits raw-scale objective mixing")
    if not checkpoint.get("query_reference_encoder_shared") or checkpoint.get("P2b_used"):
        raise RuntimeError("checkpoint is not a shared P2b-free adapter")
    zero = decision["history"][0]["inner_full"]
    if zero["corrected"] != 0 or zero["introduced"] != 0 or zero["delta_recall1"] != 0:
        raise RuntimeError("E4-M1 epoch zero does not reproduce official embedding")
    zero_action = decision["history"][0]["inner_action"]
    expected_action_keys = {f"{family}_mean_gain" for family in active_families}
    if not expected_action_keys.issubset(zero_action):
        raise RuntimeError("E4-M1 family realization fields are incomplete")
    if set(decision["gates"]) != {
        "selected_nonzero_epoch", "outer_clean_risk_net_positive",
        "outer_clean_recall1_positive", "outer_clean_mrr_nonnegative",
        "outer_near_nonnegative", "outer_preservation", "outer_action_realized_all_families",
    }:
        raise RuntimeError("E4-M1 gate schema mismatch")
    print(
        "[validate_noise_final_e4_shared_adapter] PASS "
        f"best_epoch={decision['best_epoch']} replicate={decision['pass_to_multifold_replication']}"
    )


if __name__ == "__main__":
    main()
