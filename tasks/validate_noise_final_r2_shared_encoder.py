from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    decision = json.loads((args.output_dir / "decision.json").read_text(encoding="utf-8"))
    if decision.get("status") != "noise_final_r2_shared_encoder_pilot_complete":
        raise RuntimeError("not an R2 decision")
    contract = decision.get("contracts", {})
    if contract.get("P2b") != "forbidden" or not contract.get("shared_query_reference_encoder"):
        raise RuntimeError("R2 crossed the embedding/reranker boundary")
    if contract.get("dropout_disabled_during_gradient_training") is not True:
        raise RuntimeError("R2 reintroduced train-mode dropout")
    if decision["zero_change_gate"]["preservation_mean"] < 0.9999:
        raise RuntimeError("R2 zero-change gate failed")
    if not (args.output_dir / "final_shared_encoder.pt").is_file():
        raise RuntimeError("R2 shared-encoder checkpoint is missing")
    print(
        "[validate_noise_final_r2_shared_encoder] PASS "
        f"dR1={decision['held_clean']['delta_recall1']:+.4f} "
        f"C/I={decision['held_clean']['corrected']}/{decision['held_clean']['introduced']}"
    )


if __name__ == "__main__":
    main()
