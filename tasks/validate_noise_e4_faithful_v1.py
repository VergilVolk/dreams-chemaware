"""Fail-closed validator for an E4-A direct shared-embedding run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    decision_path = args.output_dir / "decision.json"
    checkpoint_path = args.output_dir / "final_shared_encoder.pt"
    if not decision_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("E4-A output lacks decision or shared encoder checkpoint")
    report = json.loads(decision_path.read_text(encoding="utf-8"))
    if report.get("status") != "noise_final_e4a_direct_augmentation_complete":
        raise RuntimeError("unexpected E4-A status")
    contracts = report.get("contracts", {})
    expected = {
        "shared_query_reference_encoder": True,
        "model_weights_changed": True,
        "clean_and_augmented_raw_spectra_train_same_encoder": True,
        "action_outcomes_used_for_weights_or_selection": False,
        "inference_clean_spectrum_only": True,
        "teacher": "forbidden",
        "P2b": "forbidden",
        "P3_consumed": False,
    }
    for key, value in expected.items():
        if contracts.get(key) != value:
            raise RuntimeError(f"E4-A contract failed: {key}={contracts.get(key)!r}")
    held = report.get("held_clean", {})
    for key in ("baseline_recall1", "recall1", "delta_recall1", "corrected", "introduced"):
        if key not in held:
            raise RuntimeError(f"E4-A held-clean metric missing: {key}")
    print(
        "[validate_noise_final_e4a_direct_augmentation] PASS "
        f"policy={report['configuration']['policy']} "
        f"dR1={held['delta_recall1']:+.4f} C/I={held['corrected']}/{held['introduced']}"
    )


if __name__ == "__main__":
    main()
