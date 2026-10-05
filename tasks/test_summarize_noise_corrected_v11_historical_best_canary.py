"""CPU-only transport test for the V11 four-arm summary."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

from summarize_noise_corrected_v10_safe_exact_canary import summarize
from test_summarize_noise_corrected_v10_safe_exact_canary import _fixture


def _v11_fixture(root: Path) -> dict[str, Path]:
    paths = _fixture(root)
    for path in paths.values():
        decision_path = path / "decision.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        decision["configuration"]["direct_contract"] = (
            "best_action_v11_historical_best"
        )
        decision["configuration"]["action_bank_contract"] = "historical_best_v1"
        decision_path.write_text(json.dumps(decision), encoding="utf-8")
    return paths


def test_v11_summary_requires_the_frozen_action_supplier_contract() -> None:
    with tempfile.TemporaryDirectory() as temp:
        report = summarize(
            _v11_fixture(Path(temp)),
            repeats=100,
            seed=23,
            expected_direct_contract="best_action_v11_historical_best",
            expected_action_bank_contract="historical_best_v1",
            status="noise_corrected_v11_historical_best_canary_complete",
            result_name="V11",
        )
    assert report["status"] == "noise_corrected_v11_historical_best_canary_complete"
    assert report["gates"]["advance_to_larger_direct_finetuning_pilot"] is True


def test_v11_summary_fails_if_one_arm_uses_the_v10_multi_action_bank() -> None:
    with tempfile.TemporaryDirectory() as temp:
        paths = _v11_fixture(Path(temp))
        path = paths["matched_shuffled"] / "decision.json"
        decision = json.loads(path.read_text(encoding="utf-8"))
        decision["configuration"]["action_bank_contract"] = "all_strict_top1"
        path.write_text(json.dumps(decision), encoding="utf-8")
        try:
            summarize(
                paths,
                repeats=50,
                seed=29,
                expected_direct_contract="best_action_v11_historical_best",
                expected_action_bank_contract="historical_best_v1",
                status="noise_corrected_v11_historical_best_canary_complete",
                result_name="V11",
            )
        except RuntimeError as error:
            assert "action-bank contract drifted" in str(error)
        else:
            raise AssertionError("V11 summary accepted a mixed action-bank contract")


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(
        "[test_summarize_noise_corrected_v11_historical_best_canary] "
        f"PASS tests={len(tests)}"
    )


if __name__ == "__main__":
    main()
