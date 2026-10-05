#!/usr/bin/env python
"""Tests for the GNPS-panel selector: rule semantics and sbatch contract."""
import json
import sys
import tempfile
from pathlib import Path

import summarize_noise_relation_t1_t3_v2_gnps_select as selector


def _paired_recall(delta_pp: float, ci_low_pp: float) -> dict:
    return {
        "delta": delta_pp / 100.0,
        "delta_pp": delta_pp,
        "ci_low": ci_low_pp / 100.0,
        "ci_high": 0.0,
        "ci_low_pp": ci_low_pp,
        "ci_high_pp": 0.0,
    }


def _report(identity: tuple, formula: tuple, queries: int = 1000) -> dict:
    return {
        "panels": {
            "identity_disjoint": {
                "paired": {
                    "queries": queries,
                    "formula_cluster_paired_ci": {
                        "recall@1": _paired_recall(*identity),
                    },
                },
            },
            "formula_disjoint": {
                "paired": {
                    "queries": queries,
                    "formula_cluster_paired_ci": {
                        "recall@1": _paired_recall(*formula),
                    },
                },
            },
        },
    }


def test_arm_evidence_reads_both_panels() -> None:
    evidence = selector.arm_evidence(
        _report((0.4, 0.1), (-0.2, -0.5), queries=1234), "clean-only",
    )
    assert evidence["identity_disjoint"]["delta_pp"] == 0.4
    assert evidence["identity_disjoint"]["ci_low_pp"] == 0.1
    assert evidence["formula_disjoint"]["delta_pp"] == -0.2
    assert evidence["identity_disjoint"]["queries"] == 1234


def test_min_panel_rule_selects_conservatively_and_gates_held() -> None:
    # averaged is largest on one panel but negative on the other;
    # clean-only is modestly positive on both.  The predeclared worst-panel
    # rule must select clean-only and allow the held spend.
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        reports = {
            "clean-only": _report((0.3, 0.05), (0.2, 0.02)),
            "averaged": _report((1.0, 0.6), (-0.1, -0.4)),
            "action-rotation": _report((0.1, -0.1), (0.05, -0.2)),
        }
        for arm, body in reports.items():
            (root / f"{arm}.json").write_text(
                json.dumps(body), encoding="utf-8",
            )
        output = root / "selection.json"
        sys.argv = [
            "summarize_noise_relation_t1_t3_v2_gnps_select.py",
            "--clean-only-report", str(root / "clean-only.json"),
            "--averaged-report", str(root / "averaged.json"),
            "--action-rotation-report", str(root / "action-rotation.json"),
            "--output", str(output),
        ]
        selector.main()
        selection = json.loads(output.read_text(encoding="utf-8"))
        assert selection["selected_arm"] == "clean-only"
        assert selection["proceed_to_held"] is True
        assert selection["no_go_reason"] is None

        # All-negative worst panels must refuse the held spend.
        output2 = root / "selection2.json"
        for arm, body in reports.items():
            body["panels"]["identity_disjoint"]["paired"][
                "formula_cluster_paired_ci"
            ]["recall@1"] = _paired_recall(-0.3, -0.5)
            body["panels"]["formula_disjoint"]["paired"][
                "formula_cluster_paired_ci"
            ]["recall@1"] = _paired_recall(-0.4, -0.6)
            (root / f"{arm}.json").write_text(
                json.dumps(body), encoding="utf-8",
            )
        sys.argv[-1] = str(output2)
        selector.main()
        refusal = json.loads(output2.read_text(encoding="utf-8"))
        assert refusal["selected_arm"] is None
        assert refusal["proceed_to_held"] is False
        assert "unsupported" in refusal["no_go_reason"]


def test_query_count_mismatch_aborts() -> None:
    with tempfile.TemporaryDirectory() as scratch:
        root = Path(scratch)
        (root / "clean-only.json").write_text(
            json.dumps(_report((0.1, 0.0), (0.1, 0.0), queries=1000)),
            encoding="utf-8",
        )
        (root / "averaged.json").write_text(
            json.dumps(_report((0.1, 0.0), (0.1, 0.0), queries=999)),
            encoding="utf-8",
        )
        (root / "action-rotation.json").write_text(
            json.dumps(_report((0.1, 0.0), (0.1, 0.0), queries=1000)),
            encoding="utf-8",
        )
        sys.argv = [
            "summarize_noise_relation_t1_t3_v2_gnps_select.py",
            "--clean-only-report", str(root / "clean-only.json"),
            "--averaged-report", str(root / "averaged.json"),
            "--action-rotation-report", str(root / "action-rotation.json"),
            "--output", str(root / "selection.json"),
        ]
        try:
            selector.main()
        except RuntimeError as error:
            assert "different" in str(error)
        else:
            raise AssertionError("query-count mismatch did not abort")


def test_sbatch_contract() -> None:
    text = Path(__file__).with_name(
        "run_noise_relation_t1_t3_v2_gnps_select_1gpu.sbatch",
    ).read_text(encoding="utf-8")
    # One GPU, all three arms plus Stage-1 encoded through the slim
    # conversion, predeclared summarizer, and NO training and NO held
    # evaluation inside this job.
    assert "#SBATCH --gpus=1" in text
    assert "evaluate_noise_dreams_native.py" not in text
    assert "train_noise_relation" not in text
    assert "prepare_official_embedding_checkpoint.py" in text
    assert '--checkpoint "$checkpoint"' in text  # encode() forwards its argument
    assert 'encode stage1 "$STAGE1_SLIM"' in text
    assert 'encode averaged "$AB_SLIM"' in text
    assert 'encode clean-only "$CLEAN_SLIM"' in text
    assert 'encode action-rotation "$ROTATION_SLIM"' in text
    assert "summarize_noise_relation_t1_t3_v2_gnps_select.py" in text
    assert "2347055" in text and "2347230" in text
    assert "proceed_to_held" in text


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"[test_noise_relation_t1_t3_v2_gnps_select] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
