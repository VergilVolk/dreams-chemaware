import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_directional_noise_v2_m1_decision",
    ROOT / "tasks/analyze_directional_noise_v2_m1_decision.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_summary_counts_target_and_expected_random_transitions():
    frame = pd.DataFrame({
        "ik14": ["A", "B"], "formula": ["F1", "F2"],
        "baseline_top1": [0, 1], "target_top1": [1, 0],
        "random_top1_mean": [1 / 3, 2 / 3],
        "target_minus_random_top1_delta": [2 / 3, -2 / 3],
    })
    summary = MODULE.summarize(frame, bootstrap=100, seed=1)
    assert summary["target_corrected"] == 1
    assert summary["target_introduced"] == 1
    assert abs(summary["expected_random_corrected"] - 1 / 3) < 1e-12
    assert abs(summary["expected_random_introduced"] - 1 / 3) < 1e-12


def test_as_bool_does_not_treat_false_string_as_true():
    parsed = MODULE.as_bool(pd.Series(["True", "False", "1", "0"]))
    assert parsed.tolist() == [True, False, True, False]


def test_main_separates_robustness_from_error_correction(tmp_path, monkeypatch):
    m1 = tmp_path / "m1"
    out = tmp_path / "decision"
    m1.mkdir()
    key_rows = [
        {"query_row": 1, "positive_row": 11, "negative_row": 21, "ik14": "A", "formula": "F1", "adduct": "M+H"},
        {"query_row": 2, "positive_row": 12, "negative_row": 22, "ik14": "B", "formula": "F2", "adduct": "M+H"},
    ]
    variants = []
    # Query A starts correct and targeted remains correct; query B starts wrong
    # and only targeted becomes correct, while one of three random views does.
    margins = [(0.2, [0.1, 0.1, 0.1]), (0.1, [0.1, -0.1, -0.1])]
    for base, (target_margin, random_margins) in zip(key_rows, margins):
        variants.append(base | {"condition": "targeted", "repeat": -1, "perturbed_margin": target_margin})
        for repeat, margin in enumerate(random_margins):
            variants.append(base | {"condition": "matched_random", "repeat": repeat, "perturbed_margin": margin})
    pd.DataFrame(variants).to_csv(m1 / "variant_results.csv.gz", index=False)
    paired = pd.DataFrame([
        key_rows[0] | {"baseline_margin": 0.2, "target_margin_change": 0.0, "random_margin_change": -0.1,
                       "target_minus_random_margin_change": 0.1, "cross_condition_positive": "False"},
        key_rows[1] | {"baseline_margin": -0.1, "target_margin_change": 0.2, "random_margin_change": 0.0667,
                       "target_minus_random_margin_change": 0.1333, "cross_condition_positive": "True"},
    ])
    paired.to_csv(m1 / "paired_margin_effects.csv.gz", index=False)
    pd.DataFrame(key_rows).to_csv(m1 / "selected_triples.csv.gz", index=False)
    monkeypatch.setattr(MODULE, "parse_args", lambda: Namespace(
        m1_dir=m1, output_dir=out, bootstrap=100, seed=7,
    ))
    MODULE.main()
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["top1_results"]["baseline_correct"]["target_introduced"] == 0
    assert report["top1_results"]["baseline_wrong"]["target_corrected"] == 1
    assert report["top1_results"]["baseline_wrong"]["expected_random_corrected"] == 1 / 3
    assert report["top1_results"]["cross_condition"]["queries"] == 1
