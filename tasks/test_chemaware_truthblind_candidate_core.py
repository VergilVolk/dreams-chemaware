from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from chemaware_truthblind_candidate_core import (  # noqa: E402
    FEATURE_NAMES,
    build_truthblind_candidate_features,
    arbitrate_truthblind_predictions,
    predict_truthblind_baselines,
    predict_truthblind_direct_first_backoff,
    predict_truthblind_direct_first_exclusive_backoff,
    predict_truthblind_policy,
    predict_truthblind_policy_arms,
    truthblind_scored_view,
)


class Classifier:
    def __init__(self, sign: float):
        self.sign = float(sign)

    def predict_proba(self, feature: np.ndarray) -> np.ndarray:
        score = self.sign * np.mean(np.asarray(feature, dtype=np.float64), axis=1)
        probability = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack((1.0 - probability, probability))


class Regressor:
    def __init__(self, scale: float):
        self.scale = float(scale)

    def predict(self, feature: np.ndarray) -> np.ndarray:
        return self.scale * np.mean(np.asarray(feature, dtype=np.float64), axis=1)


def scored(rule_key: str, rule_values: list[np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "global": np.asarray([
            np.asarray([0.80, 0.70, 0.75], dtype=np.float32),
            np.asarray([0.40, 0.60, 0.55, 0.58], dtype=np.float32),
        ], dtype=object),
        "mass": np.asarray([
            np.asarray([0.20, 0.10, 0.50], dtype=np.float32),
            np.asarray([0.30, 0.20, 0.50, 0.40], dtype=np.float32),
        ], dtype=object),
        rule_key: np.asarray(rule_values, dtype=object),
        "reference_ptr": np.asarray([
            np.asarray([0, 2, 3], dtype=np.int32),
            np.asarray([0, 1, 3, 4], dtype=np.int32),
        ], dtype=object),
    }


def main() -> None:
    actions = [(0.0, 0.1), (0.1, 0.0), (0.1, 0.1), (0.2, 0.1)]
    correct = scored("rule_response", [
        np.asarray([0.1, 0.2, 0.8], dtype=np.float32),
        np.asarray([0.2, 0.7, 0.6, 0.3], dtype=np.float32),
    ])
    control = scored("rule_null", [
        np.asarray([0.4, 0.3, 0.2], dtype=np.float32),
        np.asarray([0.5, 0.2, 0.1, 0.4], dtype=np.float32),
    ])
    table = build_truthblind_candidate_features(correct, actions, 2, "rule_response")
    assert table["feature"].shape == (2, 2, len(FEATURE_NAMES))
    assert table["baseline_candidate"].tolist() == [0, 1]
    assert table["valid"].tolist() == [[True, False], [True, True]]

    chemical_dim = sum(
        "rule" in name.lower() or "action" in name.lower() for name in FEATURE_NAMES
    )
    bundle = {
        "schema": "chemaware_truthblind_candidate_policy_v1",
        "feature_names": list(FEATURE_NAMES),
        "actions": [list(action) for action in actions],
        "global_action": 2,
        "channels": {
            "benefit": {
                "nuisance_model": Classifier(+1.0), "prevalence": 0.2,
                "residual_model": Regressor(+0.2),
                "active": np.arange(chemical_dim, dtype=np.int64),
            },
            "harmful": {
                "nuisance_model": Classifier(-1.0), "prevalence": 0.1,
                "residual_model": Regressor(-0.1),
                "active": np.arange(chemical_dim, dtype=np.int64),
            },
        },
        "dose": 1.0,
        "threshold": -1.0,
        "risk_penalty": 2.0,
        "rule_key": "rule_response",
        "control_rule_keys": ["rule_null"],
        "baselines": {
            "same_feature_direct": {
                "channels": {
                    "benefit": {"model": Classifier(+0.5), "prevalence": 0.2},
                    "harmful": {"model": Classifier(-0.5), "prevalence": 0.1},
                },
                "threshold": -1.0,
            },
            "nuisance_only": {"threshold": -1.0},
        },
    }
    first = predict_truthblind_policy(bundle, correct, [control])
    second = predict_truthblind_policy(bundle, correct, [control])
    assert np.array_equal(first["selected_candidate"], second["selected_candidate"])
    assert np.array_equal(first["selected_candidate_slot"], second["selected_candidate_slot"])
    assert np.array_equal(first["utility"], second["utility"])
    assert np.all(first["selected_candidate"] != first["baseline_candidate"])
    baselines = predict_truthblind_baselines(bundle, correct, [control])
    assert set(baselines) == {"same_feature_direct", "nuisance_only"}
    for result in baselines.values():
        assert np.all(result["selected_candidate"] != table["baseline_candidate"])
        assert result["utility"].shape == table["valid"].shape
    arbitration = arbitrate_truthblind_predictions(baselines["same_feature_direct"], first)
    wrapped = predict_truthblind_direct_first_backoff(bundle, correct, [control])
    assert np.array_equal(arbitration["selected_candidate"], wrapped["selected_candidate"])
    assert np.array_equal(arbitration["selected_candidate_slot"], wrapped["selected_candidate_slot"])
    assert set(np.unique(arbitration["action_source"])) <= {0, 1, 2}
    arms = predict_truthblind_policy_arms(bundle, correct, [control])
    assert set(arms) == {
        "correct", "zero_contrast", "reversed_contrast",
        "candidate_rotated_truthblind",
    }
    assert all(result["utility"].shape == table["valid"].shape for result in arms.values())
    exclusive = predict_truthblind_direct_first_exclusive_backoff(
        bundle, correct, [control],
    )
    assert exclusive["chemical_exclusive"].shape == (2,)
    assert set(np.unique(exclusive["action_source"])) <= {0, 1, 2}
    assert np.all(exclusive["action_source"][~exclusive["chemical_exclusive"]] != 2)

    control_b = scored("rule_null_b", [
        np.asarray([0.2, 0.6, 0.3], dtype=np.float32),
        np.asarray([0.1, 0.4, 0.2, 0.5], dtype=np.float32),
    ])
    control_c = scored("rule_null_c", [
        np.asarray([0.6, 0.1, 0.4], dtype=np.float32),
        np.asarray([0.3, 0.1, 0.5, 0.2], dtype=np.float32),
    ])
    symmetric = copy.deepcopy(bundle)
    symmetric["control_rule_keys"] = ["rule_null", "rule_null_b", "rule_null_c"]
    symmetric["contrast_representation"] = "symmetric_summary"
    for channel in symmetric["channels"].values():
        channel["active"] = np.arange(chemical_dim * 6, dtype=np.int64)
    symmetric_result = predict_truthblind_policy(
        symmetric, correct, [control, control_b, control_c],
    )
    symmetric_baselines = predict_truthblind_baselines(
        symmetric, correct, [control, control_b, control_c],
    )
    assert symmetric_result["utility"].shape == table["valid"].shape
    assert symmetric_baselines["same_feature_direct"]["utility"].shape == table["valid"].shape

    leaked = dict(correct)
    leaked["labels"] = np.asarray([np.asarray([True, False])], dtype=object)
    try:
        build_truthblind_candidate_features(leaked, actions, 2, "rule_response")
    except ValueError as error:
        assert "truth fields" in str(error)
    else:
        raise AssertionError("truth-bearing inference input was accepted")
    projected = truthblind_scored_view(leaked, "rule_response")
    assert set(projected) == {"global", "mass", "reference_ptr", "rule_response"}
    projected_table = build_truthblind_candidate_features(
        projected, actions, 2, "rule_response",
    )
    assert np.array_equal(projected_table["feature"], table["feature"])

    drifted_control = dict(control)
    drifted_control["reference_ptr"] = control["reference_ptr"].copy()
    drifted_control["reference_ptr"][0] = np.asarray([0, 1, 3], dtype=np.int32)
    try:
        predict_truthblind_policy(bundle, correct, [drifted_control])
    except ValueError as error:
        assert "candidate rows drifted" in str(error)
    else:
        raise AssertionError("mismatched control candidate rows were accepted")
    print("PASS: ChemAware truth-blind candidate inference contracts")


if __name__ == "__main__":
    main()
