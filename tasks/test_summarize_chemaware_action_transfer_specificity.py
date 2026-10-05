"""CPU contracts for the ChemAware A2 post-hoc specificity summarizer."""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from summarize_chemaware_action_transfer_specificity import analyze


ROUTES = (
    "action_query_only",
    "action_forward_clean_backward",
    "action_routed_clean_pair",
    "naive_action_minus_clean",
)
ARMS = ("correct", "candidate_swapped", "peak_permuted")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="chem_a2_specificity_") as temporary:
        root = Path(temporary)
        screen = root / "run_1/screen24"
        bank_dir = root / "bank"
        screen.mkdir(parents=True)
        bank_dir.mkdir()
        np.savez_compressed(
            bank_dir / "action_bank.npz",
            action_fold=np.asarray([0, 2]),
            role_code=np.asarray([[1, 2]], dtype=np.int8),
            selected_setting=np.asarray([0], dtype=np.int16),
            formula=np.asarray(["C2H4", "C3H6"]),
        )
        (bank_dir / "report.json").write_text(json.dumps({
            "split": {"discovery_folds": [0, 1], "confirmation_fold": 2},
        }), encoding="utf-8")

        fields = [
            "action_position", "formula", "route", "arm", "route_forward_margin",
            "positive_reference", "negative_reference", "raw_gradient_norm",
            "learning_rate_scaled_update_norm", "predicted_clean_margin_gain",
            "clean_margin_gain_per_unit_update_norm", "clean_margin_descent_cosine",
            "clean_loss_decrease_per_unit_update_norm", "clean_loss_gradient_cosine",
        ]
        rows = []
        arm_offset = {"correct": 0.3, "candidate_swapped": 0.1, "peak_permuted": 0.0}
        for position, formula in enumerate(("C2H4", "C3H6")):
            for route in ROUTES:
                for arm in ARMS:
                    value = 1.0 + position + arm_offset[arm]
                    if route == "naive_action_minus_clean":
                        value = -value
                    positive = 0
                    negative = 2
                    if route == "action_routed_clean_pair" and position == 1 and arm != "correct":
                        negative = 3
                    rows.append({
                        "action_position": position,
                        "formula": formula,
                        "route": route,
                        "arm": arm,
                        "route_forward_margin": value / 10,
                        "positive_reference": positive if route == "action_routed_clean_pair" else "",
                        "negative_reference": negative if route == "action_routed_clean_pair" else "",
                        "raw_gradient_norm": abs(value) + 1,
                        "learning_rate_scaled_update_norm": abs(value) / 100,
                        "predicted_clean_margin_gain": value / 1000,
                        "clean_margin_gain_per_unit_update_norm": value,
                        "clean_margin_descent_cosine": value / 10,
                        "clean_loss_decrease_per_unit_update_norm": value / 2,
                        "clean_loss_gradient_cosine": value / 20,
                    })
        with (screen / "per_action.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        bank_hash = digest(bank_dir / "action_bank.npz")
        source = {
            "status": "CHEMAWARE_ACTION_TRANSFER_GRADIENT_FAIL",
            "weights_updated": False,
            "optimizer_steps": 0,
            "routes": {route: {"pass": False} for route in ROUTES},
            "preflight": {
                "positions": [0, 1],
                "provenance": {"action_bank_sha256": bank_hash},
            },
        }
        (screen / "report.json").write_text(json.dumps(source), encoding="utf-8")
        (screen / "COMPLETE.json").write_text(json.dumps({
            "status": source["status"], "winner": None,
        }), encoding="utf-8")

        report = analyze(screen.resolve(), bank_dir.resolve(), draws=200, seed=7)
        assert report["actions"] == 2
        assert report["records"] == 2 * len(ROUTES) * len(ARMS)
        assert report["pair_selection_overlap"]["candidate_swapped"]["same_pair_fraction"] == 0.5
        assert report["contracts"]["controls_used_as_optimizer_signal"] is False
        paired = report["route_diagnostics"]["action_query_only"]["metrics"][
            "clean_margin_gain_per_unit_update_norm"
        ]["correct_minus_controls"]["candidate_swapped"]
        assert np.isclose(paired["query_mean"], 0.2)
        assert set(report["route_diagnostics"]["action_query_only"]["subgroups"]["fold_role"]) == {
            "discovery", "confirmation",
        }
    sbatch = Path(__file__).with_name(
        "run_chemaware_action_transfer_specificity_summary.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpu" in sbatch
    assert "#SBATCH --gpus=1" in sbatch
    assert "#SBATCH --mem=" not in sbatch
    assert "#SBATCH --mem-per-cpu=" not in sbatch
    assert "summarize_chemaware_action_transfer_specificity.py" in sbatch
    assert "--screen-dir" not in sbatch
    assert "python tasks/test_summarize_chemaware_action_transfer_specificity.py" in sbatch
    print("PASS: ChemAware action-transfer specificity summary contracts")


if __name__ == "__main__":
    main()
