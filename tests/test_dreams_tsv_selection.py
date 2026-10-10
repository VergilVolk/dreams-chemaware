from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def report(identity_base: float, identity_candidate: float,
           formula_base: float, formula_candidate: float) -> dict:
    def side(recall1: float) -> dict:
        retrieval = {
            "recall@1": recall1,
            "recall@3": recall1 + 0.03,
            "recall@5": recall1 + 0.04,
            "recall@10": recall1 + 0.05,
            "recall@20": recall1 + 0.06,
            "mrr": recall1 + 0.02,
            "macro_query_auroc": recall1 + 0.01,
            "macro_query_auprc": recall1 + 0.02,
        }
        return {
            "retrieval": retrieval,
            "near_subset": {
                "recall@1": recall1 - 0.02,
                "mrr": recall1,
                "macro_query_auroc": recall1 - 0.01,
                "macro_query_auprc": recall1,
            },
            "micro_candidate": {"auroc": recall1 + 0.015, "auprc": recall1 - 0.05},
            "gnps_10ppm_pooled_pairwise": {
                "auroc": recall1 + 0.012,
                "auprc": recall1 - 0.06,
            },
        }

    paired = {
        "corrected": 10,
        "introduced": 2,
        "risk_net_lambda2": 6,
        "formula_cluster_paired_ci": {"recall@1": {"ci_low": 0.01}},
        "near_formula_cluster_paired_ci": {"recall@1": {"ci_low": 0.005}},
    }
    return {
        "panels": {
            "identity_disjoint": {
                "baseline": side(identity_base),
                "candidate": side(identity_candidate),
                "paired": paired,
            },
            "formula_disjoint": {
                "baseline": side(formula_base),
                "candidate": side(formula_candidate),
                "paired": paired,
            },
        }
    }


def test_selection_uses_worst_panel_gain_and_materializes_checkpoint(tmp_path: Path) -> None:
    # Linear has the higher mean, but TSV has the stronger worst-panel gain.
    linear_report = report(0.85, 0.90, 0.87, 0.88)  # +5 pp, +1 pp
    tsv_report = report(0.85, 0.88, 0.87, 0.90)     # +3 pp, +3 pp
    linear_json = tmp_path / "linear.json"
    tsv_json = tmp_path / "tsv.json"
    linear_json.write_text(json.dumps(linear_report), encoding="utf-8")
    tsv_json.write_text(json.dumps(tsv_report), encoding="utf-8")
    linear_checkpoint = tmp_path / "linear.pt"
    tsv_checkpoint = tmp_path / "tsv.pt"
    linear_checkpoint.write_bytes(b"linear")
    tsv_checkpoint.write_bytes(b"tsv")
    output = tmp_path / "selection.json"
    subprocess.run(
        [
            sys.executable, str(ROOT / "tasks/select_dreams_tsv_gnps.py"),
            "--linear-report", str(linear_json),
            "--tsv-report", str(tsv_json),
            "--linear-checkpoint", str(linear_checkpoint),
            "--tsv-checkpoint", str(tsv_checkpoint),
            "--output", str(output),
        ],
        check=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["winner"] == "tsv"
    assert result["scores"]["tsv"]["stable_plus_3pp_target_attained"] is True
    identity = result["scores"]["tsv"]["identity_disjoint"]
    assert abs(identity["metrics"]["macro_query_auroc"]["delta_pp"] - 3.0) < 1e-12
    assert abs(identity["metrics"]["micro_candidate_auroc"]["delta_pp"] - 3.0) < 1e-12
    assert abs(identity["metrics"]["pooled_pairwise_auprc"]["delta_pp"] - 3.0) < 1e-12
    assert identity["formula_cluster_paired_ci"]["recall@1"]["ci_low"] == 0.01
    assert (tmp_path / "selected_checkpoint.pt").read_bytes() == b"tsv"


def test_sbatch_is_fixed_two_arm_gnps_only() -> None:
    text = (ROOT / "tasks/run_noise_chemaware_tsv_gnps_1gpu.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --gpus=1" in text
    assert "#SBATCH --time=24:00:00" in text
    assert "MassSpecGym" in text and "data/models/MassSpecGym" not in text
    assert "Enveda-180 remains unopened" in text
    assert "--linear-output" in text and "--tsv-output" in text
    assert "--device cuda" in text
    assert "--scale" not in text and "linspace" not in text
    assert "panel_identity_disjoint.npz" in text
    assert "panel_formula_disjoint.npz" in text
