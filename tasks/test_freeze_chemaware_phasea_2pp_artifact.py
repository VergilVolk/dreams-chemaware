"""CPU contracts for protecting the ChemAware +2.1266 pp role-2 artifact."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def metrics(recall1: float, offset: float = 0.0) -> dict[str, float | int]:
    return {
        "queries": 1975,
        "recall1": recall1,
        "recall3": 0.984 + offset,
        "recall5": 0.993 + offset,
        "recall10": 0.997,
        "recall20": 1.0,
        "recall50": 1.0,
        "mrr": 0.939 + offset,
        "mean_positive_margin": 0.34,
        "micro_auc": 0.937 + offset,
        "macro_auc": 0.955 + offset,
    }


def main() -> None:
    ledger = json.loads((
        ROOT / "docs/CHEMAWARE_PHASEA_2PP_RUN_2343962_FROZEN_LEDGER.json"
    ).read_text(encoding="utf-8"))
    assert ledger["status"] == "CHEMAWARE_PHASEA_2PP_ROLE2_EVIDENCE_FROZEN"
    assert ledger["paired_vs_official"]["corrected_at_1"] == 54
    assert ledger["paired_vs_official"]["introduced_at_1"] == 12
    assert abs(
        ledger["paired_vs_official"]["delta_recall1_percentage_points"]
        - 2.1265822784810124
    ) < 1e-12
    assert ledger["interpretation"]["independent_role3_confirmation"] is False
    with tempfile.TemporaryDirectory(prefix="chem_2pp_protect_test_") as raw:
        root = Path(raw)
        checkpoint = root / "step-002000.ckpt"
        checkpoint.write_bytes(b"synthetic checkpoint contract")
        triplet_report = root / "triplet_report.json"
        triplet_report.write_text(json.dumps({"status": "triplets"}), encoding="utf-8")
        official_recall1 = 0.8962025316455696
        delta = 42 / 1975
        evaluation = root / "evaluation.json"
        evaluation.write_text(json.dumps({
            "formula_role": 2,
            "results": [
                {
                    "name": "official",
                    "checkpoint": "official.pt",
                    "metrics": metrics(official_recall1),
                },
                {
                    "name": "phaseA_step-002000",
                    "checkpoint": str(checkpoint.resolve()),
                    "metrics": metrics(official_recall1 + delta, 0.01),
                    "paired_vs_official": {
                        "delta_recall1": delta,
                        "delta_mrr": 0.01,
                        "corrected_at_1": 54,
                        "introduced_at_1": 12,
                        "formula_cluster_bootstrap_delta_recall1_ci95": [
                            0.0127, 0.0303,
                        ],
                    },
                },
            ],
        }), encoding="utf-8")
        output = root / "protected"
        subprocess.run([
            sys.executable,
            str(ROOT / "tasks/freeze_chemaware_phasea_2pp_artifact.py"),
            "--checkpoint", str(checkpoint),
            "--evaluation", str(evaluation),
            "--triplet-report", str(triplet_report),
            "--output", str(output),
        ], check=True, capture_output=True, text=True)
        manifest = json.loads(
            (output / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["status"] == "CHEMAWARE_PHASEA_2PP_ROLE2_ARTIFACT_PROTECTED"
        assert manifest["claim_status"] == "ROLE2_DEVELOPMENT_RESULT_NOT_ROLE3_CONFIRMED"
        assert manifest["shared_embedding"] is True
        assert manifest["reranker"] is False
        assert manifest["corrected_at_1"] == 54
        assert manifest["introduced_at_1"] == 12
        assert abs(manifest["delta_recall1_percentage_points"] - 2.1265822784810124) < 1e-12
        assert manifest["role3_evaluated"] is False
        assert (output / "chemaware_phasea_max_boundary_step2000.ckpt").is_file()
        assert (output / "SHA256SUMS").is_file()

    sbatch = (ROOT / "tasks/run_chemaware_phasea_2pp_protect.sbatch").read_text(
        encoding="utf-8",
    )
    assert "#SBATCH --gpus=1" in sbatch
    assert "#SBATCH --mem" not in sbatch
    assert "--max-steps 2000" in sbatch
    assert "--lr 5e-6" in sbatch
    assert "--negative-references-per-error 2" in sbatch
    assert "--chemical-candidates-per-error 2" in sbatch
    assert "freeze_chemaware_phasea_2pp_artifact.py" in sbatch
    remine = (ROOT / "tasks/run_chemaware_max_boundary_remine.sbatch").read_text(
        encoding="utf-8",
    )
    stop = remine.split('if [[ "$SELECTED_STEP" == "0" ]]', 1)[1]
    assert "freeze_chemaware_phasea_2pp_artifact.py" in stop
    assert '/bin/rm -f -- "$PHASE_A_CHECKPOINT"' not in stop
    print("PASS: ChemAware +2.1266 pp artifact-protection contracts")


if __name__ == "__main__":
    main()
