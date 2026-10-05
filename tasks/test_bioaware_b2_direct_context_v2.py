#!/usr/bin/env python
"""Dependency-free contract checks for the BioAware B2 v2 matrix."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
for location in (ROOT, ROOT / "tasks"):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from summarize_bioaware_b2_direct_context_v2 import EXPECTED
from train_bioaware_b2_leave_study_out import EVIDENCE_MODES, FORMAL_RECIPE


def fake_report(mode: str, isolation: str, dataset_hash: str) -> dict:
    metric = {
        "queries": 100, "baseline_recall1": .8, "recall1": .81,
        "delta_recall1": .01, "baseline_mrr": .9, "mrr": .91,
        "delta_mrr": .01, "corrected": 2, "introduced": 1,
        "risk_weighted_net_lambda2": 0, "mcnemar_exact_p": 1.0,
    }
    return {
        "formal": True,
        "configuration": {"evidence_mode": mode, "training_isolation": isolation},
        "overall": metric, "context_active": metric,
        "study_formula_cluster_bootstrap": {
            "mean": .01, "ci_low": .001, "ci_high": .02,
            "clusters": 50, "resamples": 10000,
        },
        "outer_studies": {"synthetic": metric},
        "mean_preservation": .999, "mean_gate": .1, "pass": True,
        "provenance": {"dataset_sha256": dataset_hash},
    }


def main() -> None:
    assert FORMAL_RECIPE["seeds"] == [20260830, 20260831, 20260832]
    assert set(("full", "reaction_only", "smn_only", "smn_rt", "rt_only")) <= set(EVIDENCE_MODES)
    sbatch = (ROOT / "tasks/run_bioaware_b2_direct_context_v2.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --gpus=1" in sbatch
    assert "--formal --device cuda" in sbatch
    assert "replay_bioaware_b2_leave_study_out.py" in sbatch
    assert "audit_bioaware_b2_flip_mechanisms.py" in sbatch
    assert "partial.${SLURM_JOB_ID}" in sbatch
    assert "mv \"$WORK\" \"$ROOT\"" in sbatch

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name, (mode, isolation) in EXPECTED.items():
            cell = root / name
            cell.mkdir()
            (cell / "report.json").write_text(
                json.dumps(fake_report(mode, isolation, "same-dataset")), encoding="utf-8",
            )
            (cell / "replay.json").write_text(json.dumps({
                "status": "bioaware_b2_artifact_replay_passed",
                "final_rank_mismatches": 0, "baseline_rank_mismatches": 0,
            }), encoding="utf-8")
            (cell / "flip_mechanism_audit.json").write_text(json.dumps({
                "status": "bioaware_b2_flip_mechanism_audit_complete",
                "rank_mismatches": 0, "corrected": 2, "introduced": 1,
            }), encoding="utf-8")
        subprocess.run(
            [sys.executable, str(ROOT / "tasks/summarize_bioaware_b2_direct_context_v2.py"),
             "--root", str(root)], check=True, capture_output=True, text=True,
        )
        matrix = json.loads((root / "matrix_report.json").read_text(encoding="utf-8"))
        assert matrix["formal"] is True
        assert matrix["pass_to_last_block_direct_finetuning"] is True
        assert len(matrix["cells"]) == len(EXPECTED)
    print("[test_bioaware_b2_direct_context_v2] PASS")


if __name__ == "__main__":
    main()
