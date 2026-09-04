"""Synthetic end-to-end test for A4 artifact validation and analysis."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    root = ROOT / "data/validation/pytest_noise_v3_a4_artifacts"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    query = pd.DataFrame({
        "query_index": [10, 20], "query_row": [100, 200],
        "query_ik14": ["ERROR000000001", "CONTROL0000001"],
        "query_formula": ["F", "F"], "has_near": [True, True],
        "baseline_rank": [2, 1], "baseline_margin": [-0.1, 0.2],
        "candidate_molecules": [3, 3], "peak_count": [2, 2],
        "baseline_adversarial_molecule_local": [1, 1],
        "baseline_adversarial_pair_row": [101, 201],
        "baseline_adversarial_mces_grade": [0, 0],
        "scan_kind": ["official_error", "safety_control"],
        "matched_error_count": [0, 1], "scan_position": [0, 1],
        "clean_embedding_preservation": [1.0, 1.0],
    })
    query.to_csv(root / "scan_queries.csv.gz", index=False, compression="gzip")
    pd.DataFrame({
        "error_query_index": [10], "control_query_index": [20],
        "match_level": ["formula+near"], "match_distance": [0.0],
        "control_reuse_after_selection": [1],
    }).to_csv(root / "safety_control_matches.csv.gz", index=False, compression="gzip")
    report = {
        "status": "noise_v3_a4_exact_peak_scan_complete", "formal": False,
        "full_graph_queries": 2, "official_errors_scanned": 1,
        "unique_safety_controls": 1, "scan_queries": 2,
        "fragment_actions": 4, "exact_variants": 8,
        "attenuations": [0.5, 1.0],
        "control_matching": {"requested_per_error": 1},
        "official_forward_cache_preservation": {"p01": 1.0},
    }
    (root / "report.json").write_text(json.dumps(report), encoding="utf-8")
    with h5py.File(root / "exact_peak_scan.h5", "w") as handle:
        handle.attrs["status"] = "noise_v3_a4_exact_peak_scan_complete"
        handle.attrs["attenuations_json"] = json.dumps([0.5, 1.0])
        handle.attrs["query_count"] = 2
        for name, values in {
            "query_action_ptr": [0, 2, 4], "action_query": [0, 0, 1, 1],
            "action_token": [1, 2, 1, 2], "action_role": [1, 2, 1, 0],
            "action_mz": [100, 200, 100, 200], "action_intensity": [.5, .2, .5, .2],
            "action_gradient": [-1, -.5, -.2, .1], "action_predicted_gain": [.25, .1, .05, -.02],
            "action_gradient_rank": [1, 2, 1, -1],
            "action_policy_eligible": [True, True, True, False],
            "result_rank": [1, 1, 2, 1, 1, 2, 2, 2],
            "result_mrr": [1, 1, .5, 1, 1, .5, .5, .5],
            "result_positive": [.8] * 8, "result_negative": [.7] * 8,
            "result_margin": [.1, .2, -.1, .1, .1, -.1, -.1, -.1],
            "result_adversarial_molecule_local": [1] * 8,
            "result_adversarial_pair_row": [101] * 8,
        }.items():
            handle.create_dataset(name, data=np.asarray(values))
    subprocess.run([
        sys.executable, str(ROOT / "tasks/validate_noise_v3_a4_exact_peak_scan.py"),
        "--output-dir", str(root), "--allow-smoke",
    ], check=True)
    missing = root / "missing_previous"
    subprocess.run([
        sys.executable, str(ROOT / "tasks/analyze_noise_v3_a4_exact_peak_scan.py"),
        "--a4-dir", str(root), "--s1c-dir", str(missing), "--s2-dir", str(missing),
        "--s3a-dir", str(missing), "--top-actions-per-query", "2",
    ], check=True)
    decision = json.loads((root / "decision.json").read_text())
    assert decision["exact_action_oracle"]["unique_recoverable_errors"] == 1
    assert (root / "exact_action_matrix.csv").is_file()
    assert (root / "policy_candidate_actions.csv.gz").is_file()
    print("[A4 artifact test] PASS", flush=True)
    shutil.rmtree(root)


if __name__ == "__main__":
    main()
