"""Fail-closed structural validation for a completed frontier G0 audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from noise_frontier_g0_core import sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/noise_frontier_g0_20260914")
    args = parser.parse_args()
    report_path = args.output_dir / "report.json"
    cache_path = args.output_dir / "cache.npz"
    cache_report_path = args.output_dir / "cache.json"
    ranks_path = args.output_dir / "oof_ranks.npz"
    for path in (report_path, cache_path, cache_report_path, ranks_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") not in {
        "noise_frontier_g0_representation_pass", "noise_frontier_g0_representation_fail",
    }:
        raise RuntimeError("unexpected G0 status")
    if report.get("formal") is not True or int(report.get("queries", 0)) != 23876:
        raise RuntimeError("G0 is not the complete formal graph")
    contracts = report.get("contracts", {})
    required_contracts = (
        "candidate_protocol_identical_across_arms",
        "formula_group_outer_OOF",
        "weights_selected_without_outer_fold",
        "pair_fusion_before_molecule_max",
        "ties_count_against_positive",
    )
    if not all(contracts.get(name) is True for name in required_contracts):
        raise RuntimeError("G0 execution contract failed")
    if contracts.get("P2b_used") is not False or contracts.get("DreaMS_parameters_updated") is not False:
        raise RuntimeError("G0 crossed the frozen-readout boundary")
    expected_results = {
        "official_global_cosine", "raw_multichannel_strong_baseline",
        "token_late_interaction", "candidate_consensus", "raw_landmark_profile",
        "token_consensus_primary", "frontier_full_exploratory",
    }
    if set(report.get("results", {})) != expected_results:
        raise RuntimeError("G0 result arms are incomplete")
    if report.get("primary_preregistered_arm") != "token_consensus_primary":
        raise RuntimeError("G0 primary arm changed after execution")
    if report.get("provenance", {}).get("cache_sha256") != sha256_file(cache_path):
        raise RuntimeError("G0 cache hash mismatch")
    if report.get("provenance", {}).get("cache_report_sha256") != sha256_file(cache_report_path):
        raise RuntimeError("G0 cache-report hash mismatch")
    with np.load(ranks_path) as body:
        if len(body["baseline_rank"]) != 23876:
            raise RuntimeError("G0 OOF rank count mismatch")
        for name in body.files:
            if name.endswith("_rank") and np.any(body[name] < 1):
                raise RuntimeError(f"invalid ranks in {name}")
    pass_flag = bool(report.get("gates", {}).get("pass_to_g1"))
    if pass_flag != report["status"].endswith("_pass"):
        raise RuntimeError("G0 status/gate disagreement")
    print(
        f"[validate_noise_frontier_g0] PASS execution; scientific_pass={pass_flag}",
        flush=True,
    )


if __name__ == "__main__":
    main()
