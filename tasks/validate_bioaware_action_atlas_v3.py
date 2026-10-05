#!/usr/bin/env python
"""Fail-closed validation for the BioAware Action Atlas v3 artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("data/validation/bioaware_action_atlas_v3_20260906"),
    )
    args = parser.parse_args()
    report_path = args.result_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))

    if report.get("status") != "bioaware_action_atlas_v3_complete":
        raise RuntimeError("unexpected Action Atlas status")
    if report.get("opened_development") is not True:
        raise RuntimeError("Action Atlas must remain labelled opened development")
    protocol = report.get("candidate_protocol", {})
    expected_protocol = {
        "queries": 482,
        "candidate_pairs": 1800,
        "truth_identities": 145,
        "truth_formulas": 117,
    }
    if protocol != expected_protocol:
        raise RuntimeError(f"candidate protocol changed: {protocol}")

    replay = report.get("current_v4_exact_replay", {})
    if replay.get("pass") is not True or any(replay.get("mismatches", {}).values()):
        raise RuntimeError(f"historical v4 replay failed: {replay}")

    recipes = report.get("fixed_recipes", {})
    required = {
        "spectral_only_control",
        "spectral_plus_degree_control",
        "current_v4_replay",
        "current_without_degree",
        "path_reliability_without_degree",
        "current_plus_path_reliability_without_degree",
        "current_plus_path_reliability",
    }
    if set(recipes) != required:
        raise RuntimeError("fixed recipe registry changed")
    spectral = recipes["spectral_only_control"]["pooled"]
    if spectral["delta_recall1"] != 0 or spectral["interventions"] != 0:
        raise RuntimeError("spectral-only identity replay no longer preserves DreaMS")
    degree = recipes["spectral_plus_degree_control"]["pooled"]
    current = recipes["current_v4_replay"]["pooled"]
    if degree["delta_recall1"] < 0.03 or current["delta_recall1"] < 0.03:
        raise RuntimeError("historical opened-development action was not reproduced")

    null = report.get("degree_fixed_path_availability_conditioned_null", {})
    if null.get("mean_moved_fraction", 0) < 0.90:
        raise RuntimeError("conditioned permutation moved too few candidate rows")
    delta_p = null.get("delta_recall1", {}).get("empirical_p_ge_observed")
    risk_p = null.get("risk_weighted_net_lambda2", {}).get("empirical_p_ge_observed")
    if delta_p is None or risk_p is None:
        raise RuntimeError("conditioned permutation p-values missing")
    if report.get("pass_current_action_specificity") is not False:
        raise RuntimeError("current action must not pass reaction-specificity gate")

    candidate_path = args.result_dir / "candidate_features_rich.csv.gz"
    expected_candidate_hash = report["provenance"]["rich_candidate_features_sha256"]
    if sha256(candidate_path) != expected_candidate_hash:
        raise RuntimeError("rich candidate artifact hash mismatch")
    for name, expected_hash in report["provenance"]["transition_sha256"].items():
        path = args.result_dir / f"{name}__transitions.csv.gz"
        if sha256(path) != expected_hash:
            raise RuntimeError(f"transition hash mismatch: {name}")

    print(
        "[validate_bioaware_action_atlas_v3] PASS "
        f"queries={protocol['queries']} degree_delta={degree['delta_recall1']:+.6f} "
        f"current_delta={current['delta_recall1']:+.6f} "
        f"conditioned_p={delta_p:.3f}/{risk_p:.3f}"
    )


if __name__ == "__main__":
    main()
