"""Fail-closed structural validation for the A4 exact peak scan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_v3_a4_exact_peak_scan",
    )
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    required = ["report.json", "scan_queries.csv.gz", "safety_control_matches.csv.gz", "exact_peak_scan.h5"]
    for name in required:
        if not (args.output_dir / name).is_file():
            raise FileNotFoundError(args.output_dir / name)
    report = json.loads((args.output_dir / "report.json").read_text(encoding="utf-8"))
    queries = pd.read_csv(args.output_dir / "scan_queries.csv.gz")
    matches = pd.read_csv(args.output_dir / "safety_control_matches.csv.gz")
    if queries["query_index"].duplicated().any() or queries["scan_position"].duplicated().any():
        raise RuntimeError("duplicate scan queries")
    if set(queries["scan_kind"]) != {"official_error", "safety_control"}:
        raise RuntimeError("scan kinds are incomplete")
    if not (queries.loc[queries["scan_kind"] == "official_error", "baseline_rank"] > 1).all():
        raise RuntimeError("official-error cohort contains a correct query")
    if not (queries.loc[queries["scan_kind"] == "safety_control", "baseline_rank"] == 1).all():
        raise RuntimeError("safety cohort contains an official error")
    match_count = matches.groupby("error_query_index").size()
    requested = int(report["control_matching"]["requested_per_error"])
    if not match_count.eq(requested).all():
        raise RuntimeError("an error lacks complete safety controls")
    if not set(matches["error_query_index"]).issubset(set(
        queries.loc[queries["scan_kind"] == "official_error", "query_index"]
    )):
        raise RuntimeError("control table references an unscanned error")

    with h5py.File(args.output_dir / "exact_peak_scan.h5", "r") as handle:
        required_datasets = {
            "query_action_ptr", "action_query", "action_token", "action_role", "action_mz",
            "action_intensity", "action_gradient", "action_predicted_gain", "action_gradient_rank",
            "action_policy_eligible", "result_rank", "result_mrr", "result_positive",
            "result_negative", "result_margin", "result_adversarial_molecule_local",
            "result_adversarial_pair_row",
        }
        missing = required_datasets - set(handle.keys())
        if missing:
            raise RuntimeError(f"missing A4 datasets: {sorted(missing)}")
        doses = json.loads(handle.attrs["attenuations_json"])
        n_actions = len(handle["action_query"])
        ptr = handle["query_action_ptr"][:]
        if len(ptr) != len(queries) + 1 or ptr[0] != 0 or ptr[-1] != n_actions:
            raise RuntimeError("invalid query/action pointer")
        if np.any(np.diff(ptr) <= 0):
            raise RuntimeError("a scan query has no real fragment action")
        if np.any(handle["action_token"][:] <= 0):
            raise RuntimeError("precursor/padding entered A4 action space")
        if np.any(handle["action_mz"][:] <= 0) or np.any(handle["action_intensity"][:] <= 0):
            raise RuntimeError("non-real fragment entered A4 action space")
        expected = n_actions * len(doses)
        for name in (
            "result_rank", "result_mrr", "result_positive", "result_negative", "result_margin",
            "result_adversarial_molecule_local", "result_adversarial_pair_row",
        ):
            if len(handle[name]) != expected:
                raise RuntimeError(f"result length mismatch: {name}")
        if np.any(handle["result_rank"][:] < 1):
            raise RuntimeError("invalid retrieval rank")
        if int(handle.attrs["query_count"]) != len(queries):
            raise RuntimeError("HDF5/report query count mismatch")

    if report["formal"] and not args.allow_smoke:
        if report["official_errors_scanned"] != 1805:
            raise RuntimeError("formal A4 did not cover every official error")
        if report["full_graph_queries"] != 23876:
            raise RuntimeError("formal A4 used the wrong candidate graph")
        if report["official_forward_cache_preservation"]["p01"] < 0.999:
            raise RuntimeError("official embedding reproduction failed")
        if report["attenuations"] != [0.25, 0.5, 0.75, 1.0]:
            raise RuntimeError("formal attenuation matrix drifted")
    validation = {
        "status": "noise_v3_a4_exact_peak_scan_validation_passed",
        "formal": bool(report["formal"]),
        "queries": int(len(queries)),
        "errors": int((queries["scan_kind"] == "official_error").sum()),
        "controls": int((queries["scan_kind"] == "safety_control").sum()),
        "actions": int(report["fragment_actions"]),
        "variants": int(report["exact_variants"]),
    }
    (args.output_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    print(json.dumps(validation, indent=2), flush=True)


if __name__ == "__main__":
    main()
