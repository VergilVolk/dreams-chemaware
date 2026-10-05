"""Fail-closed validator for the A1b official-DreaMS coordinate result."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks"))

from pilot_reference_anchored_multi_probe_a1 import sha256  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/reference_anchored_multi_probe_a1b_official_20260913",
    )
    args = parser.parse_args()
    report_path = args.output_dir / "report.json"
    per_query_path = args.output_dir / "per_query_official_coordinate.csv.gz"
    for path in (report_path, per_query_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "reference_anchored_multi_probe_a1b_official_complete":
        raise RuntimeError("unexpected A1b status")
    if report.get("formal") is not False:
        raise RuntimeError("A1b must remain exploratory")
    required_contracts = {
        "official_dreams_shared_encoder": True,
        "query_candidate_excluded_from_landmarks": True,
        "weight_search_used": False,
        "identity_labels_used_for_scoring": False,
        "structures_used_for_scoring": False,
        "structures_used_for_evaluation_only": True,
        "phenotype_used": False,
        "sample_abundance_used": False,
        "P2b_used": False,
    }
    for key, expected in required_contracts.items():
        if report.get("contracts", {}).get(key) is not expected:
            raise RuntimeError(f"A1b contract failed: {key}")
    if report["contracts"].get("fixed_profile_fusion_weights") != [0.5, 0.5]:
        raise RuntimeError("A1b fusion weights drifted")
    frame = pd.read_csv(per_query_path)
    if len(frame) != int(report["identities"]):
        raise RuntimeError("A1b per-query count mismatch")
    if frame[["panel", "query_ik14"]].duplicated().any():
        raise RuntimeError("duplicate panel/query identity")
    if frame.query_ik14.nunique() != int(report["independent_ik14_clusters"]):
        raise RuntimeError("A1b independent-cluster count mismatch")
    if sha256(per_query_path) != report["provenance"].get("per_query_sha256"):
        raise RuntimeError("A1b per-query hash mismatch")
    primary = report["paired_ik14_bootstrap"][
        "official_profile_fusion_50_50_vs_official_direct__ndcg5"
    ]
    panel_ok = all(float(value) >= 0 for value in report["panel_primary_ndcg_delta"].values())
    expected = bool(float(primary["ci_low"]) > 0 and panel_ok)
    if report["gates"].get("pass_to_sample_level_peak_operator") is not expected:
        raise RuntimeError("A1b decision is inconsistent with frozen outcomes")
    print(
        "[validate_reference_anchored_multi_probe_a1b_official] PASS "
        f"identities={len(frame)} independent={frame.query_ik14.nunique()} proceed={expected}"
    )


if __name__ == "__main__":
    main()
