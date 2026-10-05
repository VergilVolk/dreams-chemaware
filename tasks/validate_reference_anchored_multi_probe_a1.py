"""Fail-closed validator for the A1 multi-standard coordinate screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/reference_anchored_multi_probe_a1_20260913",
    )
    args = parser.parse_args()
    report_path = args.output_dir / "report.json"
    per_query_path = args.output_dir / "per_query_coordinate_recovery.csv.gz"
    for path in (report_path, per_query_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "reference_anchored_multi_probe_a1_complete":
        raise RuntimeError("unexpected A1 status")
    if report.get("formal") is not False:
        raise RuntimeError("A1 must remain an exploratory headroom screen")
    required_contracts = {
        "query_candidate_excluded_from_landmarks": True,
        "identity_labels_used_for_scoring": False,
        "structures_used_for_scoring": False,
        "structures_used_for_evaluation_only": True,
        "phenotype_used": False,
        "sample_abundance_used": False,
        "P2b_used": False,
    }
    for key, expected in required_contracts.items():
        if report["contracts"].get(key) is not expected:
            raise RuntimeError(f"contract failed: {key}")
    frame = pd.read_csv(per_query_path)
    if len(frame) != report["identities"]:
        raise RuntimeError("per-query row count is not the panel-specific identity count")
    if frame[["panel", "query_ik14"]].duplicated().any():
        raise RuntimeError("duplicate panel/query identity")
    if sha256(per_query_path) != report["provenance"]["per_query"]:
        raise RuntimeError("per-query artifact hash mismatch")
    for panel, summary in report["panels"].items():
        observed = int(frame.panel.eq(panel).sum())
        if observed != int(summary["identities"]):
            raise RuntimeError(f"{panel}: identity count mismatch")
        for method in summary["metrics"].values():
            for key, value in method.items():
                if key != "mean_structural_regret" and not (-1.0 <= float(value) <= 1.0):
                    raise RuntimeError(f"{panel}: metric outside allowed range: {key}={value}")
                if key == "mean_structural_regret" and not (0.0 <= float(value) <= 1.0):
                    raise RuntimeError(f"{panel}: invalid structural regret: {value}")
    primary = report["paired_identity_bootstrap"][
        "multi_anchor_augmented_vs_direct_mass_fusion_50_50__ndcg5"
    ]
    panel_nonnegative = all(
        summary["metrics"]["multi_anchor_augmented"]["mean_ndcg5"]
        >= summary["metrics"]["direct_mass_fusion_50_50"]["mean_ndcg5"]
        for summary in report["panels"].values()
    )
    expected_pass = bool(float(primary["ci_low"]) > 0 and panel_nonnegative)
    if report["gates"]["pass_to_official_dreams_confirmation"] is not expected_pass:
        raise RuntimeError("A1 decision gate is inconsistent with frozen metrics")
    print(
        "[validate_reference_anchored_multi_probe_a1] PASS "
        f"identities={len(frame)} proceed={expected_pass}"
    )


if __name__ == "__main__":
    main()
