from __future__ import annotations

"""Freeze a six-candidate targeted-EIC validation panel from the broad screen."""

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/mtbls13729/ms1_consensus"
PRIORITY = ROOT / "data/mtbls13729/full_annotated_feature_audit_v1/all_priority.csv"
OUT = ROOT / "data/mtbls13729/broad_candidate_eic_targets_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    priority = pd.read_csv(PRIORITY)
    expected = {"neg_rp": {486}, "pos_rp": {41, 73, 79, 398, 732}}
    reports: dict[str, object] = {}
    for panel, ids in expected.items():
        source_targets = SOURCE / f"{panel}__requantification_targets.csv.gz"
        source_samples = SOURCE / f"{panel}__samples.csv"
        targets = pd.read_csv(source_targets)
        targets = targets.loc[targets["feature_id"].isin(ids)].copy()
        if set(targets["feature_id"].astype(int)) != ids:
            raise RuntimeError(f"{panel}: frozen target identity mismatch")
        ledger = priority.loc[
            priority["panel"].eq(panel) & priority["feature_id"].isin(ids),
            ["feature_id", "best_name", "best_ik14", "annotation_evidence_tier", "screen_fdr10"],
        ]
        targets = targets.merge(ledger, on="feature_id", validate="one_to_one")
        target_path = OUT / f"{panel}__requantification_targets.csv.gz"
        targets.to_csv(target_path, index=False, compression="gzip")
        samples = pd.read_csv(source_samples)
        sample_path = OUT / f"{panel}__samples.csv"
        samples.to_csv(sample_path, index=False)
        reports[panel] = {
            "targets": int(len(targets)),
            "samples": int(len(samples)),
            "feature_ids": sorted(ids),
            "targets_sha256": sha256(target_path),
            "samples_sha256": sha256(sample_path),
        }
    report = {
        "status": "mtbls13729_broad_candidate_eic_targets_frozen",
        "panels": reports,
        "parameters": {"ppm": 5.0, "rt_half_window_sec": 20.0, "resolve_local_peaks": True, "max_apex_delta_sec": 12.0},
        "claim_limit": "Target freezing precedes targeted-EIC extraction and contains no validation outcome.",
        "priority_sha256": sha256(PRIORITY),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
