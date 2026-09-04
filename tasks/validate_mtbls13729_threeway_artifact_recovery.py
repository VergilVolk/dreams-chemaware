"""Validate the MTBLS13729 three-way artifact recovery audit."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "data/mtbls13729/threeway_artifact_recovery_audit_v1/report.json"


def main() -> None:
    value = json.loads(REPORT.read_text(encoding="utf-8"))
    assert value["status"] == "mtbls13729_threeway_artifact_recovery_audit_complete"
    assert value["aggregate_json_recovered"] is True
    panels = value["aggregate_results"]
    assert panels["p2b_panels"]["neg_rp"]["systems"]["dreams"]["annotated_features"] == 345
    assert panels["p2b_panels"]["pos_rp"]["systems"]["dreams"]["annotated_features"] == 3072
    assert panels["p2b_panels"]["pos_rp"]["systems"]["p2b"]["annotated_features"] == 3243
    assert panels["e6_panels"]["pos_rp"]["annotated_features"] == 3081
    assert panels["threeway"]["panels"]["pos_rp"]["features_union"] == 3254
    if value["missing_candidate_files"]:
        assert value["candidate_level_artifacts_complete"] is False
        assert value["forbidden_claims_without_candidate_tables"]
    print("[validate_mtbls13729_threeway_artifact_recovery] PASS")


if __name__ == "__main__":
    main()
