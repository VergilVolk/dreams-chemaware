#!/usr/bin/env python
"""Reserve the one-time Enveda opening after every score is frozen."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from train_grand_unified_evidence_model import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--freeze-report", type=Path, required=True)
    parser.add_argument("--benchmark-report", type=Path, required=True)
    parser.add_argument("--reference-score-bundle", type=Path, required=True)
    parser.add_argument("--v2-selection-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    required = (
        args.model, args.freeze_report, args.benchmark_report,
        args.reference_score_bundle, args.v2_selection_report,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    freeze = json.loads(args.freeze_report.read_text(encoding="utf-8"))
    benchmark = json.loads(args.benchmark_report.read_text(encoding="utf-8"))
    v2 = json.loads(args.v2_selection_report.read_text(encoding="utf-8"))
    if freeze.get("freeze_contract_status") != "DEVELOPMENT_MODEL_FROZEN_EXTERNAL_UNOPENED":
        raise RuntimeError("model is not frozen before external opening")
    if freeze.get("external_test_opened") is not False:
        raise RuntimeError("freeze report says external test was opened")
    if benchmark.get("truth_status") != "frozen_unscored" or benchmark.get("performance_scores_opened") is not False:
        raise RuntimeError("Enveda benchmark is not frozen and unopened")
    if v2.get("labels_opened") is not False:
        raise RuntimeError("ChemAware V2 score generation opened labels")
    payload = {
        "schema": "candidate_energy_enveda_global_opening_v1",
        "status": "OPENING_RESERVED_ALL_SCORES_FROZEN_LABELS_NOT_YET_LOADED",
        "reserved_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "panels": ["identity_disjoint", "formula_disjoint"],
        "model_sha256": sha256_file(args.model),
        "freeze_report_sha256": sha256_file(args.freeze_report),
        "benchmark_report_sha256": sha256_file(args.benchmark_report),
        "reference_score_bundle_sha256": sha256_file(args.reference_score_bundle),
        "v2_selection_report_sha256": sha256_file(args.v2_selection_report),
        "adaptation_between_panels_prohibited": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
