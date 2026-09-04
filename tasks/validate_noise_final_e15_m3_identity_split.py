"""Fail-closed validation for the frozen E15-M3 identity split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    args = parser.parse_args()
    required = {
        "report": args.split_dir / "report.json",
        "held": args.split_dir / "held_queries.csv.gz",
        "corrective": args.split_dir / "train_corrective.csv.gz",
        "harmful": args.split_dir / "train_harmful.csv.gz",
        "sentinel": args.split_dir / "sentinel_queries.csv.gz",
        "excluded": args.split_dir / "excluded_reference_identities.txt",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    report = json.loads(required["report"].read_text(encoding="utf-8"))
    if (
        report.get("status") != "noise_final_e15_m3_identity_split_complete"
        or not report.get("formal") or not report.get("pass_to_identity_holdout_training")
        or not all(report.get("gates", {}).values())
    ):
        raise RuntimeError("E15-M3 identity split report is not passing")
    held = pd.read_csv(required["held"], low_memory=False)
    corrective = pd.read_csv(required["corrective"], low_memory=False)
    harmful = pd.read_csv(required["harmful"], low_memory=False)
    sentinel = pd.read_csv(required["sentinel"], low_memory=False)
    excluded = set(required["excluded"].read_text(encoding="utf-8").splitlines())
    held_ids = set(held["query_ik14"].astype(str))
    train_ids = set(corrective["query_ik14"].astype(str)) | set(harmful["query_ik14"].astype(str))
    sentinel_ids = set(sentinel["query_ik14"].astype(str))
    if len(held) != 256 or held["query_ik14"].nunique() != 256 or held["query_index"].nunique() != 256:
        raise RuntimeError("E15-M3 held panel is not 256 identity-unique queries")
    if held_ids & train_ids or sentinel_ids & (held_ids | train_ids):
        raise RuntimeError("E15-M3 identity isolation failed")
    if excluded != held_ids | sentinel_ids:
        raise RuntimeError("E15-M3 candidate-reference exclusion ledger drifted")
    if int((held["held_kind"] == "error").sum()) != 128 or int((held["held_kind"] == "correct").sum()) != 128:
        raise RuntimeError("E15-M3 held error/correct balance drifted")
    source_counts = held.groupby(["held_kind", "source"]).size().to_dict()
    error_counts = [int(value) for (kind, _), value in source_counts.items() if kind == "error"]
    if len(error_counts) != 4 or error_counts != [32] * 4:
        raise RuntimeError(f"E15-M3 error source balance drifted: {source_counts}")
    print(
        f"[validate_noise_final_e15_m3_identity_split] PASS held={len(held)} "
        f"train_actions={len(corrective)} sentinel={len(sentinel)}", flush=True,
    )


if __name__ == "__main__":
    main()
