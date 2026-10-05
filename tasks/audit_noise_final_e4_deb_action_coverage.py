"""Quantify where E4-DEB lost query coverage without updating model weights."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from noise_final_e4_pmt_core import strict_bool


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block_size):
            digest.update(chunk)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--advantage-threshold", type=float, default=0.01)
    return parser.parse_args()


def source_summary(
    frame: pd.DataFrame, advantage_threshold: float,
) -> dict[str, int | float]:
    error = frame["clean_rank"].to_numpy(int) != 1
    corrective = frame["corrective"].to_numpy(bool)
    introduced = frame["introduced"].to_numpy(bool)
    positive = (~introduced) & (
        frame["paired_advantage"].to_numpy(float) >= advantage_threshold
    )
    harmful = introduced | (
        frame["paired_advantage"].to_numpy(float) <= -advantage_threshold
    )
    return {
        "actions": int(len(frame)),
        "queries": int(frame["query_index"].nunique()),
        "identities": int(frame["identity"].astype(str).nunique()),
        "formulas": int(frame["formula"].astype(str).nunique()),
        "action_covered_error_queries": int(frame.loc[error, "query_index"].nunique()),
        "strict_positive_actions": int(positive.sum()),
        "strict_positive_queries": int(frame.loc[positive, "query_index"].nunique()),
        "strict_positive_error_queries": int(frame.loc[positive & error, "query_index"].nunique()),
        "corrected_actions": int(corrective.sum()),
        "corrected_error_queries": int(frame.loc[corrective, "query_index"].nunique()),
        "harmful_actions": int(harmful.sum()),
        "harmful_queries": int(frame.loc[harmful, "query_index"].nunique()),
        "introduced_actions": int(introduced.sum()),
        "introduced_queries": int(frame.loc[introduced, "query_index"].nunique()),
    }


def main() -> None:
    args = arguments()
    if args.advantage_threshold <= 0:
        raise ValueError("advantage threshold must be positive")
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite E4-DEB coverage audit: {args.output_dir}")
    paths = {
        "manifest_report": args.manifest_dir / "report.json",
        "routed": args.manifest_dir / "all_routed_actions.csv.gz",
        "selected": args.manifest_dir / "corrective_actions.csv.gz",
        "ledger_report": args.ledger_dir / "report.json",
        "ledger": args.ledger_dir / "training_actions.csv.gz",
        "replay_report": args.replay_dir / "report.json",
        "outcomes": args.replay_dir / "current_geometry_outcomes.csv.gz",
    }
    if missing := [str(path) for path in paths.values() if not path.is_file()]:
        raise FileNotFoundError(missing)
    manifest_report = json.loads(paths["manifest_report"].read_text(encoding="utf-8"))
    ledger_report = json.loads(paths["ledger_report"].read_text(encoding="utf-8"))
    replay_report = json.loads(paths["replay_report"].read_text(encoding="utf-8"))
    if (
        manifest_report.get("formal") is not True
        or ledger_report.get("formal") is not True
        or replay_report.get("formal") is not True
        or replay_report.get("optimizer_steps") != 0
    ):
        raise RuntimeError("coverage inputs are not formal no-update artifacts")

    ledger = pd.read_csv(paths["ledger"], low_memory=False)
    outcomes = pd.read_csv(paths["outcomes"], low_memory=False)
    selected = pd.read_csv(paths["selected"], low_memory=False)
    required_ledger = {"action_id", "query_index", "identity", "formula", "source"}
    required_outcomes = {
        "action_id", "clean_rank", "target_rank", "control_rank",
        "paired_advantage", "corrected", "introduced",
    }
    if missing := required_ledger - set(ledger.columns):
        raise RuntimeError(f"coverage ledger lacks columns: {sorted(missing)}")
    if missing := required_outcomes - set(outcomes.columns):
        raise RuntimeError(f"coverage replay lacks columns: {sorted(missing)}")
    if ledger["action_id"].duplicated().any() or outcomes["action_id"].duplicated().any():
        raise RuntimeError("coverage audit requires one row per action_id")
    frame = ledger[list(required_ledger)].merge(
        outcomes[list(required_outcomes)], on="action_id", how="inner", validate="one_to_one",
    )
    if len(frame) != len(ledger) or len(frame) != len(outcomes):
        raise RuntimeError("coverage ledger/replay join is incomplete")
    frame["corrective"] = strict_bool(frame["corrected"], "corrected")
    frame["introduced"] = strict_bool(frame["introduced"], "introduced")
    source_reports = {
        str(source): source_summary(group, args.advantage_threshold)
        for source, group in frame.groupby("source", sort=True)
    }
    error = frame["clean_rank"].to_numpy(int) != 1
    corrected = frame["corrective"].to_numpy(bool)
    all_error_queries = set(frame.loc[error, "query_index"].astype(int))
    corrected_by_source = {
        str(source): set(group.loc[group["corrective"], "query_index"].astype(int))
        for source, group in frame.groupby("source", sort=True)
    }
    n_corrected = corrected_by_source.get("N", set())
    p_corrected = set().union(*(
        values for source, values in corrected_by_source.items() if source.startswith("P_")
    )) if any(source.startswith("P_") for source in corrected_by_source) else set()
    union_corrected = set(frame.loc[corrected, "query_index"].astype(int))
    selected_queries = int(selected["query_index"].nunique())
    report = {
        "status": "noise_final_e4_deb_action_coverage_complete",
        "formal": True,
        "optimizer_steps": 0,
        "advantage_threshold": float(args.advantage_threshold),
        "outer_train_action_ledger": {
            "actions": int(len(frame)),
            "queries": int(frame["query_index"].nunique()),
            "identities": int(frame["identity"].astype(str).nunique()),
            "formulas": int(frame["formula"].astype(str).nunique()),
            "action_covered_error_queries": int(len(all_error_queries)),
        },
        "sources": source_reports,
        "current_e4_deb_training_scope": {
            "selected_corrective_actions": int(len(selected)),
            "selected_corrective_queries": selected_queries,
            "selected_fraction_of_action_covered_error_queries": float(
                selected_queries / max(1, len(all_error_queries))
            ),
            "N_only": True,
            "P_actions_used": False,
        },
        "unique_recoverability": {
            "N_corrected_error_queries": int(len(n_corrected)),
            "P_corrected_error_queries": int(len(p_corrected)),
            "P_unique_beyond_N": int(len(p_corrected - n_corrected)),
            "N_P_union_corrected_error_queries": int(len(union_corrected)),
            "union_fraction_of_action_covered_errors": float(
                len(union_corrected) / max(1, len(all_error_queries))
            ),
        },
        "decision": (
            "E4-DEB is a sparse N-only mechanism test; do not infer full noise capacity "
            "from its corrected count. Expand only with paired controls and source-specific "
            "losses after the current treatment is formally adjudicated."
        ),
        "contracts": {
            "read_only": True,
            "outcomes_used_only_for_diagnostic_coverage": True,
            "no_training_schedule_changed": True,
            "P2b": "forbidden",
            "P3_consumed": False,
        },
        "provenance": {key: sha256_file(path) for key, path in paths.items()},
        "claim_limit": "Outer-train diagnostic coverage; not held performance or a trained embedding gain.",
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".e4_deb_coverage_", dir=args.output_dir.parent))
    try:
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
