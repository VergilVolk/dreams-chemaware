#!/usr/bin/env python
"""Pool the frozen five-fold B4 BioAware direct-embedding pilot."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from noise_final_core import sha256_file  # noqa: E402


ARMS = ("full_bioaware_safe", "full_no_edge_high_recall")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", type=Path,
        default=ROOT / "data/validation/bioaware_b4_direct_shared_embedding_v1",
    )
    parser.add_argument(
        "--manifest-dir", type=Path,
        default=ROOT / "data/validation/bioaware_b4_direct_manifest_v1",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260905)
    return parser.parse_args()


def formula_bootstrap(frame: pd.DataFrame, repeats: int, seed: int,
                      column: str = "delta_top1") -> dict:
    grouped = frame.groupby("truth_formula", sort=True)[column].agg(["sum", "count"])
    if grouped.empty:
        raise RuntimeError("empty formula bootstrap")
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=float)
    for index in range(repeats):
        draw = rng.integers(0, len(grouped), len(grouped))
        values[index] = sums[draw].sum() / counts[draw].sum()
    return {
        "mean": float(frame[column].mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "formulas": int(len(grouped)),
        "resamples": int(repeats),
    }


def summarize(frame: pd.DataFrame, repeats: int, seed: int) -> dict:
    old_top1 = frame["old_rank"].eq(1)
    new_top1 = frame["new_rank"].eq(1)
    frame = frame.copy()
    frame["delta_top1"] = new_top1.astype(int) - old_top1.astype(int)
    frame["delta_rr"] = 1.0 / frame["new_rank"] - 1.0 / frame["old_rank"]
    corrected = int((~old_top1 & new_top1).sum())
    introduced = int((old_top1 & ~new_top1).sum())
    return {
        "queries": int(len(frame)),
        "identities": int(frame["truth_ik14"].nunique()),
        "formulas": int(frame["truth_formula"].nunique()),
        "baseline_recall1": float(old_top1.mean()),
        "recall1": float(new_top1.mean()),
        "delta_recall1": float(frame["delta_top1"].mean()),
        "baseline_mrr": float((1.0 / frame["old_rank"]).mean()),
        "mrr": float((1.0 / frame["new_rank"]).mean()),
        "delta_mrr": float(frame["delta_rr"].mean()),
        "corrected": corrected,
        "introduced": introduced,
        "risk_net_lambda2": corrected - 2 * introduced,
        "formula_cluster_top1_bootstrap": formula_bootstrap(
            frame, repeats, seed, "delta_top1",
        ),
        "formula_cluster_mrr_bootstrap": formula_bootstrap(
            frame, repeats, seed + 1, "delta_rr",
        ),
        "by_unit": {
            str(unit): {
                "queries": int(len(group)),
                "delta_recall1": float(group["delta_top1"].mean()),
                "corrected": int((group["delta_top1"] > 0).sum()),
                "introduced": int((group["delta_top1"] < 0).sum()),
            }
            for unit, group in frame.groupby("unit_id", sort=True)
        },
    }


def main() -> None:
    args = arguments()
    manifest_path = args.manifest_dir / "manifest.npz"
    manifest_report = args.manifest_dir / "report.json"
    if not manifest_path.is_file() or not manifest_report.is_file():
        raise FileNotFoundError("B4 manifest is missing")
    with np.load(manifest_path, allow_pickle=False) as loaded:
        expected_query_ids = set(map(str, loaded["query_id"]))
    arm_frames: dict[str, pd.DataFrame] = {}
    arm_reports: dict[str, dict] = {}
    source_hashes = {}
    for arm_index, arm in enumerate(ARMS):
        frames = []
        folds = []
        for fold in range(args.folds):
            directory = args.input_root / arm / f"fold_{fold}"
            report_path = directory / "report.json"
            per_query_path = directory / "held_per_query.csv.gz"
            checkpoint_path = directory / "final_trainable_state.pt"
            for path in (report_path, per_query_path, checkpoint_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (report.get("status") != "bioaware_b4_direct_shared_embedding_fold_complete"
                    or report.get("arm") != arm or int(report.get("outer_fold", -1)) != fold):
                raise RuntimeError(f"invalid fold report: {report_path}")
            if report.get("numeric_integrity") != {
                    "all_finite": True, "amp_enabled": False}:
                raise RuntimeError(
                    f"non-finite or AMP-enabled fold is forbidden: {report_path}"
                )
            history_values = [
                value for epoch in report.get("history", [])
                for key, value in epoch.items() if key != "epoch"
            ]
            if (not history_values
                    or not np.isfinite(np.asarray(history_values, dtype=float)).all()):
                raise RuntimeError(f"non-finite or empty optimization history: {report_path}")
            frame = pd.read_csv(per_query_path)
            if frame["query_id"].duplicated().any() or not frame["outer_fold"].eq(fold).all():
                raise RuntimeError(f"invalid per-query ledger: {per_query_path}")
            frames.append(frame)
            folds.append({
                "fold": fold,
                "held_queries": int(len(frame)),
                "delta_recall1": report["shared_embedding"]["delta_recall1"],
                "corrected": report["shared_embedding"]["corrected"],
                "introduced": report["shared_embedding"]["introduced"],
                "preservation": report["shared_embedding"]["preservation"],
            })
            source_hashes[f"{arm}/fold_{fold}/report"] = sha256_file(report_path)
            source_hashes[f"{arm}/fold_{fold}/ledger"] = sha256_file(per_query_path)
            source_hashes[f"{arm}/fold_{fold}/checkpoint"] = sha256_file(checkpoint_path)
        pooled = pd.concat(frames, ignore_index=True).sort_values("query_id").reset_index(drop=True)
        if pooled["query_id"].duplicated().any() or set(pooled["query_id"].astype(str)) != expected_query_ids:
            raise RuntimeError(f"{arm}: five-fold OOF does not cover the exact manifest")
        arm_frames[arm] = pooled
        summary = summarize(
            pooled, args.bootstrap_resamples, args.seed + arm_index * 100,
        )
        summary["folds"] = folds
        summary["all_fold_preservation_ge_0_995"] = all(
            fold["preservation"] >= 0.995 for fold in folds
        )
        summary["gates"] = {
            "top1_formula_ci_positive": summary["formula_cluster_top1_bootstrap"]["ci_low"] > 0,
            "mrr_nonnegative": summary["delta_mrr"] >= 0,
            "corrected_gt_introduced": summary["corrected"] > summary["introduced"],
            "risk_net_lambda2_positive": summary["risk_net_lambda2"] > 0,
            "all_fold_preservation_ge_0_995": summary["all_fold_preservation_ge_0_995"],
        }
        summary["pass_to_independent_validation"] = all(summary["gates"].values())
        arm_reports[arm] = summary
        pooled.to_csv(args.input_root / f"{arm}__oof_per_query.csv.gz", index=False, compression="gzip")

    left = arm_frames[ARMS[0]][["query_id", "truth_formula", "old_rank", "new_rank"]].copy()
    right = arm_frames[ARMS[1]][["query_id", "new_rank"]].copy()
    paired = left.merge(right, on="query_id", suffixes=("_safe", "_recall"), validate="one_to_one")
    paired["delta_arm_top1"] = (
        paired["new_rank_recall"].eq(1).astype(int)
        - paired["new_rank_safe"].eq(1).astype(int)
    )
    comparison = formula_bootstrap(
        paired.rename(columns={"delta_arm_top1": "delta_top1"}),
        args.bootstrap_resamples, args.seed + 999, "delta_top1",
    )
    report = {
        "status": "bioaware_b4_direct_shared_embedding_oof_complete",
        "formal": False,
        "protocol": "opened-cohort five truth-formula-fold OOF; fixed two-arm direct shared-encoder pilot",
        "arms": arm_reports,
        "high_recall_minus_safe_formula_bootstrap": comparison,
        "contracts": {
            "same_shared_encoder_query_reference": True,
            "inference_clean_spectrum_only": True,
            "network_context_used_at_inference": False,
            "reaction_neighbours_used_as_positives": False,
            "P2b_used": False,
            "held_fold_used_for_checkpoint_selection": False,
        },
        "provenance": {
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_report_sha256": sha256_file(manifest_report),
            "source_artifacts": source_hashes,
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": "Opened-cohort engineering evidence. Even a passing arm requires a matched routing control and a new independent external cohort.",
    }
    output = args.input_root / "summary.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
