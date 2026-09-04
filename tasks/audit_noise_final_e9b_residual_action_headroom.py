"""E9-B: query-level no-op-aware residual headroom after mature E8.

E9 reports action rows, so queries with many curriculum steps are repeated.
This audit collapses those rows to one decision per query and asks a single
necessary question: after the mature shared encoder, how many remaining clean
errors can still be corrected by choosing the best frozen mature action, while
allowing no-op?  Selection uses held outcomes and is therefore an oracle upper
bound only.  It is never a deployable selector or a training result.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from noise_final_core import json_dump, sha256_file  # noqa: E402
from train_noise_final_r2_shared_encoder import formula_bootstrap_mean  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--e9-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_e9_action_staleness",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/g8r_noise_final_e9b_residual_action_headroom",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260831)
    return parser.parse_args()


def choose(group: pd.DataFrame, mode: str) -> dict:
    if group["clean_rank"].nunique() != 1 or group["clean_margin"].nunique() != 1:
        raise RuntimeError(f"clean state drift for query {int(group['query_index'].iloc[0])}")
    clean_margin = float(group["clean_margin"].iloc[0])
    clean_rank = int(group["clean_rank"].iloc[0])
    margin_column, rank_column = f"{mode}_margin", f"{mode}_rank"
    # No-op is an explicit candidate. Stable sort gives deterministic ties and
    # leaves no-op preferred when an action has exactly the same margin.
    candidates = [{
        "margin": clean_margin, "rank": clean_rank, "selector": "no_op",
        "step": 0, "path": "",
    }]
    for row in group.sort_values(["selector", "step"], kind="stable").itertuples(index=False):
        candidates.append({
            "margin": float(getattr(row, margin_column)),
            "rank": int(getattr(row, rank_column)),
            "selector": str(row.selector), "step": int(row.step),
            "path": str(getattr(row, f"{mode}_path")),
        })
    best = max(range(len(candidates)), key=lambda index: (candidates[index]["margin"], -index))
    selected = candidates[best]
    return {
        f"{mode}_oracle_rank": selected["rank"],
        f"{mode}_oracle_margin": selected["margin"],
        f"{mode}_oracle_selector": selected["selector"],
        f"{mode}_oracle_step": selected["step"],
        f"{mode}_oracle_path": selected["path"],
    }


def summary(frame: pd.DataFrame, mode: str, args: argparse.Namespace, offset: int) -> dict:
    clean = frame["clean_rank"].to_numpy(int) == 1
    oracle = frame[f"{mode}_oracle_rank"].to_numpy(int) == 1
    effect = oracle.astype(float) - clean.astype(float)
    formulas = frame["query_formula"].astype(str).to_numpy()
    errors = int(np.sum(~clean))
    corrected = int(np.sum(~clean & oracle))
    introduced = int(np.sum(clean & ~oracle))
    return {
        "queries": int(len(frame)), "identities": int(frame["query_ik14"].nunique()),
        "formulas": int(frame["query_formula"].nunique()),
        "baseline_accuracy": float(np.mean(clean)), "baseline_errors": errors,
        "oracle_accuracy": float(np.mean(oracle)), "delta_accuracy": float(np.mean(effect)),
        "corrected": corrected, "introduced": introduced,
        "fraction_of_remaining_errors_recoverable": corrected / errors if errors else 0.0,
        "formula_cluster_delta_ci": formula_bootstrap_mean(
            effect, formulas, args.bootstrap_resamples, args.seed + offset,
        ),
        "selected_policy_counts": {
            str(key): int(value) for key, value in
            frame[f"{mode}_oracle_selector"].value_counts().sort_index().items()
        },
        "selected_step_counts": {
            str(int(key)): int(value) for key, value in
            frame[f"{mode}_oracle_step"].value_counts().sort_index().items()
        },
    }


def main() -> None:
    args = arguments()
    report_path, table_path = args.e9_dir / "report.json", args.e9_dir / "per_action.csv.gz"
    if not report_path.is_file() or not table_path.is_file():
        raise FileNotFoundError("E9 report/per-action artifact is incomplete")
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite E9-B output: {args.output_dir}")
    e9 = json.loads(report_path.read_text(encoding="utf-8"))
    if e9.get("status") != "noise_final_e9_action_staleness_complete" or not e9.get("formal"):
        raise RuntimeError("E9-B requires a formal E9 artifact")
    frame = pd.read_csv(table_path, keep_default_na=False)
    required = {
        "query_index", "query_ik14", "query_formula", "selector", "step",
        "clean_rank", "clean_margin", "frozen_rank", "frozen_margin", "frozen_path",
        "online_rank", "online_margin", "online_path",
    }
    if required - set(frame.columns):
        raise RuntimeError(f"E9 table is missing {sorted(required - set(frame.columns))}")
    records = []
    for query, group in frame.groupby("query_index", sort=True):
        base = {
            "query_index": int(query), "query_ik14": str(group["query_ik14"].iloc[0]),
            "query_formula": str(group["query_formula"].iloc[0]),
            "clean_rank": int(group["clean_rank"].iloc[0]),
            "clean_margin": float(group["clean_margin"].iloc[0]),
        }
        records.append(base | choose(group, "frozen") | choose(group, "online"))
    per_query = pd.DataFrame(records)
    subset_summaries = {
        "frozen": summary(per_query, "frozen", args, 0),
        "online": summary(per_query, "online", args, 1),
    }
    # The action table covers only 2,293 held queries, whereas the retrieval
    # task contains all 5,923 held queries. Reconstruct the full task and treat
    # every uncovered query as no-op; never compare a subset delta to a full-
    # task five-point target.
    checkpoint = Path(str(e9["student_checkpoint"]))
    source_path = checkpoint.parent / "held_per_query.csv.gz"
    if not source_path.is_file():
        raise FileNotFoundError(f"mature E8 held-query table is missing: {source_path}")
    source = pd.read_csv(source_path)
    needed_source = {"query_index", "query_formula", "baseline_rank", "final_rank"}
    if needed_source - set(source.columns) or source["query_index"].duplicated().any():
        raise RuntimeError("mature E8 held-query table is malformed")
    full = source[["query_index", "query_formula", "baseline_rank", "final_rank"]].copy()
    full = full.merge(
        per_query[["query_index", "frozen_oracle_rank", "online_oracle_rank"]],
        on="query_index", how="left", validate="one_to_one",
    )
    for mode in ("frozen", "online"):
        full[f"{mode}_oracle_rank"] = (
            full[f"{mode}_oracle_rank"].fillna(full["final_rank"]).astype(int)
        )
    official_correct = full["baseline_rank"].to_numpy(int) == 1
    student_correct = full["final_rank"].to_numpy(int) == 1
    full_summaries = {}
    for offset, mode in enumerate(("frozen", "online"), start=2):
        oracle_correct = full[f"{mode}_oracle_rank"].to_numpy(int) == 1
        incremental = oracle_correct.astype(float) - student_correct.astype(float)
        total = oracle_correct.astype(float) - official_correct.astype(float)
        formulas = full["query_formula"].astype(str).to_numpy()
        full_summaries[mode] = {
            "queries": int(len(full)),
            "official_accuracy": float(np.mean(official_correct)),
            "mature_e8_accuracy": float(np.mean(student_correct)),
            "oracle_accuracy": float(np.mean(oracle_correct)),
            "incremental_delta_over_mature_e8": float(np.mean(incremental)),
            "total_delta_over_official": float(np.mean(total)),
            "incremental_corrected": int(np.sum(~student_correct & oracle_correct)),
            "incremental_introduced": int(np.sum(student_correct & ~oracle_correct)),
            "incremental_formula_cluster_ci": formula_bootstrap_mean(
                incremental, formulas, args.bootstrap_resamples, args.seed + offset,
            ),
            "total_formula_cluster_ci": formula_bootstrap_mean(
                total, formulas, args.bootstrap_resamples, args.seed + 100 + offset,
            ),
        }
    best = max(full_summaries.values(), key=lambda item: item["total_delta_over_official"])
    report = {
        "status": "noise_final_e9b_residual_action_headroom_complete", "formal": True,
        "queries": int(len(per_query)), "baseline_errors": int(np.sum(per_query["clean_rank"] != 1)),
        "action_covered_subset_summaries": subset_summaries,
        "full_task_summaries": full_summaries,
        "best_mature_action_incremental_headroom_full_task": float(best["incremental_delta_over_mature_e8"]),
        "best_mature_action_total_headroom_vs_official": float(best["total_delta_over_official"]),
        "mature_action_space_can_reach_five_total_points": bool(best["total_delta_over_official"] >= 0.05),
        "decision": (
            "conditional action selection remains a sufficient-capacity target"
            if best["total_delta_over_official"] >= 0.05
            else "mature candidate/confounder action space is insufficient for five total points; expand action families before training a selector"
        ),
        "contracts": {"one_decision_per_query": True, "no_op_allowed": True,
                      "outcome_aware_oracle_only": True, "P2b": "forbidden", "P3_consumed": False},
        "provenance": {"e9_report_sha256": sha256_file(report_path),
                       "e9_per_action_sha256": sha256_file(table_path),
                       "mature_e8_held_per_query_sha256": sha256_file(source_path),
                       "script_sha256": sha256_file(Path(__file__))},
        "claim_limit": "Outcome-aware held-fold upper bound, not a selector, trained encoder, or deployable gain.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_query.to_csv(args.output_dir / "per_query.csv.gz", index=False, compression="gzip")
    json_dump(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
