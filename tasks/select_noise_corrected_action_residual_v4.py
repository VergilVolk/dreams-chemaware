"""Freeze errors left uncovered by the complete routed direct-action union.

This selector is read-only.  It consumes the all-row routing ledger rather than
the capped training table, so a query is called a complete-union residual only
when no already audited N/P/A4/E10B/E11/E12B action both routes corrective and
reaches Top-1 in the frozen E8 geometry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from noise_final_core import CandidateGraph, sha256_file, stable_fold


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCES = frozenset({
    "N_mature", "P_guided_original", "E10B", "E11", "E12B", "A4_exact",
})


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--routed-ledger-dir", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def residual_union_tables(
    ledger: pd.DataFrame,
    all_error_queries: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    required = {
        "query_index", "query_row", "query_ik14", "query_formula",
        "formula_fold", "clean_rank", "clean_margin", "source", "route",
        "action_rank", "selected_corrective",
    }
    if missing := required - set(ledger.columns):
        raise KeyError(f"routing ledger misses {sorted(missing)}")
    if ledger.empty:
        raise ValueError("routing ledger is empty")
    metadata_columns = [
        "query_row", "query_ik14", "query_formula", "formula_fold",
        "clean_rank", "clean_margin",
    ]
    for column in metadata_columns:
        if ledger.groupby("query_index", sort=False)[column].nunique(
            dropna=False,
        ).gt(1).any():
            raise RuntimeError(f"query-level {column} is not invariant")
    query = ledger.groupby("query_index", as_index=False, sort=True)[
        metadata_columns
    ].first()
    if all_error_queries is None:
        errors = query[query["clean_rank"] > 1].copy()
    else:
        error_required = {
            "query_index", "query_row", "query_ik14", "query_formula",
            "formula_fold", "clean_rank", "clean_margin",
        }
        if missing := error_required - set(all_error_queries.columns):
            raise KeyError(f"complete E8 error table misses {sorted(missing)}")
        errors = all_error_queries[list(error_required)].copy()
        if errors["query_index"].duplicated().any() or not errors["clean_rank"].gt(1).all():
            raise RuntimeError("complete E8 error table is duplicated or contains correct queries")
    strict = ledger[
        ledger["route"].eq("corrective") & ledger["action_rank"].eq(1)
    ].copy()
    complete_covered = set(map(int, strict["query_index"]))
    selected_covered = set(map(
        int, strict.loc[strict["selected_corrective"].astype(bool), "query_index"],
    ))
    errors["complete_union_top1_covered"] = errors["query_index"].isin(
        complete_covered
    )
    errors["selected_union_top1_covered"] = errors["query_index"].isin(
        selected_covered
    )
    residual = errors[~errors["complete_union_top1_covered"]].copy()
    residual["residual_kind"] = "no_existing_strict_corrective_top1_action"

    source_queries = {
        source: set(map(int, block["query_index"]))
        for source, block in strict.groupby("source", sort=True)
    }
    source_coverage = {}
    for source, covered in sorted(source_queries.items()):
        other = set().union(*(
            values for name, values in source_queries.items() if name != source
        )) if len(source_queries) > 1 else set()
        source_coverage[source] = {
            "strict_top1_queries": int(len(covered)),
            "unique_strict_top1_queries": int(len(covered - other)),
        }
    summary = {
        "initial_E8_error_queries_in_ledger": int(len(errors)),
        "complete_union_strict_top1_covered_queries": int(
            errors["complete_union_top1_covered"].sum()
        ),
        "selected_union_strict_top1_covered_queries": int(
            errors["selected_union_top1_covered"].sum()
        ),
        "complete_union_residual_queries": int(len(residual)),
        "complete_union_residual_formulas": int(
            residual["query_formula"].nunique()
        ),
        "source_coverage": source_coverage,
    }
    return residual, summary


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    if args.outer_fold not in range(5):
        raise ValueError("outer fold must be 0..4")
    graph_path = args.graph_dir / "candidate_graph.npz"
    graph_report_path = args.graph_dir / "report.json"
    ledger_path = args.routed_ledger_dir / "routing_ledger.csv.gz"
    ledger_report_path = args.routed_ledger_dir / "report.json"
    required = (graph_path, graph_report_path, ledger_path, ledger_report_path)
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)
    graph_report = json.loads(graph_report_path.read_text(encoding="utf-8"))
    ledger_report = json.loads(ledger_report_path.read_text(encoding="utf-8"))
    if graph_report.get("formal_training_authorized") is not True:
        raise RuntimeError("candidate graph is not training-authorized")
    if ledger_report.get("status") != "noise_corrected_routed_action_ledger_complete":
        raise RuntimeError("input is not a routed action ledger")
    if ledger_report.get("outer_formula_fold") != args.outer_fold:
        raise RuntimeError("routed ledger outer fold differs from requested fold")
    if ledger_report.get("provenance", {}).get(
        "candidate_graph_sha256"
    ) != sha256_file(graph_path):
        raise RuntimeError("routed ledger candidate-graph provenance drifted")
    graph = CandidateGraph(graph_path)
    ledger = pd.read_csv(ledger_path, low_memory=False)
    observed_sources = frozenset(map(str, ledger["source"].unique()))
    p_query_pieces = []
    p_provenance = []
    p_formal = True
    for source in ledger_report.get("provenance", {}).get("route_sources", []):
        directory = Path(str(source.get("directory", "")))
        report_path = directory / "report.json"
        per_query_path = directory / "per_query.csv.gz"
        if not report_path.is_file() or not per_query_path.is_file():
            continue
        source_report = json.loads(report_path.read_text(encoding="utf-8"))
        if source_report.get("status") != "noise_corrected_full_p_router_audit_complete":
            continue
        if source_report.get("provenance", {}).get(
            "candidate_graph_sha256"
        ) != sha256_file(graph_path):
            raise RuntimeError("P route and residual selector use different graphs")
        local = pd.read_csv(per_query_path)
        p_query_pieces.append(local[local["clean_rank"] > 1][[
            "query_index", "clean_rank",
        ]])
        p_formal = p_formal and bool(
            source_report.get("formal_training_authorized") is True
        )
        p_provenance.append({
            "directory": str(directory),
            "report_sha256": sha256_file(report_path),
            "per_query_sha256": sha256_file(per_query_path),
        })
    if not p_query_pieces:
        raise RuntimeError(
            "complete E8 error membership requires a P-route per_query table"
        )
    p_errors = pd.concat(p_query_pieces, ignore_index=True).drop_duplicates(
        "query_index", keep="first",
    )
    p_error_index = p_errors["query_index"].to_numpy(np.int64)
    all_error_queries = pd.DataFrame({
        "query_index": p_error_index,
        "query_row": graph.query_row[p_error_index],
        "query_ik14": graph.query_ik14[p_error_index],
        "query_formula": graph.query_formula[p_error_index],
        "formula_fold": [
            stable_fold(str(value), 5, args.formula_fold_seed)
            for value in graph.query_formula[p_error_index]
        ],
        "clean_rank": p_errors["clean_rank"].to_numpy(np.int64),
        # The P per-query ledger records the exact clean rank but not its
        # margin.  A missing margin is explicit and is never used to select the
        # residual; later action audits recompute it in the current geometry.
        "clean_margin": np.full(len(p_errors), np.nan),
    })
    residual, summary = residual_union_tables(ledger, all_error_queries)
    query_index = residual["query_index"].to_numpy(np.int64)
    if len(query_index):
        if (
            not np.array_equal(
                residual["query_row"].to_numpy(np.int64),
                graph.query_row[query_index],
            )
            or not np.array_equal(
                residual["query_formula"].astype(str).to_numpy(),
                graph.query_formula[query_index],
            )
        ):
            raise RuntimeError("residual query metadata do not match candidate graph")
    expected_folds = np.asarray([
        stable_fold(str(value), 5, args.formula_fold_seed)
        for value in residual["query_formula"]
    ], dtype=np.int8)
    if (
        len(residual)
        and not np.array_equal(
            residual["formula_fold"].to_numpy(np.int8), expected_folds,
        )
    ):
        raise RuntimeError("residual formula folds do not match registered seed")
    if residual["formula_fold"].eq(args.outer_fold).any():
        raise RuntimeError("outer-held formula leaked into residual ledger")

    formal = bool(
        ledger_report.get("formal_training_authorized") is True
        and observed_sources == EXPECTED_SOURCES
        and p_formal
    )
    outer_train_queries = int(sum(
        stable_fold(str(value), 5, args.formula_fold_seed) != args.outer_fold
        for value in graph.query_formula
    ))
    summary["maximum_additional_recall1_headroom_pp"] = float(
        100.0 * len(residual) / outer_train_queries
    )
    report = {
        "status": "noise_corrected_complete_union_residual_v4_complete",
        "formal": formal,
        "formal_action_discovery_authorized": formal,
        "outer_formula_fold": args.outer_fold,
        "formula_fold_seed": args.formula_fold_seed,
        "observed_sources": sorted(observed_sources),
        "expected_sources_complete": observed_sources == EXPECTED_SOURCES,
        "outer_train_queries": outer_train_queries,
        "summary": summary,
        "contracts": {
            "all_routed_rows_used_not_only_training_selection": True,
            "residual_requires_no_existing_corrective_top1_action": True,
            "selected_training_union_reported_separately": True,
            "outer_held_formulas_consumed": False,
            "action_outcomes_used_only_for_outer_train_residual_selection": True,
            "teacher_or_distillation_target_used": False,
        },
        "provenance": {
            "candidate_graph_sha256": sha256_file(graph_path),
            "candidate_graph_report_sha256": sha256_file(graph_report_path),
            "routing_ledger_sha256": sha256_file(ledger_path),
            "routing_ledger_report_sha256": sha256_file(ledger_report_path),
            "complete_E8_error_membership_from_P_routes": p_provenance,
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "Outer-train direct-action residual selection only; this is not "
            "held encoder performance or evidence of a 4 pp gain."
        ),
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent,
    ))
    try:
        residual.to_csv(staging / "residual_queries.csv.gz", index=False)
        (staging / "report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8",
        )
        staging.rename(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
