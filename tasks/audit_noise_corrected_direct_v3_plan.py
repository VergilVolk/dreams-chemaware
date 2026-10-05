"""Audit the information funnel and schedule for a proposed direct-v3 ledger."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from noise_corrected_action_routing_v3 import (
    _mechanism_block,
    select_diverse_routed_actions_v3,
)
from noise_corrected_direct_v3_core import v3_schedule_geometry


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--batch-queries", type=int, default=4)
    parser.add_argument("--protective-batch-queries", type=int, default=8)
    parser.add_argument("--maximum-protective-microbatches-per-step", type=int, default=4)
    parser.add_argument("--maximum-auxiliary-microbatches-per-step", type=int, default=4)
    parser.add_argument("--maximum-corrective-recycle-factor", type=float, default=4.0)
    parser.add_argument("--maximum-corrective-per-query", type=int, default=16)
    parser.add_argument("--maximum-harmful-per-query", type=int, default=8)
    parser.add_argument("--maximum-robust-per-query", type=int, default=8)
    parser.add_argument("--margin-transfer-fraction", type=float, default=0.50)
    parser.add_argument("--margin-transfer-cap", type=float, default=0.10)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def stable_fold(value: str, folds: int, seed: int) -> int:
    payload = f"{seed}|{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % folds


def _summary(values: pd.Series) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if not len(numeric):
        return {"count": 0, "minimum": None, "p25": None, "median": None,
                "p75": None, "maximum": None}
    quantiles = np.quantile(numeric, [0.25, 0.50, 0.75])
    return {
        "count": int(len(numeric)),
        "minimum": float(np.min(numeric)),
        "p25": float(quantiles[0]),
        "median": float(quantiles[1]),
        "p75": float(quantiles[2]),
        "maximum": float(np.max(numeric)),
    }


def _counts(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(frame)),
        "queries": int(frame.query_index.nunique()) if len(frame) else 0,
        "identities": int(frame.query_ik14.astype(str).nunique()) if len(frame) else 0,
        "formulas": int(frame.query_formula.astype(str).nunique()) if len(frame) else 0,
        "source_rows": {
            str(key): int(value)
            for key, value in frame.groupby("source").size().items()
        } if len(frame) else {},
    }


def audit(args: argparse.Namespace) -> tuple[dict[str, object], pd.DataFrame]:
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    if (
        args.outer_fold not in range(5)
        or args.batch_queries < 1
        or args.protective_batch_queries < 1
        or args.maximum_protective_microbatches_per_step < 1
        or args.maximum_auxiliary_microbatches_per_step < 1
        or args.maximum_corrective_recycle_factor < 1
    ):
        raise ValueError("outer fold and batch size are invalid")
    if not 0 <= args.margin_transfer_fraction <= 1 or args.margin_transfer_cap <= 0:
        raise ValueError("margin-transfer configuration is invalid")
    graph_path = args.graph_dir / "candidate_graph.npz"
    routed_path = args.ledger_dir / "routing_ledger.csv.gz"
    training_path = args.ledger_dir / "training_actions.csv.gz"
    report_path = args.ledger_dir / "report.json"
    if not all(
        path.is_file() for path in (graph_path, routed_path, training_path, report_path)
    ):
        raise FileNotFoundError("graph or routed ledger is incomplete")

    ledger_report = json.loads(report_path.read_text(encoding="utf-8"))
    if ledger_report.get("status") != "noise_corrected_routed_action_ledger_complete":
        raise RuntimeError("routed ledger report status failed")
    formal = bool(ledger_report.get("formal_training_authorized") is True)
    if ledger_report.get("contracts", {}).get(
        "control_semantics_explicit_and_source_validated"
    ) is not True:
        raise RuntimeError("routed ledger does not preserve source-specific controls")
    routed = pd.read_csv(routed_path, low_memory=False)
    actual_training = pd.read_csv(training_path, low_memory=False)
    if "control_semantic" not in routed or "control_semantic" not in actual_training:
        raise RuntimeError("routed ledger misses explicit control semantics")
    selected = select_diverse_routed_actions_v3(
        routed,
        maximum_corrective_per_query=args.maximum_corrective_per_query,
        maximum_harmful_per_query=args.maximum_harmful_per_query,
        maximum_robust_per_query=args.maximum_robust_per_query,
    )
    definitions = {
        "corrective": "selected_corrective",
        "harmful": "selected_harmful",
        "robust": "selected_robust",
    }
    panels = {kind: selected.loc[selected[column]].copy()
              for kind, column in definitions.items()}
    training = pd.concat([
        panel.assign(supervision_kind=kind) for kind, panel in panels.items()
    ], ignore_index=True, sort=False)
    training = training.sort_values(
        ["supervision_kind", "query_index", "source", "family", "action_id"],
        kind="stable",
    ).reset_index(drop=True)
    expected_ids = training[["action_id", "supervision_kind"]].astype(str).to_dict("list")
    observed_ids = (
        actual_training.sort_values(
            ["supervision_kind", "query_index", "source", "family", "action_id"],
            kind="stable",
        )[["action_id", "supervision_kind"]]
        .reset_index(drop=True).astype(str).to_dict("list")
    )
    if expected_ids != observed_ids:
        raise RuntimeError("prospective selector replay differs from materialized training ledger")

    with np.load(graph_path, allow_pickle=False) as body:
        query_formula = np.asarray(body["query_formula"], dtype=str)
    outer_folds = np.asarray([
        stable_fold(value, 5, args.formula_fold_seed) for value in query_formula
    ], dtype=np.int8)
    protective_queries = int(np.sum(outer_folds != args.outer_fold))
    corrective_queries = int(panels["corrective"].query_index.nunique())
    robust_queries = int(panels["robust"].query_index.nunique())
    harmful_queries = int(panels["harmful"].query_index.nunique())
    schedule_geometry = v3_schedule_geometry(
        corrective_queries=corrective_queries,
        robust_queries=robust_queries,
        harmful_queries=harmful_queries,
        protective_queries=protective_queries,
        corrective_batch_size=args.batch_queries,
        protective_batch_size=args.protective_batch_queries,
        maximum_auxiliary_microbatches_per_step=(
            args.maximum_auxiliary_microbatches_per_step
        ),
        maximum_protective_microbatches_per_step=(
            args.maximum_protective_microbatches_per_step
        ),
        maximum_corrective_recycle_factor=(
            args.maximum_corrective_recycle_factor
        ),
    ).as_dict()
    original_corrective_batches = int(
        schedule_geometry["original_corrective_batches"]
    )
    corrective_batches = int(schedule_geometry["cap_safe_corrective_batches"])
    robust_batches = int(schedule_geometry["robust_batches"])
    harmful_batches = int(schedule_geometry["harmful_batches"])
    auxiliary_batches = int(schedule_geometry["auxiliary_batches"])
    v2_protective_batches = math.ceil(protective_queries / args.batch_queries)
    protective_batches = int(schedule_geometry["protective_batches"])
    current_steps = max(original_corrective_batches, v2_protective_batches)
    current_action_steps = original_corrective_batches
    balanced_steps = int(schedule_geometry["required_optimizer_steps"])
    protective_per_balanced = (
        protective_batches / balanced_steps if balanced_steps else float("inf")
    )
    corrective_recycle_factor = (
        float(schedule_geometry["effective_corrective_recycle_factor"])
    )
    auxiliary_per_balanced = (
        auxiliary_batches / balanced_steps if balanced_steps else float("inf")
    )
    robust_epoch_scale = (
        min(1.0, balanced_steps / robust_batches) if robust_batches else 0.0
    )
    harmful_epoch_scale = (
        min(1.0, balanced_steps / harmful_batches) if harmful_batches else 0.0
    )
    schedule_plausible = bool(
        protective_per_balanced <= args.maximum_protective_microbatches_per_step
        and auxiliary_per_balanced <= args.maximum_auxiliary_microbatches_per_step
        and corrective_recycle_factor <= args.maximum_corrective_recycle_factor
    )

    corrective = panels["corrective"]
    conservative_gain = pd.to_numeric(
        corrective.conservative_gain, errors="coerce"
    )
    capped = conservative_gain.clip(lower=0, upper=args.margin_transfer_cap)
    target_shift = float(args.margin_transfer_fraction) * capped
    source_kind = {
        f"{source}|{kind}": int(count)
        for (kind, source), count in training.groupby(
            ["supervision_kind", "source"]
        ).size().items()
    }
    routed_kind = {
        str(key): int(value) for key, value in routed.groupby("route").size().items()
    }
    selected_kind = {kind: int(len(panel)) for kind, panel in panels.items()}
    actions_per_query = training.groupby("query_index").size()
    sources_per_query = training.groupby("query_index").source.nunique()
    family_coverage = {
        str(source): sorted(set(map(str, block.family)))
        for source, block in training.groupby("source", sort=True)
    }
    n_rows = training.loc[training.source.astype(str).eq("N_mature")].copy()
    n_step_rows = {
        f"{kind}|{selector}|step{int(step)}": int(count)
        for (kind, selector, step), count in n_rows.dropna(subset=["step"]).groupby(
            ["supervision_kind", "selector", "step"]
        ).size().items()
    }
    selection_fraction = {
        "corrective": float(len(panels["corrective"]) / max(routed_kind.get("corrective", 0), 1)),
        "harmful": float(len(panels["harmful"]) / max(routed_kind.get("harmful", 0), 1)),
        "robust": float(len(panels["robust"]) / max(routed_kind.get("robustness_only", 0), 1)),
    }
    clean_rank_consistency = routed.groupby("query_index").clean_rank.nunique(dropna=False)
    if bool((clean_rank_consistency > 1).any()):
        raise RuntimeError("routed sources disagree on the current-E8 clean rank")
    query_clean_rank = routed.groupby("query_index").clean_rank.first()
    current_error_queries = set(map(
        int, query_clean_rank.index[pd.to_numeric(query_clean_rank).ne(1)],
    ))
    selected_corrective_queries = set(map(
        int, panels["corrective"].query_index.unique(),
    ))
    selected_top1_correcting_queries = set(map(
        int,
        panels["corrective"].loc[
            pd.to_numeric(panels["corrective"].action_rank).eq(1),
            "query_index",
        ].unique(),
    ))
    if not selected_top1_correcting_queries <= selected_corrective_queries:
        raise RuntimeError("Top-1 correcting query is absent from corrective selection")
    full_initial_e8_error_queries = None
    for source_panel in ledger_report.get("source_action_panels", []):
        if source_panel.get("status") != "noise_corrected_full_p_router_audit_complete":
            continue
        action_panel = source_panel.get("action_panel") or {}
        if str(action_panel.get("selection", "")).startswith("all_initial_E8_errors"):
            full_initial_e8_error_queries = int(action_panel["initial_E8_error_queries"])
            break
    error_denominator = (
        full_initial_e8_error_queries
        if full_initial_e8_error_queries is not None else len(current_error_queries)
    )
    corrective_coverage = len(selected_corrective_queries)
    if corrective_coverage > error_denominator:
        raise RuntimeError("selected corrective queries exceed the current-E8 error panel")
    maximum_training_geometry_headroom_pp = float(
        100.0 * len(selected_top1_correcting_queries) / protective_queries
    )
    routed_route_name = {
        "corrective": "corrective",
        "harmful": "harmful",
        "robust": "robustness_only",
    }
    mechanism_coverage: dict[str, object] = {}
    for kind, panel in panels.items():
        available = routed.loc[
            routed.route.astype(str).eq(routed_route_name[kind])
        ].copy()
        available["mechanism"] = available.source.astype(str).map(_mechanism_block)
        chosen = panel.copy()
        chosen["mechanism"] = chosen.source.astype(str).map(_mechanism_block)
        available_sets = available.groupby("query_index").mechanism.agg(
            lambda values: frozenset(map(str, values))
        )
        chosen_sets = chosen.groupby("query_index").mechanism.agg(
            lambda values: frozenset(map(str, values))
        )
        dropped = [
            int(query) for query, mechanisms in available_sets.items()
            if not mechanisms <= chosen_sets.get(query, frozenset())
        ]
        mechanism_coverage[kind] = {
            "queries": int(len(available_sets)),
            "queries_losing_an_available_mechanism_block": int(len(dropped)),
            "no_N_P_A4_starvation_under_cap": not dropped,
        }
    report: dict[str, object] = {
        "status": "noise_corrected_direct_v3_plan_audit_complete",
        "formal": formal,
        "scope": (
            "formal pretraining plan audit; no encoder performance claim"
            if formal else
            "development routed ledger; no encoder training and no performance claim"
        ),
        "outer_fold": args.outer_fold,
        "route_storage_scope": (
            "lossless_per_mechanism_selector_frontier"
            if ledger_report.get("contracts", {}).get(
                "source_selector_frontiers_composed_losslessly",
            ) else "complete_legacy_route_rows"
        ),
        "routed_route_rows": routed_kind,
        "selected_kind_rows": selected_kind,
        "selected_kind_counts": {kind: _counts(panel) for kind, panel in panels.items()},
        "selected_source_kind_rows": source_kind,
        "selected_control_semantic_rows": {
            str(semantic): int(count)
            for semantic, count in training.groupby("control_semantic").size().items()
        },
        "action_recipe_coverage": {
            "families_by_source": family_coverage,
            "N_mature_supervision_selector_step_rows": n_step_rows,
            "actions_per_query": _summary(actions_per_query),
            "sources_per_query": _summary(sources_per_query),
            "queries_with_multiple_sources": int((sources_per_query >= 2).sum()),
        },
        "selection_fraction_of_materialized_route_or_frontier": selection_fraction,
        "direct_action_training_headroom": {
            "current_E8_error_queries_observed_in_routes": int(
                len(current_error_queries)
            ),
            "errors_with_selected_corrective_action": int(
                corrective_coverage
            ),
            "errors_with_selected_actual_top1_correcting_action": int(
                len(selected_top1_correcting_queries)
            ),
            "full_initial_E8_error_queries_from_P_panel": full_initial_e8_error_queries,
            "fraction_of_initial_E8_errors_with_selected_corrective_action": float(
                corrective_coverage / error_denominator
            ) if error_denominator else 0.0,
            "selected_corrective_query_coverage_pp_of_action_panel": float(
                100.0 * corrective_coverage / error_denominator
            ) if error_denominator else 0.0,
            "maximum_training_geometry_recall1_headroom_pp": (
                maximum_training_geometry_headroom_pp
            ),
            "recall1_headroom_uses_actual_action_rank1_not_partial_margin": True,
            "interpretation": (
                "Training-side selected actual Top-1 action capacity only; not a held encoder forecast."
            ),
        },
        "hierarchical_selection_coverage": mechanism_coverage,
        "corrective_gain_proxy": {
            "conservative_gain": _summary(conservative_gain),
            "target_shift_after_fraction_and_cap": _summary(target_shift),
            "rows_at_or_above_cap": int((conservative_gain >= args.margin_transfer_cap).sum()),
            "fraction_at_or_above_cap": float(
                (conservative_gain >= args.margin_transfer_cap).mean()
            ) if len(corrective) else 0.0,
            "note": "ledger best-negative scalar proxy; trainer must log all live candidate edges",
        },
        "schedule": {
            "protective_queries": protective_queries,
            "corrective_queries_in_development_ledger": corrective_queries,
            "batch_queries": args.batch_queries,
            "protective_batch_queries": args.protective_batch_queries,
            "protective_microbatches": protective_batches,
            "v2_protective_microbatches": v2_protective_batches,
            "corrective_microbatches": corrective_batches,
            "original_corrective_microbatches": original_corrective_batches,
            "cap_safe_corrective_microbatches": corrective_batches,
            "corrective_microbatches_added_by_cap_safe_repartition": int(
                schedule_geometry[
                    "corrective_batches_added_by_cap_safe_repartition"
                ]
            ),
            "cap_safe_corrective_batch_size_minimum": int(
                schedule_geometry["minimum_cap_safe_corrective_batch_size"]
            ),
            "cap_safe_corrective_batch_size_maximum": int(
                schedule_geometry["maximum_cap_safe_corrective_batch_size"]
            ),
            "robust_queries_in_development_ledger": robust_queries,
            "harmful_queries_in_development_ledger": harmful_queries,
            "robust_microbatches": robust_batches,
            "harmful_microbatches": harmful_batches,
            "auxiliary_microbatches": auxiliary_batches,
            "v2_optimizer_steps_per_epoch": current_steps,
            "v2_action_active_steps": current_action_steps,
            "v2_action_active_fraction": (
                float(current_action_steps / current_steps) if current_steps else 0.0
            ),
            "balanced_optimizer_steps_per_epoch": balanced_steps,
            "balanced_action_active_fraction": 1.0 if balanced_steps else 0.0,
            "protective_microbatches_per_balanced_step": protective_per_balanced,
            "auxiliary_microbatches_per_balanced_step": auxiliary_per_balanced,
            "configured_maximum_auxiliary_microbatches_per_step": (
                args.maximum_auxiliary_microbatches_per_step
            ),
            "robust_dense_panel_epoch_scale": robust_epoch_scale,
            "harmful_dense_panel_epoch_scale": harmful_epoch_scale,
            "dense_auxiliary_epoch_mass_capped_at_one_batch_per_step": bool(
                np.isclose(robust_epoch_scale * robust_batches,
                           min(robust_batches, balanced_steps))
                and np.isclose(harmful_epoch_scale * harmful_batches,
                               min(harmful_batches, balanced_steps))
            ),
            "configured_maximum_protective_microbatches_per_step": (
                args.maximum_protective_microbatches_per_step
            ),
            "corrective_batch_recycle_factor": (
                corrective_recycle_factor
            ),
            "maximum_corrective_recycle_factor": args.maximum_corrective_recycle_factor,
            "schedule_geometry": schedule_geometry,
            "cap_safe_corrective_repartition_preserves_query_action_panels": True,
            "formal_gate": (
                "PASS" if formal and schedule_plausible
                else "FAIL" if formal
                else "WAIT_FOR_FORMAL_LEDGER" if not schedule_plausible
                else "SCHEDULING_RATIO_PLAUSIBLE"
            ),
            "note": (
                "The development ledger is deliberately query-limited. A high ratio is not "
                "a formal forecast; v3 training must recompute and gate this ratio on the full ledger."
            ),
        },
        "funnel_repairs": {
            "corrective": (
                "bounded clean margin transfer plus real action-view identity rank/safety gradient"
            ),
            "robust": "real action-view rank and safety floor without corrective reward",
            "harmful": "action-content damage weighting of the clean boundary; harmful view detached",
            "uncertain": "audit only",
            "multiplicity": (
                "query/identity equal, source-family equal inside mechanism, with "
                "trainer-side inverse-incidence N/P/A4 epoch equalization"
            ),
            "schedule": "protective microbatches distributed across action-active optimizer steps",
        },
        "implemented_but_pending_full_gpu_execution": [
            "live per-edge transfer/cap/safety activation and payload-gradient instrumentation",
            "branch-specific pretraining calibration and sparse auxiliary interleaving",
            "supervision/source/family/exact-recipe-matched shuffled-action control with independent norm calibration",
            "source-explicit control diagnostics and global N/P/A4 epoch-mass equalization",
            "Torch numerical tests and bounded integration smoke inside SBATCH",
        ],
        "contracts": {
            "materialized_training_ledger_matches_selector_replay": True,
            "mechanism_then_source_family_selection": True,
            "source_then_family_exposure_balanced_before_recipe_score": bool(
                ledger_report.get("contracts", {}).get(
                    "source_then_family_exposure_balanced_before_recipe_score"
                )
            ),
            "control_semantics_explicit_and_source_validated": True,
            "dense_auxiliary_panels_cannot_multiply_calibrated_step_mass": bool(
                schedule_plausible
                and np.isclose(
                    robust_epoch_scale * robust_batches,
                    min(robust_batches, balanced_steps),
                )
                and np.isclose(
                    harmful_epoch_scale * harmful_batches,
                    min(harmful_batches, balanced_steps),
                )
            ),
            "source_selector_frontiers_composed_losslessly": bool(
                ledger_report.get("contracts", {}).get(
                    "source_selector_frontiers_composed_losslessly",
                )
            ),
            "teacher_embedding_target_used": False,
            "encoder_performance_claimed": False,
            "recall1_headroom_counts_only_selected_actual_top1_corrections": True,
        },
    }
    return report, training


def main() -> None:
    args = arguments()
    report, training = audit(args)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent,
    ))
    try:
        training.to_csv(staging / "prospective_training_actions.csv.gz", index=False,
                        compression="gzip")
        (staging / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
