"""Build the E15 multi-source, multi-action corrective/harmful/no-op ledger."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from build_noise_final_e14_crossfit_p_teacher import action_definitions  # noqa: E402
from noise_final_core import CandidateGraph, json_dump, sha256_file, stable_fold  # noqa: E402
from noise_final_e15_core import bounded_stratified_epoch, exposure_items  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--r0-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_r0_faithful_s3a")
    parser.add_argument("--a4-dir", type=Path, default=ROOT / "data/validation/g8r_noise_v3_a4_exact_peak_scan")
    parser.add_argument("--c1-dir", type=Path, default=ROOT / "data/validation/g8r_noise_v3_c1_crossfit_teacher")
    parser.add_argument("--e14-dir", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=4)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--maximum-corrective-actions", type=int, default=4)
    parser.add_argument("--maximum-harmful-actions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def check_report(
    path: Path, status: str, formal_path: tuple[str, ...] = ("formal",),
) -> dict:
    body = json.loads(path.read_text(encoding="utf-8"))
    formal: object = body
    for key in formal_path:
        formal = formal.get(key) if isinstance(formal, dict) else None
    if body.get("status") != status or formal is not True:
        raise RuntimeError(f"non-formal or wrong-status source report: {path}")
    return body


def kind(corrected: bool, introduced: bool, baseline_rank: int, result_rank: int) -> str:
    if corrected or (baseline_rank != 1 and result_rank == 1):
        return "corrective"
    if introduced or (baseline_rank == 1 and result_rank != 1):
        return "harmful"
    return "neutral"


def base_record(graph: CandidateGraph, query: int, fold_seed: int) -> dict[str, object]:
    return {
        "query_index": int(query),
        "query_row": int(graph.query_row[query]),
        "query_ik14": str(graph.query_ik14[query]),
        "query_formula": str(graph.query_formula[query]),
        "formula_fold": int(stable_fold(str(graph.query_formula[query]), 5, fold_seed)),
        "has_near": bool(graph.query_has_near[query]),
    }


def select_diverse(frame: pd.DataFrame, maximum: int, harmful: bool) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    output: list[pd.DataFrame] = []
    for _, block in frame.groupby("query_index", sort=True):
        block = block.copy()
        block["source_family"] = block["source"].astype(str) + "|" + block["action_family"].astype(str)
        block["priority"] = (
            block["replicated_formula_folds"].astype(float) * 10.0
            + np.log1p(block["conditional_identities"].astype(float))
            + ((-block["margin_delta"]) if harmful else block["margin_delta"]).astype(float)
        )
        block = block.sort_values(
            ["priority", "source_family", "action_id"], ascending=[False, True, True], kind="stable",
        )
        first = block.drop_duplicates("source_family", keep="first").head(maximum)
        if len(first) < maximum:
            remainder = block.loc[~block.index.isin(first.index)].head(maximum - len(first))
            first = pd.concat([first, remainder], ignore_index=False)
        output.append(first.drop(columns=["source_family", "priority"]))
    return pd.concat(output, ignore_index=True) if output else frame.iloc[0:0].copy()


def main() -> None:
    args = arguments()
    if not 0 <= args.outer_fold < 5:
        raise ValueError("outer fold must be in [0,4]")
    if args.maximum_corrective_actions < 2 or args.maximum_harmful_actions < 1:
        raise ValueError("E15 requires >=2 corrective and >=1 harmful slots")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite E15 ledger: {args.output_dir}")
    required = {
        "graph": args.graph,
        "r0_report": args.r0_dir / "report.json",
        "r0_actions": args.r0_dir / "outcome_audit_only.csv.gz",
        "a4_decision": args.a4_dir / "decision.json",
        "a4_actions": args.a4_dir / "policy_candidate_actions.csv.gz",
        "c1_decision": args.c1_dir / "decision.json",
        "c1_examples": args.c1_dir / "crossfit_examples.csv.gz",
        "e14_report": args.e14_dir / "report.json",
        "e14_outcomes": args.e14_dir / "action_outcomes.npz",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    r0_report = check_report(required["r0_report"], "noise_final_r0_faithful_s3a_manifest_complete")
    # A4 predates the newer top-level ``formal`` convention. Its immutable
    # decision records formality under ``integrity.formal``. Do not reject the
    # genuine artifact merely because its schema is older; validate the full
    # integrity ledger instead.
    a4_report = check_report(
        required["a4_decision"], "noise_v3_a4_exact_peak_scan_decision",
        formal_path=("integrity", "formal"),
    )
    c1_report = check_report(required["c1_decision"], "noise_v3_c1_crossfit_teacher_complete")
    e14_report = check_report(required["e14_report"], "noise_final_e14_crossfit_p_teacher_complete")
    expected_a4_integrity = {
        "official_errors_scanned": 1805,
        "safety_controls": 3193,
        "fragment_actions": 206288,
        "exact_variants": 825152,
    }
    observed_a4_integrity = a4_report.get("integrity", {})
    drifted_a4 = {
        key: (observed_a4_integrity.get(key), expected)
        for key, expected in expected_a4_integrity.items()
        if int(observed_a4_integrity.get(key, -1)) != expected
    }
    if drifted_a4:
        raise RuntimeError(f"A4 integrity ledger drifted: {drifted_a4}")
    if int(e14_report.get("outer_formula_fold", -1)) != args.outer_fold:
        raise RuntimeError("E14 source fold does not match E15 outer fold")
    graph = CandidateGraph(args.graph)
    rows: list[dict[str, object]] = []

    r0 = pd.read_csv(required["r0_actions"])
    for row in r0.itertuples(index=False):
        query = int(row.query_index)
        record = base_record(graph, query, args.formula_fold_seed)
        record.update({
            "source": "R0_N", "geometry": "official",
            "action_family": str(row.selector),
            "action_id": f"R0|{row.selector}|dose={float(row.attenuation):.2f}|step={int(row.step)}",
            "action_payload": str(row.target_path),
            "dose": float(row.attenuation), "step": int(row.step),
            "baseline_rank": int(row.baseline_rank), "baseline_margin": float(row.baseline_margin),
            "result_rank": int(row.target_rank), "result_margin": float(row.target_margin),
            "margin_delta": float(row.target_margin - row.baseline_margin),
            "supervision_kind": kind(bool(row.corrected), bool(row.introduced), int(row.baseline_rank), int(row.target_rank)),
        })
        rows.append(record)

    a4 = pd.read_csv(required["a4_actions"])
    a4 = a4.loc[a4["policy_eligible"].astype(bool) & a4["gradient_rank"].astype(int).le(50)].copy()
    for row in a4.itertuples(index=False):
        query = int(row.query_index)
        record = base_record(graph, query, args.formula_fold_seed)
        record.update({
            "source": "A4_exact", "geometry": "official",
            "action_family": f"peak_attenuation|{row.role}|{row.gradient_rank_bin}",
            "action_id": f"A4|q={query}|token={int(row.token)}|dose={float(row.attenuation):.2f}",
            "action_payload": json.dumps({"token": int(row.token), "mz": float(row.mz), "role": str(row.role)}),
            "dose": float(row.attenuation), "step": 1,
            "baseline_rank": int(row.baseline_rank), "baseline_margin": float(row.baseline_margin),
            "result_rank": int(row.result_rank), "result_margin": float(row.result_margin),
            "margin_delta": float(row.margin_change),
            "supervision_kind": kind(bool(row.corrected), bool(row.introduced), int(row.baseline_rank), int(row.result_rank)),
            "score_error_family": str(row.score_error_family),
            "positive_deficit": bool(row.positive_deficit), "negative_excess": bool(row.negative_excess),
            "rules_favor_positive": bool(row.rules_favor_positive), "rules_favor_wrong": bool(row.rules_favor_wrong),
        })
        rows.append(record)

    c1 = pd.read_csv(required["c1_examples"])
    for row in c1.itertuples(index=False):
        query = int(row.query_index)
        record = base_record(graph, query, args.formula_fold_seed)
        record.update({
            "source": "C1_support_disjoint", "geometry": "official_support_disjoint",
            "action_family": "positive_identity_prototype",
            "action_id": f"C1|q={query}|eval={int(row.evaluation_positive_row)}|teachers={row.teacher_rows}",
            "action_payload": json.dumps({
                "evaluation_positive_row": int(row.evaluation_positive_row),
                "teacher_rows": str(row.teacher_rows),
            }),
            "dose": float(c1_report.get("parameters", {}).get("alpha", 0.25)), "step": 1,
            "baseline_rank": int(row.baseline_rank), "baseline_margin": float(row.baseline_margin),
            "result_rank": int(row.teacher_rank), "result_margin": float(row.teacher_margin),
            "margin_delta": float(row.teacher_margin - row.baseline_margin),
            "supervision_kind": kind(bool(row.corrected), bool(row.introduced), int(row.baseline_rank), int(row.teacher_rank)),
            "score_error_family": "positive_evidence",
        })
        rows.append(record)

    definitions = {definition.action_id: definition for definition in action_definitions()}
    with np.load(required["e14_outcomes"], allow_pickle=True) as body:
        queries = np.asarray(body["queries"], dtype=np.int64)
        action_ids = np.asarray(body["action_ids"], dtype=str)
        clean_rank = np.asarray(body["clean_rank"], dtype=np.int16)
        clean_margin = np.asarray(body["clean_margin"], dtype=np.float32)
        result_rank = np.asarray(body["result_rank"], dtype=np.int16)
        result_margin = np.asarray(body["result_margin"], dtype=np.float32)
    if result_rank.shape != (len(queries), len(action_ids)) or result_margin.shape != result_rank.shape:
        raise RuntimeError("E14 outcome tensor shape mismatch")
    missing_definitions = sorted(set(action_ids) - set(definitions))
    if missing_definitions:
        raise RuntimeError(f"E14 action ids cannot be reconstructed: {missing_definitions}")
    for local, query in enumerate(queries):
        for action_index, action_id in enumerate(action_ids):
            baseline_rank = int(clean_rank[local])
            target_rank = int(result_rank[local, action_index])
            supervision = kind(False, False, baseline_rank, target_rank)
            if supervision == "neutral":
                continue
            definition = definitions[str(action_id)]
            record = base_record(graph, int(query), args.formula_fold_seed)
            record.update({
                "source": "E14_mature_P", "geometry": "mature_crossfit",
                "action_family": str(definition.family), "action_id": str(action_id),
                "action_payload": json.dumps({
                    "reference_policy": definition.reference_policy,
                    "auxiliary_dose": definition.auxiliary_dose,
                    "prevalence": definition.prevalence,
                    "maximum_peaks": definition.maximum_peaks,
                    "support_weighted": definition.support_weighted,
                }),
                "dose": float(definition.dose), "step": 1,
                "baseline_rank": baseline_rank, "baseline_margin": float(clean_margin[local]),
                "result_rank": target_rank, "result_margin": float(result_margin[local, action_index]),
                "margin_delta": float(result_margin[local, action_index] - clean_margin[local]),
                "supervision_kind": supervision,
            })
            rows.append(record)

    ledger = pd.DataFrame(rows)
    if ledger.empty or ledger.duplicated(["source", "query_index", "action_id"]).any():
        raise RuntimeError("E15 action ledger is empty or contains duplicate source/query/action rows")
    ledger = ledger.loc[ledger["formula_fold"].astype(int).ne(args.outer_fold)].copy()
    ledger["score_error_family"] = ledger.get("score_error_family", pd.Series(index=ledger.index, dtype=object)).fillna("unknown")
    ledger["conditional_key"] = (
        ledger["source"].astype(str) + "|" + ledger["action_family"].astype(str)
        + "|" + ledger["score_error_family"].astype(str)
        + "|near=" + ledger["has_near"].astype(int).astype(str)
    )
    support = ledger.groupby(["conditional_key", "supervision_kind"], as_index=False).agg(
        conditional_queries=("query_index", "nunique"),
        conditional_identities=("query_ik14", "nunique"),
        conditional_formulas=("query_formula", "nunique"),
        replicated_formula_folds=("formula_fold", "nunique"),
    )
    ledger = ledger.merge(support, on=["conditional_key", "supervision_kind"], validate="many_to_one")
    corrective = ledger.loc[ledger["supervision_kind"].eq("corrective")].copy()
    harmful = ledger.loc[ledger["supervision_kind"].eq("harmful")].copy()
    selected_corrective = select_diverse(corrective, args.maximum_corrective_actions, harmful=False)
    selected_harmful = select_diverse(harmful, args.maximum_harmful_actions, harmful=True)
    candidates = pd.concat([selected_corrective, selected_harmful], ignore_index=True)
    no_op = pd.DataFrame([
        base_record(graph, int(query), args.formula_fold_seed) | {
            "source": "no_op", "geometry": "clean", "action_family": "no_op",
            "action_id": f"no_op|q={int(query)}", "action_payload": "{}", "dose": 0.0,
            "step": 0, "supervision_kind": "no_op",
        }
        for query in range(graph.n_queries)
        if stable_fold(str(graph.query_formula[query]), 5, args.formula_fold_seed) != args.outer_fold
    ])
    if candidates.empty or selected_corrective.empty or selected_harmful.empty:
        raise RuntimeError("E15 failed to construct both corrective and harmful candidates")
    corrective_per_query = selected_corrective.groupby("query_index").size()
    harmful_per_query = selected_harmful.groupby("query_index").size()
    if corrective_per_query.max() > args.maximum_corrective_actions or harmful_per_query.max() > args.maximum_harmful_actions:
        raise RuntimeError("E15 per-query action cap failed")
    if selected_corrective["formula_fold"].eq(args.outer_fold).any() or selected_harmful["formula_fold"].eq(args.outer_fold).any():
        raise RuntimeError("E15 outer formula fold leaked into training candidates")
    _, exposure_report = bounded_stratified_epoch(
        exposure_items(candidates.to_dict("records")), np.random.default_rng(args.seed), maximum_exposure=1,
    )
    source_counts = ledger["source"].value_counts().astype(int).to_dict()
    gates = {
        "all_four_action_sources_present": set(source_counts) == {"R0_N", "A4_exact", "C1_support_disjoint", "E14_mature_P"},
        "a4_full_scan_loaded": int(source_counts.get("A4_exact", 0)) >= 100000,
        "c1_full_teacher_loaded": int(source_counts.get("C1_support_disjoint", 0)) >= 10000,
        "multiple_corrective_actions_exist": bool((corrective_per_query >= 2).any()),
        "harmful_controls_exist": bool(len(selected_harmful) > 0),
        "no_op_for_every_train_query": bool(len(no_op) == int(np.sum([
            stable_fold(str(value), 5, args.formula_fold_seed) != args.outer_fold
            for value in graph.query_formula
        ]))),
        "bounded_exposure_is_one": exposure_report["maximum_exposure"] == 1,
        "outer_fold_overlap_zero": True,
        "P2b_forbidden": True,
    }
    report = {
        "status": "noise_final_e15_multi_action_ledger_complete",
        "formal": True,
        "outer_formula_fold": int(args.outer_fold),
        "full_action_rows": int(len(ledger)),
        "source_rows": source_counts,
        "corrective_rows_all": int(len(corrective)),
        "harmful_rows_all": int(len(harmful)),
        "training_corrective_rows": int(len(selected_corrective)),
        "training_harmful_rows": int(len(selected_harmful)),
        "training_queries": int(candidates["query_index"].nunique()),
        "training_identities": int(candidates["query_ik14"].nunique()),
        "training_formulas": int(candidates["query_formula"].nunique()),
        "queries_with_multiple_corrective_actions": int((corrective_per_query >= 2).sum()),
        "maximum_corrective_actions_per_query": int(corrective_per_query.max()),
        "maximum_harmful_actions_per_query": int(harmful_per_query.max()),
        "no_op_rows": int(len(no_op)),
        "exposure_dry_run": exposure_report,
        "gates": gates,
        "pass_to_loss_and_sampler_smoke": bool(all(gates.values())),
        "contracts": {
            "single_action_argmax_forbidden": True,
            "multiple_actions_retained": True,
            "harmful_actions_separate": True,
            "no_op_always_available": True,
            "within_epoch_recycling": False,
            "outer_formula_fold_excluded": True,
            "P2b": "forbidden", "P3_consumed": False,
        },
        "provenance": {name: sha256_file(path) for name, path in required.items()},
        "claim_limit": "Training-only multi-action ledger; no shared-embedding gain.",
    }
    if not report["pass_to_loss_and_sampler_smoke"]:
        raise RuntimeError(f"E15 ledger gates failed: {gates}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="noise_e15_ledger_", dir=args.output_dir.parent))
    try:
        ledger.to_csv(staging / "all_action_ledger.csv.gz", index=False, compression="gzip")
        selected_corrective.to_csv(staging / "corrective_actions.csv.gz", index=False, compression="gzip")
        selected_harmful.to_csv(staging / "harmful_actions.csv.gz", index=False, compression="gzip")
        no_op.to_csv(staging / "no_op.csv.gz", index=False, compression="gzip")
        support.to_csv(staging / "conditional_support.csv.gz", index=False, compression="gzip")
        json_dump(staging / "report.json", report)
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
