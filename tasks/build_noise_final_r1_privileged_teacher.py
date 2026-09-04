"""R1: materialise the strongest locally reproducible noise teacher.

This is a *training-only* privileged teacher.  It may use identity labels and
observed action outcomes inside the training graph to select a correcting peak
intervention.  It is never a deployable policy and never an evaluation result.

The output keeps three quantities separate:
  1. fixed S3A policy effects (deployable only as an action definition);
  2. outcome-selected train-only corrective views (student supervision);
  3. the historical 3.853 pp union oracle (headroom only).

P2b and every downstream reranker are prohibited.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from noise_final_core import CandidateGraph, json_dump, sha256_file, stable_fold, strict_rank


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRAPH = ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz"
DEFAULT_R0 = ROOT / "data/validation/g8r_noise_final_r0_faithful_s3a"
DEFAULT_S3A = ROOT / "data/validation/g8r_noise_v3_s3a_extended_matrix"
DEFAULT_A4 = ROOT / "data/validation/g8r_noise_v3_a4_exact_peak_scan"
DEFAULT_OUTPUT = ROOT / "data/validation/g8r_noise_final_r1_privileged_teacher"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--r0-dir", type=Path, default=DEFAULT_R0)
    parser.add_argument("--s3a-dir", type=Path, default=DEFAULT_S3A)
    parser.add_argument("--a4-dir", type=Path, default=DEFAULT_A4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--formula-folds", type=int, default=5)
    # Keep the already frozen formula partition used by the noise programme.
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def baseline_ledger(graph: CandidateGraph, folds: int, seed: int) -> pd.DataFrame:
    rows = []
    for query in range(graph.n_queries):
        scores = graph.official_molecule_scores(query)
        rank = strict_rank(scores)
        rows.append({
            "query_index": query,
            "query_row": int(graph.query_row[query]),
            "query_ik14": str(graph.query_ik14[query]),
            "query_formula": str(graph.query_formula[query]),
            "has_near": bool(graph.query_has_near[query]),
            "baseline_rank": rank,
            "baseline_margin": float(scores[0] - np.max(scores[1:])),
            "formula_fold": stable_fold(str(graph.query_formula[query]), folds, seed),
        })
    return pd.DataFrame(rows)


def _prefix(value: object, step: int) -> str:
    tokens = [token for token in str(value).split(",") if token]
    if len(tokens) < step:
        raise RuntimeError("S3A path is shorter than its reported step")
    return ",".join(tokens[:step])


def _control_prefixes(value: object, step: int) -> str:
    paths = str(value).split(";")
    if len(paths) != 2:
        raise RuntimeError("S3A oracle requires two matched controls")
    return ";".join(_prefix(path, step) for path in paths)


def best_s3a(s3a: Path) -> pd.DataFrame:
    table = pd.read_csv(s3a / "paired_interventions.csv.gz")
    table = table.loc[
        table["baseline_rank"].astype(int).gt(1)
        & table["target_rank"].astype(int).eq(1)
    ].copy()
    # Highest corrected margin is the train-only oracle target.  Stable
    # tie-breaking prefers the conservative confounder action and fewer steps.
    table["source_priority"] = np.where(
        table["selector"].astype(str).eq("role_confounder"), 0, 1,
    )
    table = table.sort_values(
        ["query_index", "target_margin", "source_priority", "step"],
        ascending=[True, False, True, True], kind="mergesort",
    ).drop_duplicates("query_index", keep="first")
    sequences = pd.read_csv(s3a / "selected_sequences.csv.gz")
    sequences = sequences[[
        "query_index", "selector", "attenuation", "steps", "control_paths",
    ]]
    table = table.merge(
        sequences, on=["query_index", "selector", "attenuation"],
        how="left", validate="many_to_one",
    )
    if table["control_paths"].isna().any() or table["steps"].lt(table["step"]).any():
        raise RuntimeError("S3A oracle lost its dynamic trajectory controls")
    table["matched_control_paths"] = [
        _control_prefixes(path, int(step))
        for path, step in zip(table["control_paths"], table["step"])
    ]
    output = pd.DataFrame({
        "query_index": table["query_index"].astype(np.int32),
        "teacher_source": "s3a_dynamic",
        "selector": table["selector"].astype(str),
        "attenuation": table["attenuation"].astype(np.float32),
        "step": table["step"].astype(np.int8),
        "target_path": table["target_path"].astype(str),
        "matched_control_paths": table["matched_control_paths"].astype(str),
        "teacher_rank": table["target_rank"].astype(np.int16),
        "teacher_margin": table["target_margin"].astype(np.float32),
        "teacher_hard_negative_row": table["hard_negative_row"].astype(np.int64),
    })
    if len(output) != 520:
        raise RuntimeError(f"S3A recoverable count drifted: {len(output)} != 520")
    return output


def best_a4(a4: Path) -> pd.DataFrame:
    scan = pd.read_csv(a4 / "scan_queries.csv.gz")
    selected: list[dict] = []
    with h5py.File(a4 / "exact_peak_scan.h5", "r") as handle:
        doses = np.asarray(json.loads(handle.attrs["attenuations_json"]), dtype=np.float32)
        ptr = np.asarray(handle["query_action_ptr"], dtype=np.int64)
        action_query = np.asarray(handle["action_query"], dtype=np.int32)
        token = np.asarray(handle["action_token"], dtype=np.int16)
        role = np.asarray(handle["action_role"], dtype=np.int8)
        gradient_rank = np.asarray(handle["action_gradient_rank"], dtype=np.int16)
        eligible = np.asarray(handle["action_policy_eligible"], dtype=bool)
        rank = np.asarray(handle["result_rank"], dtype=np.int16).reshape(-1, len(doses))
        margin = np.asarray(handle["result_margin"], dtype=np.float32).reshape(-1, len(doses))
        adversary = np.asarray(
            handle["result_adversarial_pair_row"], dtype=np.int64,
        ).reshape(-1, len(doses))
    for position, item in scan.iterrows():
        if str(item["scan_kind"]) != "official_error":
            continue
        left, right = int(ptr[position]), int(ptr[position + 1])
        candidates: list[tuple] = []
        for action in range(left, right):
            if not eligible[action]:
                continue
            for dose_index, dose in enumerate(doses):
                if int(rank[action, dose_index]) != 1:
                    continue
                candidates.append((
                    -float(margin[action, dose_index]), float(dose),
                    int(gradient_rank[action]), int(token[action]), action, dose_index,
                ))
        if not candidates:
            continue
        _, dose, grad_rank, peak_token, action, dose_index = min(candidates)
        if int(action_query[action]) != int(position):
            raise RuntimeError("A4 action/query alignment failed")
        selected.append({
            "query_index": int(item["query_index"]),
            "teacher_source": "a4_exact",
            "selector": f"a4_role_{int(role[action])}_gradient_rank_{grad_rank}",
            "attenuation": float(dose),
            "step": 1,
            "target_path": str(peak_token),
            "matched_control_paths": "",
            "teacher_rank": 1,
            "teacher_margin": float(margin[action, dose_index]),
            "teacher_hard_negative_row": int(adversary[action, dose_index]),
        })
    output = pd.DataFrame(selected)
    if len(output) != 776 or output["query_index"].nunique() != 776:
        raise RuntimeError(f"A4 eligible recoverable count drifted: {len(output)} != 776")
    return output


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite R1: {args.output_dir}")
    required = [
        args.graph, args.r0_dir / "report.json", args.r0_dir / "outcome_audit_only.csv.gz",
        args.s3a_dir / "decision.json", args.s3a_dir / "selected_sequences.csv.gz",
        args.s3a_dir / "paired_interventions.csv.gz",
        args.a4_dir / "decision.json", args.a4_dir / "scan_queries.csv.gz",
        args.a4_dir / "exact_peak_scan.h5",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    r0_report = json.loads((args.r0_dir / "report.json").read_text(encoding="utf-8"))
    a4_decision = json.loads((args.a4_dir / "decision.json").read_text(encoding="utf-8"))
    if not r0_report.get("formal") or r0_report["contracts"].get("P2b") != "forbidden":
        raise RuntimeError("R1 requires formal P2b-free R0")
    if a4_decision.get("status") != "noise_v3_a4_exact_peak_scan_decision":
        raise RuntimeError("R1 requires the formal A4 decision")

    graph = CandidateGraph(args.graph)
    ledger = baseline_ledger(graph, args.formula_folds, args.seed)
    if len(ledger) != 23876 or int(ledger["baseline_rank"].gt(1).sum()) != 1805:
        raise RuntimeError("official baseline ledger does not reproduce 23,876 / 1,805")
    s3a = best_s3a(args.s3a_dir)
    a4 = best_a4(args.a4_dir)
    actions = pd.concat((s3a, a4), ignore_index=True)
    # Cross-source oracle: choose the corrected view with the largest positive
    # margin.  This is legal only as train-fold supervision.
    actions = actions.sort_values(
        ["query_index", "teacher_margin", "teacher_source"],
        ascending=[True, False, True], kind="mergesort",
    ).drop_duplicates("query_index", keep="first")
    if len(actions) != 882:
        raise RuntimeError(f"locally materialised S3A+A4 union drifted: {len(actions)} != 882")
    actions = actions.merge(
        ledger, on="query_index", how="left", validate="one_to_one",
    )
    if actions["baseline_rank"].le(1).any() or actions["teacher_rank"].ne(1).any():
        raise RuntimeError("privileged corrective table contains a non-correction")

    # Conservative robustness views: deepest role-confounder trajectory that
    # remains correct.  These are kept separate from corrective-oracle views.
    audit = pd.read_csv(args.r0_dir / "outcome_audit_only.csv.gz")
    safety = audit.loc[
        audit["selector"].astype(str).eq("role_confounder")
        & audit["baseline_rank"].astype(int).eq(1)
        & audit["target_rank"].astype(int).eq(1)
    ].sort_values(
        ["query_index", "step"], ascending=[True, False], kind="mergesort",
    ).drop_duplicates("query_index", keep="first")
    if "formula_fold" in safety.columns:
        safety = safety.drop(columns=["formula_fold"])
    safety = safety.merge(
        ledger[["query_index", "formula_fold"]], on="query_index", how="left",
        validate="one_to_one",
    )
    safety = safety[[
        "query_index", "query_row", "query_ik14", "query_formula", "has_near",
        "formula_fold", "selector", "attenuation", "step", "target_path",
        "matched_control_paths", "hard_negative_row",
    ]].rename(columns={"hard_negative_row": "teacher_hard_negative_row"})

    ledger["privileged_teacher_available"] = ledger["query_index"].isin(
        set(actions["query_index"].astype(int))
    )
    ledger["unrecovered_official_error"] = (
        ledger["baseline_rank"].gt(1) & ~ledger["privileged_teacher_available"]
    )

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="noise_r1_", dir=args.output_dir.parent))
    try:
        actions.to_csv(temporary / "corrective_teacher_actions.csv.gz", index=False, compression="gzip")
        safety.to_csv(temporary / "robustness_teacher_actions.csv.gz", index=False, compression="gzip")
        ledger.to_csv(temporary / "query_ledger.csv.gz", index=False, compression="gzip")
        body = {
            "status": "noise_final_r1_privileged_teacher_complete",
            "formal": True,
            "queries": int(len(ledger)),
            "official_errors": int(ledger["baseline_rank"].gt(1).sum()),
            "s3a_recoverable": int(len(s3a)),
            "a4_eligible_recoverable": int(len(a4)),
            "locally_materialised_union_recoverable": int(len(actions)),
            "locally_materialised_union_upper_bound_delta": float(len(actions) / len(ledger)),
            "historical_full_union_recoverable": 920,
            "historical_full_union_upper_bound_delta": float(920 / len(ledger)),
            "historical_trajectories_not_local": int(920 - len(actions)),
            "robustness_views": int(len(safety)),
            "corrective_identities": int(actions["query_ik14"].nunique()),
            "corrective_formulas": int(actions["query_formula"].nunique()),
            "contracts": {
                "teacher_is_training_only": True,
                "teacher_uses_outcomes": True,
                "evaluation_may_not_use_teacher_action_or_outcome": True,
                "shared_student_inference": "clean spectrum only",
                "P2b": "forbidden",
            },
            "provenance": {
                "graph_sha256": sha256_file(args.graph),
                "r0_report_sha256": sha256_file(args.r0_dir / "report.json"),
                "r0_actions_sha256": sha256_file(args.r0_dir / "outcome_audit_only.csv.gz"),
                "s3a_decision_sha256": sha256_file(args.s3a_dir / "decision.json"),
                "s3a_paired_sha256": sha256_file(args.s3a_dir / "paired_interventions.csv.gz"),
                "a4_decision_sha256": sha256_file(args.a4_dir / "decision.json"),
                "a4_scan_sha256": sha256_file(args.a4_dir / "exact_peak_scan.h5"),
                "script_sha256": sha256_file(Path(__file__)),
            },
            "claim_limit": (
                "3.853 pp is an outcome-selected training-graph upper bound. "
                "Only held-out clean shared-encoder retrieval is model performance."
            ),
            "next_stage": (
                "R2 shared encoder learns clean/action groupwise ranking, "
                "noisy-to-clean robustness and protected-correct preservation"
            ),
        }
        json_dump(temporary / "report.json", body)
        temporary.replace(args.output_dir)
        print(json.dumps(body, indent=2), flush=True)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
