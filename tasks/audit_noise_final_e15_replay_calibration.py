"""E15-M1: replay every selected ledger row against its immutable source.

This stage performs no model training and does not use P3.  It proves that the
multi-action training ledger is a faithful, executable transcription of R0,
A4, C1 and E14, then freezes source-local strength calibration and a balanced
128-observation panel for the first gradient smoke test.
"""
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
from noise_final_core import CandidateGraph, json_dump, sha256_file  # noqa: E402
from noise_final_e15_calibration import (  # noqa: E402
    calibrate_source_local, diverse_panel, inverse_source_weights,
)


SOURCES = ("R0_N", "A4_exact", "C1_support_disjoint", "E14_mature_P")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--r0-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_r0_faithful_s3a")
    parser.add_argument("--a4-dir", type=Path, default=ROOT / "data/validation/g8r_noise_v3_a4_exact_peak_scan")
    parser.add_argument("--c1-dir", type=Path, default=ROOT / "data/validation/g8r_noise_v3_c1_crossfit_teacher")
    parser.add_argument("--e14-dir", type=Path, required=True)
    parser.add_argument("--per-source-kind", type=int, default=16)
    parser.add_argument("--margin-tolerance", type=float, default=2e-6)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def strict_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1"}:
        return True
    if text in {"false", "0"}:
        return False
    raise ValueError(f"not a strict boolean: {value!r}")


def supervision(baseline_rank: int, result_rank: int) -> str:
    if baseline_rank != 1 and result_rank == 1:
        return "corrective"
    if baseline_rank == 1 and result_rank != 1:
        return "harmful"
    return "neutral"


def r0_expected(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    output = pd.DataFrame({
        "source": "R0_N", "query_index": frame["query_index"].astype(np.int64),
        "action_id": [
            f"R0|{selector}|dose={float(dose):.2f}|step={int(step)}"
            for selector, dose, step in zip(frame["selector"], frame["attenuation"], frame["step"])
        ],
        "expected_baseline_rank": frame["baseline_rank"].astype(np.int16),
        "expected_baseline_margin": frame["baseline_margin"].astype(float),
        "expected_result_rank": frame["target_rank"].astype(np.int16),
        "expected_result_margin": frame["target_margin"].astype(float),
        "expected_payload": frame["target_path"].astype(str),
    })
    return output


def a4_expected(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[
        frame["policy_eligible"].map(strict_bool) & frame["gradient_rank"].astype(int).le(50)
    ].copy()
    return pd.DataFrame({
        "source": "A4_exact", "query_index": frame["query_index"].astype(np.int64),
        "action_id": [
            f"A4|q={int(query)}|token={int(token)}|dose={float(dose):.2f}"
            for query, token, dose in zip(frame["query_index"], frame["token"], frame["attenuation"])
        ],
        "expected_baseline_rank": frame["baseline_rank"].astype(np.int16),
        "expected_baseline_margin": frame["baseline_margin"].astype(float),
        "expected_result_rank": frame["result_rank"].astype(np.int16),
        "expected_result_margin": frame["result_margin"].astype(float),
        "expected_token": frame["token"].astype(np.int16),
        "expected_dose": frame["attenuation"].astype(float),
    })


def c1_expected(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return pd.DataFrame({
        "source": "C1_support_disjoint", "query_index": frame["query_index"].astype(np.int64),
        "action_id": [
            f"C1|q={int(query)}|eval={int(evaluation)}|teachers={teachers}"
            for query, evaluation, teachers in zip(
                frame["query_index"], frame["evaluation_positive_row"], frame["teacher_rows"],
            )
        ],
        "expected_baseline_rank": frame["baseline_rank"].astype(np.int16),
        "expected_baseline_margin": frame["baseline_margin"].astype(float),
        "expected_result_rank": frame["teacher_rank"].astype(np.int16),
        "expected_result_margin": frame["teacher_margin"].astype(float),
        "expected_eval_row": frame["evaluation_positive_row"].astype(np.int64),
        "expected_teacher_rows": frame["teacher_rows"].astype(str),
    })


def e14_expected(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=True) as body:
        queries = np.asarray(body["queries"], dtype=np.int64)
        action_ids = np.asarray(body["action_ids"], dtype=str)
        clean_rank = np.asarray(body["clean_rank"], dtype=np.int16)
        clean_margin = np.asarray(body["clean_margin"], dtype=np.float64)
        result_rank = np.asarray(body["result_rank"], dtype=np.int16)
        result_margin = np.asarray(body["result_margin"], dtype=np.float64)
    if result_rank.shape != (len(queries), len(action_ids)) or result_margin.shape != result_rank.shape:
        raise RuntimeError("E14 source tensor has invalid shape")
    query_grid = np.repeat(queries, len(action_ids))
    action_grid = np.tile(action_ids, len(queries))
    base_rank_grid = np.repeat(clean_rank, len(action_ids))
    base_margin_grid = np.repeat(clean_margin, len(action_ids))
    rank_grid = result_rank.reshape(-1)
    margin_grid = result_margin.reshape(-1)
    changed = ((base_rank_grid != 1) & (rank_grid == 1)) | ((base_rank_grid == 1) & (rank_grid != 1))
    return pd.DataFrame({
        "source": "E14_mature_P", "query_index": query_grid[changed],
        "action_id": action_grid[changed],
        "expected_baseline_rank": base_rank_grid[changed],
        "expected_baseline_margin": base_margin_grid[changed],
        "expected_result_rank": rank_grid[changed],
        "expected_result_margin": margin_grid[changed],
    })


def validate_payloads(frame: pd.DataFrame, graph: CandidateGraph) -> dict[str, int]:
    counts: dict[str, int] = {}
    definitions = {item.action_id: item for item in action_definitions()}
    for source, block in frame.groupby("source", sort=True):
        for row in block.itertuples(index=False):
            if source == "R0_N":
                path = tuple(int(value) for value in str(row.action_payload).split(",") if value != "")
                if not path or len(path) != len(set(path)) or any(value <= 0 for value in path):
                    raise RuntimeError(f"invalid R0 peak path: {row.action_payload}")
            elif source == "A4_exact":
                payload = json.loads(str(row.action_payload))
                if int(payload["token"]) <= 0 or not np.isfinite(float(payload["mz"])):
                    raise RuntimeError("invalid A4 exact-token payload")
            elif source == "C1_support_disjoint":
                payload = json.loads(str(row.action_payload))
                evaluation = int(payload["evaluation_positive_row"])
                teachers = tuple(int(value) for value in str(payload["teacher_rows"]).split(";") if value)
                if not teachers or evaluation in teachers or len(teachers) != len(set(teachers)):
                    raise RuntimeError("C1 teacher is empty, duplicated, or not support-disjoint")
                _, candidate_rows, ptr, _ = graph.query_block(int(row.query_index))
                positive = set(map(int, candidate_rows[int(ptr[0]):int(ptr[1])]))
                if evaluation not in positive or not set(teachers).issubset(positive):
                    raise RuntimeError("C1 payload rows do not belong to the positive molecule")
            elif source == "E14_mature_P":
                if str(row.action_id) not in definitions:
                    raise RuntimeError(f"unknown E14 action definition: {row.action_id}")
                payload = json.loads(str(row.action_payload))
                definition = definitions[str(row.action_id)]
                if str(payload["reference_policy"]) != definition.reference_policy:
                    raise RuntimeError("E14 payload/reference policy mismatch")
            else:
                raise RuntimeError(f"unexpected action source: {source}")
        counts[str(source)] = int(len(block))
    return counts


def composition(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    return (
        frame.groupby(columns, dropna=False, sort=True).agg(
            actions=("action_id", "size"), queries=("query_index", "nunique"),
            identities=("query_ik14", "nunique"), formulas=("query_formula", "nunique"),
        ).reset_index().to_dict("records")
    )


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite E15-M1 output: {args.output_dir}")
    required = {
        "ledger_report": args.ledger_dir / "report.json",
        "corrective": args.ledger_dir / "corrective_actions.csv.gz",
        "harmful": args.ledger_dir / "harmful_actions.csv.gz",
        "graph": args.graph,
        "r0": args.r0_dir / "outcome_audit_only.csv.gz",
        "a4": args.a4_dir / "policy_candidate_actions.csv.gz",
        "c1": args.c1_dir / "crossfit_examples.csv.gz",
        "e14": args.e14_dir / "action_outcomes.npz",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    ledger_report = json.loads(required["ledger_report"].read_text(encoding="utf-8"))
    if (
        ledger_report.get("status") != "noise_final_e15_multi_action_ledger_complete"
        or not ledger_report.get("formal") or not ledger_report.get("pass_to_loss_and_sampler_smoke")
    ):
        raise RuntimeError("E15-M0 ledger is not formally authorized")
    source_hash_contract = {
        "graph": "graph", "r0": "r0_actions", "a4": "a4_actions",
        "c1": "c1_examples", "e14": "e14_outcomes",
    }
    source_hash_match = {
        source: sha256_file(required[source]) == str(
            ledger_report.get("provenance", {}).get(m0_name, "")
        )
        for source, m0_name in source_hash_contract.items()
    }
    if not all(source_hash_match.values()):
        raise RuntimeError(f"E15-M1 source artifacts drifted after M0: {source_hash_match}")
    corrective = pd.read_csv(required["corrective"])
    harmful = pd.read_csv(required["harmful"])
    selected = pd.concat([corrective, harmful], ignore_index=True)
    selected["supervision_kind"] = selected["supervision_kind"].astype(str)
    key = ["source", "query_index", "action_id"]
    if selected.empty or selected.duplicated(key).any():
        raise RuntimeError("selected E15 actions are empty or duplicated")

    expected = pd.concat([
        r0_expected(required["r0"]), a4_expected(required["a4"]),
        c1_expected(required["c1"]), e14_expected(required["e14"]),
    ], ignore_index=True)
    if expected.duplicated(key).any():
        duplicate = expected.loc[expected.duplicated(key, keep=False), key].head()
        raise RuntimeError(f"source artifacts contain duplicate action keys: {duplicate.to_dict('records')}")
    replay = selected.merge(expected, on=key, how="left", validate="one_to_one", indicator=True)
    if not replay["_merge"].eq("both").all():
        missing_rows = replay.loc[replay["_merge"].ne("both"), key].head()
        raise RuntimeError(f"selected actions missing from immutable sources: {missing_rows.to_dict('records')}")
    rank_match = (
        replay["baseline_rank"].astype(int).eq(replay["expected_baseline_rank"].astype(int))
        & replay["result_rank"].astype(int).eq(replay["expected_result_rank"].astype(int))
    )
    expected_kind = [
        supervision(int(base), int(result))
        for base, result in zip(replay["expected_baseline_rank"], replay["expected_result_rank"])
    ]
    kind_match = replay["supervision_kind"].astype(str).eq(expected_kind)
    baseline_error = np.abs(
        replay["baseline_margin"].to_numpy(float) - replay["expected_baseline_margin"].to_numpy(float)
    )
    result_error = np.abs(
        replay["result_margin"].to_numpy(float) - replay["expected_result_margin"].to_numpy(float)
    )
    if not rank_match.all() or not kind_match.all():
        raise RuntimeError(
            f"E15 source replay rank/kind mismatch: rank={int((~rank_match).sum())} "
            f"kind={int((~kind_match).sum())}"
        )
    if float(max(baseline_error.max(), result_error.max())) > args.margin_tolerance:
        raise RuntimeError(
            f"E15 source replay margin mismatch: baseline={baseline_error.max():.3g} "
            f"result={result_error.max():.3g} tolerance={args.margin_tolerance:.3g}"
        )
    graph = CandidateGraph(args.graph)
    payload_counts = validate_payloads(selected, graph)
    calibrated, calibration_table = calibrate_source_local(selected)
    panel = diverse_panel(calibrated, args.per_source_kind, args.seed)
    source_weights = inverse_source_weights(calibrated)
    strata = calibrated.groupby(["source", "supervision_kind"]).size()
    expected_strata = pd.MultiIndex.from_product([SOURCES, ("corrective", "harmful")])
    gates = {
        "all_selected_actions_replayed": bool(len(replay) == len(selected)),
        "M0_source_hashes_unchanged": bool(all(source_hash_match.values())),
        "rank_replay_exact": bool(rank_match.all()),
        "supervision_replay_exact": bool(kind_match.all()),
        "margin_replay_within_tolerance": bool(max(baseline_error.max(), result_error.max()) <= args.margin_tolerance),
        "all_payloads_executable": bool(set(payload_counts) == set(SOURCES)),
        "all_source_kind_strata_present": bool(expected_strata.isin(strata.index).all()),
        "balanced_panel_has_128_unique_actions": bool(
            len(panel) == 8 * args.per_source_kind == 128
            and not panel.duplicated(key + ["supervision_kind"]).any()
        ),
        "source_local_calibration_only": True,
        "outer_formula_fold_absent": bool(
            not calibrated["formula_fold"].astype(int).eq(int(ledger_report["outer_formula_fold"])).any()
        ),
        "P2b_forbidden": True,
        "P3_not_consumed": True,
    }
    report = {
        "status": "noise_final_e15_m1_replay_calibration_complete",
        "formal": True, "outer_formula_fold": int(ledger_report["outer_formula_fold"]),
        "selected_actions": int(len(selected)),
        "corrective_actions": int(len(corrective)), "harmful_actions": int(len(harmful)),
        "replay": {
            "rank_mismatches": int((~rank_match).sum()),
            "supervision_mismatches": int((~kind_match).sum()),
            "maximum_baseline_margin_abs_error": float(baseline_error.max()),
            "maximum_result_margin_abs_error": float(result_error.max()),
            "margin_tolerance": float(args.margin_tolerance), "payload_rows": payload_counts,
            "M0_source_hash_match": source_hash_match,
        },
        "composition_by_source_kind": composition(calibrated, ["source", "supervision_kind"]),
        "composition_by_source_family_kind": composition(
            calibrated, ["source", "action_family", "supervision_kind"],
        ),
        "calibration": {
            "groups": int(len(calibration_table)),
            "rule": "within source, supervision kind and sufficiently populated family; small families borrow only source-kind scale",
            "raw_margin_cross_geometry_comparison_forbidden": True,
        },
        "gradient_panel": {
            "actions": int(len(panel)), "per_source_kind": int(args.per_source_kind),
            "unique_queries": int(panel["query_index"].nunique()),
            "unique_identities": int(panel["query_ik14"].nunique()),
            "unique_formulas": int(panel["query_formula"].nunique()),
        },
        "sampling_contract": {
            "within_epoch_recycling": False, "maximum_action_exposure": 1,
            "source_branch_total_mass": "equal", "loss_weights": source_weights,
            "calibration_strength_used_as_outcome_label": False,
        },
        "gates": gates, "pass_to_32_query_overfit": bool(all(gates.values())),
        "contracts": {
            "source_outcomes_only_for_fidelity_and_training_calibration": True,
            "no_posthoc_action_deletion": True, "no_op_retained_in_M0": True,
            "P2b": "forbidden", "P3_consumed": False,
        },
        "provenance": {name: sha256_file(path) for name, path in required.items()},
        "claim_limit": "Source-artifact replay and training calibration only; no trained embedding gain.",
    }
    if not report["pass_to_32_query_overfit"]:
        raise RuntimeError(f"E15-M1 gates failed: {gates}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="noise_e15_m1_", dir=args.output_dir.parent))
    try:
        calibrated.loc[calibrated["supervision_kind"].eq("corrective")].to_csv(
            staging / "calibrated_corrective_actions.csv.gz", index=False, compression="gzip",
        )
        calibrated.loc[calibrated["supervision_kind"].eq("harmful")].to_csv(
            staging / "calibrated_harmful_actions.csv.gz", index=False, compression="gzip",
        )
        calibration_table.to_csv(staging / "source_local_calibration.csv", index=False)
        panel.to_csv(staging / "gradient_calibration_panel.csv.gz", index=False, compression="gzip")
        replay[key + ["supervision_kind", "baseline_rank", "result_rank"]].to_csv(
            staging / "source_replay_ledger.csv.gz", index=False, compression="gzip",
        )
        json_dump(staging / "report.json", report)
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
