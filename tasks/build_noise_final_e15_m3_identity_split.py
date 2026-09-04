"""Freeze the E15-M3 identity-held transfer split; do not train weights.

The split is constructed in the mature E4-A initialization geometry.  Held
identities are absent from every training query and action-teacher reference.
They are also written to an exclusion ledger that the M3 trainer must apply to
all trainable candidate references.  P3 and P2b are forbidden.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from noise_final_core import CandidateGraph, json_dump, seed_everything, sha256_file  # noqa: E402
from train_e1_identity import load_base_model, torch_load_compat  # noqa: E402
from train_noise_final_r2_shared_encoder import SpectrumStore, encode_rows  # noqa: E402
from train_noise_final_e15_m2_overfit import (  # noqa: E402
    SOURCES, evaluate_queries, parse_ints,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m2-run-dir", type=Path, required=True)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--initial-student-checkpoint", type=Path, required=True)
    parser.add_argument("--held-errors-per-source", type=int, default=32)
    parser.add_argument("--held-correct", type=int, default=128)
    parser.add_argument("--sentinel-identities", type=int, default=256)
    parser.add_argument("--outer-fold", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def stable_order(value: str, seed: int, salt: str) -> int:
    payload = f"{seed}|{salt}|{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def query_table(frame: pd.DataFrame, ranks: np.ndarray, graph: CandidateGraph) -> pd.DataFrame:
    output = (
        frame.groupby(["source", "query_index", "query_ik14", "query_formula"], as_index=False)
        .agg(
            actions=("action_id", "nunique"),
            strength=("source_kind_percentile", "max"),
        )
    )
    query = output["query_index"].to_numpy(np.int64)
    output["initial_rank"] = ranks[query]
    output["has_near"] = graph.query_has_near[query]
    return output


def held_errors(
    frame: pd.DataFrame, ranks: np.ndarray, graph: CandidateGraph,
    per_source: int, seed: int,
) -> pd.DataFrame:
    table = query_table(frame, ranks, graph)
    table = table.loc[table["initial_rank"].ne(1)].copy()
    selected: list[pd.DataFrame] = []
    used: set[str] = set()
    availability = table.groupby("source")["query_ik14"].nunique().sort_values()
    if set(availability.index.astype(str)) != set(SOURCES):
        raise RuntimeError("held-error pool does not contain all E15 sources")
    for source in availability.index.astype(str):
        block = table.loc[table["source"].astype(str).eq(source)].copy()
        block["stable_order"] = [
            stable_order(f"{identity}|{query}", seed, f"held-error|{source}")
            for identity, query in zip(block["query_ik14"].astype(str), block["query_index"].astype(int))
        ]
        block = (
            block.sort_values(["stable_order", "query_index"], kind="stable")
            .drop_duplicates("query_ik14", keep="first")
        )
        block = block.loc[~block["query_ik14"].astype(str).isin(used)].head(per_source)
        if len(block) != per_source:
            raise RuntimeError(
                f"source {source} has {len(block)} identity-disjoint held errors; need {per_source}"
            )
        used.update(block["query_ik14"].astype(str))
        selected.append(block)
    output = pd.concat(selected, ignore_index=True)
    output["held_kind"] = "error"
    return output


def held_correct(
    frame: pd.DataFrame, ranks: np.ndarray, graph: CandidateGraph,
    excluded_identities: set[str], total: int, seed: int,
) -> pd.DataFrame:
    table = query_table(frame, ranks, graph)
    table = table.loc[
        table["initial_rank"].eq(1)
        & ~table["query_ik14"].astype(str).isin(excluded_identities)
    ].copy()
    queues: dict[str, list[pd.Series]] = {}
    for source in SOURCES:
        block = table.loc[table["source"].astype(str).eq(source)].copy()
        block["stable_order"] = [
            stable_order(f"{identity}|{query}", seed, f"held-correct|{source}")
            for identity, query in zip(block["query_ik14"].astype(str), block["query_index"].astype(int))
        ]
        block = block.sort_values(["stable_order", "query_index"], kind="stable")
        queues[source] = [row for _, row in block.iterrows()]
    cursor = {source: 0 for source in SOURCES}
    used = set(excluded_identities)
    rows: list[pd.Series] = []
    while len(rows) < total:
        progressed = False
        for source in SOURCES:
            values = queues[source]
            while cursor[source] < len(values):
                row = values[cursor[source]]; cursor[source] += 1
                identity = str(row["query_ik14"])
                if identity in used:
                    continue
                rows.append(row); used.add(identity); progressed = True
                break
            if len(rows) == total:
                break
        if not progressed:
            raise RuntimeError(f"only {len(rows)} identity-disjoint held-correct queries; need {total}")
    output = pd.DataFrame(rows).reset_index(drop=True)
    output["held_kind"] = "correct"
    return output


def decode(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)


def row_identities(data: Path, rows: np.ndarray) -> dict[int, str]:
    unique = np.asarray(sorted(set(map(int, rows))), dtype=np.int64)
    with h5py.File(data, "r") as handle:
        values = handle["INCHIKEY"][unique]
    return {int(row): decode(value)[:14] for row, value in zip(unique, values)}


def teacher_rows(row: pd.Series) -> tuple[int, ...]:
    source = str(row["source"])
    if source == "C1_support_disjoint":
        payload = json.loads(str(row["action_payload"]))
        return parse_ints(payload["teacher_rows"], ";")
    if source == "E14_mature_P":
        return parse_ints(row["positive_reference_rows"], ";")
    return ()


def choose_sentinels(
    graph: CandidateGraph, excluded: set[str], count: int, seed: int,
) -> pd.DataFrame:
    frame = pd.DataFrame({
        "query_index": np.arange(graph.n_queries, dtype=np.int64),
        "query_row": graph.query_row,
        "query_ik14": graph.query_ik14,
        "query_formula": graph.query_formula,
        "has_near": graph.query_has_near,
    })
    frame = frame.loc[~frame["query_ik14"].astype(str).isin(excluded)].copy()
    frame["stable_order"] = [
        stable_order(f"{identity}|{query}", seed, "sentinel")
        for identity, query in zip(frame["query_ik14"].astype(str), frame["query_index"].astype(int))
    ]
    frame = (
        frame.sort_values(["stable_order", "query_index"], kind="stable")
        .drop_duplicates("query_ik14", keep="first")
        .head(count)
    )
    if len(frame) != count:
        raise RuntimeError(f"only {len(frame)} untouched sentinel identities; need {count}")
    return frame.reset_index(drop=True)


def main() -> None:
    args = arguments()
    seed_everything(args.seed)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite E15-M3 split: {args.output_dir}")
    required = {
        "m2_panel_report": args.m2_run_dir / "panel/report.json",
        "corrective": args.m2_run_dir / "panel/executable_corrective.csv.gz",
        "harmful": args.m2_run_dir / "panel/executable_harmful.csv.gz",
        "m2_result_report": args.m2_run_dir / "result/report.json",
        "m2_capacity_checkpoint": args.m2_run_dir / "result/shared_encoder.pt",
        "graph": args.graph,
        "data": args.data,
        "official_checkpoint": args.official_checkpoint,
        "architecture_checkpoint": args.architecture_checkpoint,
        "initial_student_checkpoint": args.initial_student_checkpoint,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    panel_report = json.loads(required["m2_panel_report"].read_text(encoding="utf-8"))
    result_report = json.loads(required["m2_result_report"].read_text(encoding="utf-8"))
    if (
        not panel_report.get("pass_to_shared_encoder_overfit")
        or not result_report.get("pass_to_identity_holdout")
        or int(result_report.get("outer_formula_fold", -1)) != args.outer_fold
    ):
        raise RuntimeError("M3 requires the passing immutable M2 capacity run")
    corrective = pd.read_csv(required["corrective"], low_memory=False)
    harmful = pd.read_csv(required["harmful"], low_memory=False)

    graph = CandidateGraph(args.graph)
    reachable = np.unique(np.concatenate([graph.query_row, graph.pair_candidate_row])).astype(np.int64)
    store = SpectrumStore(args.data, reachable, args.n_highest_peaks)
    device = torch.device(args.device)
    model, _ = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks,
    )
    package = torch_load_compat(args.initial_student_checkpoint, map_location="cpu")
    m2_provenance = result_report.get("provenance", {})
    expected_provenance = {
        "initial_student_checkpoint": sha256_file(args.initial_student_checkpoint),
        "official_checkpoint": sha256_file(args.official_checkpoint),
        "architecture_checkpoint": sha256_file(args.architecture_checkpoint),
    }
    provenance_mismatches = {
        name: {"expected": expected, "observed": m2_provenance.get(name)}
        for name, expected in expected_provenance.items()
        if m2_provenance.get(name) != expected
    }
    if (
        package.get("status") != "noise_final_e4a_direct_shared_dreams_encoder"
        or int(package.get("outer_fold", -1)) != args.outer_fold
        or package.get("P2b_used") or not package.get("inference_clean_only")
        or provenance_mismatches
    ):
        raise RuntimeError(
            "M3 initialization does not match the exact mature clean-only E4-A encoder "
            f"used by passing M2: status={package.get('status')!r}, "
            f"fold={package.get('outer_fold')!r}, clean_only={package.get('inference_clean_only')!r}, "
            f"P2b={package.get('P2b_used')!r}, provenance_mismatches={provenance_mismatches}"
        )
    model.load_state_dict(package["model_state"], strict=True); model.eval()
    initial = encode_rows(model, store, store.rows, device, args.eval_batch_size, False, "E15-M3-split")
    index = {int(row): position for position, row in enumerate(store.rows)}
    ranks, margins = evaluate_queries(
        graph, np.arange(graph.n_queries, dtype=np.int64), initial, index,
    )

    errors = held_errors(corrective, ranks, graph, args.held_errors_per_source, args.seed)
    correct = held_correct(
        harmful, ranks, graph, set(errors["query_ik14"].astype(str)),
        args.held_correct, args.seed,
    )
    held = pd.concat([errors, correct], ignore_index=True)
    held_identities = set(held["query_ik14"].astype(str))
    if held["query_ik14"].duplicated().any() or held["query_index"].duplicated().any():
        raise RuntimeError("M3 held panel repeats an identity or query")

    train_corrective = corrective.loc[
        ~corrective["query_ik14"].astype(str).isin(held_identities)
    ].copy()
    train_harmful = harmful.loc[
        ~harmful["query_ik14"].astype(str).isin(held_identities)
    ].copy()
    train_identities = set(train_corrective["query_ik14"].astype(str)) | set(
        train_harmful["query_ik14"].astype(str)
    )
    sentinels = choose_sentinels(
        graph, held_identities | train_identities, args.sentinel_identities, args.seed,
    )
    sentinel_identities = set(sentinels["query_ik14"].astype(str))
    excluded_reference_identities = held_identities | sentinel_identities

    reference_records: list[tuple[int, str, int]] = []
    query_records = list(zip(
        train_corrective.index.astype(int),
        train_corrective["query_ik14"].astype(str),
        train_corrective["query_row"].astype(int),
    ))
    query_records.extend(zip(
        train_harmful.index.astype(int),
        train_harmful["query_ik14"].astype(str),
        train_harmful["query_row"].astype(int),
    ))
    referenced_rows: list[int] = [record[2] for record in query_records]
    for local, row in train_corrective.iterrows():
        for reference in teacher_rows(row):
            reference_records.append((int(local), str(row["query_ik14"]), int(reference)))
            referenced_rows.append(int(reference))
    identities_by_row = row_identities(args.data, np.asarray(referenced_rows, dtype=np.int64))
    bad_queries = [
        {"row_index": local, "query_ik14": identity, "query_row": query_row,
         "hdf5_ik14": identities_by_row[query_row]}
        for local, identity, query_row in query_records
        if identities_by_row[query_row] != identity
    ]
    if bad_queries:
        raise RuntimeError(f"ledger query row crosses identity: {bad_queries[:5]}")
    bad_references = [
        {"row_index": local, "query_ik14": identity, "reference_row": reference,
         "reference_ik14": identities_by_row[reference]}
        for local, identity, reference in reference_records
        if identities_by_row[reference] != identity
    ]
    if bad_references:
        raise RuntimeError(f"action teacher reference crosses identity: {bad_references[:5]}")
    if any(identities_by_row[reference] in excluded_reference_identities for _, _, reference in reference_records):
        raise RuntimeError("held/sentinel identity leaked through an action teacher reference")

    held["initial_margin"] = margins[held["query_index"].to_numpy(np.int64)]
    train_counts = {
        "corrective_actions": int(len(train_corrective)),
        "corrective_queries": int(train_corrective["query_index"].nunique()),
        "corrective_identities": int(train_corrective["query_ik14"].nunique()),
        "corrective_formulas": int(train_corrective["query_formula"].nunique()),
        "harmful_actions": int(len(train_harmful)),
        "harmful_queries": int(train_harmful["query_index"].nunique()),
        "harmful_identities": int(train_harmful["query_ik14"].nunique()),
        "harmful_formulas": int(train_harmful["query_formula"].nunique()),
    }
    source_capacity: dict[str, dict[str, int | float]] = {}
    for source in SOURCES:
        before_corrective = corrective.loc[corrective["source"].astype(str).eq(source)]
        after_corrective = train_corrective.loc[train_corrective["source"].astype(str).eq(source)]
        before_harmful = harmful.loc[harmful["source"].astype(str).eq(source)]
        after_harmful = train_harmful.loc[train_harmful["source"].astype(str).eq(source)]
        source_capacity[source] = {
            "corrective_actions": int(len(after_corrective)),
            "corrective_queries": int(after_corrective["query_index"].nunique()),
            "corrective_identities": int(after_corrective["query_ik14"].nunique()),
            "corrective_action_retention": float(len(after_corrective) / max(1, len(before_corrective))),
            "harmful_actions": int(len(after_harmful)),
            "harmful_queries": int(after_harmful["query_index"].nunique()),
            "harmful_identities": int(after_harmful["query_ik14"].nunique()),
            "harmful_action_retention": float(len(after_harmful) / max(1, len(before_harmful))),
        }
    held_source_counts = held.groupby(["held_kind", "source"]).size().astype(int).to_dict()
    gates = {
        "held_queries_eq_256": len(held) == 4 * args.held_errors_per_source + args.held_correct == 256,
        "held_identities_eq_queries": held["query_ik14"].nunique() == len(held),
        "held_errors_eq_128": int((held["held_kind"] == "error").sum()) == 128,
        "held_correct_eq_128": int((held["held_kind"] == "correct").sum()) == 128,
        "held_error_sources_balanced": bool(all(
            held_source_counts.get(("error", source), 0) == args.held_errors_per_source
            for source in SOURCES
        )),
        "train_held_identity_overlap_zero": not bool(train_identities & held_identities),
        "sentinel_disjoint": not bool(sentinel_identities & (train_identities | held_identities)),
        "ledger_query_row_identity_exact": not bad_queries,
        "action_teacher_reference_identity_exact": not bad_references,
        "all_corrective_sources_remain_nonempty": bool(all(
            source_capacity[source]["corrective_actions"] > 0
            and source_capacity[source]["corrective_queries"] > 0
            and source_capacity[source]["corrective_identities"] > 0
            for source in SOURCES
        )),
        "harmful_training_pool_nonempty": bool(
            train_counts["harmful_actions"] > 0
            and train_counts["harmful_queries"] > 0
            and train_counts["harmful_identities"] > 0
        ),
        "P2b_forbidden": True,
        "P3_not_consumed": True,
    }
    report = {
        "status": "noise_final_e15_m3_identity_split_complete",
        "formal": True,
        "outer_formula_fold": args.outer_fold,
        "held": {
            "queries": int(len(held)), "identities": int(len(held_identities)),
            "formulas": int(held["query_formula"].nunique()),
            "errors": int((held["held_kind"] == "error").sum()),
            "correct": int((held["held_kind"] == "correct").sum()),
            "near": int(held["has_near"].sum()),
            "source_counts": {f"{kind}|{source}": int(value)
                              for (kind, source), value in held_source_counts.items()},
        },
        "training_capacity": train_counts,
        "training_capacity_by_source": source_capacity,
        "sentinel_identities": int(len(sentinel_identities)),
        "action_teacher_references_checked": int(len(reference_records)),
        "ledger_query_rows_checked": int(len(query_records)),
        "candidate_reference_exclusion_contract": {
            "excluded_identities": int(len(excluded_reference_identities)),
            "rule": "every trainable candidate reference whose IK14 is held or sentinel must be removed",
        },
        "gates": gates,
        "pass_to_identity_holdout_training": bool(all(gates.values())),
        "contracts": {
            "initialization": "mature E4-A, never M2 overfit checkpoint",
            "held_identity_absent_from_training_queries": True,
            "held_identity_absent_from_action_teacher_references": True,
            "held_and_sentinel_identity_must_be_filtered_from_trainable_candidate_references": True,
            "held_panel_not_used_for_checkpoint_selection": True,
            "P2b": "forbidden", "P3_consumed": False,
        },
        "provenance": {name: sha256_file(path) for name, path in required.items()},
        "claim_limit": "Frozen identity-held split and capacity audit; no M3 model has been trained.",
    }
    if not report["pass_to_identity_holdout_training"]:
        raise RuntimeError(f"E15-M3 split gates failed: {gates}")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="noise_e15_m3_split_", dir=args.output_dir.parent))
    try:
        held.to_csv(staging / "held_queries.csv.gz", index=False, compression="gzip")
        train_corrective.to_csv(staging / "train_corrective.csv.gz", index=False, compression="gzip")
        train_harmful.to_csv(staging / "train_harmful.csv.gz", index=False, compression="gzip")
        sentinels.to_csv(staging / "sentinel_queries.csv.gz", index=False, compression="gzip")
        (staging / "excluded_reference_identities.txt").write_text(
            "\n".join(sorted(excluded_reference_identities)) + "\n", encoding="utf-8",
        )
        json_dump(staging / "report.json", report)
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
