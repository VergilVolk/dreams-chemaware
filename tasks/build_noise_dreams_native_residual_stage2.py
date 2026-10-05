"""Build a champion-current Noise residual curriculum without held outcomes.

The Stage-1 hard-positive checkpoint is frozen before this builder runs.  Only
outer-train formulas are inspected.  For each action query, one representable
action is retained when it is still native-hinge active and its exact current
boundary is better than both the clean query and its registered same-query
control.  The same query also receives its current clean max-positive versus
max-negative boundary whenever that boundary is active.  A small set of
currently correct, active, action-free clean queries acts as protection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from audit_noise_dreams_native_official_replay import encode_actions, load_npz
from build_noise_dreams_native_triplets import Registry, fixed_unicode, stable_fold
from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA
from e1_checkpoint_io import official_backbone_state, official_head_state, torch_load_compat
from evaluate_noise_dreams_native import encode_rows
from noise_final_core import CandidateGraph
from train_e1_identity import load_base_model


STAGE2_BUILDER_VERSION = "noise_champion_current_query_residual_v1"


def assert_exact_checkpoint_reconstruction(model, checkpoint: Path) -> None:
    package = torch_load_compat(checkpoint, map_location="cpu")
    panels = (
        ("backbone", model.backbone.state_dict(), official_backbone_state(package)),
        ("head", model.head.state_dict(), official_head_state(package)),
    )
    for label, observed, expected in panels:
        if set(observed) != set(expected):
            raise RuntimeError(f"Stage-2 geometry changed {label} state keys")
        for key in observed:
            if not torch.equal(observed[key].detach().cpu(), expected[key].cpu()):
                raise RuntimeError(
                    f"Stage-2 geometry changed champion tensor: {label}.{key}"
                )
    del package


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-run", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--architecture-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--minimum-action-advantage", type=float, default=5e-6)
    parser.add_argument("--protection-sentinels", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def graph_boundary(
    graph: CandidateGraph,
    query: int,
    embedding_rows: dict[int, int],
    embeddings: np.ndarray,
    anchor: np.ndarray,
) -> dict[str, object]:
    pair_slice, rows, pointers, _ = graph.query_block(query)
    del pair_slice
    positions = np.asarray([embedding_rows[int(row)] for row in rows], dtype=np.int64)
    scores = embeddings[positions] @ anchor
    positive_end = int(pointers[1])
    positive_local = int(np.argmax(scores[:positive_end]))
    negative_local = positive_end + int(np.argmax(scores[positive_end:]))
    return {
        "positive_row": int(rows[positive_local]),
        "negative_row": int(rows[negative_local]),
        "positive_score": float(scores[positive_local]),
        "negative_score": float(scores[negative_local]),
        "margin": float(scores[positive_local] - scores[negative_local]),
        "positive_embedding": embeddings[positions[positive_local]],
        "negative_embedding": embeddings[positions[negative_local]],
    }


def make_pool(
    selected: pd.DataFrame,
    sentinels: pd.DataFrame,
) -> dict[str, np.ndarray]:
    registry = Registry()
    anchors: list[int] = []
    positives: list[int] = []
    negatives: list[int] = []
    positive_ptr = [0]
    negative_ptr = [0]
    event_kind: list[int] = []
    event_query: list[int] = []
    event_action: list[int] = []
    event_formula: list[str] = []

    def append(
        anchor: int, positive: int, negative: int, *, kind: int,
        query: int, action: int, formula: str,
    ) -> None:
        if len({anchor, positive, negative}) != 3:
            raise RuntimeError("Stage-2 triplet roles overlap")
        anchors.append(anchor)
        positives.append(positive)
        negatives.append(negative)
        positive_ptr.append(len(positives))
        negative_ptr.append(len(negatives))
        event_kind.append(kind)
        event_query.append(query)
        event_action.append(action)
        event_formula.append(formula)

    for action, row in selected.reset_index(drop=True).iterrows():
        query = int(row.query_index)
        formula = str(row.query_formula)
        append(
            registry.add(Registry.ACTION, action),
            registry.add(Registry.HDF5, int(row.action_positive_row)),
            registry.add(Registry.HDF5, int(row.action_negative_row)),
            kind=2, query=query, action=action, formula=formula,
        )
        if bool(row.clean_boundary_active):
            append(
                registry.add(Registry.HDF5, int(row.query_row)),
                registry.add(Registry.HDF5, int(row.clean_positive_row)),
                registry.add(Registry.HDF5, int(row.clean_negative_row)),
                kind=0, query=query, action=-1, formula=formula,
            )
    for row in sentinels.itertuples(index=False):
        append(
            registry.add(Registry.HDF5, int(row.query_row)),
            registry.add(Registry.HDF5, int(row.clean_positive_row)),
            registry.add(Registry.HDF5, int(row.clean_negative_row)),
            kind=0, query=int(row.query_index), action=-1,
            formula=str(row.query_formula),
        )
    registry_kind, registry_source_index = registry.arrays()
    return {
        "registry_kind": registry_kind,
        "registry_source_index": registry_source_index,
        "anchor_idx": np.asarray(anchors, dtype=np.int64),
        "positive_ptr": np.asarray(positive_ptr, dtype=np.int64),
        "positive_idx": np.asarray(positives, dtype=np.int64),
        "negative_ptr": np.asarray(negative_ptr, dtype=np.int64),
        "negative_idx": np.asarray(negatives, dtype=np.int64),
        "event_kind": np.asarray(event_kind, dtype=np.int8),
        "event_query": np.asarray(event_query, dtype=np.int64),
        "event_action_index": np.asarray(event_action, dtype=np.int64),
        "event_formula": fixed_unicode(event_formula),
    }


def main() -> None:
    args = arguments()
    if (
        args.margin <= 0
        or args.minimum_action_advantage <= 0
        or args.protection_sentinels < 4
        or args.batch_size < 1
    ):
        raise ValueError("invalid Stage-2 residual selection settings")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage-2 residual construction requires an allocated GPU")
    stage1_triplets = args.stage1_run / "triplets"
    checkpoint = args.stage1_run / "checkpoint/targeted_final.ckpt"
    summary_path = args.stage1_run / "summary/report.json"
    evaluation_path = args.stage1_run / "evaluation/targeted/report.json"
    required = (
        checkpoint, summary_path, evaluation_path,
        stage1_triplets / "train_pool.npz",
        stage1_triplets / "validation_pool.npz",
        stage1_triplets / "action_spectra.npz",
        stage1_triplets / "selected_actions.csv.gz",
        stage1_triplets / "report.json",
        args.graph, args.embedding_cache, args.data, args.architecture_checkpoint,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("targeted_vs_control_recall1_pp", 0) <= 0
        or summary.get("targeted_vs_control_formula_cluster_ci", {}).get(
            "ci_low_pp", 0
        ) <= 0
        or not all(summary.get("targeted_directional_metric_checks", {}).values())
    ):
        raise RuntimeError("Stage-1 champion lacks the registered positive evidence")
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    if evaluation.get("provenance", {}).get("checkpoint_sha256") != sha256_file(checkpoint):
        raise RuntimeError("Stage-1 evaluated checkpoint differs from warm start")

    graph = CandidateGraph(args.graph)
    with np.load(args.embedding_cache, allow_pickle=False) as cache:
        rows = np.asarray(cache["rows"], dtype=np.int64)
    row_position = {int(row): index for index, row in enumerate(rows)}
    actions = pd.read_csv(stage1_triplets / "selected_actions.csv.gz", low_memory=False)
    bank = load_npz(stage1_triplets / "action_spectra.npz")
    targeted_bank = np.asarray(bank["targeted_action_spectra"], dtype=np.float32)
    control_bank = np.asarray(bank["control_action_spectra"], dtype=np.float32)
    capable = np.asarray(bank["native_action_view_representable"], dtype=bool)
    if len(actions) != len(targeted_bank) or capable.shape != (len(actions),):
        raise RuntimeError("Stage-1 action artifacts are not aligned")

    model, initialization_kind = load_base_model(
        checkpoint, args.architecture_checkpoint, torch.device("cuda"), 100,
    )
    assert_exact_checkpoint_reconstruction(model, checkpoint)
    preprocessor = SpectrumPreprocessor(
        DataFormatA(), prec_intens=1.1, n_highest_peaks=100,
        spec_entropy_cleaning=False, precision=32,
        mz_shift_aug_p=0, mz_shift_aug_max=0,
    )
    embeddings = encode_rows(
        model, rows, args.data, preprocessor,
        batch_size=args.batch_size, device=torch.device("cuda"),
        label="noise-stage1-residual-measured",
    )
    target_embedding = encode_actions(
        model, targeted_bank[capable], batch_size=args.batch_size,
        device=torch.device("cuda"), label="noise-stage1-targeted-actions",
        preprocessor=preprocessor,
    )
    control_embedding = encode_actions(
        model, control_bank[capable], batch_size=args.batch_size,
        device=torch.device("cuda"), label="noise-stage1-control-actions",
        preprocessor=preprocessor,
    )
    del model
    capable_indices = np.flatnonzero(capable)
    target_by_action = {
        int(action): target_embedding[index]
        for index, action in enumerate(capable_indices)
    }
    control_by_action = {
        int(action): control_embedding[index]
        for index, action in enumerate(capable_indices)
    }
    by_query: dict[int, list[int]] = defaultdict(list)
    for action in capable_indices:
        by_query[int(actions.at[int(action), "query_index"])].append(int(action))

    selected_rows: list[dict[str, object]] = []
    for query in sorted(by_query):
        if stable_fold(str(graph.query_formula[query]), 5, args.formula_fold_seed) == args.outer_fold:
            raise RuntimeError("outer-held formula reached Stage-2 residual construction")
        clean_vector = embeddings[row_position[int(graph.query_row[query])]]
        clean = graph_boundary(graph, query, row_position, embeddings, clean_vector)
        candidates = []
        pair_slice, candidate_rows, _, _ = graph.query_block(query)
        del pair_slice
        pair_embeddings = embeddings[[row_position[int(row)] for row in candidate_rows]]
        for action in by_query[query]:
            targeted_vector = target_by_action[action]
            targeted_scores = pair_embeddings @ targeted_vector
            _, _, pointers, _ = graph.query_block(query)
            p_end = int(pointers[1])
            p_local = int(np.argmax(targeted_scores[:p_end]))
            n_local = p_end + int(np.argmax(targeted_scores[p_end:]))
            p_embedding = pair_embeddings[p_local]
            n_embedding = pair_embeddings[n_local]
            target_margin = float(targeted_vector @ p_embedding - targeted_vector @ n_embedding)
            control_vector = control_by_action[action]
            control_margin = float(control_vector @ p_embedding - control_vector @ n_embedding)
            clean_same_margin = float(clean_vector @ p_embedding - clean_vector @ n_embedding)
            control_advantage = target_margin - control_margin
            clean_advantage = target_margin - clean_same_margin
            if (
                target_margin < args.margin
                and control_advantage >= args.minimum_action_advantage
                and clean_advantage >= args.minimum_action_advantage
            ):
                candidates.append((
                    min(control_advantage, clean_advantage),
                    control_advantage + clean_advantage,
                    str(actions.at[action, "action_id"]), action,
                    int(candidate_rows[p_local]), int(candidate_rows[n_local]),
                    target_margin, control_margin, clean_same_margin,
                ))
        if not candidates:
            continue
        best = max(candidates, key=lambda body: (body[0], body[1], body[2]))
        action = int(best[3])
        source = actions.loc[action].to_dict()
        source.update({
            "stage1_action_index": action,
            "action_positive_row": int(best[4]),
            "action_negative_row": int(best[5]),
            "stage1_target_margin": float(best[6]),
            "stage1_control_margin": float(best[7]),
            "stage1_clean_same_boundary_margin": float(best[8]),
            "target_minus_control_margin": float(best[6] - best[7]),
            "target_minus_clean_margin": float(best[6] - best[8]),
            "clean_positive_row": int(clean["positive_row"]),
            "clean_negative_row": int(clean["negative_row"]),
            "stage1_clean_current_margin": float(clean["margin"]),
            "clean_boundary_active": bool(float(clean["margin"]) < args.margin),
        })
        selected_rows.append(source)
    if not selected_rows:
        raise RuntimeError("Stage-2 current geometry has no qualified residual action")
    selected = pd.DataFrame(selected_rows).sort_values(
        ["query_index", "action_id"], kind="stable"
    ).reset_index(drop=True)
    if selected["query_index"].duplicated().any():
        raise RuntimeError("Stage-2 did not produce one unique action per query")

    action_queries = set(map(int, selected["query_index"]))
    sentinel_rows: list[dict[str, object]] = []
    train_pool = load_npz(stage1_triplets / "train_pool.npz")
    eligible_queries = np.unique(np.asarray(train_pool["event_query"], dtype=np.int64))
    for query in eligible_queries:
        query = int(query)
        if query in action_queries:
            continue
        formula = str(graph.query_formula[query])
        if stable_fold(formula, 5, args.formula_fold_seed) == args.outer_fold:
            raise RuntimeError("outer-held formula reached protection selection")
        clean_vector = embeddings[row_position[int(graph.query_row[query])]]
        clean = graph_boundary(graph, query, row_position, embeddings, clean_vector)
        margin = float(clean["margin"])
        if 0 < margin < args.margin:
            sentinel_rows.append({
                "query_index": query,
                "query_row": int(graph.query_row[query]),
                "query_formula": formula,
                "clean_positive_row": int(clean["positive_row"]),
                "clean_negative_row": int(clean["negative_row"]),
                "stage1_clean_current_margin": margin,
            })
    if len(sentinel_rows) < 4:
        raise RuntimeError("Stage-2 lacks action-free protection/filler queries")
    sentinels = pd.DataFrame(sentinel_rows).sort_values(
        ["stage1_clean_current_margin", "query_index"], kind="stable"
    ).head(args.protection_sentinels).reset_index(drop=True)

    pool = make_pool(selected, sentinels)
    validation_pool = load_npz(stage1_triplets / "validation_pool.npz")
    selected_targeted = targeted_bank[selected["stage1_action_index"].to_numpy(np.int64)]
    selected_control = control_bank[selected["stage1_action_index"].to_numpy(np.int64)]
    action_events = np.asarray(pool["event_action_index"], dtype=np.int64)
    action_events = action_events[action_events >= 0]
    event_queries, event_query_counts = np.unique(
        np.asarray(pool["event_query"], dtype=np.int64), return_counts=True,
    )
    gates = {
        "stage1_checkpoint_is_evaluated_champion": True,
        "stage1_checkpoint_tensor_reconstruction_is_exact": True,
        "outer_held_formulas_are_absent": bool(all(
            stable_fold(str(value), 5, args.formula_fold_seed) != args.outer_fold
            for value in pool["event_formula"]
        )),
        "one_residual_action_per_query": bool(
            selected["query_index"].nunique() == len(selected)
        ),
        "all_residual_actions_are_native_hinge_active": bool(
            np.all(selected["stage1_target_margin"] < args.margin)
        ),
        "all_residual_actions_beat_same_query_control": bool(
            np.all(selected["target_minus_control_margin"] > 0)
        ),
        "all_residual_actions_beat_clean_same_boundary": bool(
            np.all(selected["target_minus_clean_margin"] > 0)
        ),
        "each_residual_action_is_exposed_once": bool(
            np.array_equal(action_events, np.arange(len(selected), dtype=np.int64))
        ),
        "each_query_contributes_at_most_action_plus_clean": bool(
            len(event_queries) and int(event_query_counts.max()) <= 2
        ),
        "all_protection_sentinels_are_correct_and_active": bool(
            np.all((sentinels["stage1_clean_current_margin"] > 0)
                   & (sentinels["stage1_clean_current_margin"] < args.margin))
        ),
        "targeted_and_control_action_payloads_are_distinct": bool(
            np.all(np.any(selected_targeted != selected_control, axis=(1, 2)))
        ),
        "validation_pool_is_action_free": bool(
            np.all(np.asarray(validation_pool["event_kind"], dtype=np.int8) == 0)
            and np.all(
                np.asarray(validation_pool["event_action_index"], dtype=np.int64) == -1
            )
            and not np.any(
                np.asarray(validation_pool["registry_kind"], dtype=np.int8)
                == Registry.ACTION
            )
        ),
        "train_and_validation_formulas_are_disjoint": not bool(
            set(map(str, pool["event_formula"]))
            & set(map(str, validation_pool["event_formula"]))
        ),
    }
    if not all(gates.values()):
        raise RuntimeError(f"Noise Stage-2 residual gates failed: {gates}")
    report = {
        "status": "NOISE_DREAMS_NATIVE_RESIDUAL_STAGE2_COMPLETE",
        "builder_version": STAGE2_BUILDER_VERSION,
        "initialization_kind": initialization_kind,
        "scientific_contract": (
            "Champion-current outer-train residual mining; one best active "
            "hard-positive action per query plus direct clean current boundary "
            "and correct active protection sentinels; native DreaMS loss/model."
        ),
        "selection_thresholds": {
            "native_triplet_margin": args.margin,
            "minimum_action_advantage": args.minimum_action_advantage,
            "protection_sentinels": args.protection_sentinels,
        },
        "selected_action_queries": int(len(selected)),
        "direct_clean_boundary_events": int(selected["clean_boundary_active"].sum()),
        "protection_sentinels": int(len(sentinels)),
        "train_events": int(len(pool["anchor_idx"])),
        "sources": dict(sorted(selected["source"].astype(str).value_counts().items())),
        "margin_summary": {
            key: {
                "minimum": float(selected[key].min()),
                "median": float(selected[key].median()),
                "maximum": float(selected[key].max()),
            }
            for key in (
                "stage1_target_margin", "target_minus_control_margin",
                "target_minus_clean_margin", "stage1_clean_current_margin",
            )
        },
        "gates": gates,
        "provenance": {
            "stage1_checkpoint_sha256": sha256_file(checkpoint),
            "stage1_summary_sha256": sha256_file(summary_path),
            "stage1_triplet_report_sha256": sha256_file(stage1_triplets / "report.json"),
            "candidate_graph_sha256": sha256_file(args.graph),
            "embedding_registry_sha256": sha256_file(args.embedding_cache),
        },
        "outer_performance_claimed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="noise_stage2_", dir=args.output.parent))
    try:
        np.savez_compressed(staging / "train_pool.npz", **pool)
        shutil.copy2(stage1_triplets / "validation_pool.npz", staging / "validation_pool.npz")
        np.savez_compressed(
            staging / "action_spectra.npz",
            targeted_action_spectra=selected_targeted,
            control_action_spectra=selected_control,
            native_action_view_representable=np.ones(len(selected), dtype=bool),
            stage1_action_index=selected["stage1_action_index"].to_numpy(np.int64),
        )
        selected.to_csv(staging / "selected_actions.csv.gz", index=False, compression="gzip")
        sentinels.to_csv(staging / "protection_sentinels.csv.gz", index=False, compression="gzip")
        report["output_artifacts"] = {
            "train_pool_sha256": sha256_file(staging / "train_pool.npz"),
            "validation_pool_sha256": sha256_file(staging / "validation_pool.npz"),
            "action_spectra_sha256": sha256_file(staging / "action_spectra.npz"),
            "selected_actions_sha256": sha256_file(staging / "selected_actions.csv.gz"),
            "protection_sentinels_sha256": sha256_file(staging / "protection_sentinels.csv.gz"),
        }
        (staging / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
