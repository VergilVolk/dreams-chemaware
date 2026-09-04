"""E15-M2: prove multi-action transfer into one shared DreaMS encoder.

This is a deliberately small capacity test.  It trains the final transformer
block and projection head of one shared encoder.  Corrective actions and risk
controls have different loss functions; harmful actions are never encoded or
imitated.  Inference remains a clean spectrum -> embedding operation.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from build_noise_final_e14_crossfit_p_teacher import (  # noqa: E402
    action_definitions, build_variant as build_e14_variant,
)
from noise_final_core import (  # noqa: E402
    CandidateGraph, json_dump, seed_everything, sha256_file, strict_rank,
)
from noise_final_e15_core import project_corrective_against_risk  # noqa: E402
from noise_v3_core import attenuate_and_renormalize, attenuate_sequence  # noqa: E402
from train_e1_identity import load_base_model, torch_load_compat  # noqa: E402
from train_noise_final_r2_shared_encoder import (  # noqa: E402
    SpectrumStore, encode_rows, forward_embeddings, unfreeze_last_block,
)


SOURCES = ("R0_N", "A4_exact", "C1_support_disjoint", "E14_mature_P")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, required=True)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--initial-student-checkpoint", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--queries-per-source", type=int, default=8)
    parser.add_argument(
        "--minimum-multiaction-queries-total", type=int, default=6,
        help="minimum genuine multi-action queries across the 32-query capacity panel",
    )
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--maximum-actions-per-query", type=int, default=16)
    parser.add_argument("--positive-spectra", type=int, default=2)
    parser.add_argument(
        "--negative-molecules", type=int, default=0,
        help="0 protects every negative molecule in the frozen candidate graph",
    )
    parser.add_argument("--head-lr", type=float, default=1e-5)
    parser.add_argument("--backbone-lr", type=float, default=2e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--target-delta-min", type=float, default=0.01)
    parser.add_argument("--target-delta-max", type=float, default=0.05)
    parser.add_argument("--lambda-action-rank", type=float, default=1.0)
    parser.add_argument("--lambda-transfer", type=float, default=0.5)
    parser.add_argument("--lambda-teacher", type=float, default=0.5)
    parser.add_argument("--lambda-preserve", type=float, default=1.0)
    parser.add_argument("--lambda-risk", type=float, default=1.0)
    parser.add_argument("--risk-margin-slack", type=float, default=0.005)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_ints(value: object, separator: str) -> tuple[int, ...]:
    text = "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)
    output = tuple(int(part) for part in text.split(separator) if part.strip())
    if not output or len(output) != len(set(output)):
        raise RuntimeError(f"invalid unique integer payload: {value!r}")
    return output


def query_rank_margin(
    graph: CandidateGraph, query: int, query_vector: np.ndarray,
    embeddings: np.ndarray, index: dict[int, int],
) -> tuple[int, float]:
    _, rows, ptr, _ = graph.query_block(query)
    candidate = embeddings[[index[int(row)] for row in rows]]
    scores = candidate @ query_vector
    molecule = np.maximum.reduceat(scores, ptr[:-1])
    return strict_rank(molecule), float(molecule[0] - np.max(molecule[1:]))


def fixed_references(
    graph: CandidateGraph, query: int, query_vector: np.ndarray,
    embeddings: np.ndarray, index: dict[int, int], positives: int, negatives: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    _, rows, ptr, _ = graph.query_block(query)
    candidate = embeddings[[index[int(row)] for row in rows]]
    scores = candidate @ query_vector
    left, right = map(int, ptr[:2])
    pos_order = np.argsort(-scores[left:right], kind="stable")[:positives]
    pos = tuple(map(int, rows[left:right][pos_order]))
    neg: list[tuple[float, int]] = []
    for molecule in range(1, len(ptr) - 1):
        start, stop = map(int, ptr[molecule:molecule + 2])
        local = int(np.argmax(scores[start:stop]))
        neg.append((float(scores[start + local]), int(rows[start + local])))
    neg.sort(key=lambda item: (-item[0], item[1]))
    if not pos or not neg:
        raise RuntimeError(f"query {query} has no fixed positive/negative references")
    selected_negatives = neg if negatives <= 0 else neg[:negatives]
    return pos, tuple(row for _, row in selected_negatives)


def action_tensor(store: SpectrumStore, row: pd.Series) -> torch.Tensor | None:
    source = str(row["source"])
    clean = store.one(int(row["query_row"]))
    if source == "R0_N":
        return attenuate_sequence(clean, parse_ints(row["action_payload"], ","), float(row["dose"]))
    if source == "A4_exact":
        payload = json.loads(str(row["action_payload"]))
        return attenuate_and_renormalize(clean, int(payload["token"]), float(row["dose"]))
    if source == "C1_support_disjoint":
        return None
    if source == "E14_mature_P":
        definitions = {item.action_id: item for item in action_definitions()}
        definition = definitions.get(str(row["action_id"]))
        if definition is None:
            raise RuntimeError(f"unknown E14 action {row['action_id']}")
        if str(row["guided_family"]) != definition.family:
            raise RuntimeError("E14 action id/family drift")
        references = [
            store.one(value) for value in parse_ints(row["positive_reference_rows"], ";")
        ]
        # This is the same executor used to create the immutable E14 outcomes.
        return build_e14_variant(clean, references, definition, 0.02)
    raise RuntimeError(f"unregistered action source {source}")


def c1_target(row: pd.Series, initial: np.ndarray, index: dict[int, int]) -> np.ndarray:
    payload = json.loads(str(row["action_payload"]))
    teachers = parse_ints(payload["teacher_rows"], ";")
    clean = initial[index[int(row["query_row"])]]
    prototype = np.mean(initial[[index[value] for value in teachers]], axis=0)
    prototype /= max(float(np.linalg.norm(prototype)), 1e-12)
    alpha = float(row["dose"])
    target = (1.0 - alpha) * clean + alpha * prototype
    target /= max(float(np.linalg.norm(target)), 1e-12)
    return target.astype(np.float32)


def frame_references(
    frame: pd.DataFrame, graph: CandidateGraph, initial: np.ndarray,
    index: dict[int, int], positives: int, negatives: int,
) -> dict[int, tuple[tuple[int, ...], tuple[int, ...]]]:
    output = {}
    for query in sorted(set(frame["query_index"].astype(int))):
        qrow = int(graph.query_row[query])
        output[query] = fixed_references(
            graph, query, initial[index[qrow]], initial, index, positives, negatives,
        )
    return output


def select_queries(
    frame: pd.DataFrame, ranks: np.ndarray, margins: np.ndarray,
    per_source: int, want_wrong: bool,
) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    for source in SOURCES:
        block = frame.loc[frame["source"].astype(str).eq(source)].copy()
        block["initial_rank"] = block["query_index"].astype(int).map(lambda q: int(ranks[q]))
        block["initial_margin"] = block["query_index"].astype(int).map(lambda q: float(margins[q]))
        desired = block["initial_rank"].ne(1) if want_wrong else block["initial_rank"].eq(1)
        block = block.loc[desired].copy()
        # Query selection is deterministic and diverse.  Strength is used only
        # to order a capacity panel, never to score a held result.
        query = (
            block.groupby(["query_index", "query_ik14", "query_formula"], as_index=False)
            .agg(
                strength=("source_kind_percentile", "max"),
                boundary=("initial_margin", "min"),
                available_actions=("action_id", "nunique"),
            )
        )
        if want_wrong:
            query["is_multiaction"] = query["available_actions"].ge(2)
        else:
            query["is_multiaction"] = False
        query = (
            query.sort_values(
                ["is_multiaction", "strength", "boundary", "query_formula", "query_ik14", "query_index"],
                ascending=[False, False, True, True, True, True], kind="stable",
            )
            .drop_duplicates("query_ik14", keep="first")
            .head(per_source)
        )
        if len(query) != per_source:
            raise RuntimeError(
                f"source {source} has {len(query)} eligible unique identities; need {per_source}"
            )
        chosen = set(query["query_index"].astype(int))
        selected.append(block.loc[block["query_index"].astype(int).isin(chosen)])
    output = pd.concat(selected, ignore_index=True)
    if output.empty:
        raise RuntimeError("query selection produced no rows")
    return output


def limit_query_actions(
    frame: pd.DataFrame, maximum: int, seed: int,
) -> pd.DataFrame:
    """Keep a bounded, family-diverse action set for each capacity-test query."""
    if maximum < 2:
        raise ValueError("E15-M2 requires at least two actions per corrective query")
    rng = np.random.default_rng(seed)
    selected: list[pd.DataFrame] = []
    for (_, _), block in frame.groupby(["source", "query_index"], sort=True):
        block = block.copy()
        block["_jitter"] = rng.random(len(block))
        block = block.sort_values(
            ["source_kind_percentile", "action_family", "_jitter", "action_id"],
            ascending=[False, True, True, True], kind="stable",
        )
        first = block.drop_duplicates("action_family", keep="first").head(maximum)
        if len(first) < maximum:
            first = pd.concat([
                first,
                block.loc[~block.index.isin(first.index)].head(maximum - len(first)),
            ])
        selected.append(first.drop(columns="_jitter"))
    output = pd.concat(selected, ignore_index=True)
    if output.duplicated(["source", "query_index", "action_id"]).any():
        raise RuntimeError("bounded E15-M2 action panel contains duplicates")
    return output


def query_batches(frame: pd.DataFrame, rng: np.random.Generator) -> dict[str, list[pd.DataFrame]]:
    """Exactly one optimizer batch per source/query; actions never create extra steps."""
    output: dict[str, list[pd.DataFrame]] = {}
    for source, source_block in frame.groupby("source", sort=True):
        groups = [block.copy() for _, block in source_block.groupby("query_index", sort=True)]
        rng.shuffle(groups)
        output[str(source)] = groups
    return output


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sum(values * weights) / torch.clamp(torch.sum(weights), min=1e-12)


def corrective_loss(
    model, store: SpectrumStore, frame: pd.DataFrame,
    refs: dict[int, tuple[tuple[int, ...], tuple[int, ...]]],
    initial: np.ndarray, index: dict[int, int], device: torch.device, args,
) -> tuple[torch.Tensor, dict[str, float]]:
    spectra: list[torch.Tensor] = []
    layout: list[dict[str, object]] = []
    preserve_positions: list[int] = []
    preserve_rows: list[int] = []
    c1_targets: list[np.ndarray | None] = []
    for _, row in frame.iterrows():
        query = int(row["query_index"])
        positive, negative = refs[query]
        item: dict[str, object] = {"clean": len(spectra)}
        spectra.append(store.one(int(row["query_row"])))
        preserve_positions.append(int(item["clean"]))
        preserve_rows.append(int(row["query_row"]))
        action = action_tensor(store, row)
        item["action"] = None
        if action is not None:
            item["action"] = len(spectra)
            spectra.append(action)
        item["positive"] = list(range(len(spectra), len(spectra) + len(positive)))
        spectra.extend(store.get(positive))
        preserve_positions.extend(item["positive"]); preserve_rows.extend(positive)
        item["negative"] = list(range(len(spectra), len(spectra) + len(negative)))
        spectra.extend(store.get(negative))
        preserve_positions.extend(item["negative"]); preserve_rows.extend(negative)
        count = int(row.get("query_action_count", 1))
        if count < 1:
            raise RuntimeError("invalid global query action count")
        training_weight = float(row.get("training_weight", 1.0))
        if not np.isfinite(training_weight) or training_weight <= 0:
            raise RuntimeError("invalid corrective training weight")
        item["weight"] = training_weight / count
        item["baseline"] = float(row["initial_margin"])
        item["source_percentile"] = float(row["source_kind_percentile"])
        layout.append(item)
        c1_targets.append(c1_target(row, initial, index) if str(row["source"]) == "C1_support_disjoint" else None)
    encoded = forward_embeddings(model, torch.stack(spectra).to(device), args.amp)
    clean_margin, action_margin, transfer, teacher = [], [], [], []
    for item, target in zip(layout, c1_targets):
        clean = encoded[int(item["clean"])]
        pos = torch.max(encoded[item["positive"]] @ clean)
        neg = torch.max(encoded[item["negative"]] @ clean)
        clean_margin.append(pos - neg)
        if item["action"] is not None:
            action = encoded[int(item["action"])]
            action_margin.append(torch.max(encoded[item["positive"]] @ action) - torch.max(encoded[item["negative"]] @ action))
            transfer.append(1.0 - torch.sum(clean * action.detach()))
            teacher.append(clean.sum() * 0.0)
        else:
            action_margin.append(clean.sum() * 0.0)
            transfer.append(clean.sum() * 0.0)
            fixed = torch.from_numpy(target).to(device=device, dtype=clean.dtype)
            teacher.append(1.0 - torch.sum(clean * fixed))
    clean_margin_t = torch.stack(clean_margin)
    action_margin_t = torch.stack(action_margin)
    transfer_t = torch.stack(transfer)
    teacher_t = torch.stack(teacher)
    weights = torch.tensor([float(item["weight"]) for item in layout], device=device, dtype=clean_margin_t.dtype)
    baseline = torch.tensor([float(item["baseline"]) for item in layout], device=device, dtype=clean_margin_t.dtype)
    percentile = torch.tensor(
        [float(item["source_percentile"]) for item in layout],
        device=device, dtype=clean_margin_t.dtype,
    )
    raw = torch.tensor([item["action"] is not None for item in layout], device=device)
    # Raw margin deltas are not comparable across official, support-disjoint
    # and mature geometries.  Only the within-source percentile is transferred;
    # its magnitude is mapped into the current initialization's margin scale.
    desired_delta = args.target_delta_min + (
        args.target_delta_max - args.target_delta_min
    ) * torch.clamp(percentile, min=0.0, max=1.0)
    target = baseline + desired_delta
    clean_rank = F.softplus((args.rank_margin - clean_margin_t) / args.temperature)
    target_floor = F.relu(target - clean_margin_t)
    action_rank = torch.where(raw, F.softplus((args.rank_margin - action_margin_t) / args.temperature), torch.zeros_like(action_margin_t))
    preserve_target = torch.from_numpy(np.stack([
        initial[index[int(row)]] for row in preserve_rows
    ])).to(device=device, dtype=encoded.dtype)
    preserve = 1.0 - torch.sum(encoded[preserve_positions] * preserve_target, dim=1)
    loss = (
        weighted_mean(clean_rank + target_floor, weights)
        + args.lambda_action_rank * weighted_mean(action_rank, weights)
        + args.lambda_transfer * weighted_mean(transfer_t, weights)
        + args.lambda_teacher * weighted_mean(teacher_t, weights)
        + args.lambda_preserve * preserve.mean()
    )
    return loss, {
        "loss": float(loss.detach()), "clean_margin": float(clean_margin_t.mean().detach()),
        "action_margin": float(action_margin_t[raw].mean().detach()) if bool(raw.any()) else float("nan"),
        "preservation": float((1.0 - preserve).mean().detach()),
    }


def risk_loss(
    model, store: SpectrumStore, frame: pd.DataFrame,
    refs: dict[int, tuple[tuple[int, ...], tuple[int, ...]]],
    initial: np.ndarray, index: dict[int, int], device: torch.device, args,
) -> tuple[torch.Tensor, dict[str, float]]:
    # Harmful action payloads are intentionally ignored here.
    unique = frame.drop_duplicates(["source", "query_index"], keep="first")
    spectra: list[torch.Tensor] = []
    layout = []
    preserve_positions: list[int] = []
    preserve_rows: list[int] = []
    for _, row in unique.iterrows():
        query = int(row["query_index"]); positive, negative = refs[query]
        item = {"clean": len(spectra)}; spectra.append(store.one(int(row["query_row"])))
        preserve_positions.append(int(item["clean"])); preserve_rows.append(int(row["query_row"]))
        item["positive"] = list(range(len(spectra), len(spectra) + len(positive)))
        spectra.extend(store.get(positive)); preserve_positions.extend(item["positive"]); preserve_rows.extend(positive)
        item["negative"] = list(range(len(spectra), len(spectra) + len(negative)))
        spectra.extend(store.get(negative)); preserve_positions.extend(item["negative"]); preserve_rows.extend(negative)
        item["query_row"] = int(row["query_row"])
        item["baseline"] = float(row["initial_margin"]); layout.append(item)
    encoded = forward_embeddings(model, torch.stack(spectra).to(device), args.amp)
    margins = []
    for item in layout:
        clean = encoded[int(item["clean"])]
        margins.append(torch.max(encoded[item["positive"]] @ clean) - torch.max(encoded[item["negative"]] @ clean))
    margin = torch.stack(margins)
    preserve_target = torch.from_numpy(np.stack([
        initial[index[int(row)]] for row in preserve_rows
    ])).to(device=device, dtype=encoded.dtype)
    preserve = 1.0 - torch.sum(encoded[preserve_positions] * preserve_target, dim=1)
    baseline = torch.tensor([float(item["baseline"]) for item in layout], device=device, dtype=margin.dtype)
    floor = F.relu(baseline - args.risk_margin_slack - margin)
    loss = floor.mean() + args.lambda_preserve * preserve.mean()
    return loss, {
        "loss": float(loss.detach()), "margin": float(margin.mean().detach()),
        "preservation": float((1.0 - preserve).mean().detach()),
    }


def gradient_vector(model) -> torch.Tensor:
    values = [
        (
            torch.zeros(parameter.numel(), dtype=torch.float32)
            if parameter.grad is None else
            parameter.grad.detach().flatten().to(device="cpu", dtype=torch.float32)
        )
        for parameter in model.parameters() if parameter.requires_grad
    ]
    if not values:
        raise RuntimeError("gradient calibration produced no trainable gradient")
    return torch.cat(values)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    a = left.double(); b = right.double()
    denom = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
    if denom <= 0:
        return float("nan")
    return float(np.clip(float(torch.dot(a, b)) / denom, -1.0, 1.0))


def finite_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if len(array) else float("nan")


def evaluate_queries(
    graph: CandidateGraph, queries: np.ndarray, embeddings: np.ndarray, index: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    rank = np.empty(len(queries), dtype=np.int16); margin = np.empty(len(queries), dtype=np.float32)
    for local, query in enumerate(queries):
        qrow = int(graph.query_row[int(query)])
        rank[local], margin[local] = query_rank_margin(graph, int(query), embeddings[index[qrow]], embeddings, index)
    return rank, margin


def main() -> None:
    args = arguments()
    seed_everything(args.seed)
    if not 0 < args.target_delta_min <= args.target_delta_max:
        raise ValueError("target delta range must be positive and monotone")
    if not 0 <= args.minimum_multiaction_queries_total <= 4 * args.queries_per_source:
        raise ValueError("multi-action minimum must lie within the total capacity query count")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite E15-M2 result: {args.output_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("formal E15-M2 requires CUDA")
    required = {
        "panel_report": args.panel_dir / "report.json",
        "corrective": args.panel_dir / "executable_corrective.csv.gz",
        "harmful": args.panel_dir / "executable_harmful.csv.gz",
        "gradient_panel": args.panel_dir / "gradient_panel.csv.gz",
        "graph": args.graph, "data": args.data,
        "official_checkpoint": args.official_checkpoint,
        "architecture_checkpoint": args.architecture_checkpoint,
        "initial_student_checkpoint": args.initial_student_checkpoint,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    panel_report = json.loads(required["panel_report"].read_text(encoding="utf-8"))
    if (
        panel_report.get("status") != "noise_final_e15_m2_executable_panel_complete"
        or not panel_report.get("formal") or not panel_report.get("pass_to_shared_encoder_overfit")
        or int(panel_report.get("outer_formula_fold", -1)) != args.outer_fold
    ):
        raise RuntimeError("E15-M2 panel is not formally authorized for this fold")
    corrective = pd.read_csv(required["corrective"], low_memory=False)
    harmful = pd.read_csv(required["harmful"], low_memory=False)
    gradient_panel = pd.read_csv(required["gradient_panel"], low_memory=False)
    if len(gradient_panel) != 128:
        raise RuntimeError("E15-M2 requires the frozen 128-action gradient panel")

    graph = CandidateGraph(args.graph)
    reachable = np.unique(np.concatenate([graph.query_row, graph.pair_candidate_row])).astype(np.int64)
    store = SpectrumStore(args.data, reachable, args.n_highest_peaks)
    device = torch.device(args.device)
    model, initialization = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks,
    )
    package = torch_load_compat(args.initial_student_checkpoint, map_location="cpu")
    if (
        package.get("status") != "noise_final_e4a_direct_shared_dreams_encoder"
        or int(package.get("outer_fold", -1)) != args.outer_fold
        or package.get("P2b_used") or not package.get("inference_clean_only")
    ):
        raise RuntimeError("initial checkpoint is not the mature clean-only shared encoder")
    model.load_state_dict(package["model_state"], strict=True)
    capacity = unfreeze_last_block(model)
    model.eval()  # gradients on, dropout off
    initial = encode_rows(model, store, store.rows, device, args.eval_batch_size, False, "E15-M2-init")
    index = {int(row): position for position, row in enumerate(store.rows)}
    all_rank, all_margin = evaluate_queries(
        graph, np.arange(graph.n_queries, dtype=np.int64), initial, index,
    )
    corrective = select_queries(
        corrective, all_rank, all_margin, args.queries_per_source, True,
    )
    corrective = limit_query_actions(corrective, args.maximum_actions_per_query, args.seed)
    harmful_routes = select_queries(
        harmful, all_rank, all_margin, args.queries_per_source, False,
    )
    # A harmful action is a routing label, not an imitation target.  Multiple
    # harmful actions on one source/query therefore collapse to one clean risk
    # observation; otherwise the same no-op protection would be repeated.
    harmful = harmful_routes.sort_values(
        ["source", "query_index", "source_kind_percentile", "action_id"],
        ascending=[True, True, False, True], kind="stable",
    ).drop_duplicates(["source", "query_index"], keep="first").reset_index(drop=True)
    corrective["query_action_count"] = corrective.groupby(
        ["source", "query_index"]
    )["action_id"].transform("size").astype(int)
    for frame in (corrective, harmful, gradient_panel):
        frame["initial_margin"] = frame["query_index"].astype(int).map(lambda q: float(all_margin[q]))
        frame["initial_rank"] = frame["query_index"].astype(int).map(lambda q: int(all_rank[q]))
    union = pd.concat([corrective, harmful, gradient_panel], ignore_index=True)
    refs = frame_references(
        union, graph, initial, index, args.positive_spectra, args.negative_molecules,
    )

    # Mandatory 32 distinct microbatches (4 actions each), one observation per
    # action.  This replaces the legacy first-four-example calibration.
    gradients: dict[str, torch.Tensor] = {}
    calibration_records = []
    for (source, kind), block in gradient_panel.groupby(["source", "supervision_kind"], sort=True):
        if len(block) != 16:
            raise RuntimeError(f"gradient stratum {source}|{kind} has {len(block)}, expected 16")
        model.zero_grad(set_to_none=True)
        for batch in [block.iloc[left:left + 4] for left in range(0, 16, 4)]:
            if kind == "corrective":
                loss, _ = corrective_loss(model, store, batch, refs, initial, index, device, args)
            else:
                loss, _ = risk_loss(model, store, batch, refs, initial, index, device, args)
            (loss / 4.0).backward()
        vector = gradient_vector(model)
        key = f"{source}|{kind}"; gradients[key] = vector
        calibration_records.append({
            "branch": key, "actions": 16, "microbatches": 4,
            "gradient_norm": float(torch.linalg.vector_norm(vector.float())),
        })
    if sum(record["microbatches"] for record in calibration_records) != 32:
        raise RuntimeError("gradient calibration did not execute exactly 32 microbatches")
    cosine_matrix = {
        left: {right: cosine(gradients[left], gradients[right]) for right in sorted(gradients)}
        for left in sorted(gradients)
    }
    corr_norm = {
        source: next(record["gradient_norm"] for record in calibration_records if record["branch"] == f"{source}|corrective")
        for source in SOURCES
    }
    median_norm = float(np.median(list(corr_norm.values())))
    source_weight = {
        source: float(np.clip(median_norm / max(norm, 1e-12), 0.25, 4.0))
        for source, norm in corr_norm.items()
    }
    if max(source_weight.values()) / min(source_weight.values()) > 16.0001:
        raise RuntimeError("gradient calibration escaped the preregistered 16x weight range")
    del gradients

    head = [p for p in model.head.parameters() if p.requires_grad]
    backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    trainable = head + backbone
    optimizer = torch.optim.AdamW([
        {"params": head, "lr": args.head_lr, "weight_decay": args.weight_decay},
        {"params": backbone, "lr": args.backbone_lr, "weight_decay": 0.0},
    ])
    rng = np.random.default_rng(args.seed)
    history = []
    for epoch in range(args.epochs):
        correction_by_source = query_batches(corrective, rng)
        risk_by_source = query_batches(harmful, rng)
        schedule: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        for source in SOURCES:
            correction_batches = correction_by_source.get(source, [])
            risk_batches = risk_by_source.get(source, [])
            if len(correction_batches) != args.queries_per_source or len(risk_batches) != args.queries_per_source:
                raise RuntimeError(f"source {source} does not have one corrective/risk step per selected query")
            schedule.extend(zip(correction_batches, risk_batches, strict=True))
        rng.shuffle(schedule)
        # No cursor wrap: every row appears once, and every query creates one step.
        correction_seen: set[tuple[str, int, str]] = set()
        risk_seen: set[tuple[str, int, str]] = set()
        correction_query_steps: set[tuple[str, int]] = set()
        risk_query_steps: set[tuple[str, int]] = set()
        epoch_logs = defaultdict(list)
        for corr, risk in schedule:
            optimizer.zero_grad(set_to_none=True)
            corr_loss = None
            safe_loss = None
            correction_weight = 1.0
            if corr is not None:
                keys = list(zip(corr["source"].astype(str), corr["query_index"].astype(int), corr["action_id"].astype(str)))
                if correction_seen.intersection(keys):
                    raise RuntimeError("corrective action recycled within epoch")
                correction_seen.update(keys)
                query_key = (str(corr["source"].iloc[0]), int(corr["query_index"].iloc[0]))
                if query_key in correction_query_steps or corr["query_index"].nunique() != 1:
                    raise RuntimeError("corrective query received more than one optimizer step")
                correction_query_steps.add(query_key)
                corr_loss, log = corrective_loss(model, store, corr, refs, initial, index, device, args)
                batch_sources = set(corr["source"].astype(str))
                if len(batch_sources) != 1:
                    raise RuntimeError("corrective optimizer batch mixed source geometries")
                correction_weight = source_weight[next(iter(batch_sources))]
                source = next(iter(batch_sources))
                for key, value in log.items():
                    epoch_logs[f"corrective_{key}"].append(value)
                    epoch_logs[f"corrective_{source}_{key}"].append(value)
            if risk is not None:
                keys = list(zip(risk["source"].astype(str), risk["query_index"].astype(int), risk["action_id"].astype(str)))
                if risk_seen.intersection(keys):
                    raise RuntimeError("risk action recycled within epoch")
                risk_seen.update(keys)
                query_key = (str(risk["source"].iloc[0]), int(risk["query_index"].iloc[0]))
                if query_key in risk_query_steps or risk["query_index"].nunique() != 1:
                    raise RuntimeError("risk query received more than one optimizer step")
                risk_query_steps.add(query_key)
                if str(corr["source"].iloc[0]) != str(risk["source"].iloc[0]):
                    raise RuntimeError("corrective/risk gradient pair crossed source geometries")
                safe_loss, log = risk_loss(model, store, risk, refs, initial, index, device, args)
                source = str(risk["source"].iloc[0])
                for key, value in log.items():
                    epoch_logs[f"risk_{key}"].append(value)
                    epoch_logs[f"risk_{source}_{key}"].append(value)
            if corr_loss is not None and not torch.isfinite(corr_loss):
                raise RuntimeError("non-finite E15-M2 corrective loss")
            if safe_loss is not None and not torch.isfinite(safe_loss):
                raise RuntimeError("non-finite E15-M2 risk loss")
            corr_grad = (
                list(torch.autograd.grad(corr_loss, trainable, allow_unused=True))
                if corr_loss is not None else [None] * len(trainable)
            )
            risk_grad = (
                list(torch.autograd.grad(safe_loss, trainable, allow_unused=True))
                if safe_loss is not None else [None] * len(trainable)
            )
            corr_grad = [
                None if value is None else correction_weight * value
                for value in corr_grad
            ]
            risk_grad = [
                None if value is None else args.lambda_risk * value
                for value in risk_grad
            ]
            projected, projection = project_corrective_against_risk(corr_grad, risk_grad)
            for parameter, left, right in zip(trainable, projected, risk_grad):
                if left is None and right is None:
                    parameter.grad = None
                elif left is None:
                    parameter.grad = right
                elif right is None:
                    parameter.grad = left
                else:
                    parameter.grad = left + right
            if np.isfinite(float(projection["gradient_cosine"])):
                epoch_logs["gradient_cosine"].append(float(projection["gradient_cosine"]))
            epoch_logs["gradient_conflict"].append(float(bool(projection["conflict"])))
            epoch_logs["gradient_projection_scale"].append(float(projection["projection_scale"]))
            epoch_logs["corrective_gradient_norm"].append(float(projection["corrective_gradient_norm"]))
            epoch_logs["risk_gradient_norm"].append(float(projection["risk_gradient_norm"]))
            epoch_logs["risk_projection_active"].append(float(bool(projection["risk_projection_active"])))
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step(); model.eval()
        if len(correction_seen) != len(corrective) or len(risk_seen) != len(harmful):
            raise RuntimeError("E15-M2 epoch did not consume every action exactly once")
        if len(correction_query_steps) != 4 * args.queries_per_source or len(risk_query_steps) != 4 * args.queries_per_source:
            raise RuntimeError("E15-M2 epoch did not use exactly one step per selected source/query")
        history.append({
            "epoch": epoch + 1, "corrective_actions": len(correction_seen),
            "risk_actions": len(risk_seen),
            "corrective_query_steps": len(correction_query_steps),
            "risk_query_steps": len(risk_query_steps),
            **{key: finite_mean(value) for key, value in epoch_logs.items()},
        })
        print(json.dumps(history[-1]), flush=True)

    final = encode_rows(model, store, store.rows, device, args.eval_batch_size, False, "E15-M2-final")
    evaluation_queries = np.asarray(sorted(set(corrective["query_index"].astype(int)) | set(harmful["query_index"].astype(int))), dtype=np.int64)
    initial_rank = all_rank[evaluation_queries]
    initial_margin = all_margin[evaluation_queries]
    final_rank, final_margin = evaluate_queries(graph, evaluation_queries, final, index)
    corrective_queries = set(corrective["query_index"].astype(int))
    risk_queries = set(harmful["query_index"].astype(int))
    corrective_mask = np.asarray([int(q) in corrective_queries for q in evaluation_queries])
    risk_mask = np.asarray([int(q) in risk_queries for q in evaluation_queries])
    corrected = int(np.sum(corrective_mask & (initial_rank != 1) & (final_rank == 1)))
    introduced = int(np.sum(risk_mask & (initial_rank == 1) & (final_rank != 1)))
    preservation = np.sum(initial * final, axis=1)
    per_query = pd.DataFrame({
        "query_index": evaluation_queries,
        "query_row": graph.query_row[evaluation_queries],
        "query_ik14": graph.query_ik14[evaluation_queries],
        "query_formula": graph.query_formula[evaluation_queries],
        "corrective_panel": corrective_mask, "risk_panel": risk_mask,
        "initial_rank": initial_rank, "final_rank": final_rank,
        "initial_margin": initial_margin, "final_margin": final_margin,
        "margin_delta": final_margin - initial_margin,
    })
    action_counts = corrective.groupby(["source", "query_index"]).size()
    multiaction_queries_by_source = {
        source: int(action_counts.loc[source].ge(2).sum())
        for source in SOURCES
    }
    gates = {
        "exactly_32_corrective_source_queries": int(corrective.groupby(["source", "query_index"]).ngroups) == 4 * args.queries_per_source == 32,
        "minimum_multiaction_queries_retained_globally": bool(
            set(multiaction_queries_by_source) == set(SOURCES)
            and sum(multiaction_queries_by_source.values())
            >= args.minimum_multiaction_queries_total
        ),
        "gradient_microbatches_ge_32": sum(record["microbatches"] for record in calibration_records) >= 32,
        "gradient_observations_ge_128": sum(record["actions"] for record in calibration_records) >= 128,
        "branch_gradient_cosines_finite": bool(all(
            np.isfinite(value) for row in cosine_matrix.values() for value in row.values()
        )),
        "risk_projection_audited_each_epoch": bool(all(
            "gradient_conflict" in epoch and "gradient_projection_scale" in epoch
            for epoch in history
        )),
        "one_optimizer_step_per_source_query": bool(all(
            int(epoch.get("corrective_query_steps", -1)) == 4 * args.queries_per_source
            and int(epoch.get("risk_query_steps", -1)) == 4 * args.queries_per_source
            for epoch in history
        )),
        "all_candidate_negative_molecules_protected": args.negative_molecules <= 0,
        "at_least_one_training_error_corrected": corrected > 0,
        "corrective_mean_margin_positive": float(np.mean((final_margin - initial_margin)[corrective_mask])) > 0,
        "risk_introduced_zero": introduced == 0,
        "initialization_preservation_ge_0_99": float(np.mean(preservation)) >= 0.99,
        "P2b_forbidden": True, "P3_not_consumed": True,
    }
    report = {
        "status": "noise_final_e15_m2_shared_encoder_overfit_complete",
        "formal": True, "outer_formula_fold": args.outer_fold,
        "initialization": initialization, "capacity": capacity,
        "corrective_source_queries": int(corrective.groupby(["source", "query_index"]).ngroups),
        "corrective_actions": int(len(corrective)),
        "multiaction_corrective_queries_by_source": multiaction_queries_by_source,
        "minimum_multiaction_queries_total": int(
            args.minimum_multiaction_queries_total
        ),
        "maximum_corrective_actions_per_query": int(
            corrective.groupby(["source", "query_index"]).size().max()
        ),
        "risk_routes": int(len(harmful_routes)), "risk_training_queries": int(len(harmful)),
        "evaluation_queries": int(len(evaluation_queries)),
        "corrected": corrected, "introduced": introduced,
        "mean_corrective_margin_delta": float(np.mean((final_margin - initial_margin)[corrective_mask])),
        "mean_risk_margin_delta": float(np.mean((final_margin - initial_margin)[risk_mask])),
        "initialization_preservation_mean": float(np.mean(preservation)),
        "gradient_calibration": {
            "records": calibration_records, "cosines": cosine_matrix,
            "corrective_source_weights": source_weight,
        },
        "training_projection": {
            "epochs_with_any_conflict": int(sum(
                float(epoch.get("gradient_conflict", 0.0)) > 0 for epoch in history
            )),
            "mean_conflict_fraction": float(np.mean([
                float(epoch.get("gradient_conflict", 0.0)) for epoch in history
            ])),
            "mean_projection_scale": float(np.mean([
                float(epoch.get("gradient_projection_scale", 0.0)) for epoch in history
            ])),
        },
        "history": history, "gates": gates,
        "pass_to_identity_holdout": bool(all(gates.values())),
        "contracts": {
            "shared_query_reference_encoder": True,
            "clean_spectrum_only_at_inference": True,
            "harmful_action_payload_encoded": False,
            "within_epoch_action_recycling": False,
            "one_optimizer_step_per_source_query": True,
            "all_candidate_negative_molecules_protected": args.negative_molecules <= 0,
            "P2b": "forbidden", "P3_consumed": False,
        },
        "provenance": {name: sha256_file(path) for name, path in required.items()},
        "claim_limit": "Training-panel capacity test; not identity/formula holdout or P3 performance.",
    }
    if not report["pass_to_identity_holdout"]:
        raise RuntimeError(f"E15-M2 overfit gates failed: {gates}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="noise_e15_m2_train_", dir=args.output_dir.parent))
    try:
        corrective.to_csv(staging / "selected_corrective_actions.csv.gz", index=False, compression="gzip")
        harmful_routes.to_csv(staging / "selected_risk_actions.csv.gz", index=False, compression="gzip")
        harmful.to_csv(staging / "selected_risk_queries.csv.gz", index=False, compression="gzip")
        per_query.to_csv(staging / "per_query.csv.gz", index=False, compression="gzip")
        torch.save({
            "status": "noise_final_e15_m2_shared_dreams_encoder",
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "outer_fold": args.outer_fold, "seed": args.seed,
            "inference_clean_only": True, "P2b_used": False,
        }, staging / "shared_encoder.pt")
        json_dump(staging / "report.json", report)
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
