#!/usr/bin/env python
"""Opened full-candidate bridge for direct BioAware shared-embedding training.

B32 proved that the selected BioAware actions reach the final DreaMS block and
projection head.  B33 adds the missing deployment constraint: every official-
correct physical query contributes a one-sided boundary floor, and evaluation
re-maximises over every reference spectrum for every candidate molecule.

This is deliberately an opened engineering bridge.  It can validate a training
kernel and save a checkpoint, but it cannot establish formula-isolated or blind
generalisation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from noise_final_core import sha256_file  # noqa: E402
from train_bioaware_b26_direct_shared_embedding_canary import (  # noqa: E402
    component_gradient,
    encode,
    gradient_summary,
)
from train_e1_identity import load_base_model  # noqa: E402
from train_noise_final_e4a_direct_augmentation import unfreeze_last_blocks  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument(
        "--official-checkpoint", type=Path,
        default=ROOT / "data/e1/official_embedding_slim.pt",
    )
    parser.add_argument(
        "--architecture-checkpoint", type=Path,
        default=ROOT / "dreams/models/pretrained/ssl_model_server.pt",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--batch-corrective-identities", type=int, default=8)
    parser.add_argument("--batch-known-safety-identities", type=int, default=4)
    parser.add_argument("--batch-broad-safety-identities", type=int, default=8)
    parser.add_argument("--unfreeze-blocks", type=int, default=1)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--head-lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--rank-margin", type=float, default=0.03)
    parser.add_argument("--lambda-corrective", type=float, default=1.0)
    parser.add_argument("--lambda-known-safety", type=float, default=4.0)
    parser.add_argument("--lambda-broad-safety", type=float, default=2.0)
    parser.add_argument("--lambda-preserve", type=float, default=10.0)
    parser.add_argument("--known-safety-slack", type=float, default=0.002)
    parser.add_argument("--broad-safety-slack", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--encode-batch-size", type=int, default=96)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-threads", type=int, default=8)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8",
    )
    temporary.replace(path)


def load_graph(directory: Path) -> tuple[dict, dict[str, np.ndarray]]:
    report_path = directory / "report.json"
    graph_path = directory / "full_candidate_graph.npz"
    for path in (report_path, graph_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "bioaware_b33_full_candidate_graph_complete"
        or report.get("pass_to_full_graph_bridge") is not True
        or report.get("provenance", {}).get("graph_sha256") != sha256_file(graph_path)
    ):
        raise RuntimeError("B33 candidate graph did not pass its frozen gate")
    with np.load(graph_path, allow_pickle=False) as handle:
        body = {name: handle[name] for name in handle.files}
    required = {
        "query_id", "source", "physical_query_id", "truth_candidate_id",
        "truth_formula", "baseline_correct", "direct_corrective", "direct_safety",
        "query_tensor", "candidate_ptr", "candidate_id", "candidate_official_score",
        "candidate_all_reference_ptr", "candidate_all_reference_position",
        "candidate_reference_tensor", "truth_candidate_position",
        "action_candidate_position", "baseline_rank_full_graph",
    }
    missing = required - set(body)
    if missing:
        raise RuntimeError(f"B33 graph lacks {sorted(missing)}")
    n = len(body["query_id"])
    if n != 860 or body["query_tensor"].shape != (n, 101, 2):
        raise RuntimeError("B33 query contract changed")
    if len(body["candidate_ptr"]) != n + 1:
        raise RuntimeError("B33 candidate pointer contract changed")
    candidates = len(body["candidate_id"])
    if int(body["candidate_ptr"][-1]) != candidates:
        raise RuntimeError("B33 candidate pointer terminal changed")
    if len(body["candidate_all_reference_ptr"]) != candidates + 1:
        raise RuntimeError("B33 candidate-reference pointer contract changed")
    if int(body["candidate_all_reference_ptr"][-1]) != len(body["candidate_all_reference_position"]):
        raise RuntimeError("B33 candidate-reference pointer terminal changed")
    if np.any(np.diff(body["candidate_ptr"].astype(np.int64)) < 2):
        raise RuntimeError("B33 query lacks a negative candidate")
    if np.any(np.diff(body["candidate_all_reference_ptr"].astype(np.int64)) < 1):
        raise RuntimeError("B33 candidate lacks a reference spectrum")
    return report, body


def candidate_max_scores(
    query_embedding: np.ndarray,
    reference_embedding: np.ndarray,
    body: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return max score and winning reference position per candidate molecule."""
    candidate_ptr = body["candidate_ptr"].astype(np.int64)
    reference_ptr = body["candidate_all_reference_ptr"].astype(np.int64)
    reference_position = body["candidate_all_reference_position"].astype(np.int64)
    score = np.empty(len(body["candidate_id"]), dtype=np.float32)
    winner = np.empty(len(score), dtype=np.int64)
    for query in range(len(query_embedding)):
        for candidate in range(int(candidate_ptr[query]), int(candidate_ptr[query + 1])):
            positions = reference_position[
                int(reference_ptr[candidate]):int(reference_ptr[candidate + 1])
            ]
            values = reference_embedding[positions] @ query_embedding[query]
            best = int(np.argmax(values))
            score[candidate] = float(values[best])
            winner[candidate] = int(positions[best])
    return score, winner


def graph_ranks(
    candidate_score: np.ndarray, body: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Rank truth first and count every score tie against it."""
    pointer = body["candidate_ptr"].astype(np.int64)
    ranks = np.empty(len(pointer) - 1, dtype=np.int64)
    top_position = np.empty(len(ranks), dtype=np.int64)
    for query, (left, right) in enumerate(zip(pointer[:-1], pointer[1:], strict=True)):
        left, right = int(left), int(right)
        truth = float(candidate_score[left])
        negatives = candidate_score[left + 1:right]
        ranks[query] = 1 + int(np.sum(negatives >= truth))
        # Deterministic reporting only.  Correctness always comes from rank==1.
        top_position[query] = left + int(np.argmax(candidate_score[left:right]))
    return ranks, top_position


def physical_positions(body: dict[str, np.ndarray]) -> np.ndarray:
    """Collapse evaluation repeats after checking graph/tensor identity."""
    source = body["source"].astype(str)
    physical = body["physical_query_id"].astype(str)
    candidate_ptr = body["candidate_ptr"].astype(np.int64)
    candidate_id = body["candidate_id"].astype(str)
    all_ref_ptr = body["candidate_all_reference_ptr"].astype(np.int64)
    all_ref_pos = body["candidate_all_reference_position"].astype(np.int64)
    groups: dict[tuple[str, str], list[int]] = {}
    for position, key in enumerate(zip(source, physical, strict=True)):
        groups.setdefault(key, []).append(position)
    chosen = []
    for key in sorted(groups):
        positions = groups[key]
        first = positions[0]
        first_candidates = candidate_id[candidate_ptr[first]:candidate_ptr[first + 1]]
        for position in positions[1:]:
            if not np.array_equal(body["query_tensor"][first], body["query_tensor"][position]):
                raise RuntimeError(f"physical repeat tensor mismatch: {key}")
            observed = candidate_id[candidate_ptr[position]:candidate_ptr[position + 1]]
            if not np.array_equal(first_candidates, observed):
                raise RuntimeError(f"physical repeat candidate mismatch: {key}")
            for a, b in zip(
                range(int(candidate_ptr[first]), int(candidate_ptr[first + 1])),
                range(int(candidate_ptr[position]), int(candidate_ptr[position + 1])),
                strict=True,
            ):
                refs_a = all_ref_pos[all_ref_ptr[a]:all_ref_ptr[a + 1]]
                refs_b = all_ref_pos[all_ref_ptr[b]:all_ref_ptr[b + 1]]
                if not np.array_equal(refs_a, refs_b):
                    raise RuntimeError(f"physical repeat reference mismatch: {key}")
        chosen.append(first)
    return np.asarray(chosen, dtype=np.int64)


def build_training_units(
    body: dict[str, np.ndarray],
    candidate_score: np.ndarray,
    winning_reference: np.ndarray,
    physical: np.ndarray,
) -> pd.DataFrame:
    pointer = body["candidate_ptr"].astype(np.int64)
    action_position = body["action_candidate_position"].astype(np.int64)
    rows = []
    for query in physical:
        left, right = int(pointer[query]), int(pointer[query + 1])
        truth = left
        negatives = np.arange(left + 1, right, dtype=np.int64)
        hardest = int(negatives[int(np.argmax(candidate_score[negatives]))])
        corrective = bool(body["direct_corrective"][query])
        known_safety = bool(body["direct_safety"][query])
        if corrective or known_safety:
            action = int(action_position[query])
            if action < left or action >= right or action == truth:
                raise RuntimeError(f"invalid B32 action candidate position: {query}")
            margin = float(candidate_score[truth] - candidate_score[action])
            rows.append({
                "kind": "corrective" if corrective else "known_safety",
                "query": int(query), "truth_candidate": truth, "negative_candidate": action,
                "positive_reference": int(winning_reference[truth]),
                "negative_reference": int(winning_reference[action]),
                "official_margin": margin,
                "identity": str(body["truth_candidate_id"][query]),
            })
        if int(body["baseline_rank_full_graph"][query]) == 1:
            rows.append({
                "kind": "broad_safety", "query": int(query),
                "truth_candidate": truth, "negative_candidate": hardest,
                "positive_reference": int(winning_reference[truth]),
                "negative_reference": int(winning_reference[hardest]),
                "official_margin": float(candidate_score[truth] - candidate_score[hardest]),
                "identity": str(body["truth_candidate_id"][query]),
            })
    output = pd.DataFrame(rows)
    expected = {
        "corrective": int(np.sum(body["direct_corrective"][physical].astype(bool))),
        "known_safety": int(np.sum(body["direct_safety"][physical].astype(bool))),
        "broad_safety": int(np.sum(body["baseline_rank_full_graph"][physical] == 1)),
    }
    observed = output["kind"].value_counts().to_dict()
    if any(int(observed.get(key, 0)) != value for key, value in expected.items()):
        raise RuntimeError(f"B33 training-unit counts changed: {observed} != {expected}")
    if np.any(output.loc[output["kind"].eq("corrective"), "official_margin"] >= 0):
        raise RuntimeError("B33 corrective action is not an official pairwise error")
    if np.any(output.loc[~output["kind"].eq("corrective"), "official_margin"] <= 0):
        raise RuntimeError("B33 safety action is not officially correct")
    return output


def pool_by_identity(units: pd.DataFrame, kind: str) -> dict[str, np.ndarray]:
    local = units.index[units["kind"].eq(kind)].to_numpy(np.int64)
    output: dict[str, list[int]] = {}
    for index in local:
        output.setdefault(str(units.at[index, "identity"]), []).append(int(index))
    return {key: np.asarray(value, dtype=np.int64) for key, value in output.items()}


def sample_pool(
    pool: dict[str, np.ndarray], count: int, rng: np.random.Generator,
) -> list[int]:
    identities = sorted(pool)
    if not identities:
        raise RuntimeError("empty B33 sampling pool")
    selected = []
    for _ in range(count):
        identity = identities[int(rng.integers(0, len(identities)))]
        values = pool[identity]
        selected.append(int(values[int(rng.integers(0, len(values)))]))
    return selected


def make_batch(
    units: pd.DataFrame,
    indices: list[int],
    body: dict[str, np.ndarray],
    official_query: np.ndarray,
    official_reference: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    spectra, target, layout = [], [], []
    reference_tensor = body["candidate_reference_tensor"]
    for index in indices:
        row = units.loc[int(index)]
        query = int(row["query"])
        positive = int(row["positive_reference"])
        negative = int(row["negative_reference"])
        q = len(spectra)
        spectra.append(torch.from_numpy(np.asarray(body["query_tensor"][query], np.float32)))
        target.append(torch.from_numpy(official_query[query]))
        p = len(spectra)
        spectra.append(torch.from_numpy(np.asarray(reference_tensor[positive], np.float32)))
        target.append(torch.from_numpy(official_reference[positive]))
        n = len(spectra)
        spectra.append(torch.from_numpy(np.asarray(reference_tensor[negative], np.float32)))
        target.append(torch.from_numpy(official_reference[negative]))
        layout.append({
            "kind": str(row["kind"]), "unit": int(index),
            "query": q, "positive": p, "negative": n,
        })
    return torch.stack(spectra), torch.stack(target), layout


def metric_summary(ranks: np.ndarray, positions: np.ndarray | None = None) -> dict:
    local = ranks if positions is None else ranks[positions]
    return {
        "queries": int(len(local)),
        "recall1": float(np.mean(local == 1)),
        "mrr": float(np.mean(1.0 / local.astype(float))),
        "errors": int(np.sum(local != 1)),
    }


def main() -> None:
    args = arguments()
    started = time.time()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite: {args.output}")
    positive = (
        args.steps, args.batch_corrective_identities, args.batch_known_safety_identities,
        args.batch_broad_safety_identities, args.backbone_lr, args.head_lr,
        args.temperature, args.lambda_corrective, args.lambda_known_safety,
        args.lambda_broad_safety, args.lambda_preserve, args.grad_clip,
    )
    if any(value <= 0 for value in positive) or args.head_lr < args.backbone_lr:
        raise ValueError("invalid B33 optimization configuration")
    graph_report, body = load_graph(args.graph_dir)
    for path in (args.official_checkpoint, args.architecture_checkpoint):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    torch.set_num_threads(args.torch_threads)
    seed_everything(args.seed)
    device = torch.device(args.device)
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("formal B33 bridge requires CUDA")
    query_tensor = torch.from_numpy(np.asarray(body["query_tensor"], dtype=np.float32))
    reference_tensor = torch.from_numpy(
        np.asarray(body["candidate_reference_tensor"], dtype=np.float32)
    )
    model, initialization = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint,
        device, args.n_highest_peaks,
    )
    model.eval()
    official_query = encode(
        model, query_tensor, device, args.encode_batch_size, "B33-official-query-fp32",
    )
    official_reference = encode(
        model, reference_tensor, device, args.encode_batch_size, "B33-official-reference-fp32",
    )
    official_score, official_winner = candidate_max_scores(
        official_query, official_reference, body,
    )
    official_rank, _ = graph_ranks(official_score, body)
    stored_score = body["candidate_official_score"].astype(np.float32)
    maximum_score_error = float(np.max(np.abs(official_score - stored_score)))
    rank_mismatches = int(np.sum(
        official_rank != body["baseline_rank_full_graph"].astype(np.int64)
    ))
    if maximum_score_error > 2e-6 or rank_mismatches:
        raise RuntimeError(
            f"B33 official full-graph replay failed: max_score_error={maximum_score_error} "
            f"rank_mismatches={rank_mismatches}"
        )
    physical = physical_positions(body)
    units = build_training_units(body, official_score, official_winner, physical)
    pools = {
        kind: pool_by_identity(units, kind)
        for kind in ("corrective", "known_safety", "broad_safety")
    }
    print(
        "[B33 training units] "
        + json.dumps(units["kind"].value_counts().to_dict(), sort_keys=True),
        flush=True,
    )

    capacity = unfreeze_last_blocks(model, args.unfreeze_blocks)
    model.eval()  # keep dropout disabled while preserving autograd
    head_parameters = [p for p in model.head.parameters() if p.requires_grad]
    head_ids = {id(p) for p in head_parameters}
    backbone_parameters = [
        p for p in model.parameters() if p.requires_grad and id(p) not in head_ids
    ]
    parameters = backbone_parameters + head_parameters
    optimizer = torch.optim.AdamW([
        {"params": backbone_parameters, "lr": args.backbone_lr, "weight_decay": 0.0},
        {"params": head_parameters, "lr": args.head_lr, "weight_decay": args.weight_decay},
    ])
    initial_backbone = [p.detach().cpu().clone() for p in backbone_parameters]
    initial_head = [p.detach().cpu().clone() for p in head_parameters]
    rng = np.random.default_rng(args.seed)
    history = []
    first_gradient_audit = None
    for step in range(args.steps):
        chosen = (
            sample_pool(pools["corrective"], args.batch_corrective_identities, rng)
            + sample_pool(pools["known_safety"], args.batch_known_safety_identities, rng)
            + sample_pool(pools["broad_safety"], args.batch_broad_safety_identities, rng)
        )
        chosen = [chosen[int(i)] for i in rng.permutation(len(chosen))]
        spectra, preservation_target, layout = make_batch(
            units, chosen, body, official_query, official_reference,
        )
        spectra = spectra.to(device)
        preservation_target = preservation_target.to(device)
        z = model(spectra)
        loss_by_kind: dict[str, list[torch.Tensor]] = {
            "corrective": [], "known_safety": [], "broad_safety": [],
        }
        margins = []
        for item in layout:
            margin = (
                (z[item["query"]] * z[item["positive"]]).sum()
                - (z[item["query"]] * z[item["negative"]]).sum()
            )
            margins.append(margin)
            row = units.loc[item["unit"]]
            if item["kind"] == "corrective":
                value = F.softplus((args.rank_margin - margin) / args.temperature)
            else:
                slack = (
                    args.known_safety_slack
                    if item["kind"] == "known_safety"
                    else args.broad_safety_slack
                )
                floor = float(row["official_margin"] - slack)
                value = F.relu(torch.as_tensor(floor, device=device, dtype=z.dtype) - margin)
            loss_by_kind[item["kind"]].append(value)
        losses = {key: torch.stack(value).mean() for key, value in loss_by_kind.items()}
        preserve_loss = (1.0 - torch.sum(z * preservation_target, dim=1)).mean()
        loss = (
            args.lambda_corrective * losses["corrective"]
            + args.lambda_known_safety * losses["known_safety"]
            + args.lambda_broad_safety * losses["broad_safety"]
            + args.lambda_preserve * preserve_loss
        )
        if first_gradient_audit is None:
            gradients = {
                key: component_gradient(value, parameters, True)
                for key, value in losses.items()
            }
            gradients["preserve"] = component_gradient(preserve_loss, parameters, True)
            split = len(backbone_parameters)
            first_gradient_audit = {
                key: gradient_summary(value) for key, value in gradients.items()
            }
            first_gradient_audit.update({
                "corrective_backbone": gradient_summary(gradients["corrective"][:split]),
                "corrective_head": gradient_summary(gradients["corrective"][split:]),
                "corrective_vs_known_safety": gradient_summary(
                    gradients["corrective"], gradients["known_safety"],
                ),
                "corrective_vs_broad_safety": gradient_summary(
                    gradients["corrective"], gradients["broad_safety"],
                ),
                "corrective_vs_preserve": gradient_summary(
                    gradients["corrective"], gradients["preserve"],
                ),
            })
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(
            p.grad is not None and not bool(torch.isfinite(p.grad).all().item())
            for p in parameters
        ):
            raise RuntimeError(f"non-finite B33 gradient at step {step + 1}")
        raw_norm = float(torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip))
        if not np.isfinite(raw_norm):
            raise RuntimeError(f"non-finite B33 gradient norm at step {step + 1}")
        optimizer.step()
        clip_scale = min(1.0, args.grad_clip / max(raw_norm, 1e-12))
        record = {
            "step": step + 1, "loss": float(loss.detach()),
            "corrective": float(losses["corrective"].detach()),
            "known_safety": float(losses["known_safety"].detach()),
            "broad_safety": float(losses["broad_safety"].detach()),
            "preserve": float(preserve_loss.detach()),
            "mean_margin": float(torch.stack(margins).mean().detach()),
            "raw_grad_norm": raw_norm, "clip_scale": float(clip_scale),
            "clipped": bool(raw_norm > args.grad_clip),
        }
        history.append(record)
        if step == 0 or (step + 1) % 16 == 0 or step + 1 == args.steps:
            print(f"[B33 step {step + 1}/{args.steps}] {record}", flush=True)

    post_query = encode(
        model, query_tensor, device, args.encode_batch_size, "B33-adapted-query-fp32",
    )
    post_reference = encode(
        model, reference_tensor, device, args.encode_batch_size, "B33-adapted-reference-fp32",
    )
    post_score, _ = candidate_max_scores(post_query, post_reference, body)
    post_rank, _ = graph_ranks(post_score, body)
    baseline_correct = official_rank == 1
    post_correct = post_rank == 1
    corrected_mask = ~baseline_correct & post_correct
    introduced_mask = baseline_correct & ~post_correct
    physical_corrected = corrected_mask[physical]
    physical_introduced = introduced_mask[physical]

    action_margin_pre, action_margin_post, action_kind = [], [], []
    pointer = body["candidate_ptr"].astype(np.int64)
    for row in units.loc[units["kind"].isin(["corrective", "known_safety"])].itertuples():
        q = int(row.query)
        truth, negative = int(row.truth_candidate), int(row.negative_candidate)
        action_margin_pre.append(float(official_score[truth] - official_score[negative]))
        action_margin_post.append(float(post_score[truth] - post_score[negative]))
        action_kind.append(str(row.kind))
    action_margin_pre = np.asarray(action_margin_pre, np.float32)
    action_margin_post = np.asarray(action_margin_post, np.float32)
    action_kind = np.asarray(action_kind, str)
    corrective_action = action_kind == "corrective"
    safety_action = action_kind == "known_safety"

    query_preservation = np.sum(official_query * post_query, axis=1)
    reference_preservation = np.sum(official_reference * post_reference, axis=1)
    backbone_delta = math.sqrt(sum(
        float(torch.sum((p.detach().cpu() - initial) ** 2))
        for p, initial in zip(backbone_parameters, initial_backbone, strict=True)
    ))
    head_delta = math.sqrt(sum(
        float(torch.sum((p.detach().cpu() - initial) ** 2))
        for p, initial in zip(head_parameters, initial_head, strict=True)
    ))
    official_all = metric_summary(official_rank)
    adapted_all = metric_summary(post_rank)
    official_physical = metric_summary(official_rank, physical)
    adapted_physical = metric_summary(post_rank, physical)
    row_corrected = int(corrected_mask.sum())
    row_introduced = int(introduced_mask.sum())
    physical_corrected_n = int(physical_corrected.sum())
    physical_introduced_n = int(physical_introduced.sum())
    gates = {
        "official_full_graph_replay_exact": rank_mismatches == 0 and maximum_score_error <= 2e-6,
        "shared_embedding_row_recall1_positive": bool(
            adapted_all["recall1"] > official_all["recall1"]
        ),
        "shared_embedding_physical_recall1_positive": bool(
            adapted_physical["recall1"] > official_physical["recall1"]
        ),
        "row_corrected_minus_2x_introduced_positive": row_corrected - 2 * row_introduced > 0,
        "physical_corrected_minus_2x_introduced_positive": (
            physical_corrected_n - 2 * physical_introduced_n > 0
        ),
        "row_mrr_nonnegative": adapted_all["mrr"] >= official_all["mrr"],
        "known_harm_introduced_le_1": bool(np.sum(action_margin_post[safety_action] <= 0) <= 1),
        "corrective_actions_realised_ge_20": bool(
            np.sum(action_margin_post[corrective_action] > 0) >= 20
        ),
        "minimum_query_preservation_ge_0_98": bool(np.min(query_preservation) >= 0.98),
        "minimum_reference_preservation_ge_0_98": bool(np.min(reference_preservation) >= 0.98),
    }
    result = {
        "status": "bioaware_b33_full_graph_bridge_complete",
        "formal": True,
        "scope": "opened full-candidate engineering bridge; not held-out performance",
        "official_replay": {
            "maximum_candidate_score_error": maximum_score_error,
            "rank_mismatches": rank_mismatches,
        },
        "training_units": {
            key: int(value) for key, value in units["kind"].value_counts().to_dict().items()
        },
        "row_weighted": {
            "official": official_all, "adapted": adapted_all,
            "delta_recall1": float(adapted_all["recall1"] - official_all["recall1"]),
            "delta_mrr": float(adapted_all["mrr"] - official_all["mrr"]),
            "corrected": row_corrected, "introduced": row_introduced,
            "risk_net_lambda2": int(row_corrected - 2 * row_introduced),
        },
        "physical_query_weighted": {
            "official": official_physical, "adapted": adapted_physical,
            "delta_recall1": float(
                adapted_physical["recall1"] - official_physical["recall1"]
            ),
            "delta_mrr": float(adapted_physical["mrr"] - official_physical["mrr"]),
            "corrected": physical_corrected_n, "introduced": physical_introduced_n,
            "risk_net_lambda2": int(physical_corrected_n - 2 * physical_introduced_n),
        },
        "supervised_action_bridge": {
            "corrective_actions": int(np.sum(corrective_action)),
            "corrective_realised": int(np.sum(action_margin_post[corrective_action] > 0)),
            "corrective_mean_margin_before": float(np.mean(action_margin_pre[corrective_action])),
            "corrective_mean_margin_after": float(np.mean(action_margin_post[corrective_action])),
            "known_safety_actions": int(np.sum(safety_action)),
            "known_safety_introduced": int(np.sum(action_margin_post[safety_action] <= 0)),
        },
        "preservation": {
            "query_mean": float(np.mean(query_preservation)),
            "query_minimum": float(np.min(query_preservation)),
            "reference_mean": float(np.mean(reference_preservation)),
            "reference_minimum": float(np.min(reference_preservation)),
            "backbone_parameter_l2_delta": backbone_delta,
            "head_parameter_l2_delta": head_delta,
        },
        "gradient_audit": {
            "first_step": first_gradient_audit,
            "clipped_steps": int(sum(record["clipped"] for record in history)),
            "median_clip_scale": float(np.median([record["clip_scale"] for record in history])),
            "interpretation": (
                "global norm clipping rescales the complete update and does not erase its "
                "direction; it is reported because persistent clipping suppresses magnitude "
                "differences and makes the effective step controlled by the fixed clip bound"
            ),
        },
        "capacity": capacity,
        "history": history,
        "gates": gates,
        "pass_to_formula_isolated_training": bool(all(gates.values())),
        "contracts": {
            "query_reference_encoder_shared": True,
            "candidate_max_recomputed_over_all_reference_spectra": True,
            "all_official_correct_physical_queries_supply_boundary_safety": True,
            "physical_query_duplicates_collapsed_for_training": True,
            "dropout_disabled": True,
            "P2b_used": False,
            "phenotype_used": False,
        },
        "parameters": vars(args) | {
            "graph_dir": str(args.graph_dir),
            "official_checkpoint": str(args.official_checkpoint),
            "architecture_checkpoint": str(args.architecture_checkpoint),
            "output": str(args.output),
        },
        "provenance": {
            "initialization": initialization,
            "graph_sha256": sha256_file(args.graph_dir / "full_candidate_graph.npz"),
            "graph_report_sha256": sha256_file(args.graph_dir / "report.json"),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "architecture_checkpoint_sha256": sha256_file(args.architecture_checkpoint),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "elapsed_seconds": float(time.time() - started),
        "claim_limit": (
            "B33 tests whether the direct B32 action survives complete opened-candidate "
            "retrieval with broad safety. It does not establish held-out, formula-isolated, "
            "external, blind, or SOTA improvement."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "report.json", result)
    pd.DataFrame({
        "query_id": body["query_id"].astype(str),
        "source": body["source"].astype(str),
        "physical_query_id": body["physical_query_id"].astype(str),
        "truth_candidate_id": body["truth_candidate_id"].astype(str),
        "truth_formula": body["truth_formula"].astype(str),
        "official_rank": official_rank, "adapted_rank": post_rank,
        "corrected": corrected_mask, "introduced": introduced_mask,
        "direct_corrective": body["direct_corrective"].astype(bool),
        "direct_safety": body["direct_safety"].astype(bool),
    }).to_csv(args.output / "per_query.csv.gz", index=False, compression="gzip")
    torch.save({
        "state_dict": model.state_dict(),
        "initialization": initialization,
        "graph_sha256": result["provenance"]["graph_sha256"],
        "parameters": result["parameters"],
        "scope": result["scope"],
    }, args.output / "checkpoint.pt")
    if not result["pass_to_formula_isolated_training"]:
        raise RuntimeError(f"B33 full-graph bridge gate failed: {gates}")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
