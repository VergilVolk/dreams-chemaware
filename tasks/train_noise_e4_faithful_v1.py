"""E4-A: direct peak-noise fine-tuning of the shared DreaMS embedding.

This stage deliberately has no teacher, action selector, P2b score or
post-embedding reranker.  Two fixed, previously frozen S3A policies are used
exactly like image augmentations:

* candidate_gradient, attenuation 0.50, terminal step 6;
* role_confounder, attenuation 1.00, terminal step 5.

For a training query the clean and perturbed spectra are both encoded by the
same trainable DreaMS model.  Positive and negative reference spectra are also
encoded by that model.  The objective combines clean groupwise ranking,
perturbed-view groupwise ranking, clean/perturbed consistency, an official
margin floor and clean-embedding preservation.  At inference the saved model
receives only an ordinary clean spectrum and emits one new embedding.

Formula folds are fixed before training.  The held formula fold is evaluated
once after a fixed epoch count and is never used for checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from noise_final_core import (  # noqa: E402
    CandidateGraph, json_dump, load_embedding_cache, seed_everything,
    sha256_file, stable_fold, strict_rank,
)
from train_e1_identity import load_base_model  # noqa: E402
from train_noise_final_r2_shared_encoder import (  # noqa: E402
    SpectrumStore, encode_rows, evaluate_embeddings, formula_bootstrap_delta,
    forward_embeddings, margins, parse_path, representatives,
)
from noise_v3_core import attenuate_sequence  # noqa: E402


FIXED_POLICY = {
    "candidate": (("candidate_gradient", 0.50, 6),),
    "confounder": (("role_confounder", 1.00, 5),),
    "combined": (
        ("candidate_gradient", 0.50, 6),
        ("role_confounder", 1.00, 5),
    ),
    # Every cell below already exists in the frozen, outcome-free R0 table.
    # This is a dose curriculum, not per-query post-outcome action selection.
    "curriculum": (
        ("candidate_gradient", 0.50, 3),
        ("candidate_gradient", 0.50, 4),
        ("candidate_gradient", 0.50, 5),
        ("candidate_gradient", 0.50, 6),
        ("role_confounder", 1.00, 1),
        ("role_confounder", 1.00, 2),
        ("role_confounder", 1.00, 3),
        ("role_confounder", 1.00, 4),
        ("role_confounder", 1.00, 5),
    ),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--r0-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_r0_faithful_s3a")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--embedding-cache", type=Path, default=ROOT / "data/validation/g8r_p2_official_embeddings.npz")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/validation/g8r_noise_final_e4a_direct")
    parser.add_argument("--policy", choices=tuple(FIXED_POLICY), default="candidate")
    parser.add_argument("--action-scope", choices=("errors", "all"), default="errors")
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-actions", type=int, default=4)
    parser.add_argument("--views-per-identity", type=int, default=2)
    parser.add_argument("--positive-spectra", type=int, default=4)
    parser.add_argument("--negative-molecules", type=int, default=8)
    parser.add_argument("--unfreeze-blocks", type=int, default=1)
    parser.add_argument("--head-lr", type=float, default=5e-6)
    parser.add_argument("--backbone-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--lambda-clean-rank", type=float, default=1.0)
    parser.add_argument("--lambda-aug-rank", type=float, default=1.0)
    parser.add_argument("--lambda-consistency", type=float, default=0.25)
    parser.add_argument("--lambda-margin-floor", type=float, default=2.0)
    parser.add_argument("--lambda-preserve", type=float, default=5.0)
    parser.add_argument("--margin-floor-slack", type=float, default=0.005)
    parser.add_argument("--safety-ratio", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--run-suffix", default="",
        help="Optional filesystem-safe suffix for preregistered optimizer scans.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


@dataclass(frozen=True)
class DirectExample:
    query_index: int
    query_row: int
    identity: str
    formula: str
    positive_rows: tuple[int, ...]
    negative_rows: tuple[int, ...]
    official_margin: float
    official_rank: int
    sample_weight: float
    policy: str = "clean_safety"
    target_path: tuple[int, ...] = ()
    attenuation: float = 0.0


def official_rank_margin(graph: CandidateGraph) -> tuple[np.ndarray, np.ndarray]:
    score = graph.features[:, graph.dreams_column]
    molecule_score = np.maximum.reduceat(score, graph.molecule_ptr[:-1])
    rank = np.empty(graph.n_queries, dtype=np.int16)
    margin = np.empty(graph.n_queries, dtype=np.float32)
    for query in range(graph.n_queries):
        left, right = map(int, graph.query_ptr[query:query + 2])
        values = molecule_score[left:right]
        if len(values) < 2 or not np.all(np.isfinite(values)):
            raise RuntimeError(f"invalid frozen candidate scores for query {query}")
        rank[query] = strict_rank(values)
        margin[query] = float(values[0] - np.max(values[1:]))
    return rank, margin


def unfreeze_last_blocks(model, blocks: int) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.head.parameters():
        parameter.requires_grad = True
    encoder = model.backbone.transformer_encoder
    if blocks < 1 or blocks > int(encoder.n_layers):
        raise ValueError(f"unfreeze-blocks must be in 1..{int(encoder.n_layers)}")
    layers = list(range(int(encoder.n_layers) - blocks, int(encoder.n_layers)))
    count = 0
    for layer in layers:
        for module in (encoder.atts[layer], encoder.ffs[layer],
                       encoder.scales[2 * layer], encoder.scales[2 * layer + 1]):
            for parameter in module.parameters():
                if not parameter.requires_grad:
                    parameter.requires_grad = True
                    count += parameter.numel()
    if getattr(encoder, "pre_norm", False):
        for parameter in encoder.scales[-1].parameters():
            if not parameter.requires_grad:
                parameter.requires_grad = True
                count += parameter.numel()
    return {
        "transformer_layers": int(encoder.n_layers),
        "unfrozen_layers": layers,
        "unfrozen_backbone_parameters": int(count),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
    }


def make_examples(graph: CandidateGraph, frame: pd.DataFrame, rank: np.ndarray,
                  margin: np.ndarray, positives: int, negatives: int,
                  action: bool) -> list[DirectExample]:
    if frame.empty:
        return []
    output: list[DirectExample] = []
    for row in frame.itertuples(index=False):
        query = int(row.query_index)
        forced = int(row.hard_negative_row) if action and hasattr(row, "hard_negative_row") else None
        positive_rows, negative_rows = representatives(
            graph, query, positives, negatives, forced,
        )
        identity = str(row.query_ik14)
        output.append(DirectExample(
            query_index=query,
            query_row=int(row.query_row),
            identity=identity,
            formula=str(row.query_formula),
            positive_rows=positive_rows,
            negative_rows=negative_rows,
            official_margin=float(margin[query]),
            official_rank=int(rank[query]),
            # Exact identity balance is imposed by identity_balanced_epoch.
            sample_weight=1.0,
            policy=(
                f"{row.selector}|step={int(row.step)}" if action
                else "clean_safety"
            ),
            target_path=parse_path(row.target_path) if action else (),
            attenuation=float(row.attenuation) if action else 0.0,
        ))
    return output


def flatten_direct(store: SpectrumStore, examples: list[DirectExample], action: bool):
    tensors: list[torch.Tensor] = []
    clean_rows: list[int] = []
    layout: list[dict] = []
    for example in examples:
        item: dict[str, object] = {"clean": len(tensors)}
        tensors.append(store.one(example.query_row))
        clean_rows.append(example.query_row)
        if action:
            item["action"] = len(tensors)
            tensors.append(attenuate_sequence(
                store.one(example.query_row), example.target_path, example.attenuation,
            ))
        item["positive"] = list(range(len(tensors), len(tensors) + len(example.positive_rows)))
        tensors.extend(store.get(example.positive_rows))
        clean_rows.extend(example.positive_rows)
        item["negative"] = list(range(len(tensors), len(tensors) + len(example.negative_rows)))
        tensors.extend(store.get(example.negative_rows))
        clean_rows.extend(example.negative_rows)
        layout.append(item)
    return torch.stack(tensors), layout, clean_rows


def weighted_mean(values: torch.Tensor, examples: list[DirectExample]) -> torch.Tensor:
    weights = torch.tensor(
        [example.sample_weight for example in examples],
        device=values.device, dtype=values.dtype,
    )
    return torch.sum(values * weights) / torch.sum(weights)


def gradient_l2_norm(parameters: list[torch.nn.Parameter]) -> float:
    """Return the pre-clipping L2 norm without modifying gradients."""
    squared = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            value = parameter.grad.detach().float().norm(2).item()
            squared += value * value
    return math.sqrt(squared)


def direct_action_loss(model, store: SpectrumStore, examples: list[DirectExample],
                       official_by_row: dict[int, np.ndarray], device: torch.device,
                       args) -> tuple[torch.Tensor, dict[str, float]]:
    spectra, layout, clean_rows = flatten_direct(store, examples, True)
    encoded = forward_embeddings(model, spectra.to(device), args.amp)
    clean_margin = margins(encoded, layout, "clean")
    aug_margin = margins(encoded, layout, "action")
    clean_rank_each = F.softplus((args.rank_margin - clean_margin) / args.temperature)
    aug_rank_each = F.softplus((args.rank_margin - aug_margin) / args.temperature)
    clean_z = torch.stack([encoded[int(item["clean"])] for item in layout])
    aug_z = torch.stack([encoded[int(item["action"])] for item in layout])
    consistency_each = 1.0 - torch.sum(clean_z * aug_z, dim=1)
    floors = torch.tensor(
        [example.official_margin - args.margin_floor_slack for example in examples],
        device=device, dtype=clean_margin.dtype,
    )
    floor_each = F.relu(floors - clean_margin)

    clean_indices: list[int] = []
    for item in layout:
        clean_indices.append(int(item["clean"]))
        clean_indices.extend(item["positive"])
        clean_indices.extend(item["negative"])
    official = torch.from_numpy(np.stack([official_by_row[row] for row in clean_rows])).to(device)
    preserve = (1.0 - torch.sum(encoded[clean_indices] * official, dim=1)).mean()
    clean_rank = weighted_mean(clean_rank_each, examples)
    aug_rank = weighted_mean(aug_rank_each, examples)
    consistency = weighted_mean(consistency_each, examples)
    floor = weighted_mean(floor_each, examples)
    loss = (
        args.lambda_clean_rank * clean_rank
        + args.lambda_aug_rank * aug_rank
        + args.lambda_consistency * consistency
        + args.lambda_margin_floor * floor
        + args.lambda_preserve * preserve
    )
    return loss, {
        "action_clean_rank": float(clean_rank.detach()),
        "action_aug_rank": float(aug_rank.detach()),
        "action_consistency": float(consistency.detach()),
        "action_margin_floor": float(floor.detach()),
        "action_preserve": float(preserve.detach()),
        "action_clean_margin": float(clean_margin.mean().detach()),
        "action_aug_margin": float(aug_margin.mean().detach()),
        "action_clean_margin_pass": float((clean_margin > 0).float().mean().detach()),
        "action_aug_margin_pass": float((aug_margin > 0).float().mean().detach()),
    }


def safety_loss(model, store: SpectrumStore, examples: list[DirectExample],
                official_by_row: dict[int, np.ndarray], device: torch.device,
                args) -> tuple[torch.Tensor, dict[str, float]]:
    spectra, layout, clean_rows = flatten_direct(store, examples, False)
    encoded = forward_embeddings(model, spectra.to(device), args.amp)
    clean_margin = margins(encoded, layout, "clean")
    floors = torch.tensor(
        [example.official_margin - args.margin_floor_slack for example in examples],
        device=device, dtype=clean_margin.dtype,
    )
    floor = weighted_mean(F.relu(floors - clean_margin), examples)
    clean_indices: list[int] = []
    for item in layout:
        clean_indices.append(int(item["clean"]))
        clean_indices.extend(item["positive"])
        clean_indices.extend(item["negative"])
    official = torch.from_numpy(np.stack([official_by_row[row] for row in clean_rows])).to(device)
    preserve = (1.0 - torch.sum(encoded[clean_indices] * official, dim=1)).mean()
    loss = args.lambda_margin_floor * floor + args.lambda_preserve * preserve
    return loss, {
        "safety_margin_floor": float(floor.detach()),
        "safety_preserve": float(preserve.detach()),
        "safety_margin": float(clean_margin.mean().detach()),
    }


def batched(values: list[DirectExample], size: int):
    for left in range(0, len(values), size):
        yield values[left:left + size]


def identity_balanced_epoch(examples: list[DirectExample], rng: np.random.Generator,
                            views_per_identity: int) -> list[DirectExample]:
    """Draw exactly K views per identity and round-robin available policies."""
    if views_per_identity < 1:
        raise ValueError("views-per-identity must be positive")
    groups: dict[str, list[DirectExample]] = {}
    for example in examples:
        groups.setdefault(example.identity, []).append(example)
    output: list[DirectExample] = []
    for identity in sorted(groups):
        values = groups[identity]
        by_policy: dict[str, list[DirectExample]] = {}
        for value in values:
            by_policy.setdefault(value.policy, []).append(value)
        policies = sorted(by_policy)
        policy_order = np.asarray(policies, dtype=object)[rng.permutation(len(policies))]
        local_orders = {
            policy: rng.permutation(len(by_policy[policy])) for policy in policies
        }
        for offset in range(views_per_identity):
            policy = str(policy_order[offset % len(policy_order)])
            order = local_orders[policy]
            local = int(order[(offset // len(policy_order)) % len(order)])
            output.append(by_policy[policy][local])
    rng.shuffle(output)
    return output


def main() -> None:
    args = arguments()
    if args.outer_fold not in range(5):
        raise ValueError("outer-fold must be 0..4")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("E4-A direct augmentation requires CUDA")
    if args.head_lr < args.backbone_lr or args.backbone_lr <= 0:
        raise ValueError("require head-lr >= backbone-lr > 0")
    seed_everything(args.seed)
    device = torch.device(args.device)
    required = [
        args.graph, args.data, args.embedding_cache, args.official_checkpoint,
        args.architecture_checkpoint, args.r0_dir / "report.json",
        args.r0_dir / "training_actions.csv.gz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    r0 = json.loads((args.r0_dir / "report.json").read_text(encoding="utf-8"))
    if not r0.get("formal") or r0.get("contracts", {}).get("P2b") != "forbidden":
        raise RuntimeError("E4-A requires the formal, P2b-free R0 manifest")
    if not r0.get("contracts", {}).get("action_outcomes_absent_from_training_manifest"):
        raise RuntimeError("R0 does not certify outcome-free training actions")

    if args.run_suffix and not all(
        character.isalnum() or character in "-_" for character in args.run_suffix
    ):
        raise ValueError("run-suffix may contain only letters, digits, '-' and '_'")
    tag = (
        f"{args.policy}_{args.action_scope}_views{args.views_per_identity}_blocks{args.unfreeze_blocks}_"
        f"blr_{args.backbone_lr:.0e}_hlr_{args.head_lr:.0e}"
    )
    if args.run_suffix:
        tag += f"_{args.run_suffix}"
    output = args.output_root / tag / f"seed_{args.seed}" / f"fold_{args.outer_fold}"
    if output.exists():
        raise RuntimeError(f"refusing to overwrite E4-A result: {output}")

    graph = CandidateGraph(args.graph)
    official_rank, official_margin = official_rank_margin(graph)
    actions = pd.read_csv(args.r0_dir / "training_actions.csv.gz")
    forbidden_columns = {"corrected", "introduced", "target_rank", "target_margin", "random_margin"}
    leaked = forbidden_columns.intersection(actions.columns)
    if leaked:
        raise RuntimeError(f"post-outcome columns leaked into direct training: {sorted(leaked)}")
    selected = []
    for selector, attenuation, step in FIXED_POLICY[args.policy]:
        block = actions.loc[
            actions["selector"].astype(str).eq(selector)
            & np.isclose(actions["attenuation"].astype(float), attenuation)
            & actions["step"].astype(int).eq(step)
        ].copy()
        if block.empty:
            raise RuntimeError(f"missing fixed policy cell {selector}|{attenuation}|{step}")
        selected.append(block)
    actions = pd.concat(selected, ignore_index=True)
    actions["baseline_rank"] = official_rank[actions["query_index"].to_numpy(np.int64)]
    if args.action_scope == "errors":
        actions = actions.loc[actions["baseline_rank"].astype(int).ne(1)].copy()
    train_actions = actions.loc[actions["formula_fold"].astype(int).ne(args.outer_fold)].copy()
    held_action = actions.loc[actions["formula_fold"].astype(int).eq(args.outer_fold)].copy()
    held_formulas = set(
        actions.loc[actions["formula_fold"].astype(int).eq(args.outer_fold), "query_formula"].astype(str)
    )
    if train_actions["query_formula"].astype(str).isin(held_formulas).any():
        raise RuntimeError("formula isolation failed in action manifest")

    # Full clean ledger is reconstructed from frozen graph labels/scores.  It
    # is not a teacher: it only provides ground-truth ranking and safety replay.
    formula_fold_by_query = {
        query: stable_fold(str(formula), 5, args.formula_fold_seed)
        for query, formula in enumerate(graph.query_formula)
    }
    # The independently recomputed split must reproduce every R0 action row.
    observed_fold = actions["query_index"].astype(int).map(formula_fold_by_query).to_numpy(np.int8)
    if not np.array_equal(observed_fold, actions["formula_fold"].to_numpy(np.int8)):
        raise RuntimeError("formula-fold reconstruction does not reproduce frozen R0")
    held_queries = np.asarray(
        [query for query, fold in formula_fold_by_query.items() if fold == args.outer_fold],
        dtype=np.int64,
    )
    safety_queries = np.asarray([
        query for query, fold in formula_fold_by_query.items()
        if fold != args.outer_fold and official_rank[query] == 1
    ], dtype=np.int64)
    safety_frame = pd.DataFrame({
        "query_index": safety_queries,
        "query_row": graph.query_row[safety_queries],
        "query_ik14": graph.query_ik14[safety_queries],
        "query_formula": graph.query_formula[safety_queries],
    })

    action_examples = make_examples(
        graph, train_actions, official_rank, official_margin,
        args.positive_spectra, args.negative_molecules, True,
    )
    safety_examples = make_examples(
        graph, safety_frame, official_rank, official_margin,
        args.positive_spectra, args.negative_molecules, False,
    )
    if len(action_examples) < 100 or len(set(x.identity for x in action_examples)) < 100:
        raise RuntimeError("direct action training pool is unexpectedly small")

    reachable_rows = np.unique(np.concatenate((graph.query_row, graph.pair_candidate_row))).astype(np.int64)
    store = SpectrumStore(args.data, reachable_rows, args.n_highest_peaks)
    _, cache_embeddings, cache_index = load_embedding_cache(args.embedding_cache)
    if set(map(int, reachable_rows)) - set(cache_index):
        raise RuntimeError("official embedding cache does not cover candidate graph")
    official_by_row = {
        int(row): cache_embeddings[index]
        for row, index in cache_index.items() if int(row) in store.position
    }
    official_encoded = np.stack([official_by_row[int(row)] for row in reachable_rows])

    model, initialization = load_base_model(
        args.official_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks,
    )
    capacity = unfreeze_last_blocks(model, args.unfreeze_blocks)
    # Gradients stay on, stochastic dropout stays off.
    model.eval()
    head_parameters = [parameter for parameter in model.head.parameters() if parameter.requires_grad]
    backbone_parameters = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": head_parameters, "lr": args.head_lr, "weight_decay": args.weight_decay},
        {"params": backbone_parameters, "lr": args.backbone_lr, "weight_decay": 0.0},
    ])
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    initial_encoded = encode_rows(
        model, store, reachable_rows, device, args.eval_batch_size, False, "E4A-init-fp32",
    )
    initial_cosine = np.einsum("ij,ij->i", initial_encoded, official_encoded)
    baseline_rank, baseline_summary = evaluate_embeddings(
        graph, reachable_rows, official_encoded, held_queries,
    )
    initial_rank, _ = evaluate_embeddings(graph, reachable_rows, initial_encoded, held_queries)
    initial_mismatches = int(np.sum(initial_rank != baseline_rank))
    if float(np.mean(initial_cosine)) < 0.9999 or initial_mismatches / len(held_queries) > 0.001:
        raise RuntimeError(
            f"zero-change gate failed: cos={np.mean(initial_cosine):.7f}, "
            f"rank mismatches={initial_mismatches}/{len(held_queries)}"
        )
    del initial_encoded
    if device.type == "cuda":
        torch.cuda.empty_cache()

    rng = np.random.default_rng(args.seed)
    history = []
    epochs = 1 if args.smoke else args.epochs
    for epoch in range(1, epochs + 1):
        model.eval()
        epoch_actions = identity_balanced_epoch(
            action_examples, rng, args.views_per_identity,
        )
        safety_epoch = identity_balanced_epoch(safety_examples, rng, 1)
        if args.smoke:
            epoch_actions = epoch_actions[:16]
            safety_epoch = safety_epoch[:32]
        steps = math.ceil(len(epoch_actions) / args.batch_actions)
        totals: dict[str, float] = {}
        started = time.time()
        safety_cursor = 0
        for step, action_batch in enumerate(batched(epoch_actions, args.batch_actions), start=1):
            safety_size = max(1, int(round(len(action_batch) * args.safety_ratio)))
            if safety_cursor + safety_size > len(safety_epoch):
                rng.shuffle(safety_epoch)
                safety_cursor = 0
            safe_batch = safety_epoch[safety_cursor:safety_cursor + safety_size]
            safety_cursor += safety_size
            optimizer.zero_grad(set_to_none=True)
            action_loss, action_log = direct_action_loss(
                model, store, action_batch, official_by_row, device, args,
            )
            scaler.scale(action_loss).backward()
            safe_loss, safe_log = safety_loss(
                model, store, safe_batch, official_by_row, device, args,
            )
            scaler.scale(safe_loss).backward()
            scaler.unscale_(optimizer)
            head_grad_norm = gradient_l2_norm(head_parameters)
            backbone_grad_norm = gradient_l2_norm(backbone_parameters)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.grad_clip,
            )
            grad_norm_value = float(grad_norm)
            clip_applied = float(grad_norm_value > args.grad_clip)
            clip_scale = min(1.0, args.grad_clip / max(grad_norm_value, 1e-12))
            scaler.step(optimizer)
            scaler.update()
            log = {
                "loss": float(action_loss.detach()) + float(safe_loss.detach()),
                "gradient_norm": grad_norm_value,
                "head_gradient_norm": head_grad_norm,
                "backbone_gradient_norm": backbone_grad_norm,
                "gradient_clip_applied": clip_applied,
                "gradient_clip_scale": clip_scale,
                **action_log, **safe_log,
            }
            for key, value in log.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            if step % 25 == 0 or step == steps:
                print(
                    f"[E4A epoch={epoch}] {step}/{steps} "
                    f"loss={totals['loss']/step:.5f} grad={totals['gradient_norm']/step:.4f}",
                    flush=True,
                )
        record = {key: value / steps for key, value in totals.items()}
        record.update({"epoch": epoch, "steps": steps, "seconds": time.time() - started})
        history.append(record)
        print(json.dumps(record, indent=2), flush=True)

    final_encoded = encode_rows(
        model, store, reachable_rows, device, args.eval_batch_size, False, "E4A-final-fp32",
    )
    final_rank, final_summary = evaluate_embeddings(
        graph, reachable_rows, final_encoded, held_queries,
    )
    final_preservation = np.einsum("ij,ij->i", final_encoded, official_encoded)
    old_correct, new_correct = baseline_rank == 1, final_rank == 1
    delta_ci = formula_bootstrap_delta(
        baseline_rank, final_rank, graph.query_formula[held_queries],
        args.bootstrap_resamples, args.seed,
    )
    final_summary.update({
        "baseline_recall1": baseline_summary["recall1"],
        "delta_recall1": float(final_summary["recall1"] - baseline_summary["recall1"]),
        "baseline_mrr": baseline_summary["mrr"],
        "delta_mrr": float(final_summary["mrr"] - baseline_summary["mrr"]),
        "baseline_near_recall1": baseline_summary["near_recall1"],
        "delta_near_recall1": float(final_summary["near_recall1"] - baseline_summary["near_recall1"]),
        "corrected": int(np.sum(~old_correct & new_correct)),
        "introduced": int(np.sum(old_correct & ~new_correct)),
        "risk_net": int(np.sum(~old_correct & new_correct) - 2 * np.sum(old_correct & ~new_correct)),
        "preservation_mean": float(np.mean(final_preservation)),
        "preservation_p01": float(np.quantile(final_preservation, 0.01)),
        "formula_cluster_delta_recall1": delta_ci,
    })
    held_action_query = np.unique(held_action["query_index"].to_numpy(np.int64))
    held_action_mask = np.isin(held_queries, held_action_query)
    if np.any(held_action_mask):
        final_summary["held_action_clean"] = {
            "queries": int(np.sum(held_action_mask)),
            "baseline_accuracy": float(np.mean(baseline_rank[held_action_mask] == 1)),
            "student_accuracy": float(np.mean(final_rank[held_action_mask] == 1)),
            "corrected": int(np.sum((baseline_rank[held_action_mask] != 1) & (final_rank[held_action_mask] == 1))),
            "introduced": int(np.sum((baseline_rank[held_action_mask] == 1) & (final_rank[held_action_mask] != 1))),
        }

    output.mkdir(parents=True, exist_ok=False)
    checkpoint = {
        "status": "noise_final_e4a_direct_shared_dreams_encoder",
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "initialization": initialization,
        "policy": args.policy,
        "action_scope": args.action_scope,
        "seed": args.seed,
        "outer_fold": args.outer_fold,
        "capacity": capacity,
        "inference_clean_only": True,
        "P2b_used": False,
        "teacher_used": False,
    }
    torch.save(checkpoint, output / "final_shared_encoder.pt")
    gates = {
        "clean_recall_positive": bool(final_summary["delta_recall1"] > 0),
        "formula_ci_positive": bool(delta_ci["ci_low"] > 0),
        "corrected_gt_introduced": bool(final_summary["corrected"] > final_summary["introduced"]),
        "risk_net_positive": bool(final_summary["risk_net"] > 0),
        "near_nonnegative": bool(final_summary["delta_near_recall1"] >= 0),
        "mrr_nonnegative": bool(final_summary["delta_mrr"] >= 0),
        "preservation_ge_0_995": bool(final_summary["preservation_mean"] >= 0.995),
    }
    decision = {
        "status": "noise_final_e4a_direct_augmentation_complete",
        "formal": not args.smoke,
        "configuration": vars(args) | {"r0_fixed_cells": FIXED_POLICY[args.policy]},
        "capacity": capacity,
        "data": {
            "train_action_rows": len(train_actions),
            "train_action_identities": int(train_actions["query_ik14"].nunique()),
            "train_action_formulas": int(train_actions["query_formula"].nunique()),
            "train_action_baseline_errors": int(np.sum(train_actions["baseline_rank"].astype(int).ne(1))),
            "train_action_baseline_correct": int(np.sum(train_actions["baseline_rank"].astype(int).eq(1))),
            "train_action_cells": (
                train_actions.groupby(["selector", "attenuation", "step"])
                .size().rename("rows").reset_index().to_dict("records")
            ),
            "held_action_rows": len(held_action),
            "held_queries": len(held_queries),
            "safety_queries": len(safety_examples),
        },
        "zero_change_gate": {
            "preservation_mean": float(np.mean(initial_cosine)),
            "rank_mismatches": initial_mismatches,
        },
        "held_clean": final_summary,
        "gates": gates,
        "pass_to_multifold": bool(all(gates.values())),
        "history": history,
        "contracts": {
            "shared_query_reference_encoder": True,
            "model_weights_changed": True,
            "last_transformer_blocks_and_official_head_trainable": True,
            "clean_and_augmented_raw_spectra_train_same_encoder": True,
            "action_recipe_fixed_before_this_training_run": True,
            "action_outcomes_used_for_weights_or_selection": False,
            "identity_equal_action_weighting": True,
            "action_views_per_identity_per_epoch": args.views_per_identity,
            "formula_held_out": True,
            "dropout_disabled_during_gradient_training": True,
            "inference_clean_spectrum_only": True,
            "teacher": "forbidden",
            "P2b": "forbidden",
            "P3_consumed": False,
        },
        "provenance": {
            "r0_report_sha256": sha256_file(args.r0_dir / "report.json"),
            "r0_actions_sha256": sha256_file(args.r0_dir / "training_actions.csv.gz"),
            "graph_sha256": sha256_file(args.graph),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "held-formula development result for a directly fine-tuned shared embedding. "
            "Historical 3.85 pp is action-oracle headroom, not a promised weight gain."
        ),
    }
    # Path objects are not JSON serialisable.
    decision["configuration"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in decision["configuration"].items()
    }
    json_dump(output / "decision.json", decision)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
