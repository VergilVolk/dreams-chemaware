"""E4-M1: transfer validated noise actions into one shared clean-spectrum embedding.

All query and reference spectra use the same candidate-independent adapter.
Perturbed spectra are stop-gradient training targets only.  P2b and every
downstream reranker are prohibited.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from noise_final_core import CandidateGraph, ZeroInitPeakAdapter, json_dump, seed_everything, sha256_file  # noqa: E402
from train_noise_final_f1_parm import TokenStore, encode_all, evaluate_full, top_negative_rows  # noqa: E402


FAMILIES = ("candidate_gradient", "acquisition_positive_gradient", "role_confounder")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--d0-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_d0_manifest")
    parser.add_argument("--f0-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_f0_protocol")
    parser.add_argument("--target-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_e4_target_cache")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_f1_full_tokens")
    parser.add_argument("--embedding-cache", type=Path, default=ROOT / "data/validation/g8r_p2_official_embeddings.npz")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/validation/g8r_noise_final_e4_shared_adapter")
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-per-family", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--negative-count", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--delta-bound", type=float, default=0.08)
    parser.add_argument("--target-step", type=float, default=0.06)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--lambda-action", type=float, default=1.0)
    parser.add_argument("--lambda-rank", type=float, default=0.5)
    parser.add_argument("--lambda-safety", type=float, default=4.0)
    parser.add_argument("--lambda-preserve", type=float, default=1.0)
    parser.add_argument("--action-warmup-epochs", type=int, default=1)
    parser.add_argument("--family", choices=("all",) + FAMILIES, default="all")
    parser.add_argument("--minimum-preservation", type=float, default=0.995)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps-per-epoch", type=int, default=0, help="smoke only")
    parser.add_argument("--max-eval-queries", type=int, default=0, help="smoke only")
    return parser.parse_args()


def best_positive_rows(graph: CandidateGraph, allowed: np.ndarray) -> np.ndarray:
    output = np.full(graph.n_queries, -1, dtype=np.int64)
    column = graph.dreams_column
    for query in range(graph.n_queries):
        molecule = int(graph.query_ptr[query])
        if not allowed[molecule]:
            raise RuntimeError(f"positive molecule excluded at query {query}")
        left, right = map(int, graph.molecule_ptr[molecule:molecule + 2])
        local = int(np.argmax(graph.features[left:right, column]))
        output[query] = int(graph.pair_candidate_row[left + local])
    return output


def safe_target(clean: torch.Tensor, raw_target: torch.Tensor, step: float) -> torch.Tensor:
    raw_target = F.normalize(raw_target.float(), dim=-1)
    tangent = raw_target - torch.sum(raw_target * clean, dim=1, keepdim=True) * clean
    direction = tangent / tangent.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return F.normalize(clean + float(step) * direction, dim=-1)


def dot_grads(left: list[torch.Tensor], right: list[torch.Tensor]) -> torch.Tensor:
    return sum(torch.sum(a * b) for a, b in zip(left, right))


def pcgrad(task_gradients: list[list[torch.Tensor]]) -> tuple[list[torch.Tensor], int]:
    """Project only genuine negative task-gradient pairs, then average."""
    projected = [[value.clone() for value in gradient] for gradient in task_gradients]
    conflicts = 0
    for left in range(len(projected)):
        for right in range(len(task_gradients)):
            if left == right:
                continue
            dot = dot_grads(projected[left], task_gradients[right])
            if float(dot) < 0.0:
                denominator = dot_grads(task_gradients[right], task_gradients[right]).clamp_min(1e-12)
                coefficient = dot / denominator
                projected[left] = [
                    a - coefficient * b for a, b in zip(projected[left], task_gradients[right])
                ]
                conflicts += 1
    return [sum(values) / len(values) for values in zip(*projected)], conflicts


def balanced_pcgrad(task_gradients: list[list[torch.Tensor]]) -> tuple[list[torch.Tensor], int, list[float]]:
    """Equalise objective gradient norms before conflict projection.

    A 0.06-radian action target has loss about 0.0018 whereas clean ranking is
    about 0.7. Raw-loss addition therefore suppresses the noise objective even
    when its coefficient is one. Gradient balancing removes that unit mismatch.
    """
    norms = [torch.sqrt(dot_grads(gradient, gradient).clamp_min(1e-20)) for gradient in task_gradients]
    normalized = [
        [value / norm for value in gradient] for gradient, norm in zip(task_gradients, norms)
    ]
    combined, conflicts = pcgrad(normalized)
    scale = torch.median(torch.stack(norms))
    return [value * scale for value in combined], conflicts, [float(value.detach()) for value in norms]


def action_realization(adapter, store: TokenStore, actions: pd.DataFrame, targets: np.ndarray,
                       indices: np.ndarray, device: torch.device, step: float,
                       families: tuple[str, ...] = FAMILIES,
                       batch_size: int = 256) -> dict[str, float | int]:
    adapter.eval()
    deltas = []
    family_values: dict[str, list[float]] = {family: [] for family in families}
    with torch.no_grad():
        for left in range(0, len(indices), batch_size):
            block = indices[left:left + batch_size]
            rows = actions.iloc[block]["query_row"].to_numpy(np.int64)
            official, adapted = store.adapt(adapter, rows, device)
            raw = torch.from_numpy(targets[block].astype(np.float32)).to(device)
            target = safe_target(official, raw, step)
            delta = (torch.sum(adapted * target, dim=1) - torch.sum(official * target, dim=1)).cpu().numpy()
            delta[np.abs(delta) < 1e-7] = 0.0
            deltas.extend(map(float, delta))
            for value, family in zip(delta, actions.iloc[block]["e4_family"].astype(str)):
                family_values[family].append(float(value))
    return {
        "n_actions": int(len(indices)), "mean_target_cosine_gain": float(np.mean(deltas)),
        "positive_fraction": float(np.mean(np.asarray(deltas) > 0)),
        **{f"{family}_mean_gain": float(np.mean(values)) if values else float("nan")
           for family, values in family_values.items()},
    }


def main() -> None:
    args = arguments()
    if args.outer_fold not in range(5):
        raise ValueError("outer-fold must be 0..4")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("E4-M1 requires CUDA")
    seed_everything(args.seed)
    device = torch.device(args.device)
    required = [
        args.graph, args.d0_dir / "manifest.npz", args.f0_dir / "decision.json",
        args.f0_dir / "symmetric_zero_rank.npy", args.f0_dir / "allowed_molecule_mask.npy",
        args.target_dir / "report.json", args.target_dir / "actions.csv.gz",
        args.target_dir / "target_embedding_f16.npy", args.token_dir / "report.json",
        args.embedding_cache,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)
    target_report = json.loads((args.target_dir / "report.json").read_text(encoding="utf-8"))
    if not target_report.get("formal") or target_report["contracts"].get("P2b") != "forbidden":
        raise RuntimeError("E4-M1 requires formal P2b-free E4-M0")
    f0 = json.loads((args.f0_dir / "decision.json").read_text(encoding="utf-8"))
    if not f0.get("pass"):
        raise RuntimeError("E4-M1 requires passing symmetric F0")

    graph = CandidateGraph(args.graph)
    with np.load(args.d0_dir / "manifest.npz") as body:
        formula_fold = np.asarray(body["formula_fold"], dtype=np.int8)
    if len(formula_fold) != graph.n_queries:
        raise RuntimeError("formula fold/query mismatch")
    baseline_rank = np.load(args.f0_dir / "symmetric_zero_rank.npy")
    allowed = np.load(args.f0_dir / "allowed_molecule_mask.npy")
    store = TokenStore(args.token_dir, args.embedding_cache)
    actions = pd.read_csv(args.target_dir / "actions.csv.gz")
    targets = np.load(args.target_dir / "target_embedding_f16.npy", mmap_mode="r")
    if len(actions) != len(targets) or set(actions["e4_family"].astype(str)) != set(FAMILIES):
        raise RuntimeError("invalid E4-M0 action cache")
    actions["fold"] = formula_fold[actions["query_index"].to_numpy(np.int64)]
    active_families = FAMILIES if args.family == "all" else (args.family,)
    if args.family != "all":
        family_mask = actions["e4_family"].astype(str).eq(args.family).to_numpy()
        actions = actions.loc[family_mask].reset_index(drop=True)
        targets = np.asarray(targets[family_mask])
    if set(actions["e4_family"].astype(str)) != set(active_families):
        raise RuntimeError("active E4 family filtering failed")

    inner_fold = (args.outer_fold + 1) % 5
    train_mask = ~actions["fold"].isin([args.outer_fold, inner_fold])
    train_indices = np.flatnonzero(train_mask.to_numpy())
    inner_actions = np.flatnonzero(actions["fold"].to_numpy() == inner_fold)
    outer_actions = np.flatnonzero(actions["fold"].to_numpy() == args.outer_fold)
    inner_queries = np.flatnonzero(formula_fold == inner_fold)
    outer_queries = np.flatnonzero(formula_fold == args.outer_fold)
    if args.max_eval_queries:
        inner_queries = inner_queries[:args.max_eval_queries]
        outer_queries = outer_queries[:args.max_eval_queries]
        inner_actions = inner_actions[:args.max_eval_queries]
        outer_actions = outer_actions[:args.max_eval_queries]
    if not len(train_indices) or not len(inner_actions) or not len(outer_actions):
        raise RuntimeError("empty E4 split")

    positive_rows = best_positive_rows(graph, allowed)
    negative_rows = top_negative_rows(graph, allowed, args.negative_count)
    family_identity_indices: dict[str, dict[str, np.ndarray]] = {}
    for family in active_families:
        part = actions.iloc[train_indices]
        part = part.loc[part["e4_family"].astype(str) == family]
        family_identity_indices[family] = {
            str(identity): group.index.to_numpy(np.int64)
            for identity, group in part.groupby("query_ik14", sort=False)
        }
        if len(family_identity_indices[family]) < 100:
            raise RuntimeError(f"insufficient E4 train identities for {family}")

    adapter = ZeroInitPeakAdapter(store.dimension, args.hidden_dim, args.delta_bound).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    parameters = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    rng = np.random.default_rng(args.seed)
    max_identities = max(len(value) for value in family_identity_indices.values())
    steps_per_epoch = math.ceil(max_identities / args.batch_per_family)
    if args.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)

    encoded = encode_all(adapter, store, device, args.eval_batch_size)
    initial_inner = evaluate_full(encoded, store, graph, inner_queries, baseline_rank)
    if initial_inner["summary"]["corrected"] or initial_inner["summary"]["introduced"]:
        raise RuntimeError("zero-init E4 does not exactly reproduce symmetric baseline")
    initial_action = action_realization(
        adapter, store, actions, targets, inner_actions, device, args.target_step,
        active_families,
    )
    best_state = copy.deepcopy(adapter.state_dict())
    best_epoch, best_utility = 0, 0.0
    history = [{"epoch": 0, "inner_full": initial_inner["summary"],
                "inner_action": initial_action, "utility": 0.0, "eligible": True}]
    print(f"[E4 fold={args.outer_fold}] epoch=0 {history[-1]}", flush=True)

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        totals = {key: 0.0 for key in (
            "loss", "action", "rank", "safety", "preserve",
            "gradient_norm_sum", "gradient_norm_count",
        )}
        conflict_count = 0
        started = time.time()
        for _ in range(steps_per_epoch):
            action_losses = []
            component_rows = []
            for family in active_families:
                identity_map = family_identity_indices[family]
                identities = np.asarray(list(identity_map), dtype=object)
                selected_identities = rng.choice(identities, size=args.batch_per_family, replace=True)
                index = np.asarray([
                    rng.choice(identity_map[str(identity)]) for identity in selected_identities
                ], dtype=np.int64)
                query_index = actions.iloc[index]["query_index"].to_numpy(np.int64)
                qrow = actions.iloc[index]["query_row"].to_numpy(np.int64)
                prow = positive_rows[query_index]
                nrow = negative_rows[query_index]
                count, negative_count = len(index), nrow.shape[1]
                joined = np.concatenate((qrow, prow, nrow.reshape(-1)))
                unique, inverse = np.unique(joined, return_inverse=True)
                official_unique, adapted_unique = store.adapt(adapter, unique, device)
                q = adapted_unique[inverse[:count]]
                p = adapted_unique[inverse[count:2 * count]]
                n = adapted_unique[inverse[2 * count:]].reshape(count, negative_count, -1)
                q0 = official_unique[inverse[:count]]
                p0 = official_unique[inverse[count:2 * count]]
                n0 = official_unique[inverse[2 * count:]].reshape(count, negative_count, -1)
                raw_target = torch.from_numpy(targets[index].astype(np.float32)).to(device)
                target = safe_target(q0, raw_target, args.target_step)
                action_loss = torch.mean(1.0 - torch.sum(q * target, dim=1))
                spos = torch.sum(q * p, dim=1)
                sneg = torch.sum(q[:, None, :] * n, dim=2)
                old_pos = torch.sum(q0 * p0, dim=1)
                old_neg = torch.sum(q0[:, None, :] * n0, dim=2)
                new_margin, old_margin = spos[:, None] - sneg, old_pos[:, None] - old_neg
                rank_loss = torch.mean(F.softplus(
                    (args.rank_margin - new_margin) / args.temperature
                ))
                safety_mask = (old_margin > 0).float()
                safety_loss = torch.sum(F.relu(old_margin - new_margin) * safety_mask) / safety_mask.sum().clamp_min(1.0)
                preserve_loss = (
                    torch.mean(1.0 - torch.sum(q * q0, dim=1))
                    + torch.mean(1.0 - torch.sum(p * p0, dim=1))
                    + torch.mean(1.0 - torch.sum(n * n0, dim=2))
                ) / 3.0
                action_losses.append(args.lambda_action * action_loss)
                component_rows.append((action_loss, rank_loss, safety_loss, preserve_loss))
            clean_loss = (
                args.lambda_rank * torch.stack([x[1] for x in component_rows]).mean()
                + args.lambda_safety * torch.stack([x[2] for x in component_rows]).mean()
                + args.lambda_preserve * torch.stack([x[3] for x in component_rows]).mean()
            )
            task_losses = list(action_losses)
            if epoch > args.action_warmup_epochs:
                task_losses.append(clean_loss)
            gradients = []
            for loss in task_losses:
                values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
                gradients.append([
                    torch.zeros_like(parameter) if value is None else value
                    for parameter, value in zip(parameters, values)
                ])
            combined, conflicts, raw_gradient_norms = balanced_pcgrad(gradients)
            optimizer.zero_grad(set_to_none=True)
            for parameter, gradient in zip(parameters, combined):
                parameter.grad = gradient
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            conflict_count += conflicts
            totals["loss"] += float(torch.stack(task_losses).mean().detach())
            totals["action"] += float(torch.stack([x[0] for x in component_rows]).mean().detach())
            totals["rank"] += float(torch.stack([x[1] for x in component_rows]).mean().detach())
            totals["safety"] += float(torch.stack([x[2] for x in component_rows]).mean().detach())
            totals["preserve"] += float(torch.stack([x[3] for x in component_rows]).mean().detach())
            totals["gradient_norm_sum"] += float(sum(raw_gradient_norms))
            totals["gradient_norm_count"] += float(len(raw_gradient_norms))

        encoded = encode_all(adapter, store, device, args.eval_batch_size)
        inner = evaluate_full(encoded, store, graph, inner_queries, baseline_rank)
        realization = action_realization(
            adapter, store, actions, targets, inner_actions, device, args.target_step,
            active_families,
        )
        summary = inner["summary"]
        risk_net = (summary["corrected"] - 2.0 * summary["introduced"]) / summary["n_queries"]
        utility = risk_net + 0.25 * summary["delta_near_recall1"] + 0.10 * summary["delta_mrr"]
        eligible = (
            summary["preservation_mean"] >= args.minimum_preservation
            and risk_net > 0 and summary["delta_near_recall1"] >= 0
            and summary["delta_mrr"] >= 0 and realization["mean_target_cosine_gain"] > 1e-6
            and all(realization[f"{family}_mean_gain"] > 1e-6 for family in active_families)
        )
        if eligible and utility > best_utility:
            best_epoch, best_utility = epoch, float(utility)
            best_state = copy.deepcopy(adapter.state_dict())
        record = {
            "epoch": epoch, "train": {
                **{key: value / steps_per_epoch for key, value in totals.items()
                   if not key.startswith("gradient_norm_")},
                "mean_raw_task_gradient_norm": (
                    totals["gradient_norm_sum"] / max(totals["gradient_norm_count"], 1.0)
                ),
                "active_objectives": (
                    len(active_families) if epoch <= args.action_warmup_epochs
                    else len(active_families) + 1
                ),
            },
            "pcgrad_conflicts": int(conflict_count), "inner_full": summary,
            "inner_action": realization, "risk_net_per_query": float(risk_net),
            "utility": float(utility), "eligible": bool(eligible), "seconds": time.time() - started,
        }
        history.append(record)
        print(f"[E4 fold={args.outer_fold}] {record}", flush=True)

    adapter.load_state_dict(best_state)
    encoded = encode_all(adapter, store, device, args.eval_batch_size)
    outer = evaluate_full(encoded, store, graph, outer_queries, baseline_rank)
    outer_action = action_realization(
        adapter, store, actions, targets, outer_actions, device, args.target_step,
        active_families,
    )
    outer_summary = outer["summary"]
    outer_risk = (
        outer_summary["corrected"] - 2.0 * outer_summary["introduced"]
    ) / outer_summary["n_queries"]
    gates = {
        "selected_nonzero_epoch": bool(best_epoch > 0),
        "outer_clean_risk_net_positive": bool(outer_risk > 0),
        "outer_clean_recall1_positive": bool(outer_summary["delta_recall1"] > 0),
        "outer_clean_mrr_nonnegative": bool(outer_summary["delta_mrr"] >= 0),
        "outer_near_nonnegative": bool(outer_summary["delta_near_recall1"] >= 0),
        "outer_preservation": bool(outer_summary["preservation_mean"] >= args.minimum_preservation),
        "outer_action_realized_all_families": bool(
            outer_action["mean_target_cosine_gain"] > 1e-6
            and all(outer_action[f"{family}_mean_gain"] > 1e-6 for family in active_families)
        ),
    }
    config = (
        f"step_{args.target_step:.3f}_delta_{args.delta_bound:.3f}"
        if args.family == "all"
        else f"{args.family}_step_{args.target_step:.3f}_delta_{args.delta_bound:.3f}"
    )
    output = args.output_root / config / f"seed_{args.seed}" / f"fold_{args.outer_fold}"
    output.mkdir(parents=True, exist_ok=False)
    torch.save({
        "status": "noise_final_e4_shared_clean_embedding_adapter",
        "adapter_state": {key: value.cpu() for key, value in best_state.items()},
        "adapter_config": {"embedding_dim": store.dimension, "hidden_dim": args.hidden_dim,
                           "delta_bound": args.delta_bound},
        "best_epoch": best_epoch, "seed": args.seed, "outer_fold": args.outer_fold,
        "target_step": args.target_step, "P2b_used": False,
        "algorithm_version": "e4_m1b_gradient_balanced_pcgrad",
        "active_families": list(active_families),
        "query_reference_encoder_shared": True,
    }, output / "adapter.pt")
    decision = {
        "status": "noise_final_e4_shared_adapter_pilot_complete",
        "formal": args.max_steps_per_epoch == 0 and args.max_eval_queries == 0,
        "seed": args.seed, "outer_fold": args.outer_fold, "inner_fold": inner_fold,
        "best_epoch": best_epoch, "best_inner_utility": best_utility,
        "algorithm_version": "e4_m1b_gradient_balanced_pcgrad",
        "active_families": list(active_families),
        "train_actions": int(len(train_indices)), "train_identities": int(actions.iloc[train_indices]["query_ik14"].nunique()),
        "train_formulas": int(actions.iloc[train_indices]["query_formula"].nunique()),
        "outer_full": outer_summary, "outer_action_realization": outer_action,
        "outer_risk_net_per_query": float(outer_risk), "gates": gates,
        "pass_to_multifold_replication": bool(all(gates.values())),
        "history": history,
        "contracts": {"shared_query_reference_encoder": True, "inference_clean_only": True,
                      "P2b": "forbidden", "P3_consumed": False,
                      "raw_loss_scale_mixing": "forbidden",
                      "action_and_clean_objective_gradients_balanced_before_pcgrad": True},
        "provenance": {"target_report_sha256": sha256_file(args.target_dir / "report.json"),
                       "token_report_sha256": sha256_file(args.token_dir / "report.json"),
                       "training_script_sha256": sha256_file(Path(__file__))},
        "claim_limit": "one-fold one-seed E4 micro-train; requires multi-fold replication before a performance claim",
    }
    json_dump(output / "decision.json", decision)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
