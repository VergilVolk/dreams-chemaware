"""Phase-A direct N/P noise fine-tuning from a mature E4 shared encoder.

Each arm uses one shared query/reference DreaMS encoder, the same query order,
the same complete candidate molecules, optimizer and number of steps.  Arms
differ only in action payload and the preregistered action weights.  Inference
uses a clean spectrum only; P2b and every post-embedding expert are absent.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from audit_noise_final_dynamic_direct_replay import materialize_pair, load_replay_model
from noise_final_dynamic_direct_core import formula_identity_query_equal_weights
from noise_final_core import CandidateGraph, load_embedding_cache, sha256_file, stable_fold
from train_noise_final_e4a_direct_augmentation import (
    full_graph_query_details, unfreeze_last_blocks,
)
from train_noise_final_r2_shared_encoder import (
    SpectrumStore, encode_rows, evaluate_embeddings, formula_bootstrap_delta,
    forward_embeddings,
)


ARMS = ("clean_continuation", "matched_random", "static_target", "dynamic_np")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    validation = ROOT / "data/validation"
    parser.add_argument("--m2-dir", type=Path, required=True)
    parser.add_argument("--schedule-dir", type=Path, required=True)
    parser.add_argument("--preflight-dir", type=Path, required=True)
    parser.add_argument("--clean-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--graph", type=Path, default=validation / "g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--embedding-cache", type=Path, default=validation / "g8r_p2_official_embeddings.npz")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-queries", type=int, default=2)
    parser.add_argument("--unfreeze-blocks", type=int, default=1)
    parser.add_argument("--backbone-lr", type=float, default=2e-6)
    parser.add_argument("--head-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--margin-floor-slack", type=float, default=0.005)
    parser.add_argument("--lambda-clean-rank", type=float, default=1.0)
    parser.add_argument("--lambda-action-rank", type=float, default=1.0)
    parser.add_argument("--lambda-margin-floor", type=float, default=2.0)
    parser.add_argument("--lambda-preserve", type=float, default=5.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-batch-size", type=int, default=96)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--recurrence-prevalence", type=float, default=0.67)
    parser.add_argument("--recurrence-max-peaks", type=int, default=5)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def molecule_logits(
    query: torch.Tensor, candidates: torch.Tensor, ptr: np.ndarray, temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    pair = candidates @ query
    molecules = torch.stack([
        torch.max(pair[int(left):int(right)])
        for left, right in zip(ptr[:-1], ptr[1:])
    ])
    if len(molecules) < 2:
        raise RuntimeError("full candidate query has fewer than two molecules")
    loss = -F.log_softmax(molecules / temperature, dim=0)[0]
    margin = molecules[0] - torch.max(molecules[1:])
    return loss, margin


def selected_weights(block: pd.DataFrame, arm: str) -> tuple[np.ndarray, float, str]:
    if arm == "clean_continuation":
        return np.zeros(len(block), dtype=np.float32), 1.0, "clean"
    prefix = "static" if arm == "static_target" else "dynamic"
    values = block[f"{prefix}_weight"].to_numpy(np.float32)
    no_op = float(block[f"{prefix}_selected_no_op_weight"].iloc[0])
    payload = "control" if arm == "matched_random" else "target"
    if not np.isclose(float(values.sum()) + no_op, 1.0, atol=1e-5):
        raise RuntimeError(f"{arm} query action/no-op mass is not one")
    return values, no_op, payload


def query_loss(
    model: torch.nn.Module, store: SpectrumStore, graph: CandidateGraph,
    block: pd.DataFrame, initial_by_row: dict[int, np.ndarray], arm: str,
    device: torch.device, args: argparse.Namespace,
    action_cache: dict[tuple[object, ...], object],
) -> tuple[torch.Tensor, dict[str, float]]:
    query = int(block["query_index"].iloc[0])
    if block["query_index"].nunique() != 1:
        raise RuntimeError("one training group must contain exactly one query")
    query_row = int(graph.query_row[query])
    _, candidate_rows, ptr, _ = graph.query_block(query)
    clean = store.one(query_row)
    weights, no_op, payload = selected_weights(block, arm)
    views: list[torch.Tensor] = []
    if arm == "clean_continuation":
        views = [clean for _ in range(len(block))]
    else:
        for row in block.itertuples(index=False):
            target, control = materialize_pair(row, store, args, action_cache)
            views.append(control if payload == "control" else target)
    spectra = torch.stack([clean, *views, *store.get(tuple(map(int, candidate_rows)))])
    encoded = forward_embeddings(model, spectra.to(device), args.amp)
    clean_z = encoded[0]
    action_z = encoded[1:1 + len(views)]
    candidates = encoded[1 + len(views):]
    clean_loss, clean_margin = molecule_logits(clean_z, candidates, ptr, args.temperature)
    action_losses = torch.stack([
        molecule_logits(action_z[index], candidates, ptr, args.temperature)[0]
        for index in range(len(views))
    ])
    weight_tensor = torch.as_tensor(weights, device=device, dtype=action_losses.dtype)
    augmented = no_op * clean_loss + torch.sum(weight_tensor * action_losses)

    initial_query = torch.as_tensor(initial_by_row[query_row], device=device, dtype=clean_z.dtype)
    initial_candidates = torch.as_tensor(
        np.stack([initial_by_row[int(row)] for row in candidate_rows]),
        device=device, dtype=candidates.dtype,
    )
    _, initial_margin = molecule_logits(initial_query, initial_candidates, ptr, args.temperature)
    floor = F.relu(initial_margin.detach() - args.margin_floor_slack - clean_margin)
    preserve = torch.cat((
        (1.0 - torch.sum(clean_z * initial_query)).reshape(1),
        1.0 - torch.sum(candidates * initial_candidates, dim=1),
    )).mean()
    loss = (
        args.lambda_clean_rank * clean_loss
        + args.lambda_action_rank * augmented
        + args.lambda_margin_floor * floor
        + args.lambda_preserve * preserve
    )
    return loss, {
        "clean_listwise": float(clean_loss.detach()),
        "action_mixture_listwise": float(augmented.detach()),
        "margin_floor": float(floor.detach()),
        "preserve": float(preserve.detach()),
        "clean_margin": float(clean_margin.detach()),
        "action_mass": float(weights.sum()),
        "no_op_mass": no_op,
    }


def main() -> None:
    args = arguments()
    if args.outer_fold not in range(5) or args.epochs < 1 or args.batch_queries < 1:
        raise ValueError("invalid Phase-A training parameters")
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite Phase-A output: {args.output_dir}")
    required = [
        args.m2_dir / "report.json", args.m2_dir / "training_actions.csv.gz",
        args.schedule_dir / "report.json", args.schedule_dir / "epoch_schedule.csv.gz",
        args.schedule_dir / "arm_manifest.csv", args.preflight_dir / "report.json",
        args.clean_checkpoint, args.clean_checkpoint.parent / "decision.json",
        args.graph, args.data, args.embedding_cache, args.official_checkpoint,
        args.architecture_checkpoint,
    ]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)
    m2 = json.loads((args.m2_dir / "report.json").read_text(encoding="utf-8"))
    schedule_report = json.loads((args.schedule_dir / "report.json").read_text(encoding="utf-8"))
    preflight = json.loads((args.preflight_dir / "report.json").read_text(encoding="utf-8"))
    if (m2.get("pass_to_schedule") is not True
            or schedule_report.get("pass_to_gpu_replay") is not True
            or int(m2.get("outer_formula_fold", -1)) != args.outer_fold
            or int(schedule_report.get("outer_formula_fold", -1)) != args.outer_fold
            or preflight.get("initialization", {}).get("sha256") != sha256_file(args.clean_checkpoint)):
        raise RuntimeError("Phase-A input contracts are not aligned")
    schedule = pd.read_csv(args.schedule_dir / "epoch_schedule.csv.gz", low_memory=False)
    arms = pd.read_csv(args.schedule_dir / "arm_manifest.csv")
    if set(arms["arm"]) != set(ARMS) or arms["membership_sha256"].nunique() != 1:
        raise RuntimeError("Phase-A arms do not share one schedule")
    needed = {
        "action_id", "schedule_index", "epoch", "epoch_schedule_index",
        "query_index", "identity", "formula", "source",
        "target_payload", "control_payload", "dynamic_weight", "static_weight",
        "dynamic_selected_no_op_weight", "static_selected_no_op_weight",
    }
    if missing := needed - set(schedule.columns):
        raise RuntimeError(f"Phase-A schedule misses columns: {sorted(missing)}")
    if schedule.duplicated(["epoch", "action_id"]).any():
        raise RuntimeError("Phase-A schedule repeats an action within an epoch")
    scheduled_epochs = sorted(schedule["epoch"].astype(int).unique().tolist())
    expected_epochs = int(schedule_report.get("epochs", -1))
    if scheduled_epochs != list(range(1, expected_epochs + 1)):
        raise RuntimeError("Phase-A schedule epochs are incomplete")
    if not args.smoke and args.epochs != expected_epochs:
        raise RuntimeError("trainer epochs must exactly match the frozen schedule")
    if args.smoke:
        schedule = schedule.loc[schedule["epoch"].astype(int).eq(1)].copy()
        selected_queries = schedule["query_index"].drop_duplicates().head(8)
        schedule = schedule.loc[schedule["query_index"].isin(selected_queries)].copy()

    graph = CandidateGraph(args.graph)
    held_queries = np.asarray([
        query for query, formula in enumerate(graph.query_formula)
        if stable_fold(str(formula), 5, args.formula_fold_seed) == args.outer_fold
    ], dtype=np.int64)
    if set(schedule["formula"].astype(str)) & set(graph.query_formula[held_queries].astype(str)):
        raise RuntimeError("outer-held formula leaked into Phase-A training")
    payload_rows: set[int] = set()
    for row in schedule.itertuples(index=False):
        if row.source != "N":
            for payload in (row.target_payload, row.control_payload):
                payload_rows.update(int(value) for value in str(payload).split(";") if value.strip())
    graph_rows = np.unique(np.concatenate((graph.query_row, graph.pair_candidate_row))).astype(np.int64)
    reachable = np.unique(np.concatenate((graph_rows, np.asarray(sorted(payload_rows), dtype=np.int64))))
    store = SpectrumStore(args.data, reachable, args.n_highest_peaks)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal Phase-A training requires CUDA")
    model, model_provenance = load_replay_model(args, device, preflight)
    initial_encoded = encode_rows(
        model, store, reachable, device, args.eval_batch_size, args.amp,
        f"DD-{args.arm}-initial",
    )
    initial_by_row = {int(row): initial_encoded[index] for index, row in enumerate(reachable)}
    _, official_embeddings, official_index = load_embedding_cache(args.embedding_cache)
    if set(map(int, graph_rows)) - set(map(int, official_index)):
        raise RuntimeError("official embedding cache does not cover the complete graph")
    official_graph = np.stack([official_embeddings[official_index[int(row)]] for row in graph_rows])
    initial_graph = np.stack([initial_by_row[int(row)] for row in graph_rows])
    official_rank, official_summary = evaluate_embeddings(graph, graph_rows, official_graph, held_queries)
    initial_rank, initial_summary = evaluate_embeddings(graph, graph_rows, initial_graph, held_queries)

    capacity = unfreeze_last_blocks(model, args.unfreeze_blocks)
    model.eval()
    head = [parameter for parameter in model.head.parameters() if parameter.requires_grad]
    backbone = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": head, "lr": args.head_lr, "weight_decay": args.weight_decay},
        {"params": backbone, "lr": args.backbone_lr, "weight_decay": 0.0},
    ])
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    action_cache: dict[tuple[object, ...], object] = {}
    history: list[dict[str, float | int]] = []
    epochs = 1 if args.smoke else args.epochs
    for epoch in range(1, epochs + 1):
        epoch_frame = schedule.loc[schedule["epoch"].astype(int).eq(epoch)].copy()
        if epoch_frame.empty:
            raise RuntimeError(f"Phase-A schedule has no rows for epoch {epoch}")
        grouped = {
            int(query): block.sort_values("epoch_schedule_index", kind="stable")
            for query, block in epoch_frame.groupby("query_index", sort=False)
        }
        query_table = epoch_frame[["query_index", "formula", "identity"]].drop_duplicates("query_index")
        if len(query_table) != len(grouped):
            raise RuntimeError("query metadata are not unique within an epoch")
        values = formula_identity_query_equal_weights(
            query_table["formula"], query_table["identity"], query_table["query_index"],
        )
        query_weight = dict(zip(query_table["query_index"].astype(int), map(float, values)))
        base_order = list(grouped)
        rng = np.random.default_rng(args.seed + epoch)
        order = [base_order[index] for index in rng.permutation(len(base_order))]
        totals: dict[str, float] = {}
        clips = 0
        steps = 0
        optimizer.zero_grad(set_to_none=True)
        started = time.time()
        for position, query in enumerate(order, start=1):
            loss, logs = query_loss(
                model, store, graph, grouped[query], initial_by_row, args.arm,
                device, args, action_cache,
            )
            weighted = loss * float(query_weight[query]) / args.batch_queries
            scaler.scale(weighted).backward()
            for key, value in logs.items():
                totals[key] = totals.get(key, 0.0) + float(value)
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach())
            if position % args.batch_queries == 0 or position == len(order):
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    args.grad_clip,
                )
                norm_value = float(norm)
                clips += int(norm_value > args.grad_clip)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                totals["gradient_norm"] = totals.get("gradient_norm", 0.0) + norm_value
                steps += 1
            if position % 250 == 0 or position == len(order):
                print(
                    f"[DD {args.arm} epoch={epoch}] {position:,}/{len(order):,} "
                    f"loss={totals['loss']/position:.5f}", flush=True,
                )
        record: dict[str, float | int] = {
            key: value / len(order) for key, value in totals.items()
            if key != "gradient_norm"
        }
        record.update({
            "epoch": epoch, "optimizer_steps": steps,
            "training_queries": len(grouped), "scheduled_actions": len(epoch_frame),
            "mean_gradient_norm": totals.get("gradient_norm", 0.0) / max(steps, 1),
            "clip_fraction": clips / max(steps, 1),
            "seconds": time.time() - started,
        })
        history.append(record)
        print(json.dumps(record, indent=2), flush=True)

    final_graph = encode_rows(
        model, store, graph_rows, device, args.eval_batch_size, False,
        f"DD-{args.arm}-final",
    )
    final_rank, final_summary = evaluate_embeddings(graph, graph_rows, final_graph, held_queries)
    initial_correct, final_correct, official_correct = initial_rank == 1, final_rank == 1, official_rank == 1
    preservation = np.einsum("ij,ij->i", final_graph, initial_graph)
    _, initial_top, initial_margin = full_graph_query_details(graph, graph_rows, initial_graph, held_queries)
    _, final_top, final_margin = full_graph_query_details(graph, graph_rows, final_graph, held_queries)
    formula = graph.query_formula[held_queries]
    final_summary.update({
        "official_recall1": official_summary["recall1"],
        "delta_recall1_vs_official": float(np.mean(final_correct) - np.mean(official_correct)),
        "initial_recall1": initial_summary["recall1"],
        "delta_recall1_vs_initial": float(np.mean(final_correct) - np.mean(initial_correct)),
        "corrected_vs_initial": int(np.sum(~initial_correct & final_correct)),
        "introduced_vs_initial": int(np.sum(initial_correct & ~final_correct)),
        "risk_net_vs_initial": int(np.sum(~initial_correct & final_correct) - 2 * np.sum(initial_correct & ~final_correct)),
        "near_delta_vs_initial": float(final_summary["near_recall1"] - initial_summary["near_recall1"]),
        "mrr_delta_vs_initial": float(final_summary["mrr"] - initial_summary["mrr"]),
        "formula_cluster_delta_vs_initial": formula_bootstrap_delta(
            initial_rank, final_rank, formula, args.bootstrap_resamples, args.seed,
        ),
        "preservation_vs_initial_mean": float(np.mean(preservation)),
        "preservation_vs_initial_p01": float(np.quantile(preservation, 0.01)),
        "mean_margin_delta_vs_initial": float(np.mean(final_margin - initial_margin)),
        "top_molecule_changed_vs_initial": int(np.sum(final_top != initial_top)),
    })
    gates = {
        "finite_training": all(np.isfinite(float(value)) for row in history for value in row.values()),
        "preservation_ge_0_995": final_summary["preservation_vs_initial_mean"] >= 0.995,
        "P2b_forbidden": True, "P3_not_consumed": True,
    }
    report = {
        "status": "noise_final_dynamic_direct_phase_a_arm_complete",
        "formal": not args.smoke, "arm": args.arm, "outer_formula_fold": args.outer_fold,
        "seed": args.seed, "capacity": capacity,
        "training_queries": int(schedule["query_index"].nunique()),
        "scheduled_actions": int(len(schedule)), "held": final_summary,
        "history": history, "gates": gates,
        "configuration": vars(args) | {"output_dir": str(args.output_dir)},
        "contracts": {
            "mature_e4_initialization": True, "shared_query_reference_encoder": True,
            "full_candidate_molecule_list_training": True,
            "one_arm_invariant_schedule": True, "explicit_no_op": True,
            "formula_identity_query_equal_training_mass": True,
            "conditional_policy_frozen_across_epochs": True,
            "matched_control_semantics": (
                "N matched-random path; P wrong-reference direction control"
            ),
            "inference_clean_spectrum_only": True, "P2b": "forbidden", "P3_consumed": False,
        },
        "provenance": {
            **model_provenance,
            "m2_report": sha256_file(args.m2_dir / "report.json"),
            "schedule": sha256_file(args.schedule_dir / "epoch_schedule.csv.gz"),
            "script": sha256_file(Path(__file__)),
        },
        "claim_limit": "One development formula fold Phase-A arm; not multifold or P3 evidence.",
    }
    if not all(gates.values()):
        raise RuntimeError(f"Phase-A arm gates failed: {gates}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".dd_{args.arm}_", dir=args.output_dir.parent))
    try:
        pd.DataFrame({
            "query_index": held_queries, "query_row": graph.query_row[held_queries],
            "query_formula": formula, "has_near": graph.query_has_near[held_queries],
            "official_rank": official_rank, "initial_rank": initial_rank, "final_rank": final_rank,
            "initial_margin": initial_margin, "final_margin": final_margin,
        }).to_csv(staging / "held_per_query.csv.gz", index=False, compression="gzip")
        torch.save({
            "status": "noise_final_dynamic_direct_shared_dreams_encoder",
            "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "arm": args.arm, "outer_fold": args.outer_fold, "seed": args.seed,
            "inference_clean_only": True, "P2b_used": False,
            "initial_checkpoint_sha256": sha256_file(args.clean_checkpoint),
        }, staging / "final_shared_encoder.pt")
        (staging / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise
    print(json.dumps(report, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
