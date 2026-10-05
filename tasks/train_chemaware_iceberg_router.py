"""Test ICEBERG as a training-query router, never as an embedding target.

Every arm optimizes the same real-spectrum, complete-candidate hard loss.  The
only intervention is which fixed-size subset of official DreaMS errors is
replayed.  Correct and destructive-control arms rank those errors with their
respective ICEBERG candidate scores; spectrum_only uses a seeded random subset.
The held formula fold is never used for routing or checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from chemaware_iceberg_direct_core import stable_formula_folds  # noqa: E402
from dreams.models.chem_aware.global_embedding_adapter import GlobalResidualEmbeddingAdapter  # noqa: E402
from noise_final_core import sha256_file  # noqa: E402
from train_chemaware_full_candidate_alignment import (  # noqa: E402
    evaluate, formula_bootstrap, identity_balanced_queries, learning_rate_scale,
    listwise_losses, official_outcomes, project_numpy, sample_training_batch,
)


ARMS = ("spectrum_only", "correct", "candidate_swapped", "peak_permuted")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1")
    parser.add_argument("--teacher-graph-dir", type=Path, default=ROOT / "data/validation/chemaware_full_manifest_iceberg_graph_v1")
    parser.add_argument("--teacher-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--inner-fold", type=int, default=3)
    parser.add_argument("--outer-fold", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--route-queries", type=int, default=320)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--eval-every-steps", type=int, default=64)
    parser.add_argument("--batch-queries", type=int, default=64)
    parser.add_argument("--references-per-molecule", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda-inbatch-spectrum", type=float, default=0.25)
    parser.add_argument("--lambda-margin-floor", type=float, default=2.0)
    parser.add_argument("--margin-floor-slack", type=float, default=0.005)
    parser.add_argument("--lambda-preserve", type=float, default=20.0)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--max-eval-identities", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def load_teacher_queries(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    report = json.loads((args.teacher_dir / "report.json").read_text(encoding="utf-8"))
    graph_report = json.loads((args.teacher_graph_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("scope", {}).get("teacher_only") is not True:
        raise RuntimeError("router requires a passed teacher-only ledger")
    if graph_report.get("scope", {}).get("training_formula_only") is not True:
        raise RuntimeError("router graph is not training-formula-only")
    selected_graph = np.load(args.teacher_dir / "selected_queries.npy").astype(np.int64)
    source_query = np.load(args.teacher_graph_dir / "source_query_index.npy").astype(np.int64)
    scores_file = np.load(args.teacher_dir / "scores_and_ranks.npz", allow_pickle=True)
    scores = {key: np.asarray(scores_file[key]) for key in scores_file.files}
    return source_query[selected_graph], scores, report


def select_route(
    args: argparse.Namespace, teacher_query: np.ndarray, scores: dict[str, np.ndarray],
    train_error: np.ndarray,
) -> tuple[np.ndarray, dict]:
    eligible_position = np.flatnonzero(train_error[teacher_query])
    if len(eligible_position) < args.route_queries:
        raise RuntimeError("not enough official-error teacher queries for fixed router dose")
    if args.arm == "spectrum_only":
        rng = np.random.default_rng(args.seed + 709)
        chosen_position = eligible_position[rng.permutation(len(eligible_position))[:args.route_queries]]
        criterion = "seeded random official-error subset"
    else:
        rank = np.asarray(scores[f"{args.arm}_rank"], dtype=np.int64)[eligible_position]
        margin = np.asarray(scores[f"{args.arm}_margin"], dtype=np.float64)[eligible_position]
        # Strict rank first, then larger positive margin, then source query id.
        order = np.lexsort((teacher_query[eligible_position], -margin, rank))
        chosen_position = eligible_position[order[:args.route_queries]]
        criterion = "teacher rank ascending, positive margin descending"
    chosen = teacher_query[chosen_position].astype(np.int64)
    diag = {
        "eligible_official_errors": int(len(eligible_position)),
        "selected": int(len(chosen)),
        "criterion": criterion,
        "selected_teacher_hit1": None if args.arm == "spectrum_only" else int(
            np.sum(np.asarray(scores[f"{args.arm}_rank"])[chosen_position] == 1)
        ),
        "selected_teacher_margin_mean": None if args.arm == "spectrum_only" else float(
            np.mean(np.asarray(scores[f"{args.arm}_margin"])[chosen_position])
        ),
    }
    return chosen, diag


def main() -> None:
    args = arguments(); started = time.time()
    if args.max_steps < 4 or args.warmup_steps >= args.max_steps or args.route_queries < 1:
        raise ValueError("invalid router schedule")
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite output: {args.output}")
    required = [
        args.manifest, args.token_dir / "report.json", args.token_dir / "rows.npy",
        args.token_dir / "official_embeddings_f32.npy", args.teacher_graph_dir / "report.json",
        args.teacher_graph_dir / "source_query_index.npy", args.teacher_dir / "report.json",
        args.teacher_dir / "selected_queries.npy", args.teacher_dir / "scores_and_ranks.npz",
    ]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)
    with np.load(args.manifest) as loaded:
        body = {key: loaded[key] for key in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    fold = stable_formula_folds(body["query_formula"], args.folds, args.fold_seed)
    molecule_fold = stable_formula_folds(body["molecule_formula"], args.folds, args.fold_seed)
    molecule_allowed = (molecule_fold != args.inner_fold) & (molecule_fold != args.outer_fold)
    allowed_count = np.add.reduceat(molecule_allowed.astype(np.int32), body["query_ptr"][:-1])
    train_pool = np.flatnonzero((fold != args.inner_fold) & (fold != args.outer_fold) & (allowed_count >= 2))
    inner_pool = np.flatnonzero(fold == args.inner_fold)
    outer_pool = np.flatnonzero(fold == args.outer_fold)
    teacher_query, scores, teacher_report = load_teacher_queries(args)
    if not set(map(int, teacher_query)).issubset(set(map(int, train_pool))):
        raise RuntimeError("teacher query escaped training formula pool")
    train_error, _ = official_outcomes(body, train_pool, official, row_position, molecule_allowed)
    route_query, route_diag = select_route(args, teacher_query, scores, train_error)
    all_teacher_identity = set(body["query_ik14"][teacher_query].astype(str))
    clean_pool = np.asarray([
        query for query in train_pool
        if not train_error[query] and str(body["query_ik14"][query]) not in all_teacher_identity
    ], dtype=np.int64)
    inner = identity_balanced_queries(
        inner_pool, body["query_ik14"], np.random.default_rng(args.seed + 19),
        args.max_eval_identities,
    )
    args.output.mkdir(parents=True)
    np.save(args.output / "selected_route_queries.npy", route_query)
    preflight = {
        "status": "chemaware_iceberg_router_preflight_passed", "arm": args.arm,
        "route": route_diag, "clean_safety_pool": int(len(clean_pool)),
        "inner_eval_identities": int(len(inner)), "outer_queries_untouched": int(len(outer_pool)),
        "contracts": {
            "formula_disjoint": True, "teacher_training_only": True,
            "teacher_changes_sampling_only": True, "hard_labels_real_spectra_only": True,
            "same_spectrum_adapter_query_reference": True,
            "matched_role_schedule_and_optimizer_budget_across_arms": True,
            "candidate_input_at_deployment": False, "outer_fold_evaluated": False,
            "teacher_checkpoint_independent_of_massspecgym": False,
        },
        "claim_limit": (
            "ICEBERG public weights may overlap MassSpecGym; this is a development "
            "routing-mechanism gate, not independent-source generalization."
        ),
        "provenance": {
            "manifest_sha256": sha256_file(args.manifest),
            "teacher_report_sha256": sha256_file(args.teacher_dir / "report.json"),
        },
    }
    (args.output / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    if args.preflight_only:
        print(json.dumps(preflight, indent=2)); return

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads); device = torch.device(args.device)
    model = GlobalResidualEmbeddingAdapter(official.shape[1], args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed + 101)
    initial = evaluate(body, inner, official, official, row_position)
    final_eval = initial; final_preservation = 1.0; history = []; global_step = 0; epoch = 0
    while global_step < args.max_steps:
        epoch += 1; model.train()
        routed = route_query[rng.permutation(len(route_query))]
        safety = identity_balanced_queries(clean_pool, body["query_ik14"], rng, len(route_query))
        epoch_query = np.concatenate((routed, safety)); rng.shuffle(epoch_query)
        totals = {"loss": 0.0, "hard": 0.0, "inbatch": 0.0, "margin_floor": 0.0,
                  "preserve": 0.0, "grad_norm": 0.0, "clip_fraction": 0.0}
        batches = 0
        for left in range(0, len(epoch_query), args.batch_queries):
            if global_step >= args.max_steps:
                break
            queries = epoch_query[left:left + args.batch_queries]
            batch = sample_training_batch(
                body, queries, row_position, None, args.references_per_molecule, rng, molecule_allowed,
            )
            query_x = torch.from_numpy(np.array(official[batch["query_cache"]], copy=True)).to(device)
            reference_x = torch.from_numpy(np.array(official[batch["reference_cache"]], copy=True)).to(device)
            query_z = model(query_x); reference_z = model(reference_x)
            hard, _, inbatch, _, margin_floor = listwise_losses(
                query_z, reference_z, None, batch["candidate_ptr"], batch["reference_ptr"],
                batch["reference_edge"], args.temperature, query_x, reference_x,
                args.margin_floor_slack,
            )
            preserve = torch.cat((
                1 - torch.sum(query_z * query_x, dim=1),
                1 - torch.sum(reference_z * reference_x, dim=1),
            )).mean()
            loss = (hard + args.lambda_inbatch_spectrum * inbatch
                    + args.lambda_margin_floor * margin_floor + args.lambda_preserve * preserve)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
            scale = learning_rate_scale(global_step + 1, args.max_steps, args.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * scale
            optimizer.step(); global_step += 1; batches += 1
            for key, value in (("loss", loss), ("hard", hard), ("inbatch", inbatch),
                               ("margin_floor", margin_floor), ("preserve", preserve)):
                totals[key] += float(value.detach())
            totals["grad_norm"] += norm; totals["clip_fraction"] += float(norm > args.grad_clip)
        if global_step >= args.max_steps or global_step % args.eval_every_steps < batches:
            adapted = project_numpy(model, official, device, args.eval_batch_size)
            current = evaluate(body, inner, official, adapted, row_position)
            preservation = float(np.mean(np.sum(adapted * official, axis=1)))
            history.append({
                "epoch": epoch, "global_step": global_step,
                "train": {key: value / max(1, batches) for key, value in totals.items()},
                "inner": current["summary"], "preservation": preservation,
            })
            final_eval = current; final_preservation = preservation
            print(
                f"arm={args.arm} step={global_step}/{args.max_steps} "
                f"delta={current['summary']['delta_recall1']:+.4f} "
                f"corrected={current['summary']['corrected']} "
                f"introduced={current['summary']['introduced']} preserve={preservation:.6f}",
                flush=True,
            )
    delta = ((final_eval["new_rank"] == 1).astype(float) - (final_eval["old_rank"] == 1).astype(float))
    ci = formula_bootstrap(delta, body["query_formula"][inner], args.seed + 901, args.bootstrap_draws)
    checkpoint = {
        "status": "chemaware_iceberg_router_adapter", "format": "chemaware_global_embedding_adapter_v1",
        "adapter_config": {"dimension": int(official.shape[1]), "hidden_dim": args.hidden_dim,
                           "dropout": args.dropout},
        "adapter_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "arm": args.arm, "seed": args.seed, "inner_fold": args.inner_fold,
        "outer_fold": args.outer_fold, "query_reference_encoder_shared": True,
        "candidate_inputs_at_inference": False, "P2b_used": False,
        "chemical_supervision": args.arm != "spectrum_only", "teacher_role": "sampling_router",
        "formal": False, "validation_pass": False, "molecule_projector_state": None,
    }
    torch.save(checkpoint, args.output / "shared_spectrum_adapter.pt")
    np.savez_compressed(
        args.output / "inner_per_query.npz", query_index=inner.astype(np.int64),
        formula=body["query_formula"][inner], identity=body["query_ik14"][inner],
        initial_rank=final_eval["old_rank"].astype(np.int16), final_rank=final_eval["new_rank"].astype(np.int16),
    )
    report = {
        "status": "DEVELOPMENT", "preflight": preflight,
        "optimization": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "initial_inner": initial["summary"], "final_inner": final_eval["summary"],
        "formula_bootstrap": ci, "selected_step": args.max_steps,
        "selection": "fixed matched endpoint inherited from admitted adapter screen; no per-arm selection",
        "preservation": final_preservation, "history": history,
        "teacher_headroom": teacher_report.get("metrics", {}), "runtime_seconds": time.time() - started,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "DEVELOPMENT", "arm": args.arm, "route": route_diag,
                      "final_inner": final_eval["summary"], "formula_bootstrap": ci,
                      "preservation": final_preservation}, indent=2))


if __name__ == "__main__":
    main()
