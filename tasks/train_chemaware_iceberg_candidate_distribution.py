"""Inject ICEBERG as a candidate-relative soft ranking teacher.

Unlike the failed terminal synthetic-spectrum actions, this trainer never asks
DreaMS to imitate an ICEBERG spectrum.  ICEBERG supplies only a probability
distribution over the real candidate molecules for a training query.  The
deployable artifact remains one shared, candidate-free spectrum adapter.
Correct, candidate-swapped, peak-permuted and spectrum-only arms share query
orders, real reference sampling, optimizer steps and hard-label losses.
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
import torch.nn.functional as F

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
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--eval-every-steps", type=int, default=100)
    parser.add_argument("--batch-queries", type=int, default=64)
    parser.add_argument("--references-per-molecule", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--teacher-temperature", type=float, default=0.10)
    parser.add_argument("--lambda-teacher", type=float, default=1.0)
    parser.add_argument(
        "--teacher-dose-curriculum", type=float, nargs=4,
        default=(0.35, 0.45, 0.55, 0.55),
        help="Four fixed equal-duration teacher doses, matched across chemical arms.",
    )
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


def molecule_scores(
    query_z: torch.Tensor, reference_z: torch.Tensor,
    candidate_ptr: np.ndarray, reference_ptr: np.ndarray,
    reference_edge: np.ndarray,
) -> list[torch.Tensor]:
    edge = torch.as_tensor(reference_edge, device=reference_z.device)
    output = []
    for index, (left, right) in enumerate(zip(candidate_ptr[:-1], candidate_ptr[1:])):
        values = []
        for molecule in range(int(left), int(right)):
            rleft, rright = map(int, reference_ptr[molecule:molecule + 2])
            values.append((reference_z[edge[rleft:rright]] @ query_z[index]).max())
        output.append(torch.stack(values))
    return output


def teacher_kl(
    student: list[torch.Tensor], queries: np.ndarray,
    score_by_query: dict[int, np.ndarray], student_temperature: float,
    teacher_temperature: float,
) -> torch.Tensor:
    losses = []
    for query, logits in zip(map(int, queries), student):
        if query not in score_by_query:
            continue
        distance = torch.as_tensor(score_by_query[query], device=logits.device, dtype=logits.dtype)
        if len(distance) != len(logits):
            raise RuntimeError("teacher/student candidate counts differ")
        target = F.softmax(-distance / teacher_temperature, dim=0)
        losses.append(F.kl_div(
            F.log_softmax(logits / student_temperature, dim=0), target,
            reduction="sum",
        ))
    return torch.stack(losses).mean() if losses else student[0].sum() * 0.0


def load_teacher(
    args: argparse.Namespace, body: dict[str, np.ndarray], molecule_allowed: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict]:
    report = json.loads((args.teacher_dir / "report.json").read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("scope", {}).get("teacher_only") is not True:
        raise RuntimeError("candidate-distribution training requires a passed teacher-only ledger")
    graph_report = json.loads((args.teacher_graph_dir / "report.json").read_text(encoding="utf-8"))
    if graph_report.get("scope", {}).get("training_formula_only") is not True:
        raise RuntimeError("ICEBERG teacher graph is not training-formula-only")
    selected_graph = np.load(args.teacher_dir / "selected_queries.npy").astype(np.int64)
    teacher_ptr = np.load(args.teacher_dir / "query_ptr.npy").astype(np.int64)
    source_query = np.load(args.teacher_graph_dir / "source_query_index.npy").astype(np.int64)
    full_query = source_query[selected_graph]
    scores = np.load(args.teacher_dir / "scores_and_ranks.npz", allow_pickle=True)
    score_name = {
        "correct": "correct_score", "candidate_swapped": "candidate_swapped_score",
        "peak_permuted": "peak_permuted_score", "spectrum_only": "correct_score",
    }[args.arm]
    flat = np.asarray(scores[score_name], dtype=np.float32)
    if len(teacher_ptr) != len(full_query) + 1 or int(teacher_ptr[-1]) != len(flat):
        raise RuntimeError("teacher score ledger is misaligned")
    score_by_query = {}
    for index, query in enumerate(full_query):
        left, right = map(int, body["query_ptr"][query:query + 2])
        expected = int(np.sum(molecule_allowed[left:right]))
        tleft, tright = map(int, teacher_ptr[index:index + 2])
        if expected != tright - tleft:
            raise RuntimeError(f"teacher candidate count changed for full query {int(query)}")
        score_by_query[int(query)] = flat[tleft:tright]
    return full_query, score_by_query, report


def main() -> None:
    args = arguments(); started = time.time()
    if (args.max_steps < 4 or args.warmup_steps >= args.max_steps
            or any(value < 0 or value > 1 for value in args.teacher_dose_curriculum)):
        raise ValueError("invalid step schedule or teacher-dose curriculum")
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite output: {args.output}")
    required = [
        args.manifest, args.token_dir / "report.json", args.token_dir / "rows.npy",
        args.token_dir / "official_embeddings_f32.npy",
        args.teacher_graph_dir / "report.json", args.teacher_graph_dir / "graph.npz",
        args.teacher_graph_dir / "source_query_index.npy", args.teacher_dir / "report.json",
        args.teacher_dir / "selected_queries.npy", args.teacher_dir / "query_ptr.npy",
        args.teacher_dir / "scores_and_ranks.npz",
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
    train_pool = np.flatnonzero(
        (fold != args.inner_fold) & (fold != args.outer_fold) & (allowed_count >= 2)
    )
    inner_pool = np.flatnonzero(fold == args.inner_fold)
    outer_pool = np.flatnonzero(fold == args.outer_fold)
    teacher_query, score_by_query, teacher_report = load_teacher(args, body, molecule_allowed)
    if not set(map(int, teacher_query)).issubset(set(map(int, train_pool))):
        raise RuntimeError("teacher query escaped the training formula pool")
    train_error, _ = official_outcomes(body, train_pool, official, row_position, molecule_allowed)
    teacher_identity = set(body["query_ik14"][teacher_query].astype(str))
    clean_pool = np.asarray([
        query for query in train_pool
        if not train_error[query] and str(body["query_ik14"][query]) not in teacher_identity
    ], dtype=np.int64)
    inner = identity_balanced_queries(
        inner_pool, body["query_ik14"], np.random.default_rng(args.seed + 19),
        args.max_eval_identities,
    )
    args.output.mkdir(parents=True)
    preflight = {
        "status": "chemaware_iceberg_candidate_distribution_preflight_passed",
        "arm": args.arm, "teacher_queries": int(len(teacher_query)),
        "safety_clean_pool": int(len(clean_pool)), "inner_eval_identities": int(len(inner)),
        "outer_queries_untouched": int(len(outer_pool)),
        "contracts": {
            "formula_disjoint": True, "teacher_training_only": True,
            "candidate_relative_distribution_only": True,
            "same_spectrum_adapter_query_reference": True,
            "matched_query_and_reference_schedule_across_arms": True,
            "candidate_input_at_deployment": False, "outer_fold_evaluated": False,
            "teacher_checkpoint_independent_of_massspecgym": False,
        },
        "claim_limit": (
            "ICEBERG public weights may overlap MassSpecGym chemistry; this is a "
            "mechanism/development attribution gate, not independent-source generalization."
        ),
        "provenance": {
            "manifest_sha256": sha256_file(args.manifest),
            "teacher_report_sha256": sha256_file(args.teacher_dir / "report.json"),
            "teacher_graph_sha256": sha256_file(args.teacher_graph_dir / "graph.npz"),
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
    final_eval = initial; final_preservation = 1.0
    history = []; global_step = 0; epoch = 0
    while global_step < args.max_steps:
        epoch += 1
        model.train()
        teacher_order = teacher_query[rng.permutation(len(teacher_query))]
        safety = identity_balanced_queries(
            clean_pool, body["query_ik14"], rng, len(teacher_query),
        )
        epoch_query = np.concatenate((teacher_order, safety))
        rng.shuffle(epoch_query)
        totals = {"loss": 0.0, "hard": 0.0, "teacher": 0.0,
                  "inbatch": 0.0, "margin_floor": 0.0, "preserve": 0.0,
                  "teacher_dose": 0.0, "grad_norm": 0.0, "clip_fraction": 0.0}
        batches = 0
        for left in range(0, len(epoch_query), args.batch_queries):
            if global_step >= args.max_steps:
                break
            queries = epoch_query[left:left + args.batch_queries]
            batch = sample_training_batch(
                body, queries, row_position, None, args.references_per_molecule,
                rng, molecule_allowed,
            )
            query_x = torch.from_numpy(np.array(official[batch["query_cache"]], copy=True)).to(device)
            reference_x = torch.from_numpy(np.array(official[batch["reference_cache"]], copy=True)).to(device)
            query_z = model(query_x); reference_z = model(reference_x)
            hard, _, inbatch, _, margin_floor = listwise_losses(
                query_z, reference_z, None, batch["candidate_ptr"], batch["reference_ptr"],
                batch["reference_edge"], args.temperature, query_x, reference_x,
                args.margin_floor_slack,
            )
            scores = molecule_scores(
                query_z, reference_z, batch["candidate_ptr"], batch["reference_ptr"],
                batch["reference_edge"],
            )
            chemical = teacher_kl(
                scores, queries, score_by_query, args.temperature, args.teacher_temperature,
            )
            preserve = torch.cat((
                1 - torch.sum(query_z * query_x, dim=1),
                1 - torch.sum(reference_z * reference_x, dim=1),
            )).mean()
            dose_stage = min(3, (global_step * 4) // args.max_steps)
            teacher_weight = (
                0.0 if args.arm == "spectrum_only" else
                args.lambda_teacher * float(args.teacher_dose_curriculum[dose_stage])
            )
            loss = (hard + teacher_weight * chemical
                    + args.lambda_inbatch_spectrum * inbatch
                    + args.lambda_margin_floor * margin_floor
                    + args.lambda_preserve * preserve)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
            scale = learning_rate_scale(global_step + 1, args.max_steps, args.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * scale
            optimizer.step(); global_step += 1; batches += 1
            for key, value in (("loss", loss), ("hard", hard), ("teacher", chemical),
                               ("inbatch", inbatch), ("margin_floor", margin_floor),
                               ("preserve", preserve)):
                totals[key] += float(value.detach())
            totals["grad_norm"] += norm; totals["clip_fraction"] += float(norm > args.grad_clip)
            totals["teacher_dose"] += teacher_weight
        if global_step >= args.max_steps or global_step % args.eval_every_steps < batches:
            adapted = project_numpy(model, official, device, args.eval_batch_size)
            current = evaluate(body, inner, official, adapted, row_position)
            preservation = float(np.mean(np.sum(adapted * official, axis=1)))
            record = {
                "epoch": epoch, "global_step": global_step,
                "train": {key: value / max(1, batches) for key, value in totals.items()},
                "inner": current["summary"], "preservation": preservation,
            }
            history.append(record)
            final_eval = current; final_preservation = preservation
            print(
                f"arm={args.arm} step={global_step}/{args.max_steps} "
                f"delta={current['summary']['delta_recall1']:+.4f} "
                f"corrected={current['summary']['corrected']} "
                f"introduced={current['summary']['introduced']} preserve={preservation:.6f}",
                flush=True,
            )
    delta = ((final_eval["new_rank"] == 1).astype(float)
             - (final_eval["old_rank"] == 1).astype(float))
    ci = formula_bootstrap(
        delta, body["query_formula"][inner], args.seed + 901, args.bootstrap_draws,
    )
    checkpoint = {
        "status": "chemaware_iceberg_candidate_distribution_adapter",
        "format": "chemaware_global_embedding_adapter_v1",
        "adapter_config": {"dimension": int(official.shape[1]), "hidden_dim": args.hidden_dim,
                           "dropout": args.dropout},
        "adapter_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "arm": args.arm, "seed": args.seed, "inner_fold": args.inner_fold,
        "outer_fold": args.outer_fold, "query_reference_encoder_shared": True,
        "candidate_inputs_at_inference": False, "P2b_used": False,
        "chemical_supervision": args.arm != "spectrum_only",
        "teacher_control": args.arm, "formal": False, "validation_pass": False,
        "molecule_projector_state": None,
    }
    torch.save(checkpoint, args.output / "shared_spectrum_adapter.pt")
    np.savez_compressed(
        args.output / "inner_per_query.npz",
        query_index=inner.astype(np.int64),
        formula=body["query_formula"][inner],
        identity=body["query_ik14"][inner],
        initial_rank=final_eval["old_rank"].astype(np.int16),
        final_rank=final_eval["new_rank"].astype(np.int16),
    )
    report = {
        "status": "DEVELOPMENT", "preflight": preflight,
        "optimization": {key: str(value) if isinstance(value, Path) else value
                         for key, value in vars(args).items()},
        "initial_inner": initial["summary"], "final_inner": final_eval["summary"],
        "formula_bootstrap": ci, "selected_step": args.max_steps,
        "selection": "fixed matched endpoint; no per-arm inner selection",
        "preservation": final_preservation, "history": history,
        "teacher_headroom": teacher_report.get("metrics", {}),
        "runtime_seconds": time.time() - started,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "DEVELOPMENT", "arm": args.arm,
                      "selected_step": args.max_steps, "final_inner": final_eval["summary"],
                      "formula_bootstrap": ci, "preservation": final_preservation}, indent=2))


if __name__ == "__main__":
    main()
