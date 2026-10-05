"""Directly transfer strict ICEBERG corrective residuals to shared DreaMS.

The chemical target is neither a synthetic spectrum nor a full teacher
distribution.  It is a candidate-centred score residual on queries where the
official DreaMS boundary is wrong and the correct ICEBERG arm strictly beats
both matched chemical controls.  Every action query is exposed once per epoch.
An equally sized baseline-correct safety stream, official margin floors and
embedding preservation constrain off-target margin motion.

This file trains one matched arm only.  A multi-arm summary is required before
any checkpoint can be described as a causal ChemAware improvement.
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

from chemaware_direct_prior_torch import (  # noqa: E402
    candidate_centered_residual_loss,
    molecule_max_scores,
    select_official_boundary_batch,
)
from chemaware_iceberg_direct_core import stable_formula_folds  # noqa: E402
from noise_final_core import sha256_file, strict_rank  # noqa: E402
from train_chemaware_full_candidate_alignment import (  # noqa: E402
    formula_bootstrap,
    formula_identity_epoch_weights,
    identity_balanced_queries,
    listwise_losses,
)
from train_e1_identity import load_base_model  # noqa: E402
from train_noise_final_e4a_direct_augmentation import unfreeze_last_blocks  # noqa: E402
from train_noise_final_r2_shared_encoder import (  # noqa: E402
    SpectrumStore,
    encode_rows,
    forward_embeddings,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
    )
    parser.add_argument(
        "--action-graph-dir", type=Path,
        default=ROOT / "data/validation/chemaware_full_manifest_iceberg_graph_v1",
    )
    parser.add_argument(
        "--action-ledger-dir", type=Path,
        default=ROOT / "data/validation/chemaware_iceberg_corrective_residual_ledger_v2",
    )
    parser.add_argument(
        "--token-dir", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1",
    )
    parser.add_argument(
        "--data", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument(
        "--official-checkpoint", type=Path,
        default=ROOT / "data/e1/official_embedding_slim.pt",
    )
    parser.add_argument(
        "--architecture-checkpoint", type=Path,
        default=ROOT / "dreams/models/pretrained/ssl_model_server.pt",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prior-arm",
        choices=("none", "correct", "structure_swapped", "peak_permuted"),
        required=True,
    )
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--inner-fold", type=int, default=3)
    parser.add_argument("--outer-fold", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-queries", type=int, default=2)
    parser.add_argument("--references-per-molecule", type=int, default=2)
    parser.add_argument(
        "--safety-identities", type=int, default=0,
        help="Zero uses the number of active corrective queries.",
    )
    parser.add_argument("--max-eval-identities", type=int, default=2000)
    parser.add_argument("--unfreeze-blocks", type=int, default=1)
    parser.add_argument("--backbone-lr", type=float, default=2e-6)
    parser.add_argument("--head-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--lambda-listwise", type=float, default=1.0)
    parser.add_argument("--lambda-inbatch", type=float, default=0.25)
    parser.add_argument("--lambda-margin-floor", type=float, default=2.0)
    parser.add_argument("--margin-floor-slack", type=float, default=0.005)
    parser.add_argument("--lambda-preserve", type=float, default=5.0)
    parser.add_argument("--lambda-chemical", type=float, default=1.0)
    parser.add_argument("--chemical-huber", type=float, default=0.02)
    parser.add_argument(
        "--chemical-gradient-ratio", type=float, default=0.25,
        help="Chemical embedding-gradient norm as a fixed fraction of the base objective.",
    )
    parser.add_argument("--chemical-gradient-scale-cap", type=float, default=4.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=96)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def graph_official_outcomes(body: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Read the already frozen official score feature without a new model pass."""
    score = np.asarray(body["features"][:, 0], dtype=np.float64)
    error = np.zeros(len(body["query_row"]), dtype=bool)
    margin = np.zeros(len(body["query_row"]), dtype=np.float32)
    for query, (left, right) in enumerate(zip(body["query_ptr"][:-1], body["query_ptr"][1:])):
        values = []
        for molecule in range(int(left), int(right)):
            rleft, rright = map(int, body["molecule_ptr"][molecule : molecule + 2])
            values.append(float(np.max(score[rleft:rright])))
        values = np.asarray(values)
        if int(body["molecule_label"][int(left)]) != 1 or len(values) < 2:
            raise RuntimeError("action graph has an invalid candidate block")
        margin[query] = float(values[0] - np.max(values[1:]))
        error[query] = strict_rank(values) != 1
    return error, margin


def select_safety_queries(
    pool: np.ndarray,
    margin: np.ndarray,
    limit: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Half near-boundary and half representative baseline-correct safety."""
    values = np.asarray(pool, dtype=np.int64)
    if limit <= 0 or limit > len(values):
        raise ValueError("safety query request exceeds the available correct cohort")
    boundary_count = limit // 2
    ordered = values[np.argsort(margin[values], kind="stable")]
    boundary = ordered[:boundary_count]
    remaining = ordered[boundary_count:].copy()
    rng.shuffle(remaining)
    selected = np.concatenate((boundary, remaining[: limit - boundary_count]))
    rng.shuffle(selected)
    return selected


def evaluation_rows(body: dict[str, np.ndarray], queries: np.ndarray) -> np.ndarray:
    blocks = [np.asarray(body["query_row"])[np.asarray(queries, dtype=np.int64)]]
    for query in np.asarray(queries, dtype=np.int64):
        left, right = map(int, body["query_ptr"][query : query + 2])
        edge_left = int(body["molecule_ptr"][left])
        edge_right = int(body["molecule_ptr"][right])
        blocks.append(np.asarray(body["pair_candidate_row"][edge_left:edge_right]))
    return np.unique(np.concatenate(blocks)).astype(np.int64)


def evaluate_subset(
    body: dict[str, np.ndarray],
    queries: np.ndarray,
    official: np.ndarray,
    row_position: dict[int, int],
    encoded_rows: np.ndarray,
    encoded: np.ndarray,
) -> dict:
    local = {int(row): index for index, row in enumerate(encoded_rows)}
    old_rank, new_rank, old_margin, new_margin, candidate_count = [], [], [], [], []
    for query in np.asarray(queries, dtype=np.int64):
        query_row = int(body["query_row"][query])
        qold = np.asarray(official[row_position[query_row]])
        qnew = encoded[local[query_row]]
        left, right = map(int, body["query_ptr"][query : query + 2])
        old_score, new_score = [], []
        for molecule in range(left, right):
            rleft, rright = map(int, body["molecule_ptr"][molecule : molecule + 2])
            candidate_row = [
                int(row) for row in body["pair_candidate_row"][rleft:rright]
                if int(row) in local
            ]
            if not candidate_row:
                raise RuntimeError("evaluation closure omitted a candidate reference")
            old_pos = np.asarray([row_position[row] for row in candidate_row], dtype=np.int64)
            new_pos = np.asarray([local[row] for row in candidate_row], dtype=np.int64)
            old_score.append(float(np.max(np.asarray(official[old_pos]) @ qold)))
            new_score.append(float(np.max(encoded[new_pos] @ qnew)))
        old_score = np.asarray(old_score)
        new_score = np.asarray(new_score)
        old_rank.append(strict_rank(old_score))
        new_rank.append(strict_rank(new_score))
        old_margin.append(float(old_score[0] - np.max(old_score[1:])))
        new_margin.append(float(new_score[0] - np.max(new_score[1:])))
        candidate_count.append(len(old_score))
    old_rank = np.asarray(old_rank, dtype=np.int64)
    new_rank = np.asarray(new_rank, dtype=np.int64)
    old_margin = np.asarray(old_margin)
    new_margin = np.asarray(new_margin)
    candidate_count = np.asarray(candidate_count)
    old_ok, new_ok = old_rank == 1, new_rank == 1
    denominator = np.maximum(candidate_count - 1, 1)
    summary = {
        "queries": int(len(queries)),
        "baseline_recall1": float(np.mean(old_ok)),
        "recall1": float(np.mean(new_ok)),
        "delta_recall1": float(np.mean(new_ok) - np.mean(old_ok)),
        "baseline_mrr": float(np.mean(1 / old_rank)),
        "mrr": float(np.mean(1 / new_rank)),
        "delta_mrr": float(np.mean(1 / new_rank) - np.mean(1 / old_rank)),
        "baseline_macro_auc": float(np.mean((candidate_count - old_rank) / denominator)),
        "macro_auc": float(np.mean((candidate_count - new_rank) / denominator)),
        "delta_macro_auc": float(np.mean((old_rank - new_rank) / denominator)),
        "baseline_micro_auc": float(np.sum(candidate_count - old_rank) / np.sum(denominator)),
        "micro_auc": float(np.sum(candidate_count - new_rank) / np.sum(denominator)),
        "delta_micro_auc": float(np.sum(old_rank - new_rank) / np.sum(denominator)),
        "corrected": int(np.sum(~old_ok & new_ok)),
        "introduced": int(np.sum(old_ok & ~new_ok)),
        "delta_mean_margin": float(np.mean(new_margin - old_margin)),
        "mean_absolute_margin_change": float(np.mean(np.abs(new_margin - old_margin))),
        "p95_absolute_margin_change": float(np.quantile(np.abs(new_margin - old_margin), 0.95)),
    }
    for k in (5, 10, 20, 50):
        summary[f"baseline_recall{k}"] = float(np.mean(old_rank <= k))
        summary[f"recall{k}"] = float(np.mean(new_rank <= k))
        summary[f"delta_recall{k}"] = float(np.mean(new_rank <= k) - np.mean(old_rank <= k))
    return {
        "old_rank": old_rank,
        "new_rank": new_rank,
        "old_margin": old_margin,
        "new_margin": new_margin,
        "candidate_count": candidate_count,
        "summary": summary,
    }


def load_prior(
    args: argparse.Namespace, action: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, dict]:
    report_path = args.action_ledger_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "CHEMAWARE_ICEBERG_STRICT_CORRECTIVE_LEDGER_COMPLETE"
        or report.get("strict_corrective_actions") != 272
        or report.get("matched_membership_across_arms") is not True
        or report.get("noncorrective_all_arms_exact_zero") is not True
    ):
        raise RuntimeError("strict corrective ledger did not pass its frozen contract")
    with np.load(args.action_ledger_dir / "ledger.npz", allow_pickle=False) as ledger:
        if not np.array_equal(action["query_ptr"], ledger["query_ptr"]):
            raise RuntimeError("action graph and residual ledger have different query blocks")
        if not np.array_equal(action["query_row"], ledger["query_row"]):
            raise RuntimeError("action graph and residual ledger have different query rows")
        active = np.asarray(ledger["active_query"], dtype=bool)
        arrays = {
            "correct": np.asarray(ledger["centered_residual"], dtype=np.float32),
            "structure_swapped": np.asarray(
                ledger["structure_swapped_centered_residual"], dtype=np.float32
            ),
            "peak_permuted": np.asarray(
                ledger["peak_permuted_centered_residual"], dtype=np.float32
            ),
        }
        ledger_fold = np.asarray(ledger["query_formula_fold"], dtype=np.int16)
    if args.prior_arm == "none":
        prior = np.zeros_like(arrays["correct"])
    else:
        prior = arrays[args.prior_arm]
    inactive_candidate = np.repeat(~active, np.diff(action["query_ptr"]))
    if any(np.any(value[inactive_candidate] != 0) for value in arrays.values()):
        raise RuntimeError("a noncorrective query has nonzero chemical payload")
    for query in np.flatnonzero(active):
        left, right = map(int, action["query_ptr"][query : query + 2])
        if any(abs(float(np.sum(value[left:right]))) > 1e-5 for value in arrays.values()):
            raise RuntimeError("an active chemical residual is not candidate-centred")
    return active, prior, {
        "report": report,
        "report_sha256": sha256_file(report_path),
        "ledger_sha256": sha256_file(args.action_ledger_dir / "ledger.npz"),
        "formula_fold": ledger_fold,
    }


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    store: SpectrumStore,
    eval_rows: np.ndarray,
    body: dict[str, np.ndarray],
    queries: np.ndarray,
    official: np.ndarray,
    row_position: dict[int, int],
    device: torch.device,
    args: argparse.Namespace,
    label: str,
) -> tuple[dict, float]:
    encoded = encode_rows(
        model, store, eval_rows, device, args.eval_batch_size, args.amp, label
    )
    positions = np.asarray([row_position[int(row)] for row in eval_rows], dtype=np.int64)
    preservation = float(np.mean(np.sum(encoded * np.asarray(official[positions]), axis=1)))
    return evaluate_subset(body, queries, official, row_position, eval_rows, encoded), preservation


def main() -> None:
    args = arguments()
    started = time.time()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite output: {args.output}")
    if (
        args.folds != 5
        or args.inner_fold == args.outer_fold
        or args.alpha < 0
        or args.epochs < 1
        or args.batch_queries < 1
        or args.references_per_molecule < 1
        or args.head_lr < args.backbone_lr
        or args.chemical_huber <= 0
        or args.chemical_gradient_ratio < 0
        or args.chemical_gradient_scale_cap <= 0
        or args.bootstrap_draws < 10_000
    ):
        raise ValueError("invalid fixed training or split configuration")
    if (args.prior_arm == "none") != (args.alpha == 0):
        raise ValueError("the clean duplicate must use alpha=0 and treatment arms alpha>0")
    required = [
        args.manifest,
        args.action_graph_dir / "graph.npz",
        args.action_graph_dir / "report.json",
        args.action_ledger_dir / "ledger.npz",
        args.action_ledger_dir / "report.json",
        args.token_dir / "report.json",
        args.token_dir / "rows.npy",
        args.token_dir / "official_embeddings_f32.npy",
        args.data,
        args.official_checkpoint,
        args.architecture_checkpoint,
    ]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)
    token_report = json.loads((args.token_dir / "report.json").read_text(encoding="utf-8"))
    if token_report.get("status") != "chemaware_corrected_manifest_token_cache_complete":
        raise RuntimeError("training requires the complete official embedding cache")
    action_report = json.loads((args.action_graph_dir / "report.json").read_text(encoding="utf-8"))
    if action_report.get("scope", {}).get("training_formula_only") is not True:
        raise RuntimeError("action graph is not restricted to training formulas")

    with np.load(args.manifest, allow_pickle=True) as loaded:
        manifest = {key: loaded[key] for key in loaded.files}
    with np.load(args.action_graph_dir / "graph.npz", allow_pickle=True) as loaded:
        action = {key: loaded[key] for key in loaded.files}
    active, prior, prior_meta = load_prior(args, action)
    active_query = np.flatnonzero(active)
    effective_action = active_query[:4] if args.smoke else active_query
    if np.any(np.isin(prior_meta["formula_fold"][active_query], (args.inner_fold, args.outer_fold))):
        raise RuntimeError("a chemical action crosses into held formula folds")

    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    if len(row_position) != len(rows):
        raise RuntimeError("official embedding cache contains duplicate spectrum rows")
    reachable_action = set(map(int, np.unique(np.r_[action["query_row"], action["pair_candidate_row"]])))
    if not reachable_action.issubset(row_position):
        raise RuntimeError("official cache misses action-graph spectra")

    graph_error, graph_margin = graph_official_outcomes(action)
    if not np.all(graph_error[active_query]):
        raise RuntimeError("strict corrective membership contains a baseline-correct query")
    safety_pool = np.flatnonzero(~graph_error & ~active)
    safety_count = args.safety_identities or len(effective_action)
    if args.smoke:
        safety_count = min(4, safety_count)

    manifest_fold = stable_formula_folds(
        manifest["query_formula"], args.folds, args.fold_seed
    )
    inner_pool = np.flatnonzero(manifest_fold == args.inner_fold)
    effective_eval = min(args.max_eval_identities, 8) if args.smoke else args.max_eval_identities
    inner = identity_balanced_queries(
        inner_pool,
        manifest["query_ik14"],
        np.random.default_rng(args.seed + 19),
        effective_eval,
    )
    eval_rows = evaluation_rows(manifest, inner)
    if any(int(row) not in row_position for row in eval_rows):
        raise RuntimeError("official cache misses held-formula evaluation spectra")

    rng = np.random.default_rng(args.seed + 101)
    scheduled_rows = set(map(int, eval_rows))
    schedules: list[list[tuple[str, np.ndarray, dict, np.ndarray]]] = []
    coverage = np.zeros(len(action["query_row"]), dtype=np.int16)
    effective_epochs = 1 if args.smoke else args.epochs
    for _epoch in range(effective_epochs):
        action_order = effective_action.copy()
        rng.shuffle(action_order)
        safety_order = select_safety_queries(safety_pool, graph_margin, safety_count, rng)
        action_weight = formula_identity_epoch_weights(
            action_order, action["query_formula"], "formula_identity"
        )
        safety_weight = formula_identity_epoch_weights(
            safety_order, action["query_formula"], "formula_identity"
        )
        epoch_batches = []
        for kind, query_order, all_weight in (
            ("action", action_order, action_weight),
            ("safety", safety_order, safety_weight),
        ):
            for left in range(0, len(query_order), args.batch_queries):
                queries = query_order[left : left + args.batch_queries]
                batch = select_official_boundary_batch(
                    action,
                    queries,
                    row_position,
                    official,
                    args.references_per_molecule,
                    rng,
                )
                weights = all_weight[left : left + len(queries)]
                scheduled_rows.update(map(int, rows[batch["query_cache"]]))
                scheduled_rows.update(map(int, rows[batch["reference_cache"]]))
                epoch_batches.append((kind, queries.copy(), batch, weights.copy()))
                if kind == "action":
                    coverage[queries] += 1
        rng.shuffle(epoch_batches)
        schedules.append(epoch_batches)
    expected_coverage = effective_epochs
    if not np.all(coverage[effective_action] == expected_coverage):
        raise RuntimeError("coverage-first schedule omitted or duplicated a corrective action")

    replay_count = min(64, len(scheduled_rows))
    scheduled_rows_array = np.asarray(sorted(scheduled_rows), dtype=np.int64)
    replay_rows = scheduled_rows_array[
        np.linspace(0, len(scheduled_rows_array) - 1, replay_count, dtype=np.int64)
    ]
    preflight = {
        "status": "CHEMAWARE_ICEBERG_RESIDUAL_SHARED_PREFLIGHT_PASS",
        "arm": args.prior_arm,
        "alpha": args.alpha,
        "strict_corrective_queries": int(len(active_query)),
        "scheduled_corrective_queries": int(len(effective_action)),
        "corrective_formula_clusters": int(len(np.unique(action["query_formula"][effective_action]))),
        "safety_queries_per_epoch": int(safety_count),
        "inner_evaluation_queries": int(len(inner)),
        "outer_fold_consumed": False,
        "optimizer_steps": int(sum(map(len, schedules))),
        "raw_spectra_materialized": int(len(scheduled_rows_array)),
        "contracts": {
            "research_target_is_current_dreams_error_boundary": True,
            "corrective_membership_shared_across_arms": True,
            "noncorrective_chemical_weight_exact_zero": True,
            "alpha_zero_chemical_objective_exactly_absent": True,
            "official_top_reference_mandatory_per_molecule": True,
            "candidate_centered_minimum_l2_residual": True,
            "query_equal_formula_equal_action_mass": True,
            "coverage_before_recycling": True,
            "same_encoder_for_query_and_reference": True,
            "candidate_or_teacher_input_at_inference": False,
            "fixed_epoch_no_arm_specific_checkpoint_selection": True,
        },
        "provenance": {
            "manifest_sha256": sha256_file(args.manifest),
            "action_graph_sha256": sha256_file(args.action_graph_dir / "graph.npz"),
            "action_graph_report_sha256": sha256_file(args.action_graph_dir / "report.json"),
            "action_ledger_sha256": prior_meta["ledger_sha256"],
            "action_ledger_report_sha256": prior_meta["report_sha256"],
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
            "token_report_sha256": sha256_file(args.token_dir / "report.json"),
        },
    }
    args.output.mkdir(parents=True)
    (args.output / "preflight.json").write_text(
        json.dumps(preflight, indent=2) + "\n", encoding="utf-8"
    )
    if args.preflight_only:
        print(json.dumps(preflight, indent=2))
        return
    if not args.smoke and (not args.device.startswith("cuda") or not torch.cuda.is_available()):
        raise RuntimeError("formal shared-embedding training requires CUDA")

    torch.set_num_threads(args.torch_threads)
    seed_everything(args.seed)
    device = torch.device(args.device)
    store = SpectrumStore(args.data, scheduled_rows_array, args.n_highest_peaks)
    model, initialization = load_base_model(
        args.official_checkpoint,
        args.architecture_checkpoint,
        device,
        args.n_highest_peaks,
    )
    replay = encode_rows(
        model, store, replay_rows, device, args.eval_batch_size, False, "official-replay"
    )
    replay_position = np.asarray([row_position[int(row)] for row in replay_rows], dtype=np.int64)
    replay_cosine = np.sum(replay * np.asarray(official[replay_position]), axis=1)
    if float(np.min(replay_cosine)) < 0.999:
        raise RuntimeError(f"official checkpoint replay drift: min cosine={float(np.min(replay_cosine))}")

    capacity = unfreeze_last_blocks(model, args.unfreeze_blocks)
    model.eval()
    head = [parameter for parameter in model.head.parameters() if parameter.requires_grad]
    head_ids = {id(parameter) for parameter in head}
    backbone = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in head_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone, "lr": args.backbone_lr, "weight_decay": 0.0},
            {"params": head, "lr": args.head_lr, "weight_decay": args.weight_decay},
        ]
    )
    initial_encoded = np.asarray(
        official[np.asarray([row_position[int(row)] for row in eval_rows], dtype=np.int64)]
    )
    initial = evaluate_subset(
        manifest, inner, official, row_position, eval_rows, initial_encoded
    )
    del initial_encoded
    history = []
    global_step = 0
    for epoch, epoch_batches in enumerate(schedules, start=1):
        totals = {
            "loss": 0.0,
            "listwise": 0.0,
            "inbatch": 0.0,
            "margin_floor": 0.0,
            "preserve": 0.0,
            "chemical": 0.0,
            "chemical_gradient_scale": 0.0,
            "grad_norm": 0.0,
            "clip_fraction": 0.0,
        }
        chemical_batches = 0
        epoch_step = 0
        for kind, queries, batch, weights in epoch_batches:
            query_rows = rows[batch["query_cache"]]
            reference_rows = rows[batch["reference_cache"]]
            spectra = torch.cat((store.get(query_rows), store.get(reference_rows))).to(device)
            encoded = forward_embeddings(model, spectra, args.amp)
            query_z = encoded[: len(queries)]
            reference_z = encoded[len(queries) :]
            query_x = torch.from_numpy(
                np.array(official[batch["query_cache"]], dtype=np.float32, copy=True)
            ).to(device)
            reference_x = torch.from_numpy(
                np.array(official[batch["reference_cache"]], dtype=np.float32, copy=True)
            ).to(device)
            query_weight = torch.as_tensor(weights, device=device, dtype=query_z.dtype)
            listwise, _, inbatch, _, margin_floor = listwise_losses(
                query_z,
                reference_z,
                None,
                batch["candidate_ptr"],
                batch["reference_ptr"],
                batch["reference_edge"],
                args.temperature,
                query_x,
                reference_x,
                args.margin_floor_slack,
                query_weight,
            )
            preserve = torch.cat(
                (
                    1 - torch.sum(query_z * query_x, dim=1),
                    1 - torch.sum(reference_z * reference_x, dim=1),
                )
            ).mean()
            base_loss = (
                args.lambda_listwise * listwise
                + args.lambda_inbatch * inbatch
                + args.lambda_margin_floor * margin_floor
                + args.lambda_preserve * preserve
            )
            chemical = query_z.sum() * 0.0
            chemical_scale = 0.0
            if kind == "action":
                student_score = molecule_max_scores(
                    query_z,
                    reference_z,
                    batch["candidate_ptr"],
                    batch["reference_ptr"],
                    batch["reference_edge"],
                )
                official_score = molecule_max_scores(
                    query_x,
                    reference_x,
                    batch["candidate_ptr"],
                    batch["reference_ptr"],
                    batch["reference_edge"],
                )
                target = torch.from_numpy(prior[batch["molecule_index"]]).to(
                    device=device, dtype=query_z.dtype
                )
                chemical = candidate_centered_residual_loss(
                    student_score,
                    official_score,
                    target,
                    batch["candidate_ptr"],
                    alpha=args.alpha,
                    huber_delta=args.chemical_huber,
                    query_weight=query_weight,
                )
                if args.alpha > 0:
                    chemical_batches += 1
                    chemical_scale = 1.0
                    if args.chemical_gradient_ratio > 0:
                        base_gradient = torch.autograd.grad(
                            base_loss, (query_z, reference_z), retain_graph=True,
                            allow_unused=True,
                        )
                        chemical_gradient = torch.autograd.grad(
                            args.lambda_chemical * chemical,
                            (query_z, reference_z), retain_graph=True, allow_unused=True,
                        )
                        base_norm = torch.sqrt(sum(
                            torch.sum(value.detach().float() ** 2)
                            for value in base_gradient if value is not None
                        ))
                        chemical_norm = torch.sqrt(sum(
                            torch.sum(value.detach().float() ** 2)
                            for value in chemical_gradient if value is not None
                        ))
                        if float(chemical_norm) > 0:
                            desired = args.chemical_gradient_ratio * base_norm / chemical_norm
                            chemical_scale = float(torch.clamp(
                                desired, min=0.0, max=args.chemical_gradient_scale_cap
                            ))
                        else:
                            chemical_scale = 0.0
            loss = base_loss + args.lambda_chemical * chemical_scale * chemical
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.grad_clip,
            ))
            optimizer.step()
            global_step += 1
            epoch_step += 1
            for key, value in (
                ("loss", loss),
                ("listwise", listwise),
                ("inbatch", inbatch),
                ("margin_floor", margin_floor),
                ("preserve", preserve),
                ("chemical", chemical),
            ):
                totals[key] += float(value.detach())
            totals["chemical_gradient_scale"] += chemical_scale
            totals["grad_norm"] += norm
            totals["clip_fraction"] += float(norm > args.grad_clip)
            if global_step % 25 == 0:
                print(
                    f"[residual epoch={epoch}] step={global_step} "
                    f"loss={totals['loss']/max(1, epoch_step):.5f}",
                    flush=True,
                )
        batch_count = len(epoch_batches)
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "train": {key: value / max(1, batch_count) for key, value in totals.items()},
            "chemical_batches": int(chemical_batches),
        }
        history.append(record)
        print(
            f"residual epoch={epoch}/{effective_epochs} completed "
            f"mean_loss={record['train']['loss']:.6f} "
            f"clip={record['train']['clip_fraction']:.3f}",
            flush=True,
        )

    final_eval, preservation = evaluate_model(
        model,
        store,
        eval_rows,
        manifest,
        inner,
        official,
        row_position,
        device,
        args,
        "iceberg-residual-final",
    )
    delta = (final_eval["new_rank"] == 1).astype(float) - (
        final_eval["old_rank"] == 1
    ).astype(float)
    ci = formula_bootstrap(
        delta,
        manifest["query_formula"][inner],
        args.seed + 901,
        args.bootstrap_draws,
    )
    checkpoint = {
        "status": "chemaware_iceberg_residual_shared_arm_checkpoint",
        "model_state": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "initialization": f"official_dreams:{initialization}",
        "capacity": capacity,
        "prior_arm": args.prior_arm,
        "alpha": args.alpha,
        "fixed_final_epoch": effective_epochs,
        "query_reference_encoder_shared": True,
        "inference_clean_spectrum_only": True,
        "candidate_or_teacher_input_at_inference": False,
        "causal_chemistry_pass": False,
        "release_eligible": False,
        "provenance": preflight["provenance"],
    }
    temporary_checkpoint = args.output / "final_shared_encoder.pt.tmp"
    torch.save(checkpoint, temporary_checkpoint)
    temporary_checkpoint.replace(args.output / "final_shared_encoder.pt")
    mean_clip = float(np.mean([row["train"]["clip_fraction"] for row in history]))
    report = {
        "status": "CHEMAWARE_ICEBERG_RESIDUAL_SHARED_ARM_COMPLETE",
        "formal": not args.smoke,
        "arm": args.prior_arm,
        "alpha": args.alpha,
        "preflight": preflight,
        "optimization": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "initial_inner": initial["summary"],
        "final_inner": final_eval["summary"],
        "formula_bootstrap": ci,
        "preservation": preservation,
        "mean_clip_fraction": mean_clip,
        "fixed_final_epoch_no_arm_specific_selection": True,
        "causal_chemistry_status": "NOT_EVALUATED_SINGLE_ARM",
        "release_eligible": False,
        "history": history,
        "runtime_seconds": time.time() - started,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(
        args.output / "inner_per_query.npz",
        query=inner.astype(np.int64),
        formula=manifest["query_formula"][inner],
        old_rank=final_eval["old_rank"].astype(np.int16),
        new_rank=final_eval["new_rank"].astype(np.int16),
        old_margin=final_eval["old_margin"].astype(np.float32),
        new_margin=final_eval["new_margin"].astype(np.float32),
        candidate_count=final_eval["candidate_count"].astype(np.int16),
    )
    print(json.dumps({
        "status": report["status"],
        "arm": args.prior_arm,
        "alpha": args.alpha,
        "final_inner": final_eval["summary"],
        "formula_bootstrap": ci,
        "preservation": preservation,
        "causal_chemistry_status": report["causal_chemistry_status"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
