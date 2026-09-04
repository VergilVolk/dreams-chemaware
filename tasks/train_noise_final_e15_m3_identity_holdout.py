"""E15-M3: paired identity-held transfer of the multi-action noise ledger.

Two shared clean-spectrum encoders start from the exact same mature E4-A
checkpoint.  The warm control sees the same clean queries, candidate references,
optimizer steps and pooled risk protection, but never sees an action payload or
teacher target.  The noise arm additionally receives the corrective multi-action
loss.  Hyperparameter/epoch choice uses only an internal identity split and
label-free sentinel preservation.  The frozen 256-query M3 panel is read once
after selection.  P2b and P3 are forbidden.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from build_noise_final_e15_m3_identity_split import row_identities, stable_order  # noqa: E402
from noise_final_core import CandidateGraph, json_dump, seed_everything, sha256_file  # noqa: E402
from noise_final_e15_core import project_corrective_against_risk  # noqa: E402
from train_e1_identity import load_base_model, torch_load_compat  # noqa: E402
from train_noise_final_r2_shared_encoder import SpectrumStore, encode_rows, unfreeze_last_block  # noqa: E402
from train_noise_final_e15_m2_overfit import (  # noqa: E402
    SOURCES, corrective_loss, evaluate_queries, frame_references, risk_loss,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--m2-result-report", type=Path, required=True)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--initial-student-checkpoint", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--query-batch-size", type=int, default=4)
    parser.add_argument("--maximum-actions-per-query", type=int, default=12)
    parser.add_argument("--internal-dev-fraction", type=float, default=0.20)
    parser.add_argument("--positive-spectra", type=int, default=2)
    parser.add_argument("--negative-molecules", type=int, default=0)
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
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


LR_CONFIGS = {
    "conservative": {"backbone_lr": 5e-7, "head_lr": 5e-6},
    "standard": {"backbone_lr": 1e-6, "head_lr": 1e-5},
}


def limit_actions(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    selected = []
    for _, block in frame.groupby("query_index", sort=True):
        block = block.sort_values(
            ["source_kind_percentile", "source", "action_family", "action_id"],
            ascending=[False, True, True, True], kind="stable",
        )
        diverse = block.drop_duplicates(["source", "action_family"], keep="first").head(maximum)
        if len(diverse) < maximum:
            diverse = pd.concat([
                diverse, block.loc[~block.index.isin(diverse.index)].head(maximum - len(diverse)),
            ])
        selected.append(diverse)
    output = pd.concat(selected, ignore_index=True)
    output["query_action_count"] = output.groupby("query_index")["action_id"].transform("size").astype(int)
    if output.duplicated(["query_index", "source", "action_id"]).any():
        raise RuntimeError("M3 action limiter produced duplicates")
    return output


def identity_dev_split(frame: pd.DataFrame, fraction: float, seed: int) -> set[str]:
    identities = sorted(set(frame["query_ik14"].astype(str)))
    if not 0 < fraction < 0.5 or len(identities) < 2:
        raise ValueError("invalid M3 internal identity split")
    count = max(1, int(round(fraction * len(identities))))
    ordered = sorted(identities, key=lambda value: stable_order(value, seed, "M3-internal-dev"))
    return set(ordered[:count])


def query_batches(frame: pd.DataFrame, size: int, rng: np.random.Generator) -> list[pd.DataFrame]:
    groups = [block.copy() for _, block in frame.groupby("query_index", sort=True)]
    rng.shuffle(groups)
    return [pd.concat(groups[left:left + size], ignore_index=True) for left in range(0, len(groups), size)]


def filtered_references(
    frame: pd.DataFrame, graph: CandidateGraph, initial: np.ndarray,
    index: dict[int, int], row_ik14: dict[int, str], excluded: set[str],
    positives: int, negatives: int,
) -> tuple[dict[int, tuple[tuple[int, ...], tuple[int, ...]]], list[int]]:
    output = {}
    dropped = []
    for query in sorted(set(frame["query_index"].astype(int))):
        _, rows, ptr, _ = graph.query_block(query)
        qrow = int(graph.query_row[query]); qvec = initial[index[qrow]]
        scores = initial[[index[int(row)] for row in rows]] @ qvec
        left, right = map(int, ptr[:2])
        pos_allowed = [local for local in range(left, right) if row_ik14[int(rows[local])] not in excluded]
        if not pos_allowed:
            dropped.append(query)
            continue
        pos_allowed.sort(key=lambda local: (-float(scores[local]), int(rows[local])))
        pos = tuple(int(rows[local]) for local in pos_allowed[:positives])
        neg = []
        for molecule in range(1, len(ptr) - 1):
            start, stop = map(int, ptr[molecule:molecule + 2])
            allowed = [local for local in range(start, stop) if row_ik14[int(rows[local])] not in excluded]
            if not allowed:
                continue
            best = min(allowed, key=lambda local: (-float(scores[local]), int(rows[local])))
            neg.append((float(scores[best]), int(rows[best])))
        neg.sort(key=lambda item: (-item[0], item[1]))
        if not neg:
            dropped.append(query)
            continue
        chosen = neg if negatives <= 0 else neg[:negatives]
        output[query] = pos, tuple(row for _, row in chosen)
    return output, dropped


def retain_evaluable_queries(
    graph: CandidateGraph, queries: np.ndarray, row_ik14: dict[int, str],
    excluded_negative_identities: set[str],
) -> tuple[np.ndarray, list[int]]:
    kept, dropped = [], []
    for query in map(int, queries):
        _, rows, ptr, _ = graph.query_block(query)
        has_negative = any(
            row_ik14[int(rows[local])] not in excluded_negative_identities
            for molecule in range(1, len(ptr) - 1)
            for local in range(int(ptr[molecule]), int(ptr[molecule + 1]))
        )
        (kept if has_negative else dropped).append(query)
    return np.asarray(kept, dtype=np.int64), dropped


def evaluate_queries_filtered(
    graph: CandidateGraph, queries: np.ndarray, embeddings: np.ndarray,
    index: dict[int, int], row_ik14: dict[int, str], excluded_negative_identities: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate internal development queries without held/sentinel references."""
    ranks = np.empty(len(queries), dtype=np.int16); margins = np.empty(len(queries), dtype=np.float32)
    for position, query in enumerate(queries):
        _, rows, ptr, _ = graph.query_block(int(query))
        qrow = int(graph.query_row[int(query)]); scores = embeddings[[index[int(row)] for row in rows]] @ embeddings[index[qrow]]
        molecule_scores = []
        for molecule in range(len(ptr) - 1):
            start, stop = map(int, ptr[molecule:molecule + 2])
            allowed = [local for local in range(start, stop)
                       if molecule == 0 or row_ik14[int(rows[local])] not in excluded_negative_identities]
            if allowed:
                molecule_scores.append(float(np.max(scores[allowed])))
        if len(molecule_scores) < 2:
            raise RuntimeError(f"internal development query {query} has fewer than two filtered molecules")
        positive = molecule_scores[0]; hardest = max(molecule_scores[1:])
        ranks[position] = 1 + sum(value >= positive for value in molecule_scores[1:])
        margins[position] = positive - hardest
    return ranks, margins


def optimizer(model, config: dict[str, float], args: argparse.Namespace):
    head = [parameter for parameter in model.head.parameters() if parameter.requires_grad]
    backbone = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    trainable = head + backbone
    opt = torch.optim.AdamW([
        {"params": head, "lr": config["head_lr"], "weight_decay": args.weight_decay},
        {"params": backbone, "lr": config["backbone_lr"], "weight_decay": 0.0},
    ])
    return opt, trainable


def cpu_state(model) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def paired_summary(reference_rank: np.ndarray, result_rank: np.ndarray, near: np.ndarray) -> dict[str, float | int]:
    base = reference_rank == 1; result = result_rank == 1
    corrected = int(np.sum(~base & result)); introduced = int(np.sum(base & ~result))
    return {
        "baseline_recall1": float(np.mean(base)), "recall1": float(np.mean(result)),
        "delta_recall1": float(np.mean(result) - np.mean(base)),
        "corrected": corrected, "introduced": introduced,
        "risk_net_lambda2": corrected - 2 * introduced,
        "baseline_near_recall1": float(np.mean(base[near])) if np.any(near) else float("nan"),
        "near_recall1": float(np.mean(result[near])) if np.any(near) else float("nan"),
        "delta_near_recall1": float(np.mean(result[near]) - np.mean(base[near])) if np.any(near) else float("nan"),
    }


def formula_bootstrap(delta: np.ndarray, formula: np.ndarray, repeats: int, seed: int) -> dict[str, float]:
    groups = {key: delta[formula == key] for key in np.unique(formula)}
    keys = list(groups); rng = np.random.default_rng(seed); draws = np.empty(repeats)
    for repeat in range(repeats):
        sampled = rng.integers(0, len(keys), len(keys))
        draws[repeat] = np.mean(np.concatenate([groups[keys[index]] for index in sampled]))
    return {"mean": float(np.mean(delta)), "ci_low": float(np.percentile(draws, 2.5)),
            "ci_high": float(np.percentile(draws, 97.5))}


def train_step(model, opt, trainable, corr, risk, refs, store, initial, index, device, args, noise: bool):
    opt.zero_grad(set_to_none=True)
    risk_frame = risk if risk is not None else corr.drop_duplicates("query_index", keep="first")
    if noise:
        corr_loss, _ = corrective_loss(model, store, corr, refs, initial, index, device, args)
        safe_loss, _ = risk_loss(model, store, risk_frame, refs, initial, index, device, args)
        corr_grad = list(torch.autograd.grad(corr_loss, trainable, allow_unused=True))
        safe_grad = [None if value is None else args.lambda_risk * value for value in
                     torch.autograd.grad(safe_loss, trainable, allow_unused=True)]
        projected, audit = project_corrective_against_risk(corr_grad, safe_grad)
        for parameter, left, right in zip(trainable, projected, safe_grad):
            parameter.grad = right if left is None else left if right is None else left + right
        loss_value = float(corr_loss.detach() + safe_loss.detach())
    else:
        clean = pd.concat([corr.drop_duplicates("query_index", keep="first"), risk_frame], ignore_index=True)
        clean = clean.drop_duplicates("query_index", keep="first")
        loss, _ = risk_loss(model, store, clean, refs, initial, index, device, args)
        loss.backward(); audit = {"conflict": False, "risk_projection_active": False}
        loss_value = float(loss.detach())
    torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
    opt.step(); model.eval()
    return loss_value, audit


def main() -> None:
    args = arguments(); seed_everything(args.seed)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite E15-M3 result: {args.output_dir}")
    required = {
        "split_report": args.split_dir / "report.json", "held": args.split_dir / "held_queries.csv.gz",
        "corrective": args.split_dir / "train_corrective.csv.gz", "harmful": args.split_dir / "train_harmful.csv.gz",
        "sentinel": args.split_dir / "sentinel_queries.csv.gz", "excluded": args.split_dir / "excluded_reference_identities.txt",
        "m2_result_report": args.m2_result_report, "graph": args.graph, "data": args.data,
        "official_checkpoint": args.official_checkpoint, "architecture_checkpoint": args.architecture_checkpoint,
        "initial_student_checkpoint": args.initial_student_checkpoint,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing: raise FileNotFoundError(missing)
    split_report = json.loads(required["split_report"].read_text(encoding="utf-8"))
    m2_report = json.loads(args.m2_result_report.read_text(encoding="utf-8"))
    if not split_report.get("pass_to_identity_holdout_training") or not m2_report.get("pass_to_identity_holdout"):
        raise RuntimeError("M3 requires passing immutable split and M2 capacity reports")
    if split_report["provenance"]["m2_result_report"] != sha256_file(args.m2_result_report):
        raise RuntimeError("M3 split is not bound to the supplied M2 report")
    corrective = pd.read_csv(required["corrective"], low_memory=False)
    harmful = pd.read_csv(required["harmful"], low_memory=False)
    sentinel = pd.read_csv(required["sentinel"], low_memory=False)
    excluded = set(required["excluded"].read_text(encoding="utf-8").splitlines())

    graph = CandidateGraph(args.graph)
    reachable = np.unique(np.concatenate([graph.query_row, graph.pair_candidate_row])).astype(np.int64)
    store = SpectrumStore(args.data, reachable, args.n_highest_peaks)
    device = torch.device(args.device)
    base, _ = load_base_model(args.official_checkpoint, args.architecture_checkpoint, device, args.n_highest_peaks)
    package = torch_load_compat(args.initial_student_checkpoint, map_location="cpu")
    if (package.get("status") != "noise_final_e4a_direct_shared_dreams_encoder"
            or package.get("P2b_used") or not package.get("inference_clean_only")
            or int(package.get("outer_fold", -1)) != args.outer_fold):
        raise RuntimeError("M3 initialization violates the shared clean-embedding contract")
    base.load_state_dict(package["model_state"], strict=True); unfreeze_last_block(base); base.eval()
    initial = encode_rows(base, store, store.rows, device, args.eval_batch_size, False, "E15-M3-init")
    index = {int(row): position for position, row in enumerate(store.rows)}
    initial_rank, initial_margin = evaluate_queries(graph, np.arange(graph.n_queries, dtype=np.int64), initial, index)
    for frame in (corrective, harmful):
        frame["initial_rank"] = frame["query_index"].astype(int).map(lambda query: int(initial_rank[query]))
        frame["initial_margin"] = frame["query_index"].astype(int).map(lambda query: float(initial_margin[query]))
    corrective = corrective.loc[corrective["initial_rank"].ne(1)].copy()
    harmful = harmful.loc[harmful["initial_rank"].eq(1)].copy()
    eligible = pd.concat([corrective, harmful], ignore_index=True)
    dev_ids = identity_dev_split(eligible, args.internal_dev_fraction, args.seed)
    dev_queries = np.asarray(sorted(set(eligible.loc[eligible["query_ik14"].astype(str).isin(dev_ids), "query_index"].astype(int))), dtype=np.int64)
    corrective = corrective.loc[~corrective["query_ik14"].astype(str).isin(dev_ids)].copy()
    harmful = harmful.loc[~harmful["query_ik14"].astype(str).isin(dev_ids)].copy()
    corrective = limit_actions(corrective, args.maximum_actions_per_query)
    harmful = harmful.sort_values(["source_kind_percentile", "action_id"], ascending=[False, True], kind="stable").drop_duplicates("query_index")
    source_weights = m2_report["gradient_calibration"]["corrective_source_weights"]
    corrective["training_weight"] = corrective["source"].astype(str).map(source_weights).astype(float)
    if set(corrective["source"].astype(str)) != set(SOURCES) or harmful.empty or len(dev_queries) == 0:
        raise RuntimeError("M3 internal identity split erased a required training/evaluation stratum")

    row_ik14 = row_identities(args.data, store.rows)
    train_excluded = excluded | dev_ids
    refs_frame = pd.concat([corrective, harmful], ignore_index=True)
    refs, dropped_training_queries = filtered_references(
        refs_frame, graph, initial, index, row_ik14, train_excluded,
        args.positive_spectra, args.negative_molecules,
    )
    retained_training_queries = set(refs)
    corrective = corrective.loc[corrective["query_index"].astype(int).isin(retained_training_queries)].copy()
    harmful = harmful.loc[harmful["query_index"].astype(int).isin(retained_training_queries)].copy()
    if set(corrective["source"].astype(str)) != set(SOURCES) or harmful.empty:
        raise RuntimeError(
            "identity-isolated reference filtering erased an entire corrective source or pooled risk pool: "
            f"dropped_queries={dropped_training_queries}"
        )
    dev_queries, dropped_internal_dev_queries = retain_evaluable_queries(
        graph, dev_queries, row_ik14, excluded,
    )
    if len(dev_queries) == 0:
        raise RuntimeError("identity-isolated filtering erased the entire internal development panel")
    leaked_reference_rows = [
        int(row) for positive, negative in refs.values() for row in (*positive, *negative)
        if row_ik14[int(row)] in train_excluded
    ]
    if leaked_reference_rows:
        raise RuntimeError(f"held/sentinel/internal identity leaked into training references: {leaked_reference_rows[:5]}")
    sentinel_positions = np.asarray([index[int(row)] for row in sentinel["query_row"].astype(int)], dtype=np.int64)
    near_all = graph.query_has_near

    candidates = []
    for config_index, (config_name, config) in enumerate(LR_CONFIGS.items()):
        noise_model = copy.deepcopy(base); control_model = copy.deepcopy(base)
        noise_opt, noise_params = optimizer(noise_model, config, args)
        control_opt, control_params = optimizer(control_model, config, args)
        best = None
        rng = np.random.default_rng(args.seed + config_index)
        for epoch in range(1, args.epochs + 1):
            corr_batches = query_batches(corrective, args.query_batch_size, rng)
            risk_batches = query_batches(harmful, args.query_batch_size, rng)
            risk_slots = set(rng.choice(len(corr_batches), size=min(len(risk_batches), len(corr_batches)), replace=False).tolist())
            risk_iter = iter(risk_batches); losses = {"noise": [], "control": []}; conflicts = 0
            for slot, corr in enumerate(corr_batches):
                risk = next(risk_iter) if slot in risk_slots else None
                value, audit = train_step(noise_model, noise_opt, noise_params, corr, risk, refs, store,
                                          initial, index, device, args, True)
                losses["noise"].append(value); conflicts += int(bool(audit["conflict"]))
                value, _ = train_step(control_model, control_opt, control_params, corr, risk, refs, store,
                                      initial, index, device, args, False)
                losses["control"].append(value)
            noise_emb = encode_rows(noise_model, store, store.rows, device, args.eval_batch_size, False,
                                    f"E15-M3-{config_name}-noise-e{epoch}")
            control_emb = encode_rows(control_model, store, store.rows, device, args.eval_batch_size, False,
                                      f"E15-M3-{config_name}-control-e{epoch}")
            noise_rank, _ = evaluate_queries_filtered(
                graph, dev_queries, noise_emb, index, row_ik14, excluded,
            )
            control_rank, _ = evaluate_queries_filtered(
                graph, dev_queries, control_emb, index, row_ik14, excluded,
            )
            summary = paired_summary(control_rank, noise_rank, near_all[dev_queries])
            noise_pres = float(np.mean(np.sum(initial[sentinel_positions] * noise_emb[sentinel_positions], axis=1)))
            control_pres = float(np.mean(np.sum(initial[sentinel_positions] * control_emb[sentinel_positions], axis=1)))
            eligible_epoch = bool(noise_pres >= 0.995 and control_pres >= 0.995
                                  and summary["delta_near_recall1"] >= 0
                                  and summary["corrected"] > summary["introduced"])
            record = {"config": config_name, "epoch": epoch, "config_values": config,
                      "internal_dev_noise_vs_control": summary,
                      "noise_sentinel_preservation": noise_pres,
                      "control_sentinel_preservation": control_pres,
                      "noise_loss": float(np.mean(losses["noise"])),
                      "control_loss": float(np.mean(losses["control"])),
                      "projection_conflict_steps": conflicts, "eligible": eligible_epoch}
            print(json.dumps(record), flush=True)
            score = (int(eligible_epoch), int(summary["risk_net_lambda2"]),
                     float(summary["delta_recall1"]), -epoch)
            if best is None or score > best[0]:
                best = (score, record, cpu_state(noise_model), cpu_state(control_model))
            del noise_emb, control_emb
        candidates.append(best)
        del noise_model, control_model, noise_opt, control_opt
        torch.cuda.empty_cache()

    selected = max(candidates, key=lambda item: item[0])
    _, selection, noise_state, control_state = selected
    # The frozen held ledger is first materialized only after configuration and
    # epoch selection have completed on internal identities plus label-free
    # sentinel preservation.
    held = pd.read_csv(required["held"], low_memory=False)
    held_queries = held["query_index"].to_numpy(np.int64)
    final_noise = copy.deepcopy(base); final_noise.load_state_dict(noise_state, strict=True); final_noise.eval()
    final_control = copy.deepcopy(base); final_control.load_state_dict(control_state, strict=True); final_control.eval()
    noise_emb = encode_rows(final_noise, store, store.rows, device, args.eval_batch_size, False, "E15-M3-held-noise")
    control_emb = encode_rows(final_control, store, store.rows, device, args.eval_batch_size, False, "E15-M3-held-control")
    noise_rank, noise_margin = evaluate_queries(graph, held_queries, noise_emb, index)
    control_rank, control_margin = evaluate_queries(graph, held_queries, control_emb, index)
    init_held_rank = initial_rank[held_queries]; init_held_margin = initial_margin[held_queries]
    noise_vs_control = paired_summary(control_rank, noise_rank, near_all[held_queries])
    noise_vs_initial = paired_summary(init_held_rank, noise_rank, near_all[held_queries])
    control_vs_initial = paired_summary(init_held_rank, control_rank, near_all[held_queries])
    delta = (noise_rank == 1).astype(float) - (control_rank == 1).astype(float)
    ci = formula_bootstrap(delta, held["query_formula"].astype(str).to_numpy(), args.bootstrap_resamples, args.seed)
    preservation = np.sum(initial * noise_emb, axis=1)
    gates = {
        "selection_used_internal_dev_not_held": True,
        "selected_internal_epoch_eligible": bool(selection["eligible"]),
        "held_noise_beats_control": noise_vs_control["delta_recall1"] > 0,
        "held_corrected_gt_introduced": noise_vs_control["corrected"] > noise_vs_control["introduced"],
        "held_risk_net_positive": noise_vs_control["risk_net_lambda2"] > 0,
        "held_near_nonnegative": noise_vs_control["delta_near_recall1"] >= 0,
        "global_initialization_preservation_ge_0_995": float(np.mean(preservation)) >= 0.995,
        "sentinel_preservation_ge_0_995": float(np.mean(preservation[sentinel_positions])) >= 0.995,
        "control_not_catastrophic": control_vs_initial["delta_recall1"] >= -0.01,
        "P2b_forbidden": True, "P3_not_consumed": True,
    }
    report = {
        "status": "noise_final_e15_m3_identity_holdout_complete", "formal": True,
        "outer_formula_fold": args.outer_fold, "selected": selection,
        "training": {"corrective_actions": int(len(corrective)), "corrective_queries": int(corrective["query_index"].nunique()),
                     "corrective_identities": int(corrective["query_ik14"].nunique()),
                     "risk_queries": int(harmful["query_index"].nunique()), "internal_dev_queries": int(len(dev_queries)),
                     "internal_dev_identities": int(len(dev_ids)), "candidate_reference_excluded_identities": int(len(train_excluded)),
                     "dropped_training_queries_without_legal_negative": int(len(dropped_training_queries)),
                     "dropped_internal_dev_queries_without_legal_negative": int(len(dropped_internal_dev_queries))},
        "held": {"queries": int(len(held)), "identities": int(held["query_ik14"].nunique()),
                 "formulas": int(held["query_formula"].nunique()), "near": int(near_all[held_queries].sum())},
        "held_noise_vs_control": noise_vs_control, "held_noise_vs_initial": noise_vs_initial,
        "held_control_vs_initial": control_vs_initial, "held_formula_cluster_ci": ci,
        "global_initialization_preservation_mean": float(np.mean(preservation)),
        "sentinel_preservation_mean": float(np.mean(preservation[sentinel_positions])),
        "gates": gates, "pass_to_formula_fold": bool(all(gates.values())),
        "contracts": {"shared_clean_spectrum_encoder": True, "paired_warm_control": True,
                      "held_used_for_selection": False, "action_payload_visible_to_control": False,
                      "held_and_sentinel_references_excluded_from_training": True,
                      "P2b": "forbidden", "P3_consumed": False},
        "provenance": {name: sha256_file(path) for name, path in required.items()},
        "claim_limit": "Identity-held action-transfer gate; not a full formula-fold or P3 result.",
    }
    per_query = held.copy()
    per_query["initial_rank"] = init_held_rank; per_query["control_rank"] = control_rank; per_query["noise_rank"] = noise_rank
    per_query["initial_margin"] = init_held_margin; per_query["control_margin"] = control_margin; per_query["noise_margin"] = noise_margin
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="noise_e15_m3_train_", dir=args.output_dir.parent))
    try:
        per_query.to_csv(staging / "held_per_query.csv.gz", index=False, compression="gzip")
        json_dump(staging / "report.json", report)
        torch.save({"status": "noise_final_e15_m3_shared_dreams_encoder", "model_state": noise_state,
                    "outer_fold": args.outer_fold, "selected_config": selection["config"],
                    "selected_epoch": selection["epoch"], "inference_clean_only": True,
                    "P2b_used": False}, staging / "shared_encoder.pt")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
