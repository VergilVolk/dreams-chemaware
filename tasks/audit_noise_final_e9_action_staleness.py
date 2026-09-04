"""E9: audit whether frozen S3A peak paths become stale after fine-tuning.

The mature E8 encoder is evaluated on the already-consumed held formula fold.
For every preregistered curriculum cell, this script compares the frozen path
mined by official DreaMS with a path re-mined from the current student using
the original S3A sequential algorithm. Candidate identities are used only in
this training-development audit. No P2b score, sealed P3 query, outcome label,
or downstream reranker enters action selection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from audit_noise_v3_s2_sequential import current_context, first_unused  # noqa: E402
from noise_final_core import CandidateGraph, json_dump, sha256_file  # noqa: E402
from noise_v3_core import (  # noqa: E402
    CONFOUNDER_ONLY,
    attenuate_and_renormalize,
    attenuate_sequence,
    rank_gradient_targets,
    rank_role_targets,
)
from train_e1_identity import load_base_model, torch_load_compat  # noqa: E402
from train_noise_final_e4a_direct_augmentation import FIXED_POLICY  # noqa: E402
from train_noise_final_r2_shared_encoder import (  # noqa: E402
    SpectrumStore,
    encode_rows,
    formula_bootstrap_mean,
    forward_embeddings,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--r0-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_r0_faithful_s3a")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--student-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_e9_action_staleness")
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--selection-batch-size", type=int, default=32)
    parser.add_argument("--encode-batch-size", type=int, default=128)
    parser.add_argument("--top-k-negatives", type=int, default=8)
    parser.add_argument("--softmax-temperature", type=float, default=0.10)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-queries", type=int, default=0, help="Smoke only; 0 is formal")
    return parser.parse_args()


def parse_path(value: object) -> tuple[int, ...]:
    text = str(value).strip()
    if not text:
        return ()
    output = tuple(int(part) for part in text.split(",") if part != "")
    if len(output) != len(set(output)) or any(token <= 0 for token in output):
        raise RuntimeError(f"invalid frozen peak path: {text}")
    return output


def load_student(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model, kind = load_base_model(args.official_checkpoint, args.architecture_checkpoint, device, 100)
    if kind not in {"official_embedding", "official_embedding_slim"}:
        raise RuntimeError("E9 requires official DreaMS initialization")
    package = torch_load_compat(args.student_checkpoint, map_location="cpu")
    if package.get("status") != "noise_final_e4a_direct_shared_dreams_encoder":
        raise RuntimeError("student checkpoint is not an E4-A shared encoder")
    if not package.get("inference_clean_only") or package.get("P2b_used"):
        raise RuntimeError("student checkpoint violates clean-embedding/P2b contract")
    model.load_state_dict(package["model_state"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def action_cells() -> set[tuple[str, float, int]]:
    return {(str(selector), float(dose), int(step)) for selector, dose, step in FIXED_POLICY["curriculum"]}


def rank_margin(graph: CandidateGraph, query: int, query_vector: np.ndarray,
                embeddings: np.ndarray, embedding_index: dict[int, int]) -> tuple[int, float, int]:
    _, rows, ptr, _ = graph.query_block(query)
    candidate = embeddings[[embedding_index[int(row)] for row in rows]]
    pair_scores = candidate @ np.asarray(query_vector, dtype=np.float32)
    molecule_scores = np.maximum.reduceat(pair_scores, ptr[:-1])
    positive = float(molecule_scores[0])
    hard_local = int(np.argmax(molecule_scores[1:])) + 1
    rank = 1 + int(np.sum(molecule_scores[1:] >= positive))
    hard_left, hard_right = map(int, ptr[hard_local:hard_local + 2])
    hard_pair = hard_left + int(np.argmax(pair_scores[hard_left:hard_right]))
    return rank, positive - float(molecule_scores[hard_local]), int(rows[hard_pair])


def mine_paths(model: torch.nn.Module, graph: CandidateGraph, queries: np.ndarray,
               selector: str, attenuation: float, maximum_steps: int,
               embeddings: np.ndarray, embedding_index: dict[int, int],
               tensor_cache: dict[int, torch.Tensor], device: torch.device,
               args: argparse.Namespace) -> tuple[dict[int, tuple[int, ...]], dict[int, tuple[int, ...]]]:
    states = {int(q): tensor_cache[int(graph.query_row[int(q)])].clone() for q in queries}
    paths: dict[int, list[int]] = {int(q): [] for q in queries}
    hard_rows: dict[int, list[int]] = {int(q): [] for q in queries}
    active = set(map(int, queries))
    for step in range(1, maximum_steps + 1):
        ordered = np.asarray(sorted(active), dtype=np.int64)
        if not len(ordered):
            break
        for left in range(0, len(ordered), args.selection_batch_size):
            local_queries = ordered[left:left + args.selection_batch_size]
            block = torch.stack([states[int(query)] for query in local_queries]).to(device)
            block.requires_grad_(selector == "candidate_gradient")
            current = forward_embeddings(model, block, False)
            current_np = current.detach().float().cpu().numpy()
            contexts = [current_context(
                graph, int(query), vector, graph.dreams_column, embeddings,
                embedding_index, tensor_cache, states[int(query)],
                args.top_k_negatives, args.fragment_tolerance,
            ) for query, vector in zip(local_queries, current_np)]
            gradients = None
            if selector == "candidate_gradient":
                positive = torch.as_tensor(np.stack([
                    embeddings[embedding_index[int(context[1].positive_row)]] for context in contexts
                ]), device=device, dtype=current.dtype)
                max_k = max(len(context[1].negative_rows) for context in contexts)
                negatives = torch.zeros((len(local_queries), max_k, embeddings.shape[1]), device=device, dtype=current.dtype)
                valid = torch.zeros((len(local_queries), max_k), device=device, dtype=torch.bool)
                for index, context in enumerate(contexts):
                    rows = context[1].negative_rows
                    negatives[index, :len(rows)] = torch.as_tensor(np.stack([
                        embeddings[embedding_index[int(row)]] for row in rows
                    ]), device=device, dtype=current.dtype)
                    valid[index, :len(rows)] = True
                pos_similarity = torch.sum(current * positive, dim=1)
                neg_similarity = torch.einsum("bd,bkd->bk", current, negatives).masked_fill(~valid, -1e9)
                weights = torch.softmax(neg_similarity / args.softmax_temperature, dim=1).detach()
                objective = pos_similarity - torch.sum(weights * neg_similarity, dim=1)
                gradients = torch.autograd.grad(objective.sum(), block)[0][:, :, 1]
            for offset, query_value in enumerate(local_queries):
                query = int(query_value)
                roles = contexts[offset][2]
                if selector == "candidate_gradient":
                    ranked = rank_gradient_targets(
                        states[query], gradients[offset].detach().float().cpu().numpy(),
                        roles, attenuation, max_targets=100, protect_identity=True,
                    )
                elif selector == "role_confounder":
                    ranked = rank_role_targets(states[query], roles, CONFOUNDER_ONLY, max_targets=100)
                else:
                    raise RuntimeError(f"unregistered E9 selector: {selector}")
                target = first_unused(ranked, set(paths[query]))
                if target is None:
                    active.discard(query)
                    continue
                if target <= 0 or int(roles[target]) == 0:
                    raise RuntimeError("online action violated precursor/identity protection")
                paths[query].append(int(target))
                hard_rows[query].append(int(contexts[offset][1].negative_rows[0]))
                states[query] = attenuate_and_renormalize(states[query], int(target), attenuation)
        print(f"[E9 mine] {selector} dose={attenuation:.2f} step={step} active={len(active):,}/{len(queries):,}", flush=True)
    return ({q: tuple(v) for q, v in paths.items()}, {q: tuple(v) for q, v in hard_rows.items()})


@torch.no_grad()
def encode_variants(model: torch.nn.Module, variants: list[torch.Tensor], device: torch.device,
                    batch_size: int) -> np.ndarray:
    output: list[np.ndarray] = []
    for left in range(0, len(variants), batch_size):
        block = torch.stack(variants[left:left + batch_size]).to(device)
        values = forward_embeddings(model, block, False).float().cpu().numpy()
        if not np.all(np.isfinite(values)):
            raise RuntimeError("E9 variant encoding produced non-finite values")
        output.append(values)
    return np.concatenate(output, axis=0)


def jaccard(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    union = set(left) | set(right)
    return len(set(left) & set(right)) / len(union) if union else 1.0


def cluster_summary(frame: pd.DataFrame, args: argparse.Namespace, offset: int) -> dict:
    clean = frame["clean_rank"].to_numpy(int) == 1
    frozen = frame["frozen_rank"].to_numpy(int) == 1
    online = frame["online_rank"].to_numpy(int) == 1
    top1_delta = online.astype(float) - frozen.astype(float)
    margin_delta = frame["online_margin"].to_numpy(float) - frame["frozen_margin"].to_numpy(float)
    formulas = frame["query_formula"].astype(str).to_numpy()
    return {
        "rows": int(len(frame)), "identities": int(frame["query_ik14"].nunique()),
        "formulas": int(frame["query_formula"].nunique()),
        "baseline_accuracy": float(np.mean(clean)), "frozen_accuracy": float(np.mean(frozen)),
        "online_accuracy": float(np.mean(online)),
        "frozen_corrected": int(np.sum(~clean & frozen)), "frozen_introduced": int(np.sum(clean & ~frozen)),
        "online_corrected": int(np.sum(~clean & online)), "online_introduced": int(np.sum(clean & ~online)),
        "online_vs_frozen_corrected": int(np.sum(~frozen & online)),
        "online_vs_frozen_introduced": int(np.sum(frozen & ~online)),
        "online_minus_frozen_top1": float(np.mean(top1_delta)),
        "online_minus_frozen_margin": float(np.mean(margin_delta)),
        "top1_formula_cluster_ci": formula_bootstrap_mean(top1_delta, formulas, args.bootstrap_resamples, args.seed + offset),
        "margin_formula_cluster_ci": formula_bootstrap_mean(margin_delta, formulas, args.bootstrap_resamples, args.seed + 10_000 + offset),
        "path_exact_fraction": float(frame["path_exact"].mean()),
        "path_first_token_fraction": float(frame["first_token_match"].mean()),
        "mean_path_jaccard": float(frame["path_jaccard"].mean()),
        "hard_negative_same_fraction": float(frame["hard_negative_same"].mean()),
        "online_complete_fraction": float(frame["online_complete"].mean()),
    }


def main() -> None:
    args = arguments()
    if args.outer_fold not in range(5):
        raise ValueError("outer-fold must be 0..4")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("E9 requires CUDA")
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite E9 output: {args.output_dir}")
    source_per_query = args.student_checkpoint.parent / "held_per_query.csv.gz"
    required = [args.graph, args.data, args.official_checkpoint, args.architecture_checkpoint,
                args.student_checkpoint, source_per_query,
                args.r0_dir / "report.json", args.r0_dir / "training_actions.csv.gz"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    r0 = json.loads((args.r0_dir / "report.json").read_text(encoding="utf-8"))
    if not r0.get("formal") or r0.get("contracts", {}).get("P2b") != "forbidden":
        raise RuntimeError("E9 requires formal P2b-free R0")
    actions = pd.read_csv(args.r0_dir / "training_actions.csv.gz")
    required_columns = {"query_index", "query_row", "query_ik14", "query_formula", "formula_fold",
                        "selector", "attenuation", "step", "target_path", "hard_negative_row"}
    if required_columns - set(actions.columns):
        raise RuntimeError(f"R0 is missing columns: {sorted(required_columns - set(actions.columns))}")
    cells = action_cells()
    selected = np.asarray([
        (str(row.selector), round(float(row.attenuation), 8), int(row.step)) in cells
        and int(row.formula_fold) == args.outer_fold for row in actions.itertuples(index=False)
    ], dtype=bool)
    held = actions.loc[selected].copy()
    if held.empty:
        raise RuntimeError("E9 held curriculum action table is empty")
    if args.max_queries:
        keep = np.sort(held["query_index"].unique())[:args.max_queries]
        held = held.loc[held["query_index"].isin(keep)].copy()
    formal = args.max_queries == 0
    graph = CandidateGraph(args.graph)
    needed_rows = np.unique(np.concatenate([graph.query_row, graph.pair_candidate_row]))
    store = SpectrumStore(args.data, needed_rows, 100)
    tensor_cache = {int(row): store.tensor[index] for index, row in enumerate(store.rows)}
    device = torch.device(args.device)
    model = load_student(args, device)
    embeddings = encode_rows(model, store, store.rows, device, args.encode_batch_size, args.amp, "E9-student")
    embedding_index = {int(row): index for index, row in enumerate(store.rows)}
    mined = {}
    for selector, dose in sorted({(cell[0], cell[1]) for cell in cells}):
        local = held.loc[held["selector"].astype(str).eq(selector) & np.isclose(held["attenuation"].astype(float), dose)]
        queries = np.sort(local["query_index"].unique().astype(np.int64))
        maximum_steps = max(step for s, d, step in cells if s == selector and d == dose)
        mined[(selector, dose)] = mine_paths(model, graph, queries, selector, dose, maximum_steps,
                                             embeddings, embedding_index, tensor_cache, device, args)
    frozen_variants: list[torch.Tensor] = []
    online_variants: list[torch.Tensor] = []
    metadata: list[dict] = []
    for row in held.itertuples(index=False):
        query, selector, dose, step = int(row.query_index), str(row.selector), float(row.attenuation), int(row.step)
        frozen_path = parse_path(row.target_path)
        online_full, hard_full = mined[(selector, dose)]
        online_path = tuple(online_full[query][:step])
        online_hard = tuple(hard_full[query][:step])
        clean = store.one(int(row.query_row))
        frozen_variants.append(attenuate_sequence(clean, frozen_path, dose))
        online_variants.append(attenuate_sequence(clean, online_path, dose))
        metadata.append({
            "query_index": query, "query_row": int(row.query_row), "query_ik14": str(row.query_ik14),
            "query_formula": str(row.query_formula), "selector": selector, "attenuation": dose, "step": step,
            "frozen_path": ",".join(map(str, frozen_path)), "online_path": ",".join(map(str, online_path)),
            "online_complete": len(online_path) == step, "path_exact": frozen_path == online_path,
            "first_token_match": bool(frozen_path and online_path and frozen_path[0] == online_path[0]),
            "path_jaccard": jaccard(frozen_path, online_path),
            "hard_negative_same": bool(online_hard and int(row.hard_negative_row) == int(online_hard[-1])),
            "online_hard_negative_row": int(online_hard[-1]) if online_hard else -1,
            "frozen_hard_negative_row": int(row.hard_negative_row),
        })
    frozen_vectors = encode_variants(model, frozen_variants, device, args.encode_batch_size)
    online_vectors = encode_variants(model, online_variants, device, args.encode_batch_size)
    records = []
    for item, frozen_vector, online_vector in zip(metadata, frozen_vectors, online_vectors):
        query = int(item["query_index"])
        clean_vector = embeddings[embedding_index[int(item["query_row"])]]
        clean_rank, clean_margin, clean_hard = rank_margin(graph, query, clean_vector, embeddings, embedding_index)
        frozen_rank, frozen_margin, _ = rank_margin(graph, query, frozen_vector, embeddings, embedding_index)
        online_rank, online_margin, _ = rank_margin(graph, query, online_vector, embeddings, embedding_index)
        records.append(item | {"clean_rank": clean_rank, "clean_margin": clean_margin,
                               "clean_hard_negative_row": clean_hard, "frozen_rank": frozen_rank,
                               "frozen_margin": frozen_margin, "online_rank": online_rank,
                               "online_margin": online_margin})
    frame = pd.DataFrame(records)
    # Reproduce the mature E8 held ranks before interpreting any path effect.
    source = pd.read_csv(source_per_query, usecols=["query_index", "final_rank"])
    if source["query_index"].duplicated().any():
        raise RuntimeError("E8 held_per_query contains duplicate query indices")
    expected_rank = source.set_index("query_index")["final_rank"].astype(int)
    observed_clean = frame[["query_index", "clean_rank"]].drop_duplicates()
    if observed_clean["query_index"].duplicated().any():
        raise RuntimeError("E9 clean rank is inconsistent across action cells")
    missing_source = set(observed_clean["query_index"].astype(int)) - set(expected_rank.index.astype(int))
    if missing_source:
        raise RuntimeError(f"E9 held queries are absent from E8 evaluation: {sorted(missing_source)[:20]}")
    reproduced = expected_rank.loc[observed_clean["query_index"].to_numpy(int)].to_numpy(int)
    rank_mismatches = int(np.sum(reproduced != observed_clean["clean_rank"].to_numpy(int)))
    if rank_mismatches:
        raise RuntimeError(f"E9 failed to reproduce {rank_mismatches} mature E8 held ranks")
    summaries = {"overall": cluster_summary(frame, args, 0)}
    for index, (selector, group) in enumerate(frame.groupby("selector", sort=True), start=1):
        summaries[str(selector)] = cluster_summary(group, args, index)
    overall = summaries["overall"]
    top1_ci, margin_ci = overall["top1_formula_cluster_ci"], overall["margin_formula_cluster_ci"]
    gates = {
        "student_checkpoint_shared_embedding": True,
        "path_or_hard_negative_drift_ge_20pct": bool(overall["path_exact_fraction"] <= 0.8 or overall["hard_negative_same_fraction"] <= 0.8),
        "online_top1_not_worse": bool(overall["online_minus_frozen_top1"] >= 0),
        "online_correction_balance_nonnegative": bool(overall["online_vs_frozen_corrected"] >= overall["online_vs_frozen_introduced"]),
        "online_formula_evidence_positive": bool(top1_ci["ci_low"] > 0 or margin_ci["ci_low"] > 0),
    }
    report = {
        "status": "noise_final_e9_action_staleness_complete", "formal": formal,
        "student_checkpoint": str(args.student_checkpoint), "held_formula_fold": args.outer_fold,
        "held_action_rows": int(len(frame)), "held_queries": int(frame["query_index"].nunique()),
        "held_identities": int(frame["query_ik14"].nunique()), "held_formulas": int(frame["query_formula"].nunique()),
        "mature_e8_rank_reproduction_mismatches": rank_mismatches,
        "summaries": summaries, "gates": gates, "pass_to_online_remining_training": bool(all(gates.values())),
        "decision": "replace frozen S3A paths with epoch-wise student re-mining" if all(gates.values()) else "do not launch online re-mining training; staleness hypothesis did not pass",
        "contracts": {"shared_clean_embedding_checkpoint": True, "current_student_used_for_action_mining": True,
                      "outcome_labels_used_for_action_selection": False,
                      "candidate_information_training_development_only": True,
                      "P2b": "forbidden", "P3_consumed": False},
        "provenance": {"graph_sha256": sha256_file(args.graph),
                       "r0_actions_sha256": sha256_file(args.r0_dir / "training_actions.csv.gz"),
                       "student_checkpoint_sha256": sha256_file(args.student_checkpoint),
                       "student_held_per_query_sha256": sha256_file(source_per_query),
                       "script_sha256": sha256_file(Path(__file__))},
        "claim_limit": "Held-development audit of action staleness; not a newly trained encoder or deployable retrieval gain.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(args.output_dir / "per_action.csv.gz", index=False, compression="gzip")
    json_dump(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
