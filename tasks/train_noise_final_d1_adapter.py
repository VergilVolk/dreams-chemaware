"""D1: clean-only, query-side peak-token adapter for a new embedding space.

The DreaMS backbone, official projection head, and candidate reference
embeddings are frozen.  Only a zero-initialized peak-token adapter is trained.
No P2b score, chemical rule, masked view, or action-teacher field is read.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks"))
sys.path.insert(0, str(ROOT))

from noise_final_core import (  # noqa: E402
    CandidateGraph, ZeroInitPeakAdapter, json_dump, load_embedding_cache,
    molecule_scores_from_pairs, seed_everything, sha256_file, strict_rank,
)
from train_e1_identity import load_base_model, preprocess_spectrum  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_d0_manifest")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--embedding-cache", type=Path, default=ROOT / "data/validation/g8r_p2_official_embeddings.npz")
    parser.add_argument("--official-ckpt", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-ckpt", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/validation/g8r_noise_final_d1_adapter")
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--delta-bound", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=0.10)
    parser.add_argument("--lambda-distill", type=float, default=0.20)
    parser.add_argument("--lambda-preserve", type=float, default=1.0)
    parser.add_argument("--lambda-positive-floor", type=float, default=0.5)
    parser.add_argument("--lambda-safety", type=float, default=1.0)
    parser.add_argument("--safety-margin", type=float, default=0.0)
    parser.add_argument("--minimum-preservation", type=float, default=0.995)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-queries", type=int, default=0)
    parser.add_argument("--max-eval-queries", type=int, default=0)
    return parser.parse_args()


class QueryDataset(Dataset):
    def __init__(self, path: Path, graph: CandidateGraph, query_indices: np.ndarray, n_highest: int):
        self.path = path
        self.graph = graph
        self.query_indices = np.asarray(query_indices, dtype=np.int64)
        self.n_highest = n_highest
        self.handle = None

    def __len__(self) -> int:
        return len(self.query_indices)

    def __getitem__(self, position: int):
        if self.handle is None:
            self.handle = h5py.File(self.path, "r")
        query = int(self.query_indices[position])
        row = int(self.graph.query_row[query])
        raw = np.asarray(self.handle["spectrum"][row])
        precursor = float(self.handle["precursor_mz"][row])
        return query, preprocess_spectrum(raw, precursor, self.n_highest)


def candidate_positions(graph: CandidateGraph, row_index: dict[int, int]) -> np.ndarray:
    """Return one cache position per graph spectrum pair (single compact tensor)."""
    output = np.full(len(graph.pair_candidate_row), -1, dtype=np.int64)
    missing = set()
    for pair, row in enumerate(graph.pair_candidate_row):
        position = row_index.get(int(row))
        if position is None:
            missing.add(int(row))
        else:
            output[pair] = position
    if missing:
        raise RuntimeError(f"official embedding cache lacks {len(missing)} candidate rows")
    return output


def forward_queries(model, adapter, spectra, device):
    spectra = spectra.to(device, non_blocking=True)
    # A frozen backbone must stay in eval mode.  No dropout is permitted here.
    model.eval()
    with torch.no_grad():
        tokens = model.backbone(spectra, None)
        official = F.normalize(model.head(tokens[:, 0, :]), dim=-1).float()
        peaks = tokens[:, 1:, :].float()
    peak_mz = spectra[:, 1:, 0]
    peak_intensity = spectra[:, 1:, 1]
    peak_mask = peak_mz > 0
    adapted, delta, weights = adapter(official, peaks, peak_mz, peak_intensity, peak_mask)
    return official, adapted, delta, weights


def losses_for_batch(
    query_indices, official, adapted, graph, candidate_pos, reference_embeddings,
    identity_weight, temperature, args,
):
    individual = []
    diagnostics = {"ce": [], "distill": [], "preserve": [], "positive_floor": [], "safety": []}
    for local, query_tensor in enumerate(query_indices):
        query = int(query_tensor)
        pair_slice, _, ptr, _ = graph.query_block(query)
        refs = reference_embeddings[candidate_pos[pair_slice]]
        new_scores = molecule_scores_from_pairs(adapted[local], refs, ptr)
        old_scores = molecule_scores_from_pairs(official[local], refs, ptr).detach()
        ce = F.cross_entropy((new_scores / temperature).unsqueeze(0), torch.zeros(1, dtype=torch.long, device=new_scores.device))
        teacher_probability = torch.softmax(old_scores / temperature, dim=0)
        distill = F.kl_div(torch.log_softmax(new_scores / temperature, dim=0), teacher_probability, reduction="sum")
        preserve = 1.0 - torch.sum(adapted[local] * official[local])
        positive_floor = F.relu(old_scores[0] - new_scores[0])
        if strict_rank(old_scores.detach().cpu().numpy()) == 1:
            # Protect the teacher margin before a correct query flips.  The D1
            # pilot used only a post-flip hinge, which was ~1e-6 and therefore
            # supplied essentially no safety gradient.  Here every contraction
            # beyond the allowed tolerance is penalized.
            old_margin = old_scores[0] - torch.max(old_scores[1:])
            new_margin = new_scores[0] - torch.max(new_scores[1:])
            safety = F.relu(old_margin - new_margin - args.safety_margin)
        else:
            safety = torch.zeros((), device=new_scores.device)
        total = (
            ce + args.lambda_distill * distill + args.lambda_preserve * preserve
            + args.lambda_positive_floor * positive_floor + args.lambda_safety * safety
        )
        individual.append(total * float(identity_weight[query]))
        for name, value in (("ce", ce), ("distill", distill), ("preserve", preserve),
                            ("positive_floor", positive_floor), ("safety", safety)):
            diagnostics[name].append(float(value.detach()))
    return torch.stack(individual).mean(), {key: float(np.mean(value)) for key, value in diagnostics.items()}


@torch.no_grad()
def evaluate(
    model, adapter, loader, graph, candidate_pos, reference_embeddings,
    baseline_rank, device, max_queries=0,
):
    adapter.eval()
    records = []
    seen = 0
    for query_indices, spectra in loader:
        official, adapted, delta, _ = forward_queries(model, adapter, spectra, device)
        for local, query_tensor in enumerate(query_indices):
            query = int(query_tensor)
            pair_slice, _, ptr, _ = graph.query_block(query)
            refs = reference_embeddings[candidate_pos[pair_slice]]
            old_scores = molecule_scores_from_pairs(official[local], refs, ptr)
            new_scores = molecule_scores_from_pairs(adapted[local], refs, ptr)
            old_rank = strict_rank(old_scores.cpu().numpy())
            new_rank = strict_rank(new_scores.cpu().numpy())
            records.append((
                query, old_rank, new_rank,
                float(torch.sum(official[local] * adapted[local]).cpu()),
                float(delta[local].norm().cpu()),
                float(new_scores[0].cpu()), float(torch.max(new_scores[1:]).cpu()),
            ))
            seen += 1
            if max_queries and seen >= max_queries:
                break
        if max_queries and seen >= max_queries:
            break
    records.sort(key=lambda row: row[0])
    array = np.asarray(records, dtype=np.float64)
    query = array[:, 0].astype(np.int64)
    old_rank = array[:, 1].astype(np.int16)
    new_rank = array[:, 2].astype(np.int16)
    expected = baseline_rank[query]
    mismatch = float(np.mean(old_rank != expected))
    if mismatch > 0.001:
        raise RuntimeError(f"online official baseline disagrees with D0 for {mismatch:.2%} queries")
    near = graph.query_has_near[query]
    base_correct, new_correct = old_rank == 1, new_rank == 1
    return {
        "query": query, "old_rank": old_rank, "new_rank": new_rank,
        "preservation": array[:, 3].astype(np.float32),
        "delta_norm": array[:, 4].astype(np.float32),
        "positive_score": array[:, 5].astype(np.float32),
        "max_negative_score": array[:, 6].astype(np.float32),
        "summary": {
            "n_queries": int(len(query)),
            "baseline_recall1": float(np.mean(base_correct)),
            "recall1": float(np.mean(new_correct)),
            "delta_recall1": float(np.mean(new_correct) - np.mean(base_correct)),
            "baseline_mrr": float(np.mean(1.0 / old_rank)),
            "mrr": float(np.mean(1.0 / new_rank)),
            "delta_mrr": float(np.mean(1.0 / new_rank) - np.mean(1.0 / old_rank)),
            "corrected": int(np.sum(~base_correct & new_correct)),
            "introduced": int(np.sum(base_correct & ~new_correct)),
            "near_n": int(np.sum(near)),
            "baseline_near_recall1": float(np.mean(base_correct[near])) if np.any(near) else None,
            "near_recall1": float(np.mean(new_correct[near])) if np.any(near) else None,
            "delta_near_recall1": float(np.mean(new_correct[near]) - np.mean(base_correct[near])) if np.any(near) else None,
            "preservation_mean": float(np.mean(array[:, 3])),
            "preservation_min": float(np.min(array[:, 3])),
            "delta_norm_mean": float(np.mean(array[:, 4])),
            "online_baseline_mismatch_fraction": mismatch,
        },
    }


def main() -> None:
    args = arguments()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("formal D1 requires a CUDA-enabled PyTorch environment")
    if args.outer_fold not in range(5):
        raise ValueError("outer-fold must be 0..4")
    seed_everything(args.seed)
    device = torch.device(args.device)

    decision_path = args.manifest_dir / "decision.json"
    manifest_path = args.manifest_dir / "manifest.npz"
    for path in (args.graph, args.data, args.embedding_cache, args.official_ckpt,
                 args.architecture_ckpt, decision_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)
    d0 = json.loads(decision_path.read_text(encoding="utf-8"))
    if d0.get("status") != "noise_final_d0_manifest_complete" or d0.get("contains_p2b_fields") is not False:
        raise RuntimeError("D1 requires a valid P2b-free D0 contract")
    with np.load(manifest_path) as body:
        formula_fold = np.asarray(body["formula_fold"], dtype=np.int8)
        identity_weight = np.asarray(body["identity_weight"], dtype=np.float32)
        baseline_rank = np.asarray(body["baseline_rank"], dtype=np.int16)
    graph = CandidateGraph(args.graph)
    if len(formula_fold) != graph.n_queries:
        raise RuntimeError("D0 manifest and candidate graph disagree")

    inner_fold = (args.outer_fold + 1) % 5
    train_query = np.flatnonzero((formula_fold != args.outer_fold) & (formula_fold != inner_fold))
    inner_query = np.flatnonzero(formula_fold == inner_fold)
    outer_query = np.flatnonzero(formula_fold == args.outer_fold)
    if args.max_train_queries:
        train_query = train_query[:args.max_train_queries]
    if args.max_eval_queries:
        inner_query = inner_query[:args.max_eval_queries]
        outer_query = outer_query[:args.max_eval_queries]
    if not len(train_query) or not len(inner_query) or not len(outer_query):
        raise RuntimeError("empty D1 train/inner/outer split")
    train_formulas = set(graph.query_formula[train_query])
    inner_formulas = set(graph.query_formula[inner_query])
    outer_formulas = set(graph.query_formula[outer_query])
    if train_formulas & inner_formulas or train_formulas & outer_formulas or inner_formulas & outer_formulas:
        raise RuntimeError("formula-group split leakage")

    _, embeddings, embedding_index = load_embedding_cache(args.embedding_cache)
    candidate_pos = candidate_positions(graph, embedding_index)
    reference_embeddings = torch.from_numpy(embeddings).to(device)
    candidate_pos = torch.from_numpy(candidate_pos).to(device)
    model, initialization = load_base_model(
        args.official_ckpt, args.architecture_ckpt, device, args.n_highest_peaks,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    dimension = int(reference_embeddings.shape[1])
    adapter = ZeroInitPeakAdapter(dimension, args.hidden_dim, args.delta_bound).to(device)

    def loader(indices, shuffle):
        generator = torch.Generator().manual_seed(args.seed + (1 if shuffle else 2))
        return DataLoader(
            QueryDataset(args.data, graph, indices, args.n_highest_peaks),
            batch_size=args.batch_size, shuffle=shuffle, num_workers=0,
            pin_memory=device.type == "cuda", generator=generator,
        )

    train_loader = loader(train_query, True)
    inner_loader = loader(inner_query, False)
    outer_loader = loader(outer_query, False)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Epoch zero is the exact official no-op and remains eligible for selection.
    initial = evaluate(model, adapter, inner_loader, graph, candidate_pos, reference_embeddings,
                       baseline_rank, device, args.max_eval_queries)
    best_state = copy.deepcopy(adapter.state_dict())
    best_epoch = 0
    best_utility = 0.0
    history = [{"epoch": 0, "train": None, "inner": initial["summary"], "selection_utility": 0.0}]
    print(f"[D1 fold={args.outer_fold} seed={args.seed}] epoch 0 {initial['summary']}", flush=True)

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        epoch_loss, components = [], []
        started = time.time()
        for query_indices, spectra in train_loader:
            optimizer.zero_grad(set_to_none=True)
            official, adapted, _, _ = forward_queries(model, adapter, spectra, device)
            loss, detail = losses_for_batch(
                query_indices, official, adapted, graph, candidate_pos, reference_embeddings,
                identity_weight, args.temperature, args,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite D1 loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            epoch_loss.append(float(loss.detach()))
            components.append(detail)
        inner = evaluate(model, adapter, inner_loader, graph, candidate_pos, reference_embeddings,
                         baseline_rank, device, args.max_eval_queries)
        summary = inner["summary"]
        utility = (
            (summary["corrected"] - 2.0 * summary["introduced"]) / summary["n_queries"]
            + 0.25 * float(summary["delta_near_recall1"] or 0.0)
        )
        eligible = summary["preservation_mean"] >= args.minimum_preservation
        if eligible and utility > best_utility + 1e-12:
            best_utility, best_epoch = float(utility), epoch
            best_state = copy.deepcopy(adapter.state_dict())
        component_mean = {key: float(np.mean([value[key] for value in components])) for key in components[0]}
        record = {
            "epoch": epoch, "train": {"loss": float(np.mean(epoch_loss)), **component_mean},
            "inner": summary, "selection_utility": float(utility), "eligible": bool(eligible),
            "seconds": time.time() - started,
        }
        history.append(record)
        print(f"[D1 fold={args.outer_fold} seed={args.seed}] {record}", flush=True)

    adapter.load_state_dict(best_state)
    outer = evaluate(model, adapter, outer_loader, graph, candidate_pos, reference_embeddings,
                     baseline_rank, device, args.max_eval_queries)
    output_dir = args.output_root / f"seed_{args.seed}" / f"fold_{args.outer_fold}"
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.save({
        "status": "noise_final_d1_adapter", "adapter_state": {k: v.cpu() for k, v in best_state.items()},
        "adapter_config": {"embedding_dim": dimension, "hidden_dim": args.hidden_dim, "delta_bound": args.delta_bound},
        "seed": args.seed, "outer_fold": args.outer_fold, "inner_fold": inner_fold,
        "best_epoch": best_epoch, "initialization": initialization,
        "official_checkpoint_sha256": sha256_file(args.official_ckpt),
        "architecture_checkpoint_sha256": sha256_file(args.architecture_ckpt),
        "d0_manifest_sha256": sha256_file(manifest_path),
    }, output_dir / "adapter.pt")
    np.savez_compressed(
        output_dir / "outer_predictions.npz", query=outer["query"], old_rank=outer["old_rank"],
        new_rank=outer["new_rank"], preservation=outer["preservation"],
        delta_norm=outer["delta_norm"], positive_score=outer["positive_score"],
        max_negative_score=outer["max_negative_score"],
    )
    decision = {
        "status": "noise_final_d1_fold_complete", "seed": args.seed,
        "outer_fold": args.outer_fold, "inner_fold": inner_fold,
        "train_queries": int(len(train_query)), "inner_queries": int(len(inner_query)),
        "outer_queries": int(len(outer_query)), "best_epoch": best_epoch,
        "best_inner_utility": best_utility, "outer": outer["summary"], "history": history,
        "P2b_used": False,
        "claim_limit": "formula-outer-fold clean-only adapter result; no P3 and no noise teacher",
    }
    json_dump(output_dir / "decision.json", decision)
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
