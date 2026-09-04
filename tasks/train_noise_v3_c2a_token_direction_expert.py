"""Formula-OOF peak-token direction expert on support-disjoint C1 supervision.

The model updates only a lightweight query adapter.  A same-capacity global
embedding residual is trained as the preregistered ablation.  Peak-token gain
is claimed only when its paired formula-cluster CI exceeds the global control.
Repeated C1 holdouts are collapsed to one training item per query.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from audit_noise_v3_a4_exact_peak_scan import load_embeddings, query_candidate_block, strict_detail
from build_g8r_real_error_atlas import Cache
from diagnose_noise_v3_a4b_positive_evidence import cluster_bootstrap, normalized_mean


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--embeddings", type=Path, default=ROOT / "data/validation/g8r_p2_official_embeddings.npz")
    parser.add_argument("--c1-dir", type=Path, default=ROOT / "data/validation/g8r_noise_v3_c1_crossfit_teacher")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/g8r_noise_v3_c2_peak_tokens")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/g8r_noise_v3_c2a_token_direction")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260825, 20260826, 20260827])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--teacher-weight", type=float, default=2.0)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--preserve-weight", type=float, default=5.0)
    parser.add_argument("--floor-weight", type=float, default=2.0)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--rank-temperature", type=float, default=0.10)
    parser.add_argument("--safety-per-rescue", type=float, default=2.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--minimum-net-corrections", type=int, default=200)
    parser.add_argument("--max-queries", type=int, default=0, help="Smoke only")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values / np.clip(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12, None)).astype(np.float32)


class DirectionExpert(torch.nn.Module):
    def __init__(self, embedding_dim: int, token_dim: int, hidden: int, use_tokens: bool):
        super().__init__()
        self.use_tokens = use_tokens
        self.clean_context = torch.nn.Sequential(
            torch.nn.LayerNorm(embedding_dim), torch.nn.Linear(embedding_dim, hidden), torch.nn.SiLU(),
        )
        if use_tokens:
            self.token_context = torch.nn.Sequential(
                torch.nn.LayerNorm(token_dim + 2), torch.nn.Linear(token_dim + 2, hidden), torch.nn.SiLU(),
            )
            self.attention = torch.nn.Linear(hidden, 1)
            residual_input = hidden * 2
        else:
            residual_input = hidden
        self.residual = torch.nn.Sequential(
            torch.nn.Linear(residual_input, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, embedding_dim, bias=False),
        )
        torch.nn.init.zeros_(self.residual[-1].weight)
        self.gate = torch.nn.Linear(residual_input, 1)
        torch.nn.init.zeros_(self.gate.weight)
        torch.nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, clean: torch.Tensor, token: torch.Tensor | None = None,
                mz: torch.Tensor | None = None, intensity: torch.Tensor | None = None,
                valid: torch.Tensor | None = None) -> torch.Tensor:
        clean_hidden = self.clean_context(clean)
        if self.use_tokens:
            assert token is not None and mz is not None and intensity is not None and valid is not None
            token_input = torch.cat((token, (mz / 1000.0).unsqueeze(-1), torch.sqrt(intensity.clamp_min(0)).unsqueeze(-1)), dim=-1)
            token_hidden = self.token_context(token_input)
            logits = self.attention(torch.tanh(token_hidden + clean_hidden.unsqueeze(1))).squeeze(-1)
            logits = logits.masked_fill(~valid, -1e4)
            weight = torch.softmax(logits, dim=1)
            pooled = torch.sum(weight.unsqueeze(-1) * token_hidden, dim=1)
            context = torch.cat((clean_hidden, pooled), dim=1)
        else:
            context = clean_hidden
        gate = torch.sigmoid(self.gate(context))
        return torch.nn.functional.normalize(clean + gate * self.residual(context), dim=1)


def prepare(args: argparse.Namespace):
    graph = Cache(args.graph)
    score_column = graph.feature_names.index("dreams_similarity")
    embedding_rows, embedding, embedding_index = load_embeddings(args.embeddings)
    examples = pd.read_csv(args.c1_dir / "crossfit_examples.csv.gz")
    token_rows = np.load(args.token_dir / "rows.npy")
    token_index = {int(row): pos for pos, row in enumerate(token_rows)}
    if args.max_queries:
        rescue_queries = examples.loc[examples["corrected"], "query_index"].drop_duplicates().tolist()
        safety_queries = examples.loc[
            (examples["baseline_rank"] == 1) & (~examples["introduced"]), "query_index"
        ].drop_duplicates().tolist()
        half = (args.max_queries + 1) // 2
        keep_queries = rescue_queries[:half] + safety_queries[:args.max_queries - half]
        examples = examples.loc[examples["query_index"].isin(keep_queries)].copy()
    formal = args.max_queries == 0
    query_records = []
    target_values, positive_values, negative_values = [], [], []
    for query, local in examples.groupby("query_index", sort=True):
        rescue_rows = local.loc[local["corrected"]]
        if len(rescue_rows):
            selected = rescue_rows.sort_values(["baseline_margin", "evaluation_positive_row"]).iloc[0]
            role = "rescue"
            teacher_targets = []
            clean = embedding[embedding_index[int(selected.query_row)]]
            for row in rescue_rows.itertuples(index=False):
                teacher_rows = [int(value) for value in str(row.teacher_rows).split(";") if value]
                prototype = normalized_mean(embedding[[embedding_index[value] for value in teacher_rows]])
                teacher_targets.append(normalized(0.75 * clean + 0.25 * prototype))
            target = normalized_mean(np.asarray(teacher_targets))
        else:
            safety_rows = local.loc[(local["baseline_rank"] == 1) & (~local["introduced"])]
            if safety_rows.empty:
                continue
            selected = safety_rows.sort_values(["baseline_margin", "evaluation_positive_row"]).iloc[0]
            role = "safety"
            clean = embedding[embedding_index[int(selected.query_row)]]
            target = clean
        scores, candidate_rows, ptr, _ = query_candidate_block(graph, int(query), score_column)
        eval_matches = np.flatnonzero(candidate_rows == int(selected.evaluation_positive_row))
        if len(eval_matches) != 1:
            raise RuntimeError(f"C2-A evaluation-positive alignment failed: query={query}")
        eval_pair = int(eval_matches[0])
        allowed_scores = np.asarray(scores, dtype=np.float64).copy()
        allowed_scores[int(ptr[0]):int(ptr[1])] = -1e6
        allowed_scores[eval_pair] = float(scores[eval_pair])
        detail = strict_detail(allowed_scores, candidate_rows, ptr)
        neg_row = int(detail["adversarial_pair_row"])
        query_row = int(selected.query_row)
        if query_row not in token_index:
            raise RuntimeError(f"C2 token cache misses query row {query_row}")
        query_records.append({
            "query_index": int(query), "query_row": query_row,
            "query_ik14": str(selected.query_ik14), "query_formula": str(selected.query_formula),
            "formula_fold": int(selected.formula_fold), "role": role,
            "has_near": bool(selected.has_near), "baseline_margin": float(detail["margin"]),
            "evaluation_positive_row": int(selected.evaluation_positive_row),
            "adversarial_row": neg_row, "token_position": int(token_index[query_row]),
        })
        target_values.append(target)
        positive_values.append(embedding[embedding_index[int(selected.evaluation_positive_row)]])
        negative_values.append(embedding[embedding_index[neg_row]])
    query_frame = pd.DataFrame(query_records)
    return (
        graph, examples, query_frame, embedding, embedding_index,
        np.asarray(target_values, np.float32), np.asarray(positive_values, np.float32),
        np.asarray(negative_values, np.float32), formal,
    )


def train_fold(args, query_frame, clean, target, positive, negative, token_arrays,
               train_index, test_index, seed, use_tokens):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    model = DirectionExpert(clean.shape[1], token_arrays[0].shape[2], args.hidden_dim, use_tokens).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rescue = query_frame["role"].eq("rescue").to_numpy()
    rng = np.random.default_rng(seed)
    model.train()
    final_loss = math.nan
    for epoch in range(args.epochs):
        rescue_train = train_index[rescue[train_index]]
        safety_train = train_index[~rescue[train_index]]
        safety_n = min(len(safety_train), int(math.ceil(args.safety_per_rescue * len(rescue_train))))
        selected_safety = rng.choice(safety_train, size=safety_n, replace=False) if safety_n else np.empty(0, int)
        order = rng.permutation(np.concatenate((rescue_train, selected_safety)))
        total, seen = 0.0, 0
        for left in range(0, len(order), args.batch_size):
            idx_np = order[left:left + args.batch_size]
            idx = torch.as_tensor(idx_np, device=device)
            clean_b = torch.as_tensor(clean[idx_np], device=device)
            kwargs = {}
            if use_tokens:
                positions = query_frame.iloc[idx_np]["token_position"].to_numpy(int)
                kwargs = {
                    "token": torch.as_tensor(np.asarray(token_arrays[0][positions], np.float32), device=device),
                    "mz": torch.as_tensor(np.asarray(token_arrays[1][positions], np.float32), device=device),
                    "intensity": torch.as_tensor(np.asarray(token_arrays[2][positions], np.float32), device=device),
                    "valid": torch.as_tensor(np.asarray(token_arrays[3][positions], bool), device=device),
                }
            student = model(clean_b, **kwargs)
            target_b = torch.as_tensor(target[idx_np], device=device)
            pos_b = torch.as_tensor(positive[idx_np], device=device)
            neg_b = torch.as_tensor(negative[idx_np], device=device)
            rescue_b = torch.as_tensor(rescue[idx_np], device=device)
            sp = torch.sum(student * pos_b, dim=1); sn = torch.sum(student * neg_b, dim=1)
            rank_loss = args.rank_temperature * torch.nn.functional.softplus(
                (args.rank_margin + sn - sp) / args.rank_temperature
            ).mean()
            losses = [args.rank_weight * rank_loss]
            if bool(rescue_b.any()):
                losses.append(args.teacher_weight * (1 - torch.sum(student[rescue_b] * target_b[rescue_b], dim=1)).mean())
            safety_b = ~rescue_b
            if bool(safety_b.any()):
                preserve = (1 - torch.sum(student[safety_b] * clean_b[safety_b], dim=1)).mean()
                baseline_floor = torch.as_tensor(query_frame.iloc[idx_np]["baseline_margin"].to_numpy(np.float32), device=device)
                floor = torch.relu(baseline_floor[safety_b] - (sp - sn)[safety_b]).mean()
                losses.extend((args.preserve_weight * preserve, args.floor_weight * floor))
            loss = torch.stack(losses).sum()
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            total += float(loss.detach()) * len(idx_np); seen += len(idx_np)
        final_loss = total / max(seen, 1)
    model.eval(); output = []
    with torch.inference_mode():
        for left in range(0, len(test_index), args.batch_size):
            idx_np = test_index[left:left + args.batch_size]
            clean_b = torch.as_tensor(clean[idx_np], device=device)
            kwargs = {}
            if use_tokens:
                positions = query_frame.iloc[idx_np]["token_position"].to_numpy(int)
                kwargs = {
                    "token": torch.as_tensor(np.asarray(token_arrays[0][positions], np.float32), device=device),
                    "mz": torch.as_tensor(np.asarray(token_arrays[1][positions], np.float32), device=device),
                    "intensity": torch.as_tensor(np.asarray(token_arrays[2][positions], np.float32), device=device),
                    "valid": torch.as_tensor(np.asarray(token_arrays[3][positions], bool), device=device),
                }
            output.append(model(clean_b, **kwargs).cpu().numpy())
    return np.concatenate(output), {"seed": seed, "final_loss": final_loss}


def evaluate_examples(args, graph, examples, query_frame, student_by_query, embedding, embedding_index):
    score_column = graph.feature_names.index("dreams_similarity")
    query_map = {int(row.query_index): student_by_query[pos] for pos, row in enumerate(query_frame.itertuples(index=False))}
    baseline_correct, result_correct, near_values, formulas = [], [], [], []
    for row in examples.itertuples(index=False):
        scores, candidate_rows, ptr, _ = query_candidate_block(graph, int(row.query_index), score_column)
        positions = np.asarray([embedding_index[int(value)] for value in candidate_rows])
        candidate = embedding[positions]
        clean = embedding[embedding_index[int(row.query_row)]]
        # Queries outside the rescue/safety training cohort receive the exact
        # official embedding.  They remain in the full-space denominator and
        # cannot create artificial gains by being silently omitted.
        student_embedding = query_map.get(int(row.query_index), clean)
        eval_pair = int(np.flatnonzero(candidate_rows == int(row.evaluation_positive_row))[0])
        allowed = np.ones(len(scores), bool); allowed[int(ptr[0]):int(ptr[1])] = False; allowed[eval_pair] = True
        locked = np.asarray(scores, float).copy(); locked[~allowed] = -1e6
        baseline = strict_detail(locked, candidate_rows, ptr)
        student_scores = np.asarray(scores, float) + (candidate @ student_embedding - candidate @ clean)
        student_scores[~allowed] = -1e6
        result = strict_detail(student_scores, candidate_rows, ptr)
        baseline_correct.append(int(baseline["rank"]) == 1); result_correct.append(int(result["rank"]) == 1)
        near_values.append(bool(row.has_near)); formulas.append(str(row.query_formula))
    baseline_correct = np.asarray(baseline_correct, bool); result_correct = np.asarray(result_correct, bool)
    near = np.asarray(near_values, bool)
    eval_frame = pd.DataFrame({"query_formula": formulas})
    contribution = (~baseline_correct & result_correct).astype(float) - (baseline_correct & ~result_correct).astype(float)
    return {
        "examples": int(len(result_correct)), "baseline_accuracy": float(baseline_correct.mean()),
        "accuracy": float(result_correct.mean()), "delta_accuracy": float(result_correct.mean() - baseline_correct.mean()),
        "corrected": int(np.sum(~baseline_correct & result_correct)),
        "introduced": int(np.sum(baseline_correct & ~result_correct)),
        "net_corrections": int(contribution.sum()),
        "near_delta_accuracy": float(result_correct[near].mean() - baseline_correct[near].mean()),
        "formula_cluster_delta": cluster_bootstrap(eval_frame, contribution, "query_formula", args.bootstrap_resamples, 20260825),
        "contribution": contribution,
        "eval_frame": eval_frame,
    }


def main() -> None:
    args = parse_args()
    required = [args.graph, args.embeddings, args.c1_dir / "crossfit_examples.csv.gz",
                args.token_dir / "report.json", args.token_dir / "rows.npy", args.token_dir / "tokens_f16.npy",
                args.token_dir / "mz_f32.npy", args.token_dir / "intensity_f32.npy", args.token_dir / "valid.npy"]
    for path in required:
        if not path.is_file(): raise FileNotFoundError(path)
    if args.output_dir.exists(): raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    graph, examples, query_frame, embedding, embedding_index, target, positive, negative, formal = prepare(args)
    clean = embedding[[embedding_index[int(value)] for value in query_frame["query_row"]]]
    token_arrays = (
        np.load(args.token_dir / "tokens_f16.npy", mmap_mode="r"), np.load(args.token_dir / "mz_f32.npy", mmap_mode="r"),
        np.load(args.token_dir / "intensity_f32.npy", mmap_mode="r"), np.load(args.token_dir / "valid.npy", mmap_mode="r"),
    )
    fold_values = sorted(query_frame["formula_fold"].unique())
    outputs, logs = {}, {}
    for name, use_tokens in (("global_control", False), ("peak_token_expert", True)):
        oof = np.zeros_like(clean); model_logs = []
        for fold in fold_values:
            test = np.flatnonzero(query_frame["formula_fold"].to_numpy() == fold)
            train = np.flatnonzero(query_frame["formula_fold"].to_numpy() != fold)
            ensemble = np.zeros((len(test), clean.shape[1]), np.float64)
            for seed in args.seeds:
                value, log = train_fold(args, query_frame, clean, target, positive, negative, token_arrays, train, test, seed, use_tokens)
                ensemble += value / len(args.seeds); model_logs.append({"fold": int(fold), **log})
            oof[test] = normalized(ensemble)
            print(f"[C2-A] {name} fold={fold}", flush=True)
        evaluated = evaluate_examples(args, graph, examples, query_frame, oof, embedding, embedding_index)
        preservation = np.sum(oof * clean, axis=1)
        safety = query_frame["role"].eq("safety").to_numpy()
        evaluated["safety_preservation_mean"] = float(preservation[safety].mean())
        evaluated["safety_preservation_p05"] = float(np.quantile(preservation[safety], 0.05))
        evaluated["oof"] = oof; outputs[name] = evaluated; logs[name] = model_logs
    token = outputs["peak_token_expert"]; control = outputs["global_control"]
    paired = cluster_bootstrap(
        token["eval_frame"], token["contribution"] - control["contribution"],
        "query_formula", args.bootstrap_resamples, 20260829,
    )
    gates = {
        "token_formula_ci_positive": bool(token["formula_cluster_delta"]["ci_low"] > 0),
        "token_net_ge_minimum": bool(token["net_corrections"] >= args.minimum_net_corrections),
        "token_corrected_gt_twice_introduced": bool(token["corrected"] >= 2 * token["introduced"]),
        "near_nonnegative": bool(token["near_delta_accuracy"] >= 0),
        "safety_preservation_ok": bool(token["safety_preservation_mean"] >= 0.995),
        "token_beats_global_formula_ci_positive": bool(paired["ci_low"] > 0),
    }
    serial = {}
    for name, value in outputs.items():
        serial[name] = {key: item for key, item in value.items() if key not in {"oof", "contribution", "eval_frame"}}
    decision = {
        "status": "noise_v3_c2a_token_direction_complete", "formal": formal,
        "training_queries": int(len(query_frame)), "rescue_queries": int(query_frame["role"].eq("rescue").sum()),
        "safety_queries": int(query_frame["role"].eq("safety").sum()), "formulas": int(query_frame["query_formula"].nunique()),
        "models": serial, "token_vs_global_paired_formula_bootstrap": paired,
        "gates": gates, "pass_to_candidate_safety_gate": bool(all(gates.values())),
        "claim_limit": "Formula-OOF C1 development result; no sealed P3 query is evaluated.",
        "training_log": logs,
        "provenance": {str(path): sha256_file(path) for path in required},
    }
    staging = Path(tempfile.mkdtemp(prefix="c2a_", dir=args.output_dir.parent))
    try:
        query_frame.to_csv(staging / "training_queries.csv.gz", index=False, compression="gzip")
        np.savez_compressed(staging / "oof_query_embeddings.npz", clean=clean, global_control=control["oof"], peak_token_expert=token["oof"])
        (staging / "decision.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True); raise
    print(json.dumps(decision, indent=2, default=str), flush=True)


if __name__ == "__main__": main()
