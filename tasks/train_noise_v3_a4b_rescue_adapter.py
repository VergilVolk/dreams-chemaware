"""Formula-OOF residual adapter distilling A4-B0 positive-evidence targets.

The teacher target uses real same-identity spectra only during training.  The
student receives only the clean official query embedding.  Candidate/reference
embeddings remain frozen official DreaMS.  This script is a deployable-query-head
pilot and does not modify the 116M backbone.
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

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold

from audit_noise_v3_a4_exact_peak_scan import load_embeddings, query_candidate_block, strict_detail
from build_g8r_real_error_atlas import Cache
from diagnose_noise_v3_a4b_positive_evidence import normalized_mean, cluster_bootstrap


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path,
                        default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--embedding-cache", type=Path,
                        default=ROOT / "data/validation/g8r_p2_official_embeddings.npz")
    parser.add_argument("--data", type=Path,
                        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--b0-dir", type=Path,
                        default=ROOT / "data/validation/g8r_noise_v3_a4b_positive_evidence")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data/validation/g8r_noise_v3_a4b_rescue_adapter")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260825, 20260826, 20260827])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--teacher-weight", type=float, default=2.0)
    parser.add_argument("--preserve-weight", type=float, default=2.0)
    parser.add_argument("--rank-weight", type=float, default=1.0)
    parser.add_argument("--rank-margin", type=float, default=0.05)
    parser.add_argument("--rank-temperature", type=float, default=0.10)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--minimum-net-corrections", type=int, default=80)
    parser.add_argument("--max-queries", type=int, default=0, help="balanced smoke only")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def decode(value) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


class ResidualAdapter(torch.nn.Module):
    def __init__(self, dimension: int, hidden: int, nonlinear: bool):
        super().__init__()
        if nonlinear:
            self.residual = torch.nn.Sequential(
                torch.nn.Linear(dimension, hidden, bias=False),
                torch.nn.SiLU(),
                torch.nn.LayerNorm(hidden),
                torch.nn.Linear(hidden, dimension, bias=False),
            )
            torch.nn.init.zeros_(self.residual[-1].weight)
        else:
            self.residual = torch.nn.Linear(dimension, dimension, bias=False)
            torch.nn.init.zeros_(self.residual.weight)
        self.gate = torch.nn.Linear(dimension, 1)
        torch.nn.init.zeros_(self.gate.weight)
        torch.nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate(x))
        return torch.nn.functional.normalize(x + gate * self.residual(x), dim=1)


def build_arrays(args: argparse.Namespace):
    graph = Cache(args.graph)
    score_column = graph.feature_names.index("dreams_similarity")
    rows, embeddings, index = load_embeddings(args.embedding_cache)
    b0 = pd.read_csv(args.b0_dir / "paired_results.csv.gz")
    decision = json.loads((args.b0_dir / "decision.json").read_text(encoding="utf-8"))
    alpha = float(decision["primary_alpha"])
    b0 = b0.loc[b0["alpha"].eq(alpha)].sort_values("query_index").reset_index(drop=True)
    if args.max_queries:
        errors = b0.loc[b0["scan_kind"].eq("official_error")].head((args.max_queries + 1) // 2)
        controls = b0.loc[b0["scan_kind"].eq("safety_control")].head(args.max_queries // 2)
        b0 = pd.concat((errors, controls)).sort_values("query_index").reset_index(drop=True)
    formal = args.max_queries == 0
    if formal and (len(b0) != 4998 or not decision.get("pass_to_two_expert_teacher")):
        raise RuntimeError("formal B1 requires the passing full B0 panel")

    with h5py.File(args.data, "r") as handle:
        ik14 = np.asarray([decode(value)[:14] for value in handle["INCHIKEY"][rows]], dtype=object)
        adduct = np.asarray([decode(value) for value in handle["adduct"][rows]], dtype=object)
    groups: dict[tuple[str, str], list[int]] = {}
    for position, key in enumerate(zip(ik14, adduct)):
        groups.setdefault((str(key[0]), str(key[1])), []).append(position)

    clean, target, positive, negative = [], [], [], []
    candidate_blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    aligned_records = []
    for row in b0.itertuples(index=False):
        query = int(row.query_index)
        query_row = int(row.query_row)
        query_position = index.get(query_row)
        if query_position is None:
            raise RuntimeError(f"missing query embedding: {query_row}")
        key = (str(row.query_ik14)[:14], str(adduct[query_position]))
        support = [value for value in groups.get(key, []) if int(rows[value]) != query_row]
        if not support:
            raise RuntimeError(f"B1 lost a B0 support group: query={query}")
        support = sorted(support, key=lambda value: int(rows[value]))[:12]
        prototype = normalized_mean(embeddings[support])
        z = embeddings[query_position]
        teacher = (1.0 - alpha) * z.astype(np.float64) + alpha * prototype.astype(np.float64)
        teacher = (teacher / np.linalg.norm(teacher)).astype(np.float32)

        scores, candidate_rows, ptr, _ = query_candidate_block(graph, query, score_column)
        positions = np.asarray([index[int(value)] for value in candidate_rows], dtype=np.int64)
        candidate_embedding = embeddings[positions]
        baseline = strict_detail(scores, candidate_rows, ptr)
        if int(baseline["rank"]) != int(row.baseline_rank):
            raise RuntimeError(f"B0/B1 baseline mismatch: query={query}")
        pos_left, pos_right = map(int, ptr[:2])
        pos_pair = pos_left + int(np.argmax(scores[pos_left:pos_right]))
        neg_pair = int(ptr[1]) + int(np.argmax(scores[int(ptr[1]):]))
        clean.append(z)
        # Only B0-correctable errors receive the privileged teacher direction.
        # All other queries use the clean embedding as their preservation target.
        target.append(teacher if bool(row.corrected) else z)
        positive.append(candidate_embedding[pos_pair])
        negative.append(candidate_embedding[neg_pair])
        recomputed_clean = candidate_embedding @ z
        candidate_blocks.append((
            candidate_embedding, np.asarray(candidate_rows), np.asarray(ptr),
            np.asarray(scores, dtype=np.float32), np.asarray(recomputed_clean, dtype=np.float32),
        ))
        aligned_records.append(row._asdict())
    frame = pd.DataFrame(aligned_records)
    return (
        frame, np.asarray(clean, np.float32), np.asarray(target, np.float32),
        np.asarray(positive, np.float32), np.asarray(negative, np.float32),
        candidate_blocks, formal, alpha,
    )


def train_one(
    x: np.ndarray, target: np.ndarray, positive: np.ndarray, negative: np.ndarray,
    rescue: np.ndarray, control: np.ndarray, train: np.ndarray, test: np.ndarray,
    args: argparse.Namespace, seed: int, nonlinear: bool,
) -> tuple[np.ndarray, dict]:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    model = ResidualAdapter(x.shape[1], args.hidden_dim, nonlinear).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    tx = torch.as_tensor(x[train], device=device)
    tt = torch.as_tensor(target[train], device=device)
    tp = torch.as_tensor(positive[train], device=device)
    tn = torch.as_tensor(negative[train], device=device)
    tr = torch.as_tensor(rescue[train], device=device)
    tc = torch.as_tensor(control[train], device=device)
    generator = torch.Generator().manual_seed(seed)
    final_loss = math.nan
    model.train()
    for _ in range(args.epochs):
        order = torch.randperm(len(train), generator=generator)
        total = 0.0
        seen = 0
        for left in range(0, len(train), args.batch_size):
            idx = order[left:left + args.batch_size].to(device)
            student = model(tx[idx])
            sp = torch.sum(student * tp[idx], dim=1)
            sn = torch.sum(student * tn[idx], dim=1)
            rank = args.rank_temperature * torch.nn.functional.softplus(
                (args.rank_margin + sn - sp) / args.rank_temperature
            ).mean()
            losses = [args.rank_weight * rank]
            local_rescue = tr[idx]
            if bool(local_rescue.any()):
                losses.append(args.teacher_weight * (
                    1.0 - torch.sum(student[local_rescue] * tt[idx][local_rescue], dim=1)
                ).mean())
            local_control = tc[idx]
            if bool(local_control.any()):
                losses.append(args.preserve_weight * (
                    1.0 - torch.sum(student[local_control] * tx[idx][local_control], dim=1)
                ).mean())
            loss = torch.stack(losses).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(idx)
            seen += len(idx)
        final_loss = total / max(seen, 1)
    model.eval()
    output = []
    with torch.inference_mode():
        for left in range(0, len(test), args.batch_size):
            batch = torch.as_tensor(x[test[left:left + args.batch_size]], device=device)
            output.append(model(batch).cpu().numpy())
    return np.concatenate(output).astype(np.float32), {"seed": seed, "final_loss": final_loss}


def evaluate(frame: pd.DataFrame, embeddings: np.ndarray, blocks, args: argparse.Namespace) -> dict:
    baseline_correct = frame["baseline_rank"].to_numpy(int) == 1
    result_rank = np.empty(len(frame), dtype=np.int16)
    result_margin = np.empty(len(frame), dtype=np.float32)
    for index, (embedding, block) in enumerate(zip(embeddings, blocks)):
        candidate, rows, ptr, official_scores, recomputed_clean = block
        # Preserve the exact locked DreaMS scoring/tie convention and apply only
        # the student's counterfactual score displacement.
        scores = official_scores + (candidate @ embedding - recomputed_clean)
        detail = strict_detail(scores, rows, ptr)
        result_rank[index] = int(detail["rank"])
        result_margin[index] = float(detail["margin"])
    result_correct = result_rank == 1
    corrected = ~baseline_correct & result_correct
    introduced = baseline_correct & ~result_correct
    contribution = corrected.astype(float) - introduced.astype(float)
    formula_ci = cluster_bootstrap(
        frame, contribution, "query_formula", args.bootstrap_resamples, 20260825,
    )
    near = frame["has_near"].astype(bool).to_numpy()
    positive_deficit = frame["positive_deficit"].astype(bool).to_numpy()
    controls = frame["scan_kind"].eq("safety_control").to_numpy()
    return {
        "queries": int(len(frame)),
        "baseline_recall1": float(baseline_correct.mean()),
        "recall1": float(result_correct.mean()),
        "delta_recall1": float(result_correct.mean() - baseline_correct.mean()),
        "corrected": int(corrected.sum()),
        "introduced": int(introduced.sum()),
        "net_corrections": int(corrected.sum() - introduced.sum()),
        "near_baseline_recall1": float(baseline_correct[near].mean()),
        "near_recall1": float(result_correct[near].mean()),
        "near_delta_recall1": float(result_correct[near].mean() - baseline_correct[near].mean()),
        "positive_deficit_corrected": int((corrected & positive_deficit).sum()),
        "formula_cluster_delta_recall1": formula_ci,
        "result_rank": result_rank,
        "result_margin": result_margin,
        "corrected_mask": corrected,
        "introduced_mask": introduced,
        "control_mask": controls,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    frame, x, target, positive, negative, blocks, formal, alpha = build_arrays(args)
    rescue = frame["corrected"].astype(bool).to_numpy()
    controls = frame["scan_kind"].eq("safety_control").to_numpy()
    formulas = frame["query_formula"].astype(str).to_numpy()
    unique_formula = np.unique(formulas)
    folds = min(args.folds, len(unique_formula))
    if folds < 2:
        raise RuntimeError("B1 requires at least two formula folds")
    splitter = GroupKFold(n_splits=folds)
    fold_id = np.full(len(frame), -1, dtype=np.int8)
    for fold, (_, test) in enumerate(splitter.split(np.zeros(len(frame)), groups=formulas)):
        fold_id[test] = fold
    if np.any(fold_id < 0):
        raise RuntimeError("unassigned B1 formula fold")

    outputs = {}
    training_logs = {}
    for name, nonlinear in (("linear_residual", False), ("nonlinear_residual", True)):
        oof = np.zeros_like(x)
        logs = []
        for fold in range(folds):
            test = np.flatnonzero(fold_id == fold)
            train = np.flatnonzero(fold_id != fold)
            if set(formulas[train]) & set(formulas[test]):
                raise RuntimeError("formula leakage in B1")
            ensemble = np.zeros((len(test), x.shape[1]), dtype=np.float64)
            for seed in args.seeds:
                value, log = train_one(
                    x, target, positive, negative, rescue, controls,
                    train, test, args, seed, nonlinear,
                )
                ensemble += value / len(args.seeds)
                logs.append({"fold": fold, **log})
            ensemble /= np.clip(np.linalg.norm(ensemble, axis=1, keepdims=True), 1e-12, None)
            oof[test] = ensemble.astype(np.float32)
            print(f"[B1] {name} fold {fold + 1}/{folds}", flush=True)
        evaluated = evaluate(frame, oof, blocks, args)
        preservation = np.sum(oof * x, axis=1)
        evaluated["control_preservation_cosine_mean"] = float(preservation[controls].mean())
        evaluated["control_preservation_cosine_p05"] = float(np.quantile(preservation[controls], 0.05))
        outputs[name] = evaluated
        outputs[name]["oof_embedding"] = oof
        training_logs[name] = logs

    nonlinear = outputs["nonlinear_residual"]
    linear = outputs["linear_residual"]
    baseline_correct = frame["baseline_rank"].to_numpy(int) == 1
    nonlinear_correct = nonlinear["result_rank"] == 1
    linear_correct = linear["result_rank"] == 1
    nonlinear_contribution = (
        (~baseline_correct & nonlinear_correct).astype(float)
        - (baseline_correct & ~nonlinear_correct).astype(float)
    )
    linear_contribution = (
        (~baseline_correct & linear_correct).astype(float)
        - (baseline_correct & ~linear_correct).astype(float)
    )
    nonlinear_vs_linear = cluster_bootstrap(
        frame, nonlinear_contribution - linear_contribution,
        "query_formula", args.bootstrap_resamples, 20260829,
    )
    gates = {
        "nonlinear_formula_ci_positive": bool(
            nonlinear["formula_cluster_delta_recall1"]["ci_low"] > 0
        ),
        "nonlinear_net_corrections_ge_minimum": bool(
            nonlinear["net_corrections"] >= args.minimum_net_corrections
        ),
        "nonlinear_corrected_gt_introduced": bool(
            nonlinear["corrected"] > nonlinear["introduced"]
        ),
        "near_nonnegative": bool(nonlinear["near_delta_recall1"] >= 0),
        "control_preservation_ok": bool(
            nonlinear["control_preservation_cosine_mean"] >= 0.995
        ),
        "nonlinear_beats_linear_formula_ci_positive": bool(
            nonlinear_vs_linear["ci_low"] > 0
        ),
    }
    serializable_outputs = {}
    for name, evaluated in outputs.items():
        serializable_outputs[name] = {
            key: value for key, value in evaluated.items()
            if key not in {
                "result_rank", "result_margin", "corrected_mask",
                "introduced_mask", "control_mask", "oof_embedding",
            }
        }
    decision = {
        "status": "noise_v3_a4b_rescue_adapter_complete",
        "formal": formal,
        "teacher_alpha": alpha,
        "integrity": {
            "queries": int(len(frame)), "formulas": int(len(unique_formula)),
            "rescue_targets": int(rescue.sum()), "safety_controls": int(controls.sum()),
            "formula_fold_overlap": 0,
        },
        "models": serializable_outputs,
        "nonlinear_vs_linear_paired_formula_bootstrap": nonlinear_vs_linear,
        "gates": gates,
        "pass_to_token_adapter": bool(all(gates.values())),
        "claim_limit": (
            "Formula-OOF performance estimates a frozen-embedding query adapter. It is not yet "
            "a backbone fine-tune and must not be evaluated on sealed P3 before freezing."
        ),
        "parameters": vars(args) | {"graph": str(args.graph), "embedding_cache": str(args.embedding_cache),
                                    "data": str(args.data), "b0_dir": str(args.b0_dir),
                                    "output_dir": str(args.output_dir)},
        "training_log": training_logs,
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "embedding_cache_sha256": sha256_file(args.embedding_cache),
            "b0_results_sha256": sha256_file(args.b0_dir / "paired_results.csv.gz"),
            "script_sha256": sha256_file(Path(__file__)),
        },
    }
    staging = Path(tempfile.mkdtemp(prefix="a4b_adapter_", dir=args.output_dir.parent))
    try:
        oof_frame = frame.assign(
            formula_fold=fold_id,
            linear_rank=linear["result_rank"],
            linear_margin=linear["result_margin"],
            nonlinear_rank=nonlinear["result_rank"],
            nonlinear_margin=nonlinear["result_margin"],
            linear_corrected=linear["corrected_mask"],
            linear_introduced=linear["introduced_mask"],
            nonlinear_corrected=nonlinear["corrected_mask"],
            nonlinear_introduced=nonlinear["introduced_mask"],
        )
        oof_frame.to_csv(
            staging / "oof_queries.csv.gz", index=False, compression="gzip"
        )
        np.savez_compressed(
            staging / "oof_embeddings.npz",
            clean=x,
            linear=linear["oof_embedding"],
            nonlinear=nonlinear["oof_embedding"],
            formula_fold=fold_id,
        )
        (staging / "decision.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(decision, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
