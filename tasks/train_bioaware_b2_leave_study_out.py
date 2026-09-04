#!/usr/bin/env python
"""Nested study-isolated training of candidate-specific BioAware B2 embeddings."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import binomtest

from annotation.bioaware_context_adapter import BiologicalEvidenceContextAdapter


def strict_rank(scores: np.ndarray, positive: int) -> int:
    mask = np.ones(len(scores), dtype=bool)
    mask[positive] = False
    return 1 + int(np.sum(scores[mask] >= scores[positive]))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Dataset:
    def __init__(self, path: Path, device: torch.device):
        body = np.load(path)
        self.query_ids = body["query_ids"].astype(str)
        self.unit_ids = body["unit_ids"].astype(str)
        self.study_ids = body["study_ids"].astype(str)
        self.formulas = body["truth_formulas"].astype(str)
        self.truth_ids = body["truth_candidate_ids"].astype(str)
        self.query = torch.from_numpy(body["query_embeddings"]).to(device)
        self.offsets = body["offsets"].astype(np.int64)
        self.positive = body["positive_indices"].astype(np.int64)
        self.candidate_ids = body["candidate_ids"].astype(str)
        self.candidate = torch.from_numpy(body["candidate_embeddings"]).to(device)
        self.evidence_raw = body["evidence"].astype(np.float32)
        self.evidence = torch.from_numpy(self.evidence_raw).to(device)
        self.context = torch.from_numpy(body["context_mask"].astype(bool)).to(device)
        self.evidence_columns = body["evidence_columns"].astype(str)
        if len(self.offsets) != len(self.query_ids) + 1:
            raise RuntimeError("dataset offset/query mismatch")
        if self.offsets[-1] != len(self.candidate_ids):
            raise RuntimeError("dataset offset/candidate mismatch")

    def section(self, index: int) -> slice:
        return slice(int(self.offsets[index]), int(self.offsets[index + 1]))


def evidence_scaler(data: Dataset, query_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = np.concatenate([
        np.arange(data.offsets[index], data.offsets[index + 1], dtype=np.int64)
        for index in query_indices
    ])
    values = data.evidence_raw[flat]
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def standardised_evidence(
    data: Dataset, flat: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor,
) -> torch.Tensor:
    return (data.evidence[flat] - mean) / scale


def batch_loss(
    model: BiologicalEvidenceContextAdapter, data: Dataset, indices: np.ndarray,
    mean: torch.Tensor, scale: torch.Tensor, args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses: list[torch.Tensor] = []
    safety_losses: list[torch.Tensor] = []
    preservations: list[torch.Tensor] = []
    gate_values: list[torch.Tensor] = []
    for index in indices:
        section = data.section(int(index))
        flat = torch.arange(section.start, section.stop, device=data.query.device)
        universal = data.candidate[section]
        adapted, _, gates = model(
            universal, standardised_evidence(data, flat, mean, scale), data.context[section],
        )
        scores = adapted @ data.query[index]
        baseline = universal @ data.query[index]
        positive = int(data.positive[index])
        negative = torch.ones(len(scores), dtype=torch.bool, device=scores.device)
        negative[positive] = False
        baseline_margin = baseline[positive] - baseline[negative].max()
        adapted_margin = scores[positive] - scores[negative].max()
        rank_loss = F.cross_entropy(
            scores[None, :] / args.temperature,
            torch.tensor([positive], device=scores.device),
        )
        baseline_correct = bool(float(baseline_margin.detach()) > 0)
        losses.append(rank_loss * (args.correct_query_weight if baseline_correct else 1.0))
        if baseline_correct:
            safety_losses.append(F.relu(
                baseline_margin.detach() - args.safety_slack - adapted_margin
            ))
        preservations.append(torch.clamp(
            1.0 - torch.sum(adapted * universal, dim=-1), min=0,
        ).mean())
        gate_values.append(gates.mean())
    rank = torch.stack(losses).mean()
    safety = torch.stack(safety_losses).mean() if safety_losses else rank.new_zeros(())
    preservation = torch.stack(preservations).mean()
    gate = torch.stack(gate_values).mean()
    total = (
        rank + args.safety_weight * safety
        + args.preserve_weight * preservation + args.gate_weight * gate
    )
    return total, {
        "rank": float(rank.detach()), "safety": float(safety.detach()),
        "preservation": float(preservation.detach()), "gate": float(gate.detach()),
    }


@torch.no_grad()
def score_query(
    model: BiologicalEvidenceContextAdapter, data: Dataset, index: int,
    mean: torch.Tensor, scale: torch.Tensor,
) -> tuple[np.ndarray, float, float]:
    section = data.section(index)
    flat = torch.arange(section.start, section.stop, device=data.query.device)
    universal = data.candidate[section]
    adapted, _, gate = model(
        universal, standardised_evidence(data, flat, mean, scale), data.context[section],
    )
    return (
        (adapted @ data.query[index]).cpu().numpy(),
        float(torch.sum(adapted * universal, dim=-1).mean().cpu()),
        float(gate.mean().cpu()),
    )


def metrics(frame: pd.DataFrame) -> dict:
    corrected = int(frame.corrected.sum())
    introduced = int(frame.introduced.sum())
    discordant = corrected + introduced
    return {
        "queries": int(len(frame)),
        "baseline_recall1": float(frame.baseline_correct.mean()),
        "recall1": float(frame.final_correct.mean()),
        "delta_recall1": float(frame.delta.mean()),
        "corrected": corrected,
        "introduced": introduced,
        "risk_weighted_net_lambda2": corrected - 2 * introduced,
        "mcnemar_exact_p": (
            float(binomtest(min(corrected, introduced), discordant, .5).pvalue)
            if discordant else 1.0
        ),
    }


def cluster_bootstrap(frame: pd.DataFrame, repeats: int, seed: int) -> dict:
    work = frame.copy()
    work["cluster"] = work.truth_formula.astype(str)
    groups = {str(key): group for key, group in work.groupby("cluster", sort=True)}
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=float)
    for index in range(repeats):
        draw = rng.choice(keys, len(keys), replace=True)
        sample = pd.concat([groups[str(key)] for key in draw], ignore_index=True)
        values[index] = float(sample.delta.mean())
    return {
        "mean": float(work.delta.mean()),
        "ci_low": float(np.quantile(values, .025)),
        "ci_high": float(np.quantile(values, .975)),
        "clusters": len(keys), "resamples": repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("data/validation/bioaware_b2_external_context_dataset_v1/dataset.npz"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_b2_leave_study_out_v1"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260830, 20260831, 20260832])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--update-rank", type=int, default=16)
    parser.add_argument("--delta-bound", type=float, default=.05)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--temperature", type=float, default=.08)
    parser.add_argument("--correct-query-weight", type=float, default=2.0)
    parser.add_argument("--safety-weight", type=float, default=4.0)
    parser.add_argument("--safety-slack", type=float, default=.005)
    parser.add_argument("--preserve-weight", type=float, default=8.0)
    parser.add_argument("--gate-weight", type=float, default=.005)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output: {args.output_dir}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.output_dir.mkdir(parents=True)
    device = torch.device(args.device)
    data = Dataset(args.dataset, device)
    studies = sorted(set(data.study_ids))
    if len(studies) != 4:
        raise RuntimeError(f"expected four studies, found {studies}")

    transitions: list[dict] = []
    run_reports: list[dict] = []
    preservation_values: list[float] = []
    gate_values: list[float] = []
    for outer_study in studies:
        train = np.flatnonzero(data.study_ids != outer_study)
        heldout = np.flatnonzero(data.study_ids == outer_study)
        active_train = np.asarray([
            index for index in train if bool(data.context[data.section(int(index))].any())
        ], dtype=np.int64)
        if len(active_train) < 100 or len(heldout) < 50:
            raise RuntimeError(f"{outer_study}: insufficient active train or heldout queries")
        mean_np, scale_np = evidence_scaler(data, train)
        mean = torch.from_numpy(mean_np).to(device)
        scale = torch.from_numpy(scale_np).to(device)
        ensemble: dict[int, list[np.ndarray]] = {int(index): [] for index in heldout}
        for seed in args.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            model = BiologicalEvidenceContextAdapter(
                data.query.shape[1], data.evidence_raw.shape[1],
                hidden_dim=args.hidden_dim, update_rank=args.update_rank,
                delta_bound=args.delta_bound,
            ).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.learning_rate, weight_decay=1e-4,
            )
            rng = np.random.default_rng(seed)
            final_components: dict[str, float] = {}
            steps = 0
            model.train()
            for _ in range(args.epochs):
                order = rng.permutation(active_train)
                for start in range(0, len(order), args.batch_size):
                    batch = order[start:start + args.batch_size]
                    optimizer.zero_grad(set_to_none=True)
                    loss, final_components = batch_loss(
                        model, data, batch, mean, scale, args,
                    )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    steps += 1
            model.eval()
            seed_preservation: list[float] = []
            seed_gate: list[float] = []
            for index in heldout:
                scores, preservation, gate = score_query(model, data, int(index), mean, scale)
                ensemble[int(index)].append(scores)
                seed_preservation.append(preservation)
                seed_gate.append(gate)
            preservation_values.extend(seed_preservation)
            gate_values.extend(seed_gate)
            run_reports.append({
                "outer_study": outer_study, "seed": seed,
                "training_queries": int(len(train)),
                "active_training_queries": int(len(active_train)),
                "heldout_queries": int(len(heldout)), "gradient_steps": steps,
                "final_components": final_components,
                "heldout_preservation": float(np.mean(seed_preservation)),
                "heldout_gate": float(np.mean(seed_gate)),
            })

        for index in heldout:
            index = int(index)
            if len(ensemble[index]) != len(args.seeds):
                raise RuntimeError(f"{outer_study} query {index}: incomplete seed ensemble")
            adapted = np.mean(np.stack(ensemble[index]), axis=0)
            section = data.section(index)
            baseline = (data.candidate[section] @ data.query[index]).cpu().numpy()
            positive = int(data.positive[index])
            baseline_rank = strict_rank(baseline, positive)
            final_rank = strict_rank(adapted, positive)
            baseline_correct = baseline_rank == 1
            final_correct = final_rank == 1
            negative = np.ones(len(baseline), dtype=bool)
            negative[positive] = False
            transitions.append({
                "query_id": str(data.query_ids[index]),
                "unit_id": str(data.unit_ids[index]),
                "study_id": str(data.study_ids[index]),
                "truth_formula": str(data.formulas[index]),
                "truth_candidate_id": str(data.truth_ids[index]),
                "baseline_rank": baseline_rank, "final_rank": final_rank,
                "baseline_correct": baseline_correct, "final_correct": final_correct,
                "corrected": (not baseline_correct) and final_correct,
                "introduced": baseline_correct and (not final_correct),
                "delta": int(final_correct) - int(baseline_correct),
                "has_context": bool(data.context[section].any()),
                "baseline_margin": float(baseline[positive] - baseline[negative].max()),
                "final_margin": float(adapted[positive] - adapted[negative].max()),
                "max_abs_score_delta": float(np.max(np.abs(adapted - baseline))),
            })

    frame = pd.DataFrame(transitions)
    if frame.query_id.duplicated().any() or len(frame) != len(data.query_ids):
        raise RuntimeError("outer-study OOF coverage mismatch")
    transition_path = args.output_dir / "query_oof_transitions.csv.gz"
    frame.to_csv(transition_path, index=False, compression="gzip")
    overall = metrics(frame)
    active = frame[frame.has_context]
    study_results = {study: metrics(group) for study, group in frame.groupby("study_id")}
    bootstrap = cluster_bootstrap(frame, args.bootstrap_resamples, args.seeds[0])
    preservation = float(np.mean(preservation_values))
    gates = {
        "study_formula_cluster_ci_positive": bootstrap["ci_low"] > 0,
        "corrected_gt_introduced": overall["corrected"] > overall["introduced"],
        "risk_weighted_net_lambda2_positive": overall["risk_weighted_net_lambda2"] > 0,
        "every_study_nonnegative": all(value["delta_recall1"] >= 0 for value in study_results.values()),
        "preservation_ge_0_995": preservation >= .995,
    }
    report = {
        "status": "bioaware_b2_leave_study_out_complete",
        "formal": True,
        "protocol": "candidate-context embedding; outer study excluded from every model; fixed three-seed ensemble",
        "overall": overall,
        "context_active": metrics(active),
        "study_formula_cluster_bootstrap": bootstrap,
        "outer_studies": study_results,
        "mean_preservation": preservation,
        "mean_gate": float(np.mean(gate_values)),
        "mean_abs_score_delta": float(frame.max_abs_score_delta.mean()),
        "configuration": vars(args) | {"dataset": str(args.dataset), "output_dir": str(args.output_dir)},
        "run_reports": run_reports,
        "gates": gates,
        "pass": bool(all(gates.values())),
        "contracts": {
            "output_is_contextual_embedding": True,
            "query_embedding_is_unmodified_official": True,
            "no_context_exact_fallback": True,
            "outer_study_outcomes_used_for_training": False,
            "reaction_neighbour_is_positive": False,
            "P2b": "forbidden", "phenotype": "forbidden",
        },
        "provenance": {
            "dataset_sha256": sha256(args.dataset),
            "transitions_sha256": sha256(transition_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "Cross-study OOF contextual-embedding result. It is not an untouched blind study "
            "and cannot alone establish SOTA or a universal DreaMS embedding improvement."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
