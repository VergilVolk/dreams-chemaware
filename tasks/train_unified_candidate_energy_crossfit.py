#!/usr/bin/env python
"""Formula-cross-fitted unified candidate energy on consumed GNPS development.

The model is a WSE-anchored convex fusion of calibrated candidate evidence.  It has
one global non-negative residual coefficient per additional evidence family;
there is no per-query expert routing, hard promotion, voting, or truth-derived
input.  Five formula-hash folds produce predictions for every query from a
model that did not train on that query's molecular formula.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from grand_unified_evidence_core import rank_logit_evidence, strict_ranks
from train_grand_unified_evidence_model import load_bundle


BASE = "weighted_spectral_entropy"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def formula_fold(value: str, folds: int, seed: int) -> int:
    payload = f"{seed}|{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % folds


class AnchoredCandidateEnergy(nn.Module):
    """A convex WSE-anchored fusion of calibrated candidate evidence.

    The non-anchor coefficients are non-negative and have total mass at most
    one.  Consequently, correlated evidence families cannot amplify one
    another without bound: the remaining mass always belongs to WSE.  Exact
    zero is allowed for an unhelpful family.
    """

    def __init__(self, module_names: tuple[str, ...], base_name: str = BASE) -> None:
        super().__init__()
        if base_name not in module_names or len(set(module_names)) != len(module_names):
            raise ValueError("module list must be unique and contain the anchor")
        self.module_names = module_names
        self.base_index = module_names.index(base_name)
        # Start close to the strong WSE anchor.  clamp_min permits exact zero;
        # the normalization below prevents repeated correlated evidence from
        # increasing the total non-anchor mass beyond one.
        initial = torch.full((len(module_names),), 0.02)
        initial[self.base_index] = 0.0
        self.raw_alpha = nn.Parameter(initial)

    def coefficients(self) -> torch.Tensor:
        alpha = torch.clamp_min(self.raw_alpha, 0.0)
        mask = torch.ones_like(alpha)
        mask[self.base_index] = 0.0
        alpha = alpha * mask
        return alpha / torch.clamp_min(torch.sum(alpha), 1.0)

    def mixture_weights(self) -> torch.Tensor:
        """Effective weights, including the implicit remaining WSE mass."""
        weights = self.coefficients().clone()
        weights[self.base_index] = 1.0 - torch.sum(weights)
        return weights

    def forward(self, evidence: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
        if evidence.shape != availability.shape:
            raise ValueError("evidence/availability shape mismatch")
        base = evidence[self.base_index]
        residual = (evidence - base[None, :]) * availability
        return base + torch.sum(self.coefficients()[:, None] * residual, dim=0)


def candidate_indices(ptr: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    blocks = [np.arange(ptr[q], ptr[q + 1], dtype=np.int64) for q in queries]
    index = np.concatenate(blocks) if blocks else np.asarray([], dtype=np.int64)
    lengths = np.asarray([len(block) for block in blocks], dtype=np.int64)
    return index, np.concatenate(([0], np.cumsum(lengths)))


def listwise_loss(logits: torch.Tensor, ptr: np.ndarray, labels: np.ndarray) -> torch.Tensor:
    losses = []
    for left, right in zip(ptr[:-1], ptr[1:]):
        left, right = int(left), int(right)
        target = np.flatnonzero(labels[left:right] > 0)
        if len(target) != 1:
            raise RuntimeError("every training query must contain one positive")
        losses.append(torch.nn.functional.cross_entropy(
            logits[left:right][None, :],
            torch.tensor([int(target[0])], device=logits.device),
        ))
    return torch.stack(losses).mean()


def prepare(bundle: dict) -> dict:
    evidence, _ = rank_logit_evidence(
        bundle["scores"], bundle["query_ptr"], bundle["availability"],
    )
    return {
        "evidence": evidence,
        "availability": np.asarray(bundle["availability"], dtype=np.float32),
        "query_ptr": np.asarray(bundle["query_ptr"], dtype=np.int64),
        "labels": np.asarray(bundle["labels"], dtype=np.int8),
        "formulas": np.asarray(bundle["group_ids"]).astype("U"),
        "near": np.asarray(bundle["near_query"], dtype=bool),
    }


def fit_model(
    prepared: dict,
    module_names: tuple[str, ...],
    queries: np.ndarray,
    epochs: int,
    batch_queries: int,
    learning_rate: float,
    anchor_strength: float,
    seed: int,
    device: torch.device,
) -> AnchoredCandidateEnergy:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = AnchoredCandidateEnergy(module_names).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = queries[rng.permutation(len(queries))]
        for start in range(0, len(queries), batch_queries):
            batch = order[start:start + batch_queries]
            index, ptr = candidate_indices(prepared["query_ptr"], batch)
            evidence = torch.as_tensor(prepared["evidence"][:, index], device=device)
            availability = torch.as_tensor(prepared["availability"][:, index], device=device)
            labels = prepared["labels"][index]
            logits = model(evidence, availability)
            anchor = evidence[model.base_index]
            loss = listwise_loss(logits, ptr, labels)
            loss = loss + anchor_strength * torch.mean(torch.square(logits - anchor))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            # Project the directly optimized reliability parameters back onto
            # their scientifically interpretable non-negative domain.
            with torch.no_grad():
                model.raw_alpha.clamp_(min=0.0)
                model.raw_alpha[model.base_index] = 0.0
    return model


@torch.inference_mode()
def predict_queries(
    model: AnchoredCandidateEnergy,
    prepared: dict,
    queries: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    index, _ = candidate_indices(prepared["query_ptr"], queries)
    evidence = torch.as_tensor(prepared["evidence"][:, index], device=device)
    availability = torch.as_tensor(prepared["availability"][:, index], device=device)
    return index, model(evidence, availability).cpu().numpy().astype(np.float32)


def metric_summary(scores: np.ndarray, prepared: dict) -> tuple[dict, np.ndarray]:
    ptr = prepared["query_ptr"]
    labels = prepared["labels"]
    ranks = strict_ranks(scores, ptr, labels)
    macro_auc = []
    macro_auprc = []
    margins = []
    for left, right in zip(ptr[:-1], ptr[1:]):
        left, right = int(left), int(right)
        y = labels[left:right]
        s = scores[left:right]
        positive = float(s[np.flatnonzero(y > 0)[0]])
        negative = s[y == 0]
        macro_auc.append(float(np.mean(positive > negative) + 0.5 * np.mean(positive == negative)))
        macro_auprc.append(float(average_precision_score(y, s)))
        ordered = np.sort(s)[::-1]
        margins.append(float(ordered[0] - ordered[1]) if len(ordered) > 1 else 0.0)
    macro_auc = np.asarray(macro_auc, dtype=np.float64)
    macro_auprc = np.asarray(macro_auprc, dtype=np.float64)
    margins = np.asarray(margins, dtype=np.float64)
    result = {
        "queries": int(len(ranks)),
        **{f"recall@{k}": float(np.mean(ranks <= k)) for k in (1, 2, 3, 5, 10, 20)},
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
        "median_rank": float(np.median(ranks)),
        "macro_query_auroc": float(np.mean(macro_auc)),
        "macro_query_auprc": float(np.mean(macro_auprc)),
        "micro_candidate_auroc": float(roc_auc_score(labels, scores)),
        "micro_candidate_auprc": float(average_precision_score(labels, scores)),
        "mean_top1_top2_margin": float(np.mean(margins)),
        "median_top1_top2_margin": float(np.median(margins)),
    }
    near = np.asarray(prepared["near"], dtype=bool)
    if np.any(near):
        near_ranks = ranks[near]
        near_index, _ = candidate_indices(ptr, np.flatnonzero(near))
        result["near"] = {
            "queries": int(np.sum(near)),
            **{f"recall@{k}": float(np.mean(near_ranks <= k)) for k in (1, 2, 3, 5, 10, 20)},
            "mrr": float(np.mean(1.0 / near_ranks)),
            "mean_rank": float(np.mean(near_ranks)),
            "median_rank": float(np.median(near_ranks)),
            "macro_query_auroc": float(np.mean(macro_auc[near])),
            "macro_query_auprc": float(np.mean(macro_auprc[near])),
            "micro_candidate_auroc": float(roc_auc_score(labels[near_index], scores[near_index])),
            "micro_candidate_auprc": float(average_precision_score(labels[near_index], scores[near_index])),
            "mean_top1_top2_margin": float(np.mean(margins[near])),
            "median_top1_top2_margin": float(np.median(margins[near])),
        }
    else:
        result["near"] = {"queries": 0}
    return result, ranks


def clustered_ci(delta: np.ndarray, groups: np.ndarray, seed: int, resamples: int) -> list[float]:
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        draws[index] = np.mean(np.concatenate([delta[members[group]] for group in sampled]))
    return [float(value) for value in np.quantile(draws, [0.025, 0.975])]


def compare(scores: np.ndarray, prepared: dict, base_scores: np.ndarray, seed: int, resamples: int) -> dict:
    candidate_metrics, candidate_rank = metric_summary(scores, prepared)
    baseline_metrics, baseline_rank = metric_summary(base_scores, prepared)
    corrected = (candidate_rank == 1) & (baseline_rank != 1)
    introduced = (candidate_rank != 1) & (baseline_rank == 1)
    delta = (candidate_rank == 1).astype(float) - (baseline_rank == 1).astype(float)
    metric_delta = {
        key: float(candidate_metrics[key] - baseline_metrics[key])
        for key in candidate_metrics
        if isinstance(candidate_metrics[key], (int, float))
        and isinstance(baseline_metrics.get(key), (int, float))
        and key != "queries"
    }
    return {
        "candidate": candidate_metrics,
        "baseline_wse": baseline_metrics,
        "delta_candidate_minus_wse": metric_delta,
        "paired_vs_wse": {
            "delta_recall@1": float(np.mean(delta)),
            "corrected": int(corrected.sum()),
            "introduced": int(introduced.sum()),
            "risk_net_corrected_minus_2introduced": int(corrected.sum() - 2 * introduced.sum()),
            "formula_cluster_bootstrap_95ci": clustered_ci(
                delta, prepared["formulas"], seed, resamples,
            ),
        },
    }


def crossfit(
    train: dict,
    targets: dict[str, dict],
    module_names: tuple[str, ...],
    folds: int,
    epochs: int,
    batch_queries: int,
    learning_rate: float,
    anchor_strength: float,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], list[dict]]:
    outputs = {
        name: np.full(len(body["labels"]), np.nan, dtype=np.float32)
        for name, body in targets.items()
    }
    fold_reports = []
    train_folds = np.asarray([formula_fold(v, folds, seed) for v in train["formulas"]])
    target_folds = {
        name: np.asarray([formula_fold(v, folds, seed) for v in body["formulas"]])
        for name, body in targets.items()
    }
    for fold in range(folds):
        train_queries = np.flatnonzero(train_folds != fold)
        model = fit_model(
            train, module_names, train_queries, epochs, batch_queries,
            learning_rate, anchor_strength, seed + fold, device,
        )
        fold_report = {
            "fold": fold,
            "train_queries": int(len(train_queries)),
            "residual_coefficients": {
                name: float(value) for name, value in zip(
                    module_names, model.coefficients().detach().cpu().numpy(), strict=True,
                )
            },
            "effective_mixture_weights": {
                name: float(value) for name, value in zip(
                    module_names, model.mixture_weights().detach().cpu().numpy(), strict=True,
                )
            },
            "prediction_queries": {},
        }
        for name, body in targets.items():
            queries = np.flatnonzero(target_folds[name] == fold)
            index, scores = predict_queries(model, body, queries, device)
            outputs[name][index] = scores
            fold_report["prediction_queries"][name] = int(len(queries))
        fold_reports.append(fold_report)
    if any(not np.all(np.isfinite(values)) for values in outputs.values()):
        raise RuntimeError("cross-fitting did not predict every candidate exactly once")
    return outputs, fold_reports


def select_modules(bundle: dict, keep: tuple[str, ...]) -> dict:
    names = tuple(map(str, bundle["module_names"]))
    position = [names.index(name) for name in keep]
    selected = dict(bundle)
    selected["module_names"] = np.asarray(keep)
    selected["scores"] = np.asarray(bundle["scores"])[position]
    selected["availability"] = np.asarray(bundle["availability"])[position]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-bundle", type=Path, required=True)
    parser.add_argument("--formula-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-queries", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--anchor-strength", type=float, default=0.01)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--run-ablations", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    identity_raw = load_bundle(args.identity_bundle)
    formula_raw = load_bundle(args.formula_bundle)
    names = tuple(map(str, identity_raw["module_names"]))
    if tuple(map(str, formula_raw["module_names"])) != names or BASE not in names:
        raise RuntimeError("identity/formula bundles have incompatible module axes")

    arms = {"full": names}
    if args.run_ablations:
        for name in names:
            if name != BASE:
                arms[f"without_{name}"] = tuple(item for item in names if item != name)
    args.output.mkdir(parents=True)
    device = torch.device(args.device)
    report = {
        "status": "UNIFIED_CANDIDATE_ENERGY_CROSSFIT_COMPLETE",
        "dataset_role": "GNPS development consumed",
        "model": "WSE-anchored convex fusion of rank-logit candidate evidence",
        "hard_routing": False,
        "truth_derived_inputs": False,
        "formula_crossfit_folds": args.folds,
        "crossfit_rule": (
            "train each fold on identity queries whose molecular formula hashes to "
            "another fold; predict identity and formula-panel queries in the held formula fold"
        ),
        "formula_panel_labels_used_for_fitting": False,
        "panels_are_independent_confirmations": False,
        "panel_relation": (
            "identity and formula panels are correlated GNPS development strata; "
            "both are reported for robustness, not counted as two external replications"
        ),
        "optimization": {
            "epochs": args.epochs,
            "batch_queries": args.batch_queries,
            "learning_rate": args.learning_rate,
            "anchor_strength": args.anchor_strength,
            "seed": args.seed,
        },
        "bioaware": "unavailable_without_real_sample_event_context",
        "arms": {},
        "input_sha256": {
            "identity": sha256(args.identity_bundle),
            "formula": sha256(args.formula_bundle),
        },
    }
    full_predictions = None
    for arm_index, (arm, arm_names) in enumerate(arms.items()):
        identity = prepare(select_modules(identity_raw, arm_names))
        formula = prepare(select_modules(formula_raw, arm_names))
        outputs, folds = crossfit(
            identity, {"identity": identity, "formula": formula}, arm_names,
            args.folds, args.epochs, args.batch_queries, args.learning_rate,
            args.anchor_strength, args.seed, device,
        )
        base_index = arm_names.index(BASE)
        arm_report = {"modules": list(arm_names), "folds": folds, "panels": {}}
        for panel_name, body in (("identity", identity), ("formula", formula)):
            base_scores = body["evidence"][base_index]
            arm_report["panels"][panel_name] = compare(
                outputs[panel_name], body, base_scores,
                args.seed + arm_index * 100 + (0 if panel_name == "identity" else 1),
                args.bootstrap_resamples,
            )
        report["arms"][arm] = arm_report
        if arm == "full":
            full_predictions = outputs

    full = report["arms"]["full"]["panels"]
    gates = {}
    for panel in ("identity", "formula"):
        paired = full[panel]["paired_vs_wse"]
        auc_delta = (
            full[panel]["candidate"]["micro_candidate_auroc"]
            - full[panel]["baseline_wse"]["micro_candidate_auroc"]
        )
        gates[f"{panel}_positive_ci"] = paired["formula_cluster_bootstrap_95ci"][0] > 0
        gates[f"{panel}_positive_risk_net"] = paired["risk_net_corrected_minus_2introduced"] > 0
        gates[f"{panel}_micro_auc_not_materially_lower"] = auc_delta >= -0.001
    report["freeze_gates"] = gates
    report["eligible_for_enveda"] = bool(all(gates.values()))

    # Freeze one final model trained on all identity-panel formulas only.  Its
    # hyperparameters are fixed above; formula-panel labels never fit it.
    identity = prepare(identity_raw)
    all_queries = np.arange(len(identity["query_ptr"]) - 1, dtype=np.int64)
    final_model = fit_model(
        identity, names, all_queries, args.epochs, args.batch_queries,
        args.learning_rate, args.anchor_strength, args.seed + 1000, device,
    )
    model_path = args.output / "model.pt"
    torch.save({
        "state_dict": final_model.state_dict(),
        "module_names": names,
        "base_name": BASE,
        "evidence_transform": "within-query rank-logit centered and clipped at 4",
        "seed": args.seed + 1000,
    }, model_path)
    report["final_residual_coefficients"] = {
        name: float(value) for name, value in zip(
            names, final_model.coefficients().detach().cpu().numpy(), strict=True,
        )
    }
    report["final_effective_mixture_weights"] = {
        name: float(value) for name, value in zip(
            names, final_model.mixture_weights().detach().cpu().numpy(), strict=True,
        )
    }
    report["model_sha256"] = sha256(model_path)
    report["external_test_opened"] = False
    if full_predictions is not None:
        np.savez_compressed(
            args.output / "crossfit_scores.npz",
            scores_identity=full_predictions["identity"],
            scores_formula=full_predictions["formula"],
        )
        report["crossfit_scores_sha256"] = sha256(args.output / "crossfit_scores.npz")
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
