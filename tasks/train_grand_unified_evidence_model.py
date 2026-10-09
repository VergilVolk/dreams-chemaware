"""Train and evaluate the all-module joint evidence model."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from grand_unified_evidence_core import (  # noqa: E402
    AllModuleEvidenceModel,
    listwise_loss,
    rank_logit_evidence,
    strict_ranks,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_bundle(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key] for key in data.files}
    required = {"module_names", "scores", "availability", "query_ptr", "labels", "group_ids"}
    missing = required - result.keys()
    if missing:
        raise ValueError(f"{path} missing fields: {sorted(missing)}")
    return result


def scalar_text(bundle: dict, key: str, default: str = "") -> str:
    if key not in bundle:
        return default
    value = np.asarray(bundle[key])
    if value.size != 1:
        raise ValueError(f"bundle field {key} must be scalar")
    return str(value.reshape(-1)[0])


def assert_sealed_external_test(bundle: dict, path: Path) -> None:
    """Fail closed when a consumed development collection is passed as final test."""
    dataset_id = scalar_text(bundle, "dataset_id", "UNDECLARED")
    role = scalar_text(bundle, "evaluation_role", "UNDECLARED")
    truth_status = scalar_text(bundle, "truth_status", "UNDECLARED")
    allowed = bool(int(np.asarray(bundle.get("allow_final_claim", 0)).reshape(-1)[0]))
    forbidden = ("massspecgym", "gnps_gold", "gnps_silver", "massbank_2026.03")
    normalized = dataset_id.lower().replace("-", "_").replace("/", "_")
    if any(token in normalized for token in forbidden):
        raise ValueError(
            f"DATA_LEAKAGE_GUARD: {path} identifies consumed collection {dataset_id}; "
            "it cannot be used as a final external test"
        )
    accepted_truth_states = {"frozen_unscored"}
    if role != "sealed_external_test" or truth_status not in accepted_truth_states or not allowed:
        raise ValueError(
            "DATA_LEAKAGE_GUARD: external test must declare "
            "evaluation_role=sealed_external_test, truth_status in "
            "{frozen_unscored}, "
            f"allow_final_claim=1; got dataset={dataset_id}, role={role}, "
            f"truth_status={truth_status}, allow_final_claim={allowed}"
        )
    if dataset_id != "enveda_180_filtered_20260713":
        raise ValueError(
            f"DATA_LEAKAGE_GUARD: undeclared final-test dataset {dataset_id}; "
            "only the frozen Enveda-180 asset is authorized"
        )
    guard = bool(int(np.asarray(bundle.get("leakage_guard_complete", 0)).reshape(-1)[0]))
    frozen_before_scoring = bool(int(np.asarray(bundle.get("model_frozen_before_scoring", 0)).reshape(-1)[0]))
    required_hashes = (
        "benchmark_report_sha256", "benchmark_checksums_sha256",
        "benchmark_panel_sha256", "component_score_bundle_sha256",
        "freeze_contract_sha256", "module_registry_sha256", "model_sha256",
    )
    invalid_hashes = [
        key for key in required_hashes
        if len(scalar_text(bundle, key, "")) != 64
        or any(ch not in "0123456789abcdef" for ch in scalar_text(bundle, key, "").lower())
    ]
    if not guard or not frozen_before_scoring or invalid_hashes:
        raise ValueError(
            "DATA_LEAKAGE_GUARD: Enveda bundle lacks a complete pre-scoring freeze chain; "
            f"leakage_guard_complete={guard}, model_frozen_before_scoring={frozen_before_scoring}, "
            f"invalid_hashes={invalid_hashes}"
        )


def group_partition(groups: np.ndarray, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(groups).astype(str)
    unique = np.unique(groups)
    keyed = sorted(
        unique,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    n_val = max(1, int(round(len(keyed) * val_fraction)))
    val_groups = set(keyed[:n_val])
    val = np.asarray([value in val_groups for value in groups], dtype=bool)
    return ~val, val


def subset_queries(bundle: dict, queries: np.ndarray) -> dict:
    ptr = np.asarray(bundle["query_ptr"], dtype=np.int64)
    blocks = [np.arange(ptr[q], ptr[q + 1], dtype=np.int64) for q in queries]
    candidate_index = np.concatenate(blocks) if blocks else np.asarray([], dtype=np.int64)
    lengths = np.asarray([len(block) for block in blocks], dtype=np.int64)
    new_ptr = np.concatenate(([0], np.cumsum(lengths)))
    return {
        "scores": np.asarray(bundle["scores"])[:, candidate_index],
        "availability": np.asarray(bundle["availability"])[:, candidate_index],
        "query_ptr": new_ptr,
        "labels": np.asarray(bundle["labels"])[candidate_index],
        "near_query": np.asarray(bundle.get("near_query", np.zeros(len(ptr) - 1, bool)))[queries],
        "group_ids": np.asarray(bundle.get("group_ids", np.arange(len(ptr) - 1)))[queries],
    }


def tensors(prepared: dict, device: torch.device) -> tuple[torch.Tensor, ...]:
    evidence, summaries = rank_logit_evidence(
        prepared["scores"], prepared["query_ptr"], prepared["availability"]
    )
    return (
        torch.as_tensor(evidence, dtype=torch.float32, device=device),
        torch.as_tensor(prepared["query_ptr"], dtype=torch.long, device=device),
        torch.as_tensor(summaries, dtype=torch.float32, device=device),
        torch.as_tensor(prepared["availability"], dtype=torch.float32, device=device),
        torch.as_tensor(prepared["labels"], dtype=torch.long, device=device),
    )


def ranking_metrics(scores: np.ndarray, ptr: np.ndarray, labels: np.ndarray, near: np.ndarray) -> tuple[dict, np.ndarray]:
    ranks = strict_ranks(scores, ptr, labels)
    macro_auc = []
    pooled_scores, pooled_labels, pooled_weights = [], [], []
    for left, right in zip(ptr[:-1], ptr[1:]):
        left, right = int(left), int(right)
        block_y = labels[left:right]
        block_s = scores[left:right]
        if len(np.unique(block_y)) == 2:
            macro_auc.append(roc_auc_score(block_y, block_s))
            pooled_scores.extend(block_s.tolist())
            pooled_labels.extend(block_y.tolist())
            pooled_weights.extend([1.0 / len(block_y)] * len(block_y))
    result = {
        "queries": int(len(ranks)),
        **{f"recall{k}": float(np.mean(ranks <= k)) for k in (1, 3, 5, 10, 20, 50)},
        "mrr": float(np.mean(1.0 / ranks)),
        "near_recall1": float(np.mean(ranks[near] <= 1)) if near.any() else None,
        "candidate_macro_query_auroc": float(np.mean(macro_auc)) if macro_auc else None,
        "candidate_micro_query_balanced_auroc": float(roc_auc_score(pooled_labels, pooled_scores, sample_weight=pooled_weights)) if pooled_scores else None,
        "candidate_pooled_auroc": float(roc_auc_score(pooled_labels, pooled_scores)) if pooled_scores else None,
    }
    return result, ranks


def clustered_delta_ci(delta: np.ndarray, groups: np.ndarray, seed: int = 20261008, resamples: int = 2000) -> list[float]:
    groups = np.asarray(groups).astype("U")
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        values = np.concatenate([delta[members[group]] for group in sampled])
        draws[index] = np.mean(values)
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def paired_comparison(rank: np.ndarray, baseline_rank: np.ndarray, groups: np.ndarray) -> dict:
    success = (rank == 1).astype(float)
    baseline_success = (baseline_rank == 1).astype(float)
    delta = success - baseline_success
    return {
        "delta_recall1": float(np.mean(delta)),
        "corrected": int(np.sum((success == 1) & (baseline_success == 0))),
        "introduced": int(np.sum((success == 0) & (baseline_success == 1))),
        "risk_net_corrected_minus_2introduced": int(np.sum((success == 1) & (baseline_success == 0)) - 2 * np.sum((success == 0) & (baseline_success == 1))),
        "formula_cluster_bootstrap_95ci": clustered_delta_ci(delta, groups),
    }
@torch.no_grad()
def evaluate(model, bundle: dict, queries: np.ndarray, device: torch.device, comparison_baseline: str | None = None, detailed: bool = True) -> dict:
    part = subset_queries(bundle, queries)
    evidence, ptr, summaries, availability, labels = tensors(part, device)
    model.eval()
    output = model(evidence, ptr, summaries, availability)
    logits = output.logits.detach().cpu().numpy()
    fused, rank = ranking_metrics(logits, part["query_ptr"], part["labels"], np.asarray(part["near_query"], bool))
    weights = output.module_weights.detach().cpu().numpy()
    if not detailed:
        return fused
    module_names = [str(x) for x in bundle["module_names"]]
    standalone = {}
    standalone_ranks = {}
    for module, name in enumerate(module_names):
        applicable = np.asarray(part["availability"])[module] > 0
        coverage = float(np.mean(applicable))
        row = {"candidate_applicability_fraction": coverage}
        if np.all(applicable):
            metrics, method_rank = ranking_metrics(
                np.asarray(part["scores"])[module], part["query_ptr"], part["labels"], np.asarray(part["near_query"], bool)
            )
            row.update(metrics)
            standalone_ranks[name] = method_rank
        standalone[name] = row
    eligible = {name: row for name, row in standalone.items() if "mrr" in row}
    if comparison_baseline is None and eligible:
        comparison_baseline = max(eligible, key=lambda name: eligible[name]["mrr"])
    result = {
        **fused,
        "mean_module_weight": weights.mean(axis=1).tolist(),
        "zero_weight_fraction_when_applicable": float(np.mean(weights[availability.detach().cpu().numpy() > 0] == 0)),
        "standalone_modules": standalone,
    }
    if comparison_baseline:
        if comparison_baseline not in standalone_ranks:
            raise ValueError(f"comparison baseline {comparison_baseline} is absent or not fully applicable")
        result["comparison_baseline"] = comparison_baseline
        result["paired_vs_comparison_baseline"] = paired_comparison(
            rank, standalone_ranks[comparison_baseline], part["group_ids"]
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-bundle", type=Path, required=True)
    parser.add_argument(
        "--test-bundle", type=Path,
        help="forbidden: final external evaluation is performed only by evaluate_frozen_grand_unified_external.py",
    )
    parser.add_argument(
        "--module-registry", type=Path,
        default=ROOT / "tasks/grand_unified_components_v2.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-queries", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--balance-strength", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.test_bundle is not None:
        raise ValueError(
            "DATA_LEAKAGE_GUARD: training and final external evaluation must be separate; "
            "freeze the model first, then use evaluate_frozen_grand_unified_external.py"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_bundle = load_bundle(args.train_bundle)
    module_names = tuple(str(x) for x in train_bundle["module_names"])
    train_mask, val_mask = group_partition(train_bundle["group_ids"], args.val_fraction, args.seed)
    train_queries = np.flatnonzero(train_mask)
    val_queries = np.flatnonzero(val_mask)
    model = AllModuleEvidenceModel(module_names).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    history = []
    best_state = None
    best_val = -np.inf
    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(train_queries)
        epoch_losses = []
        for start in range(0, len(order), args.batch_queries):
            query_batch = order[start:start + args.batch_queries]
            part = subset_queries(train_bundle, query_batch)
            evidence, ptr, summaries, availability, labels = tensors(part, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(evidence, ptr, summaries, availability)
            loss = listwise_loss(
                output.logits, ptr, labels, output.module_weights, availability,
                balance_strength=args.balance_strength,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        val = evaluate(model, train_bundle, val_queries, device, detailed=False)
        row = {"epoch": epoch + 1, "loss": float(np.mean(epoch_losses)), "val": val}
        history.append(row)
        print(json.dumps(row), flush=True)
        if val["mrr"] > best_val:
            best_val = val["mrr"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)

    report = {
        "status": "GRAND_UNIFIED_EVIDENCE_QUALIFIED_COMPLETE",
        "schema": "grand_unified_evidence_model_v2",
        "module_names": list(module_names),
        "all_modules_jointly_scored": False,
        "qualification_boundary": "Only modules explicitly included in the bundle are scored; registry-retired and unavailable modules remain in the audit ledger.",
        "hard_expert_selection": False,
        "train_queries": int(len(train_queries)),
        "validation_queries": int(len(val_queries)),
        "validation": evaluate(model, train_bundle, val_queries, device),
        "history": history,
        "external_test_opened": False,
        "training_bundle_sha256": sha256_file(args.train_bundle),
        "module_registry_sha256": sha256_file(args.module_registry),
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_queries": args.batch_queries,
            "learning_rate": args.learning_rate,
            "val_fraction": args.val_fraction,
            "balance_strength": args.balance_strength,
            "seed": args.seed,
        },
    }

    args.out.mkdir(parents=True, exist_ok=True)
    model_path = args.out / "model.pt"
    torch.save(
        {"state_dict": best_state, "module_names": module_names, "seed": args.seed},
        model_path,
    )
    report["frozen_model_sha256"] = sha256_file(model_path)
    report["freeze_contract_status"] = "DEVELOPMENT_MODEL_FROZEN_EXTERNAL_UNOPENED"
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"written {args.out / 'report.json'}")


if __name__ == "__main__":
    main()
