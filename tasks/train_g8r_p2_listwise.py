"""Train P2: protected DreaMS + bounded molecule-listwise residual ranking.

Model selection uses formula-isolated out-of-fold predictions on the P2 cache.
The sealed P3 manifests are deliberately not accepted as inputs.  If every
trained configuration fails the safety gates, the script fails closed instead
of exporting a harmful model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from g8r_p2_listwise_core import (
    ResidualListwiseRanker,
    deterministic_formula_fold,
    evaluate_query_scores,
    query_listwise_loss,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE = ROOT / "data/validation/g8r_p2_listwise_cache.npz"
DEFAULT_HEADROOM = ROOT / "data/validation/g8r_p2_cache_headroom.json"
DEFAULT_OUTPUT = ROOT / "data/validation/g8r_p2_listwise_model.pt"


@dataclass(frozen=True)
class Configuration:
    name: str
    hidden_dim: int
    delta_bound: float
    safety_weight: float
    near_weight: float


CONFIGURATIONS = (
    Configuration("linear_conservative", 0, 0.03, 4.0, 2.0),
    Configuration("mlp_conservative", 32, 0.03, 4.0, 2.0),
    Configuration("mlp_balanced", 32, 0.06, 4.0, 2.0),
    Configuration("mlp_safe", 32, 0.06, 8.0, 2.0),
    Configuration("mlp_extended_safe", 32, 0.10, 8.0, 2.0),
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--headroom-audit", type=Path, default=DEFAULT_HEADROOM)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--query-batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--allowed-margin-drop", type=float, default=0.003)
    p.add_argument("--residual-weight", type=float, default=0.02)
    p.add_argument("--seeds", type=int, nargs="+", default=[20260824, 20260825, 20260826])
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class ListwiseCache:
    def __init__(self, path: Path):
        with np.load(path, allow_pickle=True) as body:
            for name in body.files:
                setattr(self, name, body[name])
        self.features = np.asarray(self.features, dtype=np.float32)
        self.query_ptr = np.asarray(self.query_ptr, dtype=np.int64)
        self.molecule_ptr = np.asarray(self.molecule_ptr, dtype=np.int64)
        self.query_formula = np.asarray(self.query_formula, dtype=object)
        self.query_has_near = np.asarray(self.query_has_near, dtype=bool)
        self.feature_names = list(map(str, self.feature_names))
        self.n_queries = len(self.query_ptr) - 1
        if self.n_queries != len(self.query_formula) or self.n_queries != len(self.query_has_near):
            raise RuntimeError("query arrays are not aligned")
        if self.query_ptr[0] != 0 or self.query_ptr[-1] != len(self.molecule_label):
            raise RuntimeError("query_ptr is invalid")
        if self.molecule_ptr[0] != 0 or self.molecule_ptr[-1] != len(self.features):
            raise RuntimeError("molecule_ptr is invalid")
        if self.features.shape[1] != len(self.feature_names) or self.feature_names[0] != "dreams_similarity":
            raise RuntimeError("feature schema is invalid")

    def query_arrays(self, query: int, mean: np.ndarray, scale: np.ndarray,
                     device: torch.device):
        molecule_left, molecule_right = map(int, self.query_ptr[query:query + 2])
        pair_left = int(self.molecule_ptr[molecule_left])
        pair_right = int(self.molecule_ptr[molecule_right])
        local_ptr = self.molecule_ptr[molecule_left:molecule_right + 1] - pair_left
        values = self.features[pair_left:pair_right]
        standardized = (values - mean) / scale
        return (
            torch.as_tensor(standardized, dtype=torch.float32, device=device),
            torch.as_tensor(values[:, 0], dtype=torch.float32, device=device),
            torch.as_tensor(local_ptr, dtype=torch.int64, device=device),
        )


def fit_standardizer(cache: ListwiseCache, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pair_ranges = []
    for query in queries:
        ml, mr = map(int, cache.query_ptr[int(query):int(query) + 2])
        pair_ranges.append((int(cache.molecule_ptr[ml]), int(cache.molecule_ptr[mr])))
    count = sum(right - left for left, right in pair_ranges)
    if count == 0:
        raise RuntimeError("cannot standardize an empty training fold")
    total = np.zeros(cache.features.shape[1], dtype=np.float64)
    total2 = np.zeros_like(total)
    for left, right in pair_ranges:
        block = cache.features[left:right].astype(np.float64)
        total += block.sum(axis=0)
        total2 += np.square(block).sum(axis=0)
    mean = total / count
    variance = np.maximum(total2 / count - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def evaluate(cache: ListwiseCache, model: ResidualListwiseRanker, mean: np.ndarray,
             scale: np.ndarray, queries: np.ndarray, device: torch.device) -> dict:
    model.eval()
    records = []
    with torch.inference_mode():
        for query in queries:
            query = int(query)
            features, baseline, ptr = cache.query_arrays(query, mean, scale, device)
            final, _ = model(features, baseline)
            base_result = evaluate_query_scores(baseline.cpu().numpy(), ptr.cpu().numpy(), 0)
            final_result = evaluate_query_scores(final.cpu().numpy(), ptr.cpu().numpy(), 0)
            records.append({
                "query": query,
                "formula": str(cache.query_formula[query]),
                "near": bool(cache.query_has_near[query]),
                "base_top1": bool(base_result["top1"]),
                "top1": bool(final_result["top1"]),
                "base_mrr": float(base_result["mrr"]),
                "mrr": float(final_result["mrr"]),
                "base_margin": float(base_result["margin"]),
                "margin": float(final_result["margin"]),
            })
    base = np.asarray([record["base_top1"] for record in records], dtype=bool)
    final = np.asarray([record["top1"] for record in records], dtype=bool)
    near = np.asarray([record["near"] for record in records], dtype=bool)
    base_mrr = np.asarray([record["base_mrr"] for record in records], dtype=np.float64)
    final_mrr = np.asarray([record["mrr"] for record in records], dtype=np.float64)
    result = {
        "n_queries": len(records),
        "baseline_recall1": float(base.mean()),
        "recall1": float(final.mean()),
        "delta_recall1": float(final.mean() - base.mean()),
        "baseline_mrr": float(base_mrr.mean()),
        "mrr": float(final_mrr.mean()),
        "delta_mrr": float(final_mrr.mean() - base_mrr.mean()),
        "corrected": int(np.sum((~base) & final)),
        "introduced": int(np.sum(base & (~final))),
        "mean_margin_delta": float(np.mean([r["margin"] - r["base_margin"] for r in records])),
        "records": records,
    }
    if near.any():
        result.update({
            "n_near": int(near.sum()),
            "baseline_near_recall1": float(base[near].mean()),
            "near_recall1": float(final[near].mean()),
            "delta_near_recall1": float(final[near].mean() - base[near].mean()),
        })
    else:
        result.update({"n_near": 0, "baseline_near_recall1": None,
                       "near_recall1": None, "delta_near_recall1": None})
    return result


def selection_tuple(metrics: dict) -> tuple[float, ...]:
    near_delta = metrics["delta_near_recall1"] if metrics["delta_near_recall1"] is not None else -1.0
    safe = (
        metrics["delta_recall1"] >= 0.0
        and near_delta >= 0.0
        and metrics["delta_mrr"] >= 0.0
        and metrics["corrected"] >= metrics["introduced"]
    )
    return (
        float(safe),
        metrics["delta_recall1"],
        near_delta,
        metrics["corrected"] - metrics["introduced"],
        metrics["delta_mrr"],
    )


def train_one(cache: ListwiseCache, train_queries: np.ndarray, dev_queries: np.ndarray,
              configuration: Configuration, seed: int, a: argparse.Namespace,
              device: torch.device) -> tuple[ResidualListwiseRanker, np.ndarray, np.ndarray, int, dict]:
    set_seed(seed)
    mean, scale = fit_standardizer(cache, train_queries)
    model = ResidualListwiseRanker(
        cache.features.shape[1], configuration.hidden_dim, configuration.delta_bound,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)
    rng = np.random.default_rng(seed)
    best_state = None
    best_metrics = None
    best_epoch = 0
    bad_epochs = 0
    for epoch in range(1, a.epochs + 1):
        model.train()
        shuffled = rng.permutation(train_queries)
        optimizer.zero_grad(set_to_none=True)
        batch_losses = []
        epoch_loss = 0.0
        for position, query in enumerate(shuffled, start=1):
            query = int(query)
            features, baseline, ptr = cache.query_arrays(query, mean, scale, device)
            output = query_listwise_loss(
                model, features, baseline, ptr, 0,
                temperature=a.temperature,
                safety_weight=configuration.safety_weight,
                allowed_margin_drop=a.allowed_margin_drop,
                residual_weight=a.residual_weight,
            )
            weight = configuration.near_weight if cache.query_has_near[query] else 1.0
            batch_losses.append(output.total * weight)
            if len(batch_losses) == a.query_batch_size or position == len(shuffled):
                loss = torch.stack(batch_losses).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                epoch_loss += float(loss.detach()) * len(batch_losses)
                batch_losses.clear()
        metrics = evaluate(cache, model, mean, scale, dev_queries, device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = epoch_loss / len(shuffled)
        near_text = (f"{metrics['delta_near_recall1']:+.4f}"
                     if metrics["delta_near_recall1"] is not None else "NA")
        print(
            f"[{configuration.name} seed={seed} epoch={epoch:02d}] "
            f"loss={metrics['train_loss']:.5f} R1={metrics['recall1']:.4f} "
            f"dR1={metrics['delta_recall1']:+.4f} near={near_text} "
            f"C/I={metrics['corrected']}/{metrics['introduced']}", flush=True,
        )
        if best_metrics is None or selection_tuple(metrics) > selection_tuple(best_metrics):
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            best_metrics = metrics
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= a.patience:
                break
    if best_state is None or best_metrics is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, mean, scale, best_epoch, best_metrics


def train_fixed_epochs(cache: ListwiseCache, queries: np.ndarray, mean: np.ndarray,
                       scale: np.ndarray, configuration: Configuration, seed: int,
                       epochs: int, a: argparse.Namespace, device: torch.device
                       ) -> ResidualListwiseRanker:
    """Fit the frozen final recipe on all P2 queries for exactly ``epochs``."""
    set_seed(seed)
    model = ResidualListwiseRanker(
        cache.features.shape[1], configuration.hidden_dim, configuration.delta_bound,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)
    rng = np.random.default_rng(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled = rng.permutation(queries)
        optimizer.zero_grad(set_to_none=True)
        pending = []
        running = 0.0
        for position, query in enumerate(shuffled, start=1):
            query = int(query)
            features, baseline, ptr = cache.query_arrays(query, mean, scale, device)
            output = query_listwise_loss(
                model, features, baseline, ptr, 0,
                temperature=a.temperature,
                safety_weight=configuration.safety_weight,
                allowed_margin_drop=a.allowed_margin_drop,
                residual_weight=a.residual_weight,
            )
            weight = configuration.near_weight if cache.query_has_near[query] else 1.0
            pending.append(output.total * weight)
            if len(pending) == a.query_batch_size or position == len(shuffled):
                loss = torch.stack(pending).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                running += float(loss.detach()) * len(pending)
                pending.clear()
        print(f"[final seed={seed} epoch={epoch:02d}/{epochs}] loss={running / len(shuffled):.5f}",
              flush=True)
    return model


def combine_oof(records_by_fold: list[list[dict]]) -> dict:
    records = [record for fold in records_by_fold for record in fold]
    if len({record["query"] for record in records}) != len(records):
        raise RuntimeError("OOF query appeared in more than one fold")
    base = np.asarray([record["base_top1"] for record in records], dtype=bool)
    final = np.asarray([record["top1"] for record in records], dtype=bool)
    near = np.asarray([record["near"] for record in records], dtype=bool)
    base_mrr = np.asarray([record["base_mrr"] for record in records])
    final_mrr = np.asarray([record["mrr"] for record in records])
    return {
        "n_queries": len(records),
        "baseline_recall1": float(base.mean()),
        "recall1": float(final.mean()),
        "delta_recall1": float(final.mean() - base.mean()),
        "baseline_near_recall1": float(base[near].mean()),
        "near_recall1": float(final[near].mean()),
        "delta_near_recall1": float(final[near].mean() - base[near].mean()),
        "baseline_mrr": float(base_mrr.mean()),
        "mrr": float(final_mrr.mean()),
        "delta_mrr": float(final_mrr.mean() - base_mrr.mean()),
        "corrected": int(np.sum((~base) & final)),
        "introduced": int(np.sum(base & (~final))),
        "records": records,
    }


def formula_cluster_ci(records: list[dict], n_boot: int, seed: int) -> dict:
    groups: dict[str, list[float]] = {}
    for record in records:
        groups.setdefault(record["formula"], []).append(float(record["top1"]) - float(record["base_top1"]))
    formulas = sorted(groups)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for iteration in range(n_boot):
        sampled = rng.integers(0, len(formulas), len(formulas))
        values = np.concatenate([np.asarray(groups[formulas[index]]) for index in sampled])
        draws[iteration] = values.mean()
    return {
        "mean": float(np.mean([value for values in groups.values() for value in values])),
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
    }


def main() -> None:
    a = parse_args()
    if a.output.exists() and not a.overwrite:
        raise FileExistsError(f"refusing to overwrite {a.output}")
    if a.folds < 3 or a.epochs < 1 or a.query_batch_size < 1 or a.temperature <= 0:
        raise ValueError("invalid training parameter")
    if a.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    audit_path = a.cache.with_suffix(".json")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "g8r_p2_listwise_cache_built":
        raise RuntimeError("P2 listwise cache audit is missing or invalid")
    if audit.get("cache_sha256") != sha256_file(a.cache):
        raise RuntimeError("P2 listwise cache hash mismatch")
    headroom = json.loads(a.headroom_audit.read_text(encoding="utf-8"))
    if headroom.get("status") != "g8r_p2_cache_headroom_passed" or not headroom.get("gates", {}).get("pass"):
        raise RuntimeError("P2 cache/headroom audit has not passed")
    if headroom.get("cache_sha256") != sha256_file(a.cache):
        raise RuntimeError("P2 headroom audit belongs to a different cache")
    cache = ListwiseCache(a.cache)
    device = torch.device(a.device)
    all_queries = np.arange(cache.n_queries, dtype=np.int64)
    fold = np.asarray([deterministic_formula_fold(str(value), a.folds) for value in cache.query_formula])
    if len(set(cache.query_formula[fold == 0]) & set(cache.query_formula[fold != 0])):
        raise RuntimeError("formula split leaked across folds")

    configuration_reports = []
    best_configuration = None
    best_metrics = None
    best_epochs = None
    for configuration in CONFIGURATIONS:
        fold_records = []
        fold_metrics = []
        fold_epochs = []
        for held_fold in range(a.folds):
            # Nested formula split: the outer fold is never used for early
            # stopping.  A different formula fold inside the outer-training
            # body chooses the epoch, then the model is refit on all outer
            # training queries for exactly that many epochs.
            inner_fold = (held_fold + 1) % a.folds
            inner_train = all_queries[(fold != held_fold) & (fold != inner_fold)]
            inner_dev = all_queries[fold == inner_fold]
            outer_train = all_queries[fold != held_fold]
            outer_dev = all_queries[fold == held_fold]
            if min(len(inner_train), len(inner_dev), len(outer_train), len(outer_dev)) == 0:
                raise RuntimeError("empty formula fold")
            _, _, _, epoch, inner_metrics = train_one(
                cache, inner_train, inner_dev, configuration,
                a.seeds[0] + 100 * held_fold, a, device,
            )
            outer_mean, outer_scale = fit_standardizer(cache, outer_train)
            outer_model = train_fixed_epochs(
                cache, outer_train, outer_mean, outer_scale, configuration,
                a.seeds[0] + 100 * held_fold + 1, epoch, a, device,
            )
            metrics = evaluate(cache, outer_model, outer_mean, outer_scale, outer_dev, device)
            fold_records.append(metrics["records"])
            fold_metrics.append({
                "outer_fold": held_fold,
                "inner_epoch_selection_fold": inner_fold,
                "selected_epoch": epoch,
                "inner_selection": {key: value for key, value in inner_metrics.items() if key != "records"},
                "outer_evaluation": {key: value for key, value in metrics.items() if key != "records"},
            })
            fold_epochs.append(epoch)
        pooled = combine_oof(fold_records)
        pooled["formula_cluster_bootstrap"] = formula_cluster_ci(pooled["records"], a.bootstrap, a.seeds[0])
        report = {
            "configuration": asdict(configuration),
            "fold_metrics": fold_metrics,
            "fold_best_epochs": fold_epochs,
            "pooled_oof": {key: value for key, value in pooled.items() if key != "records"},
        }
        configuration_reports.append(report)
        print(f"[OOF {configuration.name}] dR1={pooled['delta_recall1']:+.4f} "
              f"near={pooled['delta_near_recall1']:+.4f} C/I={pooled['corrected']}/{pooled['introduced']}",
              flush=True)
        if best_metrics is None or selection_tuple(pooled) > selection_tuple(best_metrics):
            best_configuration = configuration
            best_metrics = pooled
            best_epochs = fold_epochs

    assert best_configuration is not None and best_metrics is not None and best_epochs is not None
    gates = {
        "overall_recall1_nonnegative": best_metrics["delta_recall1"] >= 0.0,
        "near_recall1_nonnegative": best_metrics["delta_near_recall1"] >= 0.0,
        "mrr_nonnegative": best_metrics["delta_mrr"] >= 0.0,
        "corrected_ge_introduced": best_metrics["corrected"] >= best_metrics["introduced"],
        "overall_gain_at_least_three_points": best_metrics["delta_recall1"] >= 0.03,
        "formula_cluster_ci_positive": best_metrics["formula_cluster_bootstrap"]["ci_low"] > 0.0,
    }
    gates["pass"] = all(gates.values())
    report_path = a.output.with_suffix(".selection.json")
    selection_report = {
        "status": "g8r_p2_selection_passed" if gates["pass"] else "g8r_p2_selection_failed",
        "cache_sha256": sha256_file(a.cache),
        "configurations": configuration_reports,
        "selected_configuration": asdict(best_configuration),
        "selected_oof": {key: value for key, value in best_metrics.items() if key != "records"},
        "gates": gates,
        "important_limit": "OOF development evidence is not sealed-P3 evidence and cannot guarantee +4 pp.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(selection_report, ensure_ascii=False, indent=2), encoding="utf-8")
    if not gates["pass"]:
        raise RuntimeError(f"P2 OOF safety/efficacy gates failed; see {report_path}")

    final_epochs = max(1, int(round(float(np.median(best_epochs)))))
    full_mean, full_scale = fit_standardizer(cache, all_queries)
    states = []
    for seed in a.seeds:
        model = train_fixed_epochs(
            cache, all_queries, full_mean, full_scale, best_configuration,
            seed, final_epochs, a, device,
        )
        states.append({name: value.detach().cpu() for name, value in model.state_dict().items()})
    artifact = {
        "format": "g8r_p2_listwise_residual_v1",
        "feature_names": cache.feature_names,
        "standardizer_mean": full_mean,
        "standardizer_scale": full_scale,
        "configuration": asdict(best_configuration),
        "training_epochs": final_epochs,
        "seeds": list(a.seeds),
        "state_dicts": states,
        "cache_sha256": sha256_file(a.cache),
        "cache_audit_sha256": sha256_file(audit_path),
        "headroom_audit_sha256": sha256_file(a.headroom_audit),
        "selection_report_sha256": sha256_file(report_path),
        "selection_metrics": {key: value for key, value in best_metrics.items() if key != "records"},
        "p3_used_for_training_or_selection": False,
    }
    temporary = a.output.with_suffix(a.output.suffix + ".tmp")
    torch.save(artifact, temporary)
    temporary.replace(a.output)
    a.output.with_suffix(".sha256").write_text(sha256_file(a.output) + "\n", encoding="utf-8")
    print(f"[P2] PASS and frozen: {a.output}")


if __name__ == "__main__":
    main()
