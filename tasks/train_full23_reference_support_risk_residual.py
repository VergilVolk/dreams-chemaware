#!/usr/bin/env python
"""Repair reference-max bias with parent-risk and support-stability evidence.

This is a consumed-GNPS development experiment.  It leaves Full23 v1 frozen
and compares three formula-OOF residuals:

1. the original count-only listwise loss;
2. the same residual with parent-correct queries weighted by two;
3. the risk-aligned loss with per-expert within-candidate max-support features.

The third arm tests whether an expert maximum is supported by other reference
spectra of the same candidate, rather than merely benefiting from more draws.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_full23_reference_coherence_residual as base
from train_enveda_unified_candidate_integrator import _comparison, _metric_summary


PANELS = base.PANELS
EXPERTS = base.EXPERTS


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--identity-panel", type=Path, required=True)
    p.add_argument("--formula-panel", type=Path, required=True)
    p.add_argument("--score-bundle", type=Path, required=True)
    p.add_argument("--msg-cache", type=Path, required=True)
    p.add_argument("--embedding", action="append", default=[], metavar="NAME=PATH")
    p.add_argument("--full23-scores", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-queries", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=0.015)
    p.add_argument("--weight-decay", type=float, default=0.002)
    p.add_argument("--boundary-l2", type=float, default=0.002)
    p.add_argument("--residual-bound", type=float, default=0.25)
    p.add_argument("--bootstrap-resamples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20261031)
    return p.parse_args()


def formula_fold(value: str, folds: int, seed: int) -> int:
    token = f"{seed}|{str(value).strip()}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % folds


def support_features(graph, pair: dict[str, np.ndarray]) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return query-centred count and within-expert reference support features."""
    n = len(graph.molecule_label)
    columns = [np.empty(n, dtype=np.float32)]
    names = ["log_reference_count"]
    for expert in EXPERTS:
        columns.extend([np.zeros(n, np.float32), np.zeros(n, np.float32)])
        names.extend([f"{expert}__max_minus_second", f"{expert}__max_minus_mean"])
    for molecule, (left, right) in enumerate(zip(graph.molecule_ptr[:-1], graph.molecule_ptr[1:])):
        left, right = int(left), int(right)
        count = right - left
        columns[0][molecule] = np.log1p(count)
        if count < 2:
            continue
        for index, expert in enumerate(EXPERTS):
            values = np.asarray(pair[expert][left:right], dtype=np.float32)
            top_two = np.partition(values, -2)[-2:]
            maximum = float(np.max(top_two))
            columns[1 + 2 * index][molecule] = maximum - float(np.min(top_two))
            columns[2 + 2 * index][molecule] = maximum - float(np.mean(values))
    matrix = np.stack(columns, axis=1)
    for column in range(matrix.shape[1]):
        matrix[:, column] = base.center_by_query(matrix[:, column], graph.query_ptr)
    return matrix, tuple(names)


def query_batch(ptr: np.ndarray, labels: np.ndarray, queries: np.ndarray):
    widths = ptr[queries + 1] - ptr[queries]
    width = int(np.max(widths))
    index = np.zeros((len(queries), width), np.int64)
    valid = np.zeros((len(queries), width), bool)
    target = np.empty(len(queries), np.int64)
    for row, query in enumerate(queries):
        left, right = int(ptr[query]), int(ptr[query + 1])
        size = right - left
        index[row, :size] = np.arange(left, right)
        valid[row, :size] = True
        positive = np.flatnonzero(labels[left:right] > 0)
        if len(positive) != 1:
            raise RuntimeError("query must have exactly one positive")
        target[row] = int(positive[0])
    return index, valid, target


def fit_arm(parent: np.ndarray, features: np.ndarray, ptr: np.ndarray,
            labels: np.ndarray, train_queries: np.ndarray, columns: np.ndarray,
            loss_mode: str, args: argparse.Namespace, seed: int) -> dict:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rows = base.query_rows(ptr, train_queries)
    mean = features[rows][:, columns].mean(0).astype(np.float32)
    scale = features[rows][:, columns].std(0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    x = torch.as_tensor((features[:, columns] - mean) / scale, dtype=torch.float32)
    frozen = torch.as_tensor(parent, dtype=torch.float32)
    weight = torch.nn.Parameter(torch.zeros(len(columns), dtype=torch.float32))
    optimizer = torch.optim.Adam([weight], lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(seed)
    for _ in range(args.epochs):
        order = rng.permutation(train_queries)
        for start in range(0, len(train_queries), args.batch_queries):
            queries = order[start:start + args.batch_queries]
            index, valid, target = query_batch(ptr, labels, queries)
            take = torch.as_tensor(index, dtype=torch.long)
            valid_t = torch.as_tensor(valid)
            target_t = torch.as_tensor(target, dtype=torch.long)
            parent_logits = frozen[take].masked_fill(~valid_t, -1e9)
            logits = frozen[take] + args.residual_bound * torch.tanh(x[take] @ weight)
            logits = logits.masked_fill(~valid_t, -1e9)
            local = torch.nn.functional.cross_entropy(logits, target_t, reduction="none")
            if loss_mode == "risk_listwise":
                parent_correct = torch.argmax(parent_logits, dim=1) == target_t
                local = local * torch.where(parent_correct, 2.0, 1.0)
                loss = torch.mean(local)
            elif loss_mode == "boundary_top1":
                rows = torch.arange(len(target_t))
                true_parent = parent_logits[rows, target_t]
                true_new = logits[rows, target_t]
                wrong_parent = parent_logits.clone(); wrong_parent[rows, target_t] = -1e9
                wrong_new = logits.clone(); wrong_new[rows, target_t] = -1e9
                parent_margin = true_parent - torch.max(wrong_parent, dim=1).values
                new_margin = true_new - torch.max(wrong_new, dim=1).values
                parent_correct = parent_margin >= 0
                reachable = torch.abs(parent_margin) <= 2.0 * args.residual_bound
                # Wrong/reachable: cross the Top-1 boundary by a small fixed
                # numerical margin. Correct/reachable: do not erode the
                # already achieved parent margin; an erosion costs twice.
                repair = torch.relu(0.01 - new_margin)
                preserve = 2.0 * torch.relu(parent_margin - new_margin)
                boundary = torch.where(parent_correct, preserve, repair)
                if torch.any(reachable):
                    loss = torch.mean(boundary[reachable])
                else:
                    loss = torch.sum(weight * 0.0)
                # Boundary supervision is sparse.  Keep its support-feature
                # solution conservative rather than allowing a few reachable
                # queries to drive large, source-specific coefficients.
                loss = loss + args.boundary_l2 * torch.sum(weight.square())
            else:
                loss = torch.mean(local)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return {
        "columns": columns, "mean": mean, "scale": scale,
        "weight": weight.detach().numpy(), "loss_mode": loss_mode,
    }


def predict(model: dict, parent: np.ndarray, features: np.ndarray, ptr: np.ndarray,
            queries: np.ndarray, bound: float, output: np.ndarray) -> None:
    columns = model["columns"]
    x = (features[:, columns] - model["mean"]) / model["scale"]
    residual = bound * np.tanh(x @ model["weight"])
    for query in queries:
        left, right = int(ptr[query]), int(ptr[query + 1])
        output[left:right] = parent[left:right] + residual[left:right]


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    paths = {"identity_disjoint": args.identity_panel, "formula_disjoint": args.formula_panel}
    graphs = {name: base.graph_from_panel(path) for name, path in paths.items()}
    pair = base.pair_evidence(args, graphs)
    panels, parents, features = {}, {}, {}
    feature_names = None
    for name in PANELS:
        features[name], names = support_features(graphs[name], pair[name])
        feature_names = names if feature_names is None else feature_names
        if names != feature_names:
            raise RuntimeError("feature schema drift")
        panels[name], parents[name] = base.load_panel_data(
            paths[name], args.full23_scores / f"candidate_scores_{name}.npz",
            graphs[name], features[name],
        )
    train = panels["formula_disjoint"]
    folds = np.asarray([formula_fold(v, args.folds, args.seed) for v in train.formulas])
    arms = {
        "count_listwise": (np.asarray([0]), "listwise"),
        "count_risk_aligned": (np.asarray([0]), "risk_listwise"),
        "count_boundary_top1": (np.asarray([0]), "boundary_top1"),
        "support_risk_aligned": (np.arange(len(feature_names)), "risk_listwise"),
        "support_boundary_top1": (np.arange(len(feature_names)), "boundary_top1"),
    }
    oof = {arm: {name: np.full_like(parents[name], np.nan) for name in PANELS} for arm in arms}
    models = []
    for fold in range(args.folds):
        train_queries = np.flatnonzero(folds != fold)
        fold_models = {}
        for arm, (columns, loss_mode) in arms.items():
            fold_models[arm] = fit_arm(
                parents["formula_disjoint"], features["formula_disjoint"], train.query_ptr,
                train.labels, train_queries, columns, loss_mode, args,
                args.seed + 100 * fold + len(fold_models),
            )
        models.append({arm: model["weight"].tolist() for arm, model in fold_models.items()})
        for name in PANELS:
            panel_folds = np.asarray([formula_fold(v, args.folds, args.seed) for v in panels[name].formulas])
            queries = np.flatnonzero(panel_folds == fold)
            for arm, model in fold_models.items():
                predict(model, parents[name], features[name], panels[name].query_ptr,
                        queries, args.residual_bound, oof[arm][name])
        print(f"[reference support] completed formula fold {fold + 1}/{args.folds}", flush=True)
    report_panels = {}
    for index, name in enumerate(PANELS):
        panel = panels[name]
        methods = {"full23_v1": parents[name], **{arm: oof[arm][name] for arm in arms}}
        if any(not np.all(np.isfinite(score)) for score in methods.values()):
            raise RuntimeError(f"incomplete predictions for {name}")
        report_panels[name] = {
            "queries": int(len(panel.formulas)),
            "metrics": {method: _metric_summary(score, panel)[0] for method, score in methods.items()},
            "comparisons_vs_full23": {
                arm: _comparison(scores[name], parents[name], panel, args.bootstrap_resamples,
                                 args.seed + 1000 * index + offset)
                for offset, (arm, scores) in enumerate(oof.items(), 1)
            },
            "support_vs_risk_count": _comparison(
                oof["support_risk_aligned"][name], oof["count_risk_aligned"][name],
                panel, args.bootstrap_resamples, args.seed + 1000 * index + 99,
            ),
        }
    args.output.mkdir(parents=True)
    for name in PANELS:
        np.savez_compressed(
            args.output / f"predictions_{name}.npz", query_ptr=panels[name].query_ptr,
            labels=panels[name].labels, formulas=panels[name].formulas,
            full23_v1=parents[name], **{arm: oof[arm][name] for arm in arms},
        )
    report = {
        "status": "FULL23_REFERENCE_SUPPORT_RISK_RESIDUAL_DEVELOPMENT_COMPLETE",
        "dataset": "GNPS Gold/Silver consumed development",
        "allow_external_claim": False,
        "method": "frozen Full23 plus formula-OOF reference support residual",
        "feature_names": feature_names,
        "loss": (
            "listwise controls plus boundary Top-1 loss: repair reachable parent errors; "
            "charge twice for erosion of reachable parent-correct margins"
        ),
        "training": {key: getattr(args, key) for key in (
            "folds", "epochs", "batch_queries", "learning_rate", "weight_decay",
            "boundary_l2", "residual_bound", "bootstrap_resamples", "seed",
        )},
        "fold_weights": models,
        "panels": report_panels,
        "claim_boundary": "development diagnosis only; no untouched external claim",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
