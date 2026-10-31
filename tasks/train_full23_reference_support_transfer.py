#!/usr/bin/env python
"""Train Full23 reference-max repairs on Enveda and transfer fixed to GNPS.

Two deliberately low-capacity arms are trained from existing cached evidence:

* count_listwise: the strongest raw-ranking development repair;
* support_boundary: parent-risk Top-1 loss with per-expert support stability.

Enveda supplies 25,000 identity and 9,435 formula-panel queries.  Its existing
global-formula folds produce residual OOF predictions, while repeated queries
retain their frozen query weights.  One all-Enveda refit is then transferred
unchanged to the two consumed GNPS development panels.  No upstream encoder,
reranker, V2 policy, candidate graph, or Full23 parent is refitted here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from types import SimpleNamespace
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_full23_reference_coherence_residual as coherence
import train_full23_reference_support_risk_residual as support
from train_enveda_unified_candidate_integrator import PanelData, _comparison, _metric_summary


PANELS = coherence.PANELS


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--enveda-benchmark", type=Path, required=True)
    p.add_argument("--enveda-score-bundle", type=Path, required=True)
    p.add_argument("--enveda-msg-cache", type=Path, required=True)
    p.add_argument("--enveda-embedding", action="append", default=[], metavar="NAME=PATH")
    p.add_argument("--enveda-full23-oof", type=Path, required=True)
    p.add_argument("--gnps-identity-panel", type=Path, required=True)
    p.add_argument("--gnps-formula-panel", type=Path, required=True)
    p.add_argument("--gnps-score-bundle", type=Path, required=True)
    p.add_argument("--gnps-msg-cache", type=Path, required=True)
    p.add_argument("--gnps-embedding", action="append", default=[], metavar="NAME=PATH")
    p.add_argument("--gnps-full23-scores", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-queries", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=0.015)
    p.add_argument("--weight-decay", type=float, default=0.002)
    p.add_argument("--boundary-l2", type=float, default=0.002)
    p.add_argument("--residual-bound", type=float, default=0.25)
    p.add_argument("--bootstrap-resamples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20261031)
    return p.parse_args()


def _raw(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as body:
        return {key: np.asarray(body[key]) for key in body.files}


def _pair_args(score_bundle: Path, msg_cache: Path, embeddings: list[str]) -> SimpleNamespace:
    return SimpleNamespace(score_bundle=score_bundle, msg_cache=msg_cache, embedding=embeddings)


def _panel(raw: dict[str, np.ndarray], graph, scalar: np.ndarray, parent: np.ndarray,
           query_weights: np.ndarray, query_keys: np.ndarray, near: np.ndarray,
           name: str) -> PanelData:
    if parent.shape != raw["molecule_label"].shape:
        raise RuntimeError(f"{name}: parent candidate axis drift")
    return PanelData(
        name=name, graph=graph, scalar=scalar,
        feature_names=(), labels=raw["molecule_label"].astype(np.int8),
        query_ptr=raw["query_ptr"].astype(np.int64),
        formulas=raw["query_formula"].astype("U"),
        query_keys=query_keys.astype("U"), near=near.astype(bool),
        query_weights=query_weights.astype(np.float32), scores={"full23_v1": parent},
    )


def load_enveda(args: argparse.Namespace):
    paths = {name: args.enveda_benchmark / f"panel_{name}.npz" for name in PANELS}
    graphs = {name: coherence.graph_from_panel(path) for name, path in paths.items()}
    pair = coherence.pair_evidence(
        _pair_args(args.enveda_score_bundle, args.enveda_msg_cache, args.enveda_embedding), graphs,
    )
    with np.load(args.enveda_full23_oof, allow_pickle=False) as body:
        pred = {key: np.asarray(body[key]) for key in body.files}
    panels, features, folds = {}, {}, {}
    names = None
    for name in PANELS:
        raw = _raw(paths[name]); prefix = f"{name}__"
        checks = {
            "query_ptr": np.array_equal(raw["query_ptr"], pred[prefix + "query_ptr"]),
            "labels": np.array_equal(raw["molecule_label"], pred[prefix + "labels"]),
            "formula": np.array_equal(raw["query_formula"].astype("U"), pred[prefix + "formula"].astype("U")),
        }
        if not all(checks.values()):
            raise RuntimeError(f"Enveda axis drift for {name}: {checks}")
        features[name], current_names = support.support_features(graphs[name], pair[name])
        names = current_names if names is None else names
        if current_names != names:
            raise RuntimeError("Enveda support feature schema drift")
        panels[name] = _panel(
            raw, graphs[name], features[name], pred[prefix + "additive_linear_control"].astype(np.float32),
            pred[prefix + "query_weight"], pred[prefix + "query_keys"], pred[prefix + "near"], name,
        )
        folds[name] = pred[prefix + "fold"].astype(np.int16)
    return panels, features, folds, names


def load_gnps(args: argparse.Namespace, expected_names: tuple[str, ...]):
    paths = {
        "identity_disjoint": args.gnps_identity_panel,
        "formula_disjoint": args.gnps_formula_panel,
    }
    graphs = {name: coherence.graph_from_panel(path) for name, path in paths.items()}
    pair = coherence.pair_evidence(
        _pair_args(args.gnps_score_bundle, args.gnps_msg_cache, args.gnps_embedding), graphs,
    )
    panels, features = {}, {}
    for name in PANELS:
        raw = _raw(paths[name])
        features[name], names = support.support_features(graphs[name], pair[name])
        if names != expected_names:
            raise RuntimeError("GNPS support feature schema drift")
        with np.load(args.gnps_full23_scores / f"candidate_scores_{name}.npz", allow_pickle=False) as body:
            if not np.array_equal(raw["query_ptr"], body["query_ptr"]):
                raise RuntimeError(f"GNPS query axis drift for {name}")
            parent = np.asarray(body["additive_linear_control"], dtype=np.float32)
        panels[name] = _panel(
            raw, graphs[name], features[name], parent,
            np.ones(len(raw["query_formula"]), np.float32), raw["query_ik14"], raw["near_query"], name,
        )
    return panels, features


def occurrence_rows(panels: dict[str, PanelData], occurrences: list[tuple[str, int]]) -> np.ndarray:
    return np.concatenate([
        np.arange(int(panels[name].query_ptr[q]), int(panels[name].query_ptr[q + 1]), dtype=np.int64)
        for name, q in occurrences
    ])


def fit_model(panels: dict[str, PanelData], features: dict[str, np.ndarray],
              occurrences: list[tuple[str, int]], columns: np.ndarray, loss_mode: str,
              args: argparse.Namespace, seed: int) -> dict:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    rows_by_panel = {
        name: occurrence_rows(panels, [(p, q) for p, q in occurrences if p == name])
        for name in PANELS
    }
    train_values = np.concatenate([features[name][rows_by_panel[name]][:, columns] for name in PANELS])
    mean = train_values.mean(0).astype(np.float32)
    scale = train_values.std(0).astype(np.float32); scale[scale < 1e-6] = 1.0
    x = {name: torch.as_tensor((features[name][:, columns] - mean) / scale, dtype=torch.float32) for name in PANELS}
    parent = {name: torch.as_tensor(panels[name].scores["full23_v1"], dtype=torch.float32) for name in PANELS}
    weight = torch.nn.Parameter(torch.zeros(len(columns), dtype=torch.float32))
    optimizer = torch.optim.Adam([weight], lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(seed)
    for _ in range(args.epochs):
        order = rng.permutation(len(occurrences))
        for start in range(0, len(order), args.batch_queries):
            batch = [occurrences[i] for i in order[start:start + args.batch_queries]]
            losses, masses = [], []
            for name in PANELS:
                queries = np.asarray([q for p, q in batch if p == name], dtype=np.int64)
                if not len(queries):
                    continue
                index, valid, target = support.query_batch(
                    panels[name].query_ptr, panels[name].labels, queries,
                )
                take = torch.as_tensor(index, dtype=torch.long)
                valid_t = torch.as_tensor(valid)
                target_t = torch.as_tensor(target, dtype=torch.long)
                parent_logits = parent[name][take].masked_fill(~valid_t, -1e9)
                logits = parent[name][take] + args.residual_bound * torch.tanh(x[name][take] @ weight)
                logits = logits.masked_fill(~valid_t, -1e9)
                query_weight = torch.as_tensor(panels[name].query_weights[queries], dtype=torch.float32)
                if loss_mode == "listwise":
                    local = torch.nn.functional.cross_entropy(logits, target_t, reduction="none")
                    losses.append(torch.sum(local * query_weight)); masses.append(torch.sum(query_weight))
                else:
                    row = torch.arange(len(target_t))
                    true_parent = parent_logits[row, target_t]
                    true_new = logits[row, target_t]
                    wrong_parent = parent_logits.clone(); wrong_parent[row, target_t] = -1e9
                    wrong_new = logits.clone(); wrong_new[row, target_t] = -1e9
                    parent_margin = true_parent - torch.max(wrong_parent, dim=1).values
                    new_margin = true_new - torch.max(wrong_new, dim=1).values
                    correct = parent_margin >= 0
                    reachable = torch.abs(parent_margin) <= 2.0 * args.residual_bound
                    repair = torch.relu(0.01 - new_margin)
                    preserve = 2.0 * torch.relu(parent_margin - new_margin)
                    local = torch.where(correct, preserve, repair)
                    mass = query_weight * reachable.float()
                    losses.append(torch.sum(local * mass)); masses.append(torch.sum(mass))
            denominator = torch.sum(torch.stack(masses))
            if float(denominator) <= 0:
                continue
            loss = torch.sum(torch.stack(losses)) / denominator
            if loss_mode == "boundary_top1":
                loss = loss + args.boundary_l2 * torch.sum(weight.square())
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return {"columns": columns, "mean": mean, "scale": scale, "weight": weight.detach().numpy(), "loss_mode": loss_mode}


def predict(model: dict, panel: PanelData, features: np.ndarray, bound: float) -> np.ndarray:
    x = (features[:, model["columns"]] - model["mean"]) / model["scale"]
    residual = bound * np.tanh(x @ model["weight"])
    return panel.scores["full23_v1"] + residual.astype(np.float32)


def evaluate(panels: dict[str, PanelData], scores: dict[str, dict[str, np.ndarray]],
             args: argparse.Namespace, seed_offset: int) -> dict:
    result = {}
    for index, name in enumerate(PANELS):
        parent = panels[name].scores["full23_v1"]
        result[name] = {
            "queries": int(len(panels[name].formulas)),
            "effective_query_weight": float(np.sum(panels[name].query_weights)),
            "metrics": {
                "full23_v1": _metric_summary(parent, panels[name])[0],
                **{arm: _metric_summary(values[name], panels[name])[0] for arm, values in scores.items()},
            },
            "comparisons_vs_full23": {
                arm: _comparison(values[name], parent, panels[name], args.bootstrap_resamples,
                                 args.seed + seed_offset + 100 * index + offset)
                for offset, (arm, values) in enumerate(scores.items(), 1)
            },
        }
    return result


def serializable(model: dict) -> dict:
    return {
        "columns": model["columns"].tolist(), "mean": model["mean"].tolist(),
        "scale": model["scale"].tolist(), "weight": model["weight"].tolist(),
        "loss_mode": model["loss_mode"],
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    enveda, enveda_features, folds, feature_names = load_enveda(args)
    gnps, gnps_features = load_gnps(args, feature_names)
    arms = {
        "count_listwise": (np.asarray([0], dtype=np.int64), "listwise"),
        "support_boundary_top1": (np.arange(len(feature_names), dtype=np.int64), "boundary_top1"),
    }
    fold_ids = sorted(set(np.concatenate([folds[name] for name in PANELS]).tolist()))
    oof = {arm: {name: np.full_like(enveda[name].scores["full23_v1"], np.nan) for name in PANELS} for arm in arms}
    fold_models = []
    for held in fold_ids:
        occurrences = [
            (name, q) for name in PANELS for q in range(len(enveda[name].formulas))
            if int(folds[name][q]) != int(held)
        ]
        models = {
            arm: fit_model(enveda, enveda_features, occurrences, columns, mode, args,
                           args.seed + 100 * int(held) + index)
            for index, (arm, (columns, mode)) in enumerate(arms.items())
        }
        fold_models.append({"fold": int(held), "models": {arm: serializable(model) for arm, model in models.items()}})
        for name in PANELS:
            candidates = {arm: predict(model, enveda[name], enveda_features[name], args.residual_bound) for arm, model in models.items()}
            for query in np.flatnonzero(folds[name] == held):
                left, right = map(int, enveda[name].query_ptr[query:query + 2])
                for arm in arms:
                    oof[arm][name][left:right] = candidates[arm][left:right]
        print(f"[reference support transfer] completed Enveda formula fold {int(held) + 1}/{len(fold_ids)}", flush=True)
    if any(np.any(~np.isfinite(values)) for arm in oof.values() for values in arm.values()):
        raise RuntimeError("Enveda residual OOF prediction incomplete")
    all_occurrences = [(name, q) for name in PANELS for q in range(len(enveda[name].formulas))]
    final_models = {
        arm: fit_model(enveda, enveda_features, all_occurrences, columns, mode, args,
                       args.seed + 10000 + index)
        for index, (arm, (columns, mode)) in enumerate(arms.items())
    }
    fixed = {
        arm: {name: predict(model, gnps[name], gnps_features[name], args.residual_bound) for name in PANELS}
        for arm, model in final_models.items()
    }
    args.output.mkdir(parents=True)
    torch.save({
        "schema": "full23_reference_support_transfer_v1",
        "feature_names": feature_names, "residual_bound": args.residual_bound,
        "models": {arm: serializable(model) for arm, model in final_models.items()},
    }, args.output / "full23_reference_support_transfer.pt")
    for scope, panels, scores in (("enveda_oof", enveda, oof), ("gnps_fixed", gnps, fixed)):
        for name in PANELS:
            np.savez_compressed(
                args.output / f"{scope}_{name}.npz", query_ptr=panels[name].query_ptr,
                labels=panels[name].labels, formulas=panels[name].formulas,
                full23_v1=panels[name].scores["full23_v1"],
                **{arm: values[name] for arm, values in scores.items()},
            )
    report = {
        "status": "FULL23_REFERENCE_SUPPORT_FIXED_TRANSFER_COMPLETE",
        "dataset_roles": {
            "Enveda": "consumed development; residual OOF plus all-data fit",
            "GNPS": "consumed development; fixed transfer evaluation",
        },
        "allow_external_claim": False,
        "training": {key: getattr(args, key) for key in (
            "epochs", "batch_queries", "learning_rate", "weight_decay", "boundary_l2",
            "residual_bound", "bootstrap_resamples", "seed",
        )},
        "feature_names": feature_names,
        "fold_models": fold_models,
        "final_models": {arm: serializable(model) for arm, model in final_models.items()},
        "enveda_oof": evaluate(enveda, oof, args, 1000),
        "gnps_fixed": evaluate(gnps, fixed, args, 2000),
        "claim_boundary": (
            "both resources are consumed development; GNPS is a fixed transfer from the all-Enveda residual fit; "
            "Enveda residual OOF conditions on existing parent OOF predictions and is not a fresh external claim"
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
