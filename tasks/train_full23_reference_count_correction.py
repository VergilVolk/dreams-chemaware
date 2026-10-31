#!/usr/bin/env python
"""Correct max-over-reference opportunity bias in frozen Full23 v1.

Each candidate molecule can have a different number of reference spectra.
Because every Full23 expert is reduced by a maximum over those spectra,
high-multiplicity candidates receive more chances to obtain an extreme score.
This program learns one bounded, query-centred log-reference-count correction
from Enveda global-formula OOF predictions, then applies the all-Enveda refit
once to fixed GNPS development predictions.  No encoder or reranker is refit.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_enveda_unified_candidate_integrator import PanelData, _comparison, _metric_summary


PANELS = ("identity_disjoint", "formula_disjoint")


@dataclass
class Data:
    name: str
    query_ptr: np.ndarray
    molecule_ptr: np.ndarray
    labels: np.ndarray
    formulas: np.ndarray
    query_keys: np.ndarray
    near: np.ndarray
    query_weights: np.ndarray
    folds: np.ndarray | None
    parent: np.ndarray
    feature: np.ndarray


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--enveda-benchmark", type=Path, required=True)
    p.add_argument("--enveda-predictions", type=Path, required=True)
    p.add_argument("--gnps-identity-panel", type=Path, required=True)
    p.add_argument("--gnps-formula-panel", type=Path, required=True)
    p.add_argument("--gnps-scores", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-queries", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=0.001)
    p.add_argument("--residual-bound", type=float, default=0.25)
    p.add_argument("--bootstrap-resamples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20261031)
    return p.parse_args()


def centred_log_count(query_ptr: np.ndarray, molecule_ptr: np.ndarray) -> np.ndarray:
    count = np.diff(molecule_ptr).astype(np.float32)
    value = np.log1p(count)
    for left, right in zip(query_ptr[:-1], query_ptr[1:]):
        left, right = int(left), int(right)
        value[left:right] -= float(np.mean(value[left:right]))
    return value


def raw_panel(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as body:
        return {key: np.asarray(body[key]) for key in body.files}


def load_enveda(args: argparse.Namespace) -> dict[str, Data]:
    with np.load(args.enveda_predictions, allow_pickle=False) as body:
        pred = {key: np.asarray(body[key]) for key in body.files}
    output = {}
    for name in PANELS:
        raw = raw_panel(args.enveda_benchmark / f"panel_{name}.npz")
        prefix = f"{name}__"
        checks = {
            "query_ptr": np.array_equal(raw["query_ptr"], pred[prefix + "query_ptr"]),
            "labels": np.array_equal(raw["molecule_label"], pred[prefix + "labels"]),
            "formula": np.array_equal(raw["query_formula"].astype("U"), pred[prefix + "formula"].astype("U")),
        }
        if not all(checks.values()):
            raise RuntimeError(f"Enveda axis drift for {name}: {checks}")
        output[name] = Data(
            name=name,
            query_ptr=raw["query_ptr"].astype(np.int64),
            molecule_ptr=raw["molecule_ptr"].astype(np.int64),
            labels=raw["molecule_label"].astype(np.int8),
            formulas=raw["query_formula"].astype("U"),
            query_keys=pred[prefix + "query_keys"].astype("U"),
            near=pred[prefix + "near"].astype(bool),
            query_weights=pred[prefix + "query_weight"].astype(np.float32),
            folds=pred[prefix + "fold"].astype(np.int16),
            parent=pred[prefix + "additive_linear_control"].astype(np.float32),
            feature=centred_log_count(raw["query_ptr"], raw["molecule_ptr"]),
        )
    return output


def load_gnps(args: argparse.Namespace) -> dict[str, Data]:
    paths = {
        "identity_disjoint": args.gnps_identity_panel,
        "formula_disjoint": args.gnps_formula_panel,
    }
    output = {}
    for name, path in paths.items():
        raw = raw_panel(path)
        with np.load(args.gnps_scores / f"candidate_scores_{name}.npz", allow_pickle=False) as body:
            if not np.array_equal(raw["query_ptr"], body["query_ptr"]):
                raise RuntimeError(f"GNPS query axis drift for {name}")
            parent = np.asarray(body["additive_linear_control"], dtype=np.float32)
        output[name] = Data(
            name=name,
            query_ptr=raw["query_ptr"].astype(np.int64),
            molecule_ptr=raw["molecule_ptr"].astype(np.int64),
            labels=raw["molecule_label"].astype(np.int8),
            formulas=raw["query_formula"].astype("U"),
            query_keys=raw["query_ik14"].astype("U"),
            near=raw["near_query"].astype(bool),
            query_weights=np.ones(len(raw["query_formula"]), dtype=np.float32),
            folds=None,
            parent=parent,
            feature=centred_log_count(raw["query_ptr"], raw["molecule_ptr"]),
        )
    return output


def query_matrix(data: Data, queries: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    widths = data.query_ptr[queries + 1] - data.query_ptr[queries]
    width = int(np.max(widths))
    index = np.zeros((len(queries), width), dtype=np.int64)
    valid = np.zeros((len(queries), width), dtype=bool)
    target = np.empty(len(queries), dtype=np.int64)
    for row, query in enumerate(queries):
        left, right = int(data.query_ptr[query]), int(data.query_ptr[query + 1])
        local_width = right - left
        index[row, :local_width] = np.arange(left, right)
        valid[row, :local_width] = True
        positive = np.flatnonzero(data.labels[left:right] > 0)
        if len(positive) != 1:
            raise RuntimeError("query does not have exactly one positive")
        target[row] = int(positive[0])
    return (
        torch.as_tensor(index, dtype=torch.long),
        torch.as_tensor(valid),
        torch.as_tensor(target, dtype=torch.long),
        torch.as_tensor(data.query_weights[queries], dtype=torch.float32),
    )


def train_model(data: dict[str, Data], held_fold: int | None, args: argparse.Namespace, seed: int) -> dict:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    selected = {
        name: np.flatnonzero(np.ones(len(panel.formulas), dtype=bool) if held_fold is None else panel.folds != held_fold)
        for name, panel in data.items()
    }
    train_values = np.concatenate([
        panel.feature[np.concatenate([
            np.arange(panel.query_ptr[q], panel.query_ptr[q + 1], dtype=np.int64)
            for q in selected[name]
        ])]
        for name, panel in data.items()
    ])
    scale = float(np.std(train_values))
    if not np.isfinite(scale) or scale < 1e-6:
        raise RuntimeError("reference-count feature has no training variation")
    weight = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    optimizer = torch.optim.Adam([weight], lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(seed)
    occurrences = [(name, int(query)) for name in PANELS for query in selected[name]]
    for _ in range(args.epochs):
        order = rng.permutation(len(occurrences))
        for start in range(0, len(order), args.batch_queries):
            batch = [occurrences[i] for i in order[start:start + args.batch_queries]]
            losses, weights = [], []
            for name in PANELS:
                queries = np.asarray([q for panel, q in batch if panel == name], dtype=np.int64)
                if not len(queries):
                    continue
                panel = data[name]
                index, valid, target, query_weight = query_matrix(panel, queries)
                base = torch.as_tensor(panel.parent, dtype=torch.float32)[index]
                feature = torch.as_tensor(panel.feature / scale, dtype=torch.float32)[index]
                logits = base + args.residual_bound * torch.tanh(feature * weight)
                logits = logits.masked_fill(~valid, -1e9)
                local = torch.nn.functional.cross_entropy(logits, target, reduction="none")
                losses.append(local); weights.append(query_weight)
            loss_vector = torch.cat(losses); weight_vector = torch.cat(weights)
            loss = torch.sum(loss_vector * weight_vector) / torch.sum(weight_vector)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return {"scale": scale, "weight": float(weight.detach())}


def predict(model: dict, panel: Data, bound: float) -> np.ndarray:
    residual = bound * np.tanh(panel.feature / model["scale"] * model["weight"])
    return panel.parent + residual.astype(np.float32)


def panel_for_metrics(data: Data) -> PanelData:
    return PanelData(
        name=data.name, graph=None, scalar=np.empty((len(data.labels), 0), np.float32),
        feature_names=(), labels=data.labels, query_ptr=data.query_ptr,
        formulas=data.formulas, query_keys=data.query_keys, near=data.near,
        query_weights=data.query_weights, scores={"full23_v1": data.parent},
    )


def evaluate(data: dict[str, Data], corrected: dict[str, np.ndarray], args: argparse.Namespace, seed_offset: int) -> dict:
    report = {}
    for index, name in enumerate(PANELS):
        panel = panel_for_metrics(data[name])
        report[name] = {
            "queries": int(len(data[name].formulas)),
            "molecules": int(len(data[name].labels)),
            "reference_count": {
                "single_reference_fraction": float(np.mean(np.diff(data[name].molecule_ptr) == 1)),
                "median": float(np.median(np.diff(data[name].molecule_ptr))),
                "maximum": int(np.max(np.diff(data[name].molecule_ptr))),
            },
            "metrics": {
                "full23_v1": _metric_summary(data[name].parent, panel)[0],
                "full23_reference_count_corrected": _metric_summary(corrected[name], panel)[0],
            },
            "comparison": _comparison(
                corrected[name], data[name].parent, panel,
                args.bootstrap_resamples, args.seed + seed_offset + index,
            ),
        }
    return report


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    enveda = load_enveda(args)
    gnps = load_gnps(args)
    folds = sorted(set(np.concatenate([panel.folds for panel in enveda.values()])))
    enveda_oof = {name: np.full_like(panel.parent, np.nan) for name, panel in enveda.items()}
    fold_models = []
    for fold in folds:
        model = train_model(enveda, int(fold), args, args.seed + int(fold))
        fold_models.append({"fold": int(fold), **model})
        for name, panel in enveda.items():
            queries = np.flatnonzero(panel.folds == fold)
            candidate = predict(model, panel, args.residual_bound)
            for query in queries:
                left, right = panel.query_ptr[query:query + 2]
                enveda_oof[name][left:right] = candidate[left:right]
    if any(not np.all(np.isfinite(value)) for value in enveda_oof.values()):
        raise RuntimeError("Enveda OOF prediction is incomplete")
    final_model = train_model(enveda, None, args, args.seed + 100)
    gnps_fixed = {name: predict(final_model, panel, args.residual_bound) for name, panel in gnps.items()}
    args.output.mkdir(parents=True)
    torch.save({
        "schema": "full23_reference_count_correction_v1",
        "model": final_model,
        "residual_bound": args.residual_bound,
        "feature": "query_centered_log1p_reference_spectrum_count",
    }, args.output / "full23_reference_count_correction.pt")
    for scope, data, scores in (("enveda_oof", enveda, enveda_oof), ("gnps_fixed", gnps, gnps_fixed)):
        for name in PANELS:
            np.savez_compressed(
                args.output / f"{scope}_{name}.npz", query_ptr=data[name].query_ptr,
                labels=data[name].labels, formulas=data[name].formulas,
                full23_v1=data[name].parent, full23_reference_count_corrected=scores[name],
            )
    report = {
        "status": "FULL23_REFERENCE_COUNT_CORRECTION_COMPLETE",
        "method": "frozen Full23 + bounded query-centred log reference-count correction",
        "dataset_roles": {"Enveda": "consumed development OOF fit", "GNPS": "consumed fixed transfer"},
        "allow_external_claim": False,
        "training": {key: getattr(args, key) for key in (
            "epochs", "batch_queries", "learning_rate", "weight_decay",
            "residual_bound", "seed",
        )},
        "fold_models": fold_models,
        "final_model": final_model,
        "enveda_oof": evaluate(enveda, enveda_oof, args, 1000),
        "gnps_fixed": evaluate(gnps, gnps_fixed, args, 2000),
        "claim_boundary": "single prespecified correction; no encoder/reranker refit; both resources consumed",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
