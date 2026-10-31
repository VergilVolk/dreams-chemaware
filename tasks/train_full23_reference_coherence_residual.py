#!/usr/bin/env python
"""Test whether reference-spectrum coherence repairs frozen Full23 ranking.

Full23 v1 independently maximises each expert over all reference spectra of a
candidate molecule.  This can create a synthetic evidence vector whose entries
are supported by different spectra.  This bounded, two-parameter residual asks
whether label-blind agreement on the best supporting reference adds ranking
information beyond the number of available reference spectra.

The experiment is consumed GNPS development only.  Models are trained on the
formula panel with strict formula-hash OOF and applied to both panels with the
fold model that did not train on the query formula.
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
from evaluate_gnps_gold_silver_10ppm_embeddings import graph_from_panel, load_embeddings
from noise_corrected_fullgraph_evaluation import score_embeddings
from train_enveda_unified_candidate_integrator import (
    PanelData, _comparison, _metric_summary,
)


PANELS = ("identity_disjoint", "formula_disjoint")
EXPERTS = (
    "noise_v1", "weighted_spectral_entropy", "p2b_noise_v1_frozen",
    "noise_msg_fusion_stack_v1", "chemaware_stage1", "chemaware_phasea",
    "noise_chemaware_fullmsg",
)


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
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-queries", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=0.001)
    p.add_argument("--residual-bound", type=float, default=0.25)
    p.add_argument("--bootstrap-resamples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20261031)
    return p.parse_args()


def formula_fold(value: str, folds: int, seed: int) -> int:
    token = f"{seed}|{str(value).strip()}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % folds


def query_rows(ptr: np.ndarray, queries: np.ndarray) -> np.ndarray:
    return np.concatenate([
        np.arange(int(ptr[q]), int(ptr[q + 1]), dtype=np.int64) for q in queries
    ])


def center_by_query(values: np.ndarray, ptr: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32).copy()
    for left, right in zip(ptr[:-1], ptr[1:]):
        left, right = int(left), int(right)
        out[left:right] -= float(np.mean(out[left:right]))
    return out


def pair_evidence(args: argparse.Namespace, graphs: dict) -> dict[str, dict[str, np.ndarray]]:
    with np.load(args.score_bundle, allow_pickle=False) as body:
        methods = tuple(map(str, body["method_names"]))
        score_matrix = {
            name: np.asarray(body[f"scores_{name}"], dtype=np.float32) for name in PANELS
        }
    pair = {name: {} for name in PANELS}
    for panel in PANELS:
        for expert in EXPERTS[:3]:
            pair[panel][expert] = score_matrix[panel][methods.index(expert)]
        pair[panel]["noise_msg_fusion_stack_v1"] = np.load(
            args.msg_cache / f"pair_scores_{panel}.npy"
        ).astype(np.float32)
    manifest_rows = max(
        int(np.max(graph.query_row)) for graph in graphs.values()
    )
    manifest_rows = max(
        manifest_rows,
        max(int(np.max(graph.pair_candidate_row)) for graph in graphs.values()),
    ) + 1
    paths = {}
    for item in args.embedding:
        name, path = item.split("=", 1)
        paths[name] = Path(path)
    for expert in EXPERTS[4:]:
        if expert not in paths:
            raise RuntimeError(f"missing embedding path for {expert}")
        rows, embeddings = load_embeddings(paths[expert], manifest_rows)
        for panel in PANELS:
            pair[panel][expert] = score_embeddings(graphs[panel], rows, embeddings).pair
    for panel in PANELS:
        count = len(graphs[panel].pair_candidate_row)
        if any(values.shape != (count,) for values in pair[panel].values()):
            raise RuntimeError(f"pair evidence axis drifted for {panel}")
    return pair


def coherence_features(graph, pair: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    n = len(graph.molecule_label)
    agreement = np.ones(n, dtype=np.float32)
    log_count = np.empty(n, dtype=np.float32)
    distinct = np.ones(n, dtype=np.int16)
    for molecule, (left, right) in enumerate(zip(graph.molecule_ptr[:-1], graph.molecule_ptr[1:])):
        left, right = int(left), int(right)
        winners = [int(np.argmax(pair[name][left:right])) for name in EXPERTS]
        same = sum(
            winners[i] == winners[j]
            for i in range(len(winners)) for j in range(i + 1, len(winners))
        )
        agreement[molecule] = same / (len(winners) * (len(winners) - 1) / 2)
        distinct[molecule] = len(set(winners))
        log_count[molecule] = np.log1p(right - left)
    features = np.stack([
        center_by_query(agreement, graph.query_ptr),
        center_by_query(log_count, graph.query_ptr),
    ], axis=1)
    multi = np.diff(graph.molecule_ptr) > 1
    positive = graph.molecule_label.astype(bool)
    diagnostic = {
        "molecules": int(n),
        "multi_reference_fraction": float(np.mean(multi)),
        "all_experts_same_reference_multi": float(np.mean(distinct[multi] == 1)),
        "all_experts_same_reference_multi_positive": float(np.mean(distinct[multi & positive] == 1)),
        "all_experts_same_reference_multi_negative": float(np.mean(distinct[multi & ~positive] == 1)),
        "mean_distinct_winners_multi_positive": float(np.mean(distinct[multi & positive])),
        "mean_distinct_winners_multi_negative": float(np.mean(distinct[multi & ~positive])),
    }
    return features, diagnostic


def load_panel_data(path: Path, scores_path: Path, graph, features: np.ndarray) -> tuple[PanelData, np.ndarray]:
    with np.load(path, allow_pickle=False) as body:
        raw = {key: np.asarray(body[key]) for key in body.files}
    with np.load(scores_path, allow_pickle=False) as body:
        parent = np.asarray(body["additive_linear_control"], dtype=np.float32)
        if not np.array_equal(body["query_ptr"], raw["query_ptr"]):
            raise RuntimeError("Full23 query axis drifted")
        if not np.array_equal(body["molecule_label"], raw["molecule_label"]):
            raise RuntimeError("Full23 label axis drifted")
    panel = PanelData(
        name=path.stem,
        graph=graph,
        scalar=features,
        feature_names=("reference_agreement", "log_reference_count"),
        labels=raw["molecule_label"].astype(np.int8),
        query_ptr=raw["query_ptr"].astype(np.int64),
        formulas=raw["query_formula"].astype("U"),
        query_keys=raw["query_ik14"].astype("U"),
        near=raw["near_query"].astype(bool),
        query_weights=np.ones(len(raw["query_formula"]), dtype=np.float32),
        scores={"full23_v1": parent},
    )
    return panel, parent


def fit_arm(
    parent: np.ndarray, features: np.ndarray, ptr: np.ndarray, labels: np.ndarray,
    train_queries: np.ndarray, use_coherence: bool, args: argparse.Namespace, seed: int,
) -> dict:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    columns = np.asarray([0, 1] if use_coherence else [1], dtype=np.int64)
    rows = query_rows(ptr, train_queries)
    mean = features[rows][:, columns].mean(0).astype(np.float32)
    scale = features[rows][:, columns].std(0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    x = torch.as_tensor((features[:, columns] - mean) / scale, dtype=torch.float32)
    base = torch.as_tensor(parent, dtype=torch.float32)
    weight = torch.nn.Parameter(torch.zeros(len(columns), dtype=torch.float32))
    optimizer = torch.optim.Adam([weight], lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(seed)
    for _ in range(args.epochs):
        order = rng.permutation(train_queries)
        for start in range(0, len(order), args.batch_queries):
            batch = order[start:start + args.batch_queries]
            widths = ptr[batch + 1] - ptr[batch]
            width = int(np.max(widths))
            index = np.zeros((len(batch), width), dtype=np.int64)
            valid = np.zeros((len(batch), width), dtype=bool)
            target = np.empty(len(batch), dtype=np.int64)
            for row, query in enumerate(batch):
                left, right = int(ptr[query]), int(ptr[query + 1])
                local_width = right - left
                index[row, :local_width] = np.arange(left, right)
                valid[row, :local_width] = True
                target[row] = int(np.flatnonzero(labels[left:right] > 0)[0])
            take = torch.as_tensor(index, dtype=torch.long)
            logits = base[take] + args.residual_bound * torch.tanh(x[take] @ weight)
            logits = logits.masked_fill(~torch.as_tensor(valid), -1e9)
            loss = torch.nn.functional.cross_entropy(
                logits, torch.as_tensor(target, dtype=torch.long),
            )
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    return {"columns": columns, "mean": mean, "scale": scale, "weight": weight.detach().numpy()}


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
    panel_paths = {"identity_disjoint": args.identity_panel, "formula_disjoint": args.formula_panel}
    graphs = {name: graph_from_panel(path) for name, path in panel_paths.items()}
    pair = pair_evidence(args, graphs)
    panels, parents, diagnostics = {}, {}, {}
    for name in PANELS:
        features, diagnostics[name] = coherence_features(graphs[name], pair[name])
        panels[name], parents[name] = load_panel_data(
            panel_paths[name], args.full23_scores / f"candidate_scores_{name}.npz",
            graphs[name], features,
        )
    train_panel = panels["formula_disjoint"]
    train_folds = np.asarray([
        formula_fold(value, args.folds, args.seed) for value in train_panel.formulas
    ])
    predictions = {
        arm: {name: np.full_like(parents[name], np.nan) for name in PANELS}
        for arm in ("reference_count_control", "reference_coherence_residual")
    }
    fold_models = []
    for fold in range(args.folds):
        train_queries = np.flatnonzero(train_folds != fold)
        models = {
            "reference_count_control": fit_arm(
                parents["formula_disjoint"], train_panel.scalar, train_panel.query_ptr,
                train_panel.labels, train_queries, False, args, args.seed + fold,
            ),
            "reference_coherence_residual": fit_arm(
                parents["formula_disjoint"], train_panel.scalar, train_panel.query_ptr,
                train_panel.labels, train_queries, True, args, args.seed + fold,
            ),
        }
        fold_models.append({arm: model["weight"].tolist() for arm, model in models.items()})
        for name in PANELS:
            panel = panels[name]
            folds = np.asarray([formula_fold(v, args.folds, args.seed) for v in panel.formulas])
            queries = np.flatnonzero(folds == fold)
            for arm, model in models.items():
                predict(model, parents[name], panel.scalar, panel.query_ptr, queries,
                        args.residual_bound, predictions[arm][name])
    reports = {}
    for index, name in enumerate(PANELS):
        panel = panels[name]
        if any(not np.all(np.isfinite(predictions[arm][name])) for arm in predictions):
            raise RuntimeError(f"OOF prediction incomplete for {name}")
        methods = {"full23_v1": parents[name], **{arm: predictions[arm][name] for arm in predictions}}
        reports[name] = {
            "queries": int(len(panel.query_ptr) - 1),
            "molecules": int(len(panel.labels)),
            "reference_diagnostic": diagnostics[name],
            "metrics": {method: _metric_summary(score, panel)[0] for method, score in methods.items()},
            "comparisons": {
                "count_control_vs_full23": _comparison(methods["reference_count_control"], parents[name], panel, args.bootstrap_resamples, args.seed + index * 100 + 1),
                "coherence_vs_full23": _comparison(methods["reference_coherence_residual"], parents[name], panel, args.bootstrap_resamples, args.seed + index * 100 + 2),
                "coherence_vs_count_control": _comparison(methods["reference_coherence_residual"], methods["reference_count_control"], panel, args.bootstrap_resamples, args.seed + index * 100 + 3),
            },
        }
    args.output.mkdir(parents=True)
    for name in PANELS:
        np.savez_compressed(
            args.output / f"predictions_{name}.npz", query_ptr=panels[name].query_ptr,
            labels=panels[name].labels, formulas=panels[name].formulas,
            full23_v1=parents[name], **{arm: predictions[arm][name] for arm in predictions},
        )
    report = {
        "status": "FULL23_REFERENCE_COHERENCE_RESIDUAL_DEVELOPMENT_COMPLETE",
        "dataset": "GNPS Gold/Silver consumed development",
        "allow_external_claim": False,
        "method": "frozen Full23 + bounded reference-support coherence residual",
        "experts": list(EXPERTS),
        "training": {k: getattr(args, k) for k in (
            "folds", "epochs", "batch_queries", "learning_rate", "weight_decay",
            "residual_bound", "seed",
        )},
        "fold_weights": fold_models,
        "panels": reports,
        "claim_boundary": "development mechanism test; no Enveda or untouched external confirmation",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
