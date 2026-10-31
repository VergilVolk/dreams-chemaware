#!/usr/bin/env python
"""Jointly repair several observable Full23 v1 candidate-level defects.

Consumed GNPS development only.  Identity and formula panels are trained
together with inverse query-occurrence weights and strict formula-hash OOF.
The frozen Full23 score remains the parent.  The residual jointly sees:

* reference multiplicity and within-expert max-support stability;
* query-relative evidence from the seven deployed experts;
* candidate-specific expert top-1/top-3 voting and disagreement;
* the candidate's position relative to the frozen parent top score.

The hybrid objective combines query-balanced listwise CE with a direct Top-1
repair/preservation loss.  This is a development experiment, not external
evidence or a deployable release.
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
import train_full23_reference_coherence_residual as coherence
import train_full23_reference_support_risk_residual as support
from train_enveda_unified_candidate_integrator import PanelData, _comparison, _metric_summary


PANELS = coherence.PANELS
EXPERTS = coherence.EXPERTS


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
    p.add_argument("--learning-rate", type=float, default=0.003)
    p.add_argument("--weight-decay", type=float, default=0.002)
    p.add_argument("--boundary-weight", type=float, default=1.0)
    p.add_argument("--residual-bound", type=float, default=0.25)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--bootstrap-resamples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20261031)
    return p.parse_args()


def formula_fold(value: str, folds: int, seed: int) -> int:
    token = f"{seed}|{str(value).strip()}".encode()
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % folds


def ranks_desc(values: np.ndarray) -> np.ndarray:
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(len(values), dtype=np.float32)
    return ranks / max(1, len(values) - 1)


def build_joint_features(panel: PanelData, support_features: np.ndarray,
                         expert_scores: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build candidate-specific, query-relative, label-blind features."""
    n_experts, n_candidates = expert_scores.shape
    if n_experts != len(EXPERTS) or n_candidates != len(panel.labels):
        raise RuntimeError("expert score axis drift")
    z = np.zeros((n_candidates, n_experts), dtype=np.float32)
    rank = np.zeros_like(z)
    top1_vote = np.zeros(n_candidates, dtype=np.float32)
    top3_vote = np.zeros(n_candidates, dtype=np.float32)
    parent_relative = np.zeros(n_candidates, dtype=np.float32)
    parent = panel.scores["full23_v1"]
    for left, right in zip(panel.query_ptr[:-1], panel.query_ptr[1:]):
        left, right = int(left), int(right)
        block = expert_scores[:, left:right]
        mean = block.mean(1, keepdims=True)
        scale = block.std(1, keepdims=True); scale[scale < 1e-6] = 1.0
        z[left:right] = ((block - mean) / scale).T
        for expert in range(n_experts):
            rank[left:right, expert] = ranks_desc(block[expert])
            order = np.argsort(-block[expert], kind="stable")
            top1_vote[left + order[0]] += 1.0
            top3_vote[left + order[:min(3, right - left)]] += 1.0
        top1_vote[left:right] /= n_experts
        top3_vote[left:right] /= n_experts
        parent_relative[left:right] = parent[left:right] - float(np.max(parent[left:right]))
    z_mean = z.mean(1)
    z_std = z.std(1)
    matrix = np.concatenate([
        support_features,
        z,
        rank,
        z_mean[:, None], z_std[:, None],
        top1_vote[:, None], top3_vote[:, None], parent_relative[:, None],
    ], axis=1).astype(np.float32)
    names = (
        tuple(["log_reference_count"] + [
            item for expert in EXPERTS
            for item in (f"{expert}__max_minus_second", f"{expert}__max_minus_mean")
        ])
        + tuple(f"{expert}__query_z" for expert in EXPERTS)
        + tuple(f"{expert}__rank_fraction" for expert in EXPERTS)
        + ("expert_z_mean", "expert_z_std", "top1_vote_fraction", "top3_vote_fraction", "parent_relative_score")
    )
    if matrix.shape[1] != len(names):
        raise RuntimeError("joint feature schema mismatch")
    return matrix, names


class Residual(torch.nn.Module):
    def __init__(self, width: int, hidden: int, nonlinear: bool):
        super().__init__()
        if nonlinear:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(width, hidden), torch.nn.GELU(),
                torch.nn.Linear(hidden, 1, bias=False),
            )
            torch.nn.init.zeros_(self.net[-1].weight)
        else:
            self.net = torch.nn.Linear(width, 1, bias=False)
            torch.nn.init.zeros_(self.net.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def raw_panel(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as body:
        return {key: np.asarray(body[key]) for key in body.files}


def load_data(args: argparse.Namespace):
    paths = {"identity_disjoint": args.identity_panel, "formula_disjoint": args.formula_panel}
    graphs = {name: coherence.graph_from_panel(path) for name, path in paths.items()}
    pair = coherence.pair_evidence(args, graphs)
    panels, features, feature_names = {}, {}, None
    for name in PANELS:
        raw = raw_panel(paths[name])
        with np.load(args.full23_scores / f"candidate_scores_{name}.npz", allow_pickle=False) as body:
            parent = np.asarray(body["additive_linear_control"], dtype=np.float32)
            expert_scores = np.stack([np.asarray(body[expert], dtype=np.float32) for expert in EXPERTS])
            if not np.array_equal(raw["query_ptr"], body["query_ptr"]):
                raise RuntimeError(f"query axis drift for {name}")
        support_features, _ = support.support_features(graphs[name], pair[name])
        panel = PanelData(
            name=name, graph=graphs[name], scalar=np.empty((len(parent), 0), np.float32),
            feature_names=(), labels=raw["molecule_label"].astype(np.int8),
            query_ptr=raw["query_ptr"].astype(np.int64), formulas=raw["query_formula"].astype("U"),
            # query_row identifies the actual experimental spectrum.  Using
            # IK14 here would collapse distinct acquisitions of one molecule
            # and discard the cross-condition training signal we want to keep.
            query_keys=raw["query_row"].astype("U"), near=raw["near_query"].astype(bool),
            query_weights=np.ones(len(raw["query_formula"]), np.float32),
            scores={"full23_v1": parent},
        )
        features[name], names = build_joint_features(panel, support_features, expert_scores)
        feature_names = names if feature_names is None else feature_names
        if names != feature_names:
            raise RuntimeError("panel feature schema drift")
        panels[name] = panel
    # The formula panel overlaps the identity panel. Weight only the same
    # experimental query row inversely by its panel multiplicity. Distinct
    # spectra of the same identity remain distinct training observations.
    multiplicity = {}
    for panel in panels.values():
        for key in panel.query_keys:
            multiplicity[str(key)] = multiplicity.get(str(key), 0) + 1
    for panel in panels.values():
        panel.query_weights[:] = np.asarray([1.0 / multiplicity[str(key)] for key in panel.query_keys], np.float32)
    return panels, features, feature_names


def training_rows(panels, occurrences):
    return np.concatenate([
        np.arange(int(panels[name].query_ptr[q]), int(panels[name].query_ptr[q + 1]), dtype=np.int64)
        for name, q in occurrences
    ])


def fit_model(panels, features, occurrences, columns, nonlinear, args, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    values = np.concatenate([
        features[name][training_rows(panels, [(p, q) for p, q in occurrences if p == name])][:, columns]
        for name in PANELS
    ])
    mean = values.mean(0).astype(np.float32)
    scale = values.std(0).astype(np.float32); scale[scale < 1e-6] = 1.0
    x = {name: torch.as_tensor((features[name][:, columns] - mean) / scale) for name in PANELS}
    parent = {name: torch.as_tensor(panels[name].scores["full23_v1"]) for name in PANELS}
    model = Residual(len(columns), args.hidden, nonlinear)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(seed)
    history = []
    for _ in range(args.epochs):
        order = rng.permutation(len(occurrences)); total = mass_total = 0.0
        for start in range(0, len(order), args.batch_queries):
            batch = [occurrences[i] for i in order[start:start + args.batch_queries]]
            numerators, masses = [], []
            for name in PANELS:
                queries = np.asarray([q for p, q in batch if p == name], np.int64)
                if not len(queries): continue
                index, valid, target = support.query_batch(panels[name].query_ptr, panels[name].labels, queries)
                take = torch.as_tensor(index, dtype=torch.long); valid_t = torch.as_tensor(valid)
                target_t = torch.as_tensor(target, dtype=torch.long); row = torch.arange(len(target_t))
                parent_logits = parent[name][take].masked_fill(~valid_t, -1e9)
                # A candidate farther than 2B below the parent top cannot win:
                # even +B for it and -B for the top leaves the ordering intact.
                # Modifying such candidates cannot improve Top-1 and can only
                # damage global pairwise ranking metrics.
                reachable_candidate = (
                    torch.max(parent_logits, dim=1, keepdim=True).values - parent_logits
                    <= 2.0 * args.residual_bound
                ) & valid_t
                residual = args.residual_bound * torch.tanh(model(x[name][take]))
                residual = residual * reachable_candidate.float()
                logits = (parent[name][take] + residual).masked_fill(~valid_t, -1e9)
                query_weight = torch.as_tensor(panels[name].query_weights[queries])
                ce = torch.nn.functional.cross_entropy(logits, target_t, reduction="none")
                true_parent = parent_logits[row, target_t]; true_new = logits[row, target_t]
                wrong_parent = parent_logits.clone(); wrong_parent[row, target_t] = -1e9
                wrong_new = logits.clone(); wrong_new[row, target_t] = -1e9
                parent_margin = true_parent - torch.max(wrong_parent, 1).values
                new_margin = true_new - torch.max(wrong_new, 1).values
                reachable = (torch.abs(parent_margin) <= 2 * args.residual_bound).float()
                boundary = torch.where(
                    parent_margin >= 0,
                    2.0 * torch.relu(parent_margin - new_margin),
                    torch.relu(0.01 - new_margin),
                )
                local = ce + args.boundary_weight * boundary * reachable
                numerators.append(torch.sum(local * query_weight)); masses.append(torch.sum(query_weight))
            mass = torch.sum(torch.stack(masses)); loss = torch.sum(torch.stack(numerators)) / mass
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += float(loss.detach()) * float(mass); mass_total += float(mass)
        history.append(total / mass_total)
    return {"columns": columns, "mean": mean, "scale": scale, "state": model.state_dict(),
            "nonlinear": nonlinear, "history": history}


def predict(model_data, panel, features, args):
    model = Residual(len(model_data["columns"]), args.hidden, model_data["nonlinear"])
    model.load_state_dict(model_data["state"]); model.eval()
    x = torch.as_tensor((features[:, model_data["columns"]] - model_data["mean"]) / model_data["scale"])
    with torch.no_grad(): residual = args.residual_bound * torch.tanh(model(x)).numpy()
    reachable = np.zeros(len(residual), dtype=bool)
    parent = panel.scores["full23_v1"]
    for left, right in zip(panel.query_ptr[:-1], panel.query_ptr[1:]):
        left, right = int(left), int(right)
        reachable[left:right] = float(np.max(parent[left:right])) - parent[left:right] <= 2.0 * args.residual_bound
    residual[~reachable] = 0.0
    return panel.scores["full23_v1"] + residual.astype(np.float32)


def serializable(model):
    return {"columns": model["columns"].tolist(), "mean": model["mean"].tolist(),
            "scale": model["scale"].tolist(), "nonlinear": model["nonlinear"],
            "history": model["history"]}


def main() -> None:
    args = arguments()
    if args.output.exists(): raise FileExistsError(args.output)
    panels, features, feature_names = load_data(args)
    folds = {name: np.asarray([formula_fold(v, args.folds, args.seed) for v in panel.formulas]) for name, panel in panels.items()}
    arms = {
        "count_hybrid": (np.asarray([0]), False),
        "count_confidence_gate": (
            np.asarray([
                feature_names.index("log_reference_count"),
                feature_names.index("parent_relative_score"),
                feature_names.index("top1_vote_fraction"),
                feature_names.index("expert_z_std"),
            ]),
            True,
        ),
    }
    predictions = {arm: {name: np.full_like(panel.scores["full23_v1"], np.nan) for name, panel in panels.items()} for arm in arms}
    fold_models = []
    for fold in range(args.folds):
        occurrences = [(name, q) for name in PANELS for q in range(len(panels[name].formulas)) if folds[name][q] != fold]
        models = {arm: fit_model(panels, features, occurrences, columns, nonlinear, args,
                                 args.seed + 100 * fold + index)
                  for index, (arm, (columns, nonlinear)) in enumerate(arms.items())}
        fold_models.append({"fold": fold, "models": {arm: serializable(model) for arm, model in models.items()}})
        for name in PANELS:
            queries = np.flatnonzero(folds[name] == fold)
            for arm, model in models.items():
                candidate = predict(model, panels[name], features[name], args)
                for query in queries:
                    left, right = map(int, panels[name].query_ptr[query:query + 2])
                    predictions[arm][name][left:right] = candidate[left:right]
        print(f"[joint quality repair] completed formula fold {fold + 1}/{args.folds}", flush=True)
    if any(np.any(~np.isfinite(score)) for arm in predictions.values() for score in arm.values()):
        raise RuntimeError("joint repair OOF predictions incomplete")
    all_occurrences = [(name, q) for name in PANELS for q in range(len(panels[name].formulas))]
    final_models = {arm: fit_model(panels, features, all_occurrences, columns, nonlinear, args,
                                  args.seed + 10000 + index)
                    for index, (arm, (columns, nonlinear)) in enumerate(arms.items())}
    report_panels = {}
    for index, name in enumerate(PANELS):
        panel = panels[name]; parent = panel.scores["full23_v1"]
        report_panels[name] = {
            "queries": int(len(panel.formulas)), "effective_query_weight": float(np.sum(panel.query_weights)),
            "metrics": {"full23_v1": _metric_summary(parent, panel)[0],
                        **{arm: _metric_summary(values[name], panel)[0] for arm, values in predictions.items()}},
            "comparisons_vs_full23": {arm: _comparison(values[name], parent, panel, args.bootstrap_resamples,
                                                         args.seed + 1000 * index + offset)
                                       for offset, (arm, values) in enumerate(predictions.items(), 1)},
        }
    args.output.mkdir(parents=True)
    for name in PANELS:
        np.savez_compressed(args.output / f"predictions_{name}.npz", query_ptr=panels[name].query_ptr,
                            labels=panels[name].labels, formulas=panels[name].formulas,
                            full23_v1=panels[name].scores["full23_v1"],
                            **{arm: values[name] for arm, values in predictions.items()})
    torch.save({"schema": "full23_joint_quality_repair_local_v1", "feature_names": feature_names,
                "residual_bound": args.residual_bound, "models": final_models}, args.output / "models.pt")
    report = {
        "status": "FULL23_JOINT_QUALITY_REPAIR_LOCAL_COMPLETE",
        "dataset": "GNPS Gold/Silver consumed development; both panels joint formula-OOF",
        "allow_external_claim": False,
        "feature_names": feature_names,
        "training": {k: getattr(args, k) for k in ("folds", "epochs", "batch_queries", "learning_rate",
                                                    "weight_decay", "boundary_weight", "residual_bound", "hidden", "seed")},
        "fold_models": fold_models, "final_models": {arm: serializable(model) for arm, model in final_models.items()},
        "panels": report_panels,
        "claim_boundary": "local combined development only; any winner requires fixed cross-source evaluation",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))


if __name__ == "__main__": main()
