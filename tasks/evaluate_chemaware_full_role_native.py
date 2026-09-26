"""Evaluate shared embeddings on every query in ChemAware formula roles 2/3.

Unlike the one-query-per-identity checkpoint selector, this secondary panel
measures robustness across all available spectra and experimental conditions.
It reports both query-micro metrics and identity-equal metrics, with
formula-cluster bootstrap inference.  Formula role 4 is intentionally
unreachable from this program.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from chemaware_numpy_sampling import stable_formula_folds
from chemaware_v2_triplet_eval_core import evaluate_graph, paired_summary, summarize
from evaluate_chemaware_v2_direct_triplet import (
    DEFAULT_ARCHITECTURE,
    DEFAULT_DATA,
    DEFAULT_MANIFEST,
    DEFAULT_OFFICIAL,
    encode_rows,
    load_model,
    parse_checkpoint,
    required_rows,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--architecture-checkpoint", type=Path, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--official-checkpoint", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument(
        "--checkpoint", action="append", required=True,
        help="Named checkpoint as NAME=PATH; first checkpoint must be official.",
    )
    parser.add_argument(
        "--formula-role", type=int, choices=(2, 3), nargs="+", required=True,
        help="One or both development roles. Role 4 is deliberately unavailable.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-highest-peaks", type=int, default=100)
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def identity_equal_metrics(
    ranks: np.ndarray, positive: np.ndarray, negative: np.ndarray,
    identities: np.ndarray,
) -> dict[str, float | int]:
    identities = np.asarray(identities).astype(str)
    unique, inverse = np.unique(identities, return_inverse=True)
    count = np.bincount(inverse)

    def macro(values: np.ndarray) -> float:
        return float(np.mean(np.bincount(inverse, weights=values) / count))

    return {
        "identities": int(len(unique)),
        "recall1": macro((ranks <= 1).astype(np.float64)),
        "recall3": macro((ranks <= 3).astype(np.float64)),
        "recall5": macro((ranks <= 5).astype(np.float64)),
        "recall10": macro((ranks <= 10).astype(np.float64)),
        "recall20": macro((ranks <= 20).astype(np.float64)),
        "recall50": macro((ranks <= 50).astype(np.float64)),
        "mrr": macro(1.0 / ranks),
        "mean_positive_margin": macro(positive - negative),
    }


def identity_equal_paired(
    baseline: np.ndarray, current: np.ndarray, identities: np.ndarray,
    formulas: np.ndarray, draws: int, seed: int,
) -> dict[str, object]:
    identities = np.asarray(identities).astype(str)
    formulas = np.asarray(formulas).astype(str)
    unique_identity, inverse = np.unique(identities, return_inverse=True)
    count = np.bincount(inverse)
    query_delta = (
        (current <= 1).astype(np.float64) - (baseline <= 1).astype(np.float64)
    )
    identity_delta = np.bincount(inverse, weights=query_delta) / count
    identity_formula = np.empty(len(unique_identity), dtype=object)
    for index in range(len(unique_identity)):
        values = np.unique(formulas[inverse == index])
        if len(values) != 1:
            raise RuntimeError("one identity spans multiple formula clusters")
        identity_formula[index] = values[0]
    unique_formula, formula_inverse = np.unique(identity_formula.astype(str), return_inverse=True)
    formula_count = np.bincount(formula_inverse)
    formula_sum = np.bincount(formula_inverse, weights=identity_delta)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        selected = rng.integers(0, len(unique_formula), len(unique_formula))
        samples[draw] = formula_sum[selected].sum() / formula_count[selected].sum()
    return {
        "delta_recall1": float(np.mean(identity_delta)),
        "identities_improved": int(np.sum(identity_delta > 0)),
        "identities_harmed": int(np.sum(identity_delta < 0)),
        "identities_unchanged": int(np.sum(identity_delta == 0)),
        "formula_clusters": int(len(unique_formula)),
        "formula_cluster_bootstrap_delta_recall1_ci95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
    }


def main() -> None:
    args = arguments()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    with np.load(args.manifest, allow_pickle=False) as loaded:
        manifest = {key: np.asarray(loaded[key]) for key in loaded.files}
    roles = tuple(dict.fromkeys(map(int, args.formula_role)))
    folds = stable_formula_folds(manifest["query_formula"], args.folds, args.fold_seed)
    role_query = {
        role: np.flatnonzero(folds == role).astype(np.int64) for role in roles
    }
    if any(not len(queries) for queries in role_query.values()):
        raise RuntimeError("selected formula role has no queries")
    all_queries = np.concatenate([role_query[role] for role in roles])
    rows = required_rows(manifest, all_queries)

    reports = []
    baseline_rank: dict[int, np.ndarray] = {}
    baseline_encoded = None
    for index, raw_checkpoint in enumerate(args.checkpoint):
        name, path = parse_checkpoint(raw_checkpoint)
        if index == 0 and name != "official":
            raise ValueError("the first checkpoint must be named official")
        if not path.is_file():
            raise FileNotFoundError(path)
        print(
            f"FULL_ROLE_EVAL name={name} roles={roles} "
            f"queries={len(all_queries)} rows={len(rows)} checkpoint={path}",
            flush=True,
        )
        model, kind = load_model(
            path, args.official_checkpoint, args.architecture_checkpoint,
            device, args.n_highest_peaks,
        )
        encoded = encode_rows(
            model, rows, args.data, device, args.batch_size, args.n_highest_peaks,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        row = {
            "name": name,
            "checkpoint": str(path.resolve()),
            "kind": kind,
            "roles": {},
        }
        for role in roles:
            queries = role_query[role]
            formulas = np.asarray(manifest["query_formula"])[queries].astype(str)
            identities = np.asarray(manifest["query_ik14"])[queries].astype(str)
            ranks, positive, negative, auc = evaluate_graph(
                encoded, rows, manifest, queries,
            )
            role_row = {
                "queries": int(len(queries)),
                "identities": int(len(np.unique(identities))),
                "formulas": int(len(np.unique(formulas))),
                "query_micro": summarize(ranks, positive, negative, auc),
                "identity_equal": identity_equal_metrics(
                    ranks, positive, negative, identities,
                ),
            }
            if index == 0:
                baseline_rank[role] = ranks.copy()
            else:
                if role not in baseline_rank or baseline_encoded is None:
                    raise AssertionError("official baseline was not evaluated first")
                role_row["query_micro_vs_official"] = paired_summary(
                    baseline_rank[role], ranks, formulas, args.bootstrap_draws,
                    args.seed + 100 * role + index,
                )
                role_row["identity_equal_vs_official"] = identity_equal_paired(
                    baseline_rank[role], ranks, identities, formulas,
                    args.bootstrap_draws, args.seed + 10_000 + 100 * role + index,
                )
            row["roles"][str(role)] = role_row
        if index == 0:
            baseline_encoded = encoded.copy()
        else:
            row["mean_cosine_to_official_embedding"] = float(np.mean(np.sum(
                encoded * baseline_encoded, axis=1,
            )))
        reports.append(row)
        print(json.dumps(row, indent=2), flush=True)

    report = {
        "status": "CHEMAWARE_FULL_ROLE_SHARED_EMBEDDING_EVALUATION_COMPLETE",
        "formula_roles": list(roles),
        "outer_role_4_accessed": False,
        "queries": int(len(all_queries)),
        "unique_spectrum_rows_encoded": int(len(rows)),
        "primary_inference": "identity-equal metrics with formula-cluster bootstrap",
        "secondary_inference": "query-micro condition-robustness metrics",
        "results": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {args.output}", flush=True)


if __name__ == "__main__":
    main()
