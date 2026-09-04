"""Test whether observable confidence can safely route a raw-spectral consensus.

This is a teacher-applicability audit, not DreaMS training.  Candidate scores
are rebuilt from frozen pair tables.  Features contain only observable ranking
geometry (top-two margins, view agreement, and candidate counts).  A logistic
gate is cross-fitted by molecular formula on discovery, its threshold is chosen
once from discovery OOF risk utility, and the frozen gate is evaluated once on
formula-disjoint confirmation.  The untouched test split is never read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent.parent
RAW_VIEWS = (
    "entropy_similarity", "sqrt_cosine", "linear_cosine",
    "top10_match_fraction", "intensity_coverage_min",
)
SCORES = ("dreams_similarity", *RAW_VIEWS)
FEATURES = (
    "dreams_top2_margin",
    "consensus_votes",
    "distinct_raw_winners",
    "raw_winners_equal_dreams",
    "raw_top2_margin_mean",
    "raw_top2_margin_min",
    "raw_top2_margin_max",
    "consensus_candidate_gap_mean",
    "consensus_candidate_gap_min",
    "candidate_molecules",
    "candidate_reference_spectra",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-dir", type=Path,
        default=ROOT / "data/validation/large_observability_residual_audit",
    )
    parser.add_argument(
        "--discovery-manifest", type=Path,
        default=ROOT / "data/validation/large_observability_embeddings_discovery/manifest.csv",
    )
    parser.add_argument(
        "--confirmation-manifest", type=Path,
        default=ROOT / "data/validation/large_observability_embeddings_confirmation/manifest.csv",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--permutation-controls", type=int, default=100)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/chemaware_spectral_consensus_applicability_v4_frozen/report.json",
    )
    return parser.parse_args()


def build_query_table(pair_path: Path, manifest_path: Path, split: str) -> pd.DataFrame:
    pairs = pd.read_csv(pair_path)
    manifest = pd.read_csv(manifest_path)
    if set(pairs["split"].astype(str)) != {split}:
        raise RuntimeError(f"pair split mismatch: {pair_path}")
    adjacency: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for edge, row in enumerate(pairs[["left", "right"]].itertuples(index=False)):
        adjacency[int(row.left)].append((edge, int(row.right)))
        adjacency[int(row.right)].append((edge, int(row.left)))
    rows: list[dict] = []
    for query in range(len(manifest)):
        edges = adjacency.get(query, [])
        if not edges:
            continue
        by_identity: dict[str, list[int]] = defaultdict(list)
        for edge, candidate in edges:
            by_identity[str(manifest.at[candidate, "ik14"])].append(edge)
        truth = str(manifest.at[query, "ik14"])
        if truth not in by_identity or len(by_identity) < 2:
            continue
        identities = sorted(by_identity)
        matrix = np.empty((len(identities), len(SCORES)), dtype=np.float64)
        reference_count = 0
        for identity_index, identity in enumerate(identities):
            edge_index = by_identity[identity]
            reference_count += len(edge_index)
            matrix[identity_index] = pairs.iloc[edge_index][list(SCORES)].max(axis=0).to_numpy(float)
        winners = np.argmax(matrix, axis=0)
        winner_identity = np.asarray([identities[index] for index in winners], dtype=object)
        top2 = np.partition(matrix, -2, axis=0)[-2:]
        margins = top2[1] - top2[0]
        # Raw views abstain on an exact top-score tie. This avoids turning the
        # lexicographic identity order into apparent chemical agreement.
        raw_winners = winner_identity[1:].copy()
        raw_tied = np.sum(matrix[:, 1:] == np.max(matrix[:, 1:], axis=0), axis=0) > 1
        raw_winners[raw_tied] = None
        votes_cast = [str(value) for value in raw_winners if value is not None]
        if votes_cast:
            consensus_identity, consensus_votes = Counter(votes_cast).most_common(1)[0]
        else:
            consensus_identity, consensus_votes = "", 0
        consensus_index = identities.index(consensus_identity) if votes_cast else int(winners[0])
        raw_best = np.max(matrix[:, 1:], axis=0)
        consensus_gaps = raw_best - matrix[consensus_index, 1:]
        dreams_identity = str(winner_identity[0])
        truth_index = identities.index(truth)
        negative_index = [index for index, identity in enumerate(identities) if identity != truth]
        dreams_strict_correct = bool(
            matrix[truth_index, 0] > np.max(matrix[negative_index, 0])
        )
        row = {
            "split": split,
            "query_index": query,
            "ik14": truth,
            "formula": str(manifest.at[query, "formula"]),
            "dreams_prediction": dreams_identity,
            "consensus_prediction": consensus_identity,
            "dreams_correct": dreams_strict_correct,
            "consensus_correct": consensus_identity == truth,
            "route_candidate": consensus_votes >= 3 and consensus_identity != dreams_identity,
            "dreams_top2_margin": float(margins[0]),
            "consensus_votes": int(consensus_votes),
            "distinct_raw_winners": len(set(votes_cast)),
            "raw_winners_equal_dreams": int(np.sum(raw_winners == dreams_identity)),
            "raw_top2_margin_mean": float(np.mean(margins[1:])),
            "raw_top2_margin_min": float(np.min(margins[1:])),
            "raw_top2_margin_max": float(np.max(margins[1:])),
            "consensus_candidate_gap_mean": float(np.mean(consensus_gaps)),
            "consensus_candidate_gap_min": float(np.min(consensus_gaps)),
            "candidate_molecules": len(identities),
            "candidate_reference_spectra": reference_count,
        }
        row["beneficial_route"] = bool(row["consensus_correct"] and not row["dreams_correct"])
        rows.append(row)
    return pd.DataFrame(rows)


def fit_model(frame: pd.DataFrame, target: np.ndarray | None = None):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, max_iter=3000, random_state=20260902),
    )
    formula_counts = frame.groupby("formula")["query_index"].transform("size").to_numpy(float)
    weights = 1.0 / formula_counts
    weights *= len(weights) / weights.sum()
    labels = frame["beneficial_route"].astype(int).to_numpy() if target is None else np.asarray(target, dtype=int)
    model.fit(frame[list(FEATURES)], labels,
              logisticregression__sample_weight=weights)
    return model


def crossfit_probability(
    frame: pd.DataFrame, target: np.ndarray, folds: int,
) -> tuple[np.ndarray, object]:
    groups = frame["formula"].astype(str).to_numpy()
    oof = np.empty(len(frame), dtype=float)
    splitter = GroupKFold(n_splits=folds)
    for train, valid in splitter.split(frame, groups=groups):
        model = fit_model(frame.iloc[train], target[train])
        oof[valid] = model.predict_proba(frame.iloc[valid][list(FEATURES)])[:, 1]
    return oof, fit_model(frame, target)


def route_metrics(frame: pd.DataFrame, probability: np.ndarray, threshold: float) -> dict:
    active = frame["route_candidate"].to_numpy(bool) & (probability >= threshold)
    old = frame["dreams_correct"].to_numpy(bool)
    new = old.copy()
    new[active] = frame.loc[active, "consensus_correct"].to_numpy(bool)
    corrected = (~old) & new
    introduced = old & (~new)
    return {
        "threshold": float(threshold),
        "route_activated": int(np.sum(active)),
        "corrected": int(np.sum(corrected)),
        "introduced": int(np.sum(introduced)),
        "risk_utility_corrected_minus_2x_introduced": int(np.sum(corrected) - 2 * np.sum(introduced)),
        "dreams_recall1": float(np.mean(old)),
        "routed_recall1": float(np.mean(new)),
        "delta_pp": float(100.0 * np.mean(new.astype(float) - old.astype(float))),
        "active_mask": active,
        "new_correct": new,
    }


def select_threshold(frame: pd.DataFrame, probability: np.ndarray) -> tuple[float, dict]:
    candidates = np.r_[np.linspace(0.05, 0.95, 91), 1.01]
    scored = [route_metrics(frame, probability, float(value)) for value in candidates]
    best = max(
        scored,
        key=lambda row: (
            row["risk_utility_corrected_minus_2x_introduced"],
            -row["introduced"], row["corrected"], row["threshold"],
        ),
    )
    return float(best["threshold"]), {k: v for k, v in best.items() if k not in ("active_mask", "new_correct")}


def bootstrap_delta(frame: pd.DataFrame, new: np.ndarray, iterations: int, seed: int) -> list[float]:
    work = pd.DataFrame({
        "formula": frame["formula"].astype(str),
        "n": 1,
        "delta": new.astype(np.int8) - frame["dreams_correct"].to_numpy(np.int8),
    }).groupby("formula", sort=False).sum()
    n = work["n"].to_numpy(float)
    delta = work["delta"].to_numpy(float)
    rng = np.random.default_rng(seed)
    out = np.empty(iterations, dtype=float)
    for start in range(0, iterations, 500):
        stop = min(start + 500, iterations)
        draw = rng.integers(0, len(work), size=(stop - start, len(work)))
        out[start:stop] = delta[draw].sum(axis=1) / n[draw].sum(axis=1)
    return [float(value) for value in np.quantile(out, (0.025, 0.975))]


def main() -> None:
    args = parse_args()
    if args.folds < 2 or args.bootstrap <= 0 or args.permutation_controls <= 0:
        raise ValueError("invalid fold/bootstrap setting")
    files = {
        "discovery_pairs": args.audit_dir / "discovery_pair_features.csv",
        "confirmation_pairs": args.audit_dir / "confirmation_pair_features.csv",
        "discovery_manifest": args.discovery_manifest,
        "confirmation_manifest": args.confirmation_manifest,
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    discovery = build_query_table(files["discovery_pairs"], args.discovery_manifest, "discovery")
    confirmation = build_query_table(files["confirmation_pairs"], args.confirmation_manifest, "confirmation")
    route_discovery = discovery.loc[discovery["route_candidate"]].copy().reset_index(drop=True)
    if route_discovery["beneficial_route"].nunique() != 2:
        raise RuntimeError("discovery route candidates do not contain both classes")
    target = route_discovery["beneficial_route"].astype(int).to_numpy()
    oof, final_model = crossfit_probability(route_discovery, target, args.folds)
    threshold, oof_selection = select_threshold(route_discovery, oof)
    discovery_probability = np.zeros(len(discovery), dtype=float)
    discovery_probability[discovery["route_candidate"].to_numpy(bool)] = oof
    confirmation_probability = final_model.predict_proba(confirmation[list(FEATURES)])[:, 1]
    discovery_result = route_metrics(discovery, discovery_probability, threshold)
    confirmation_result = route_metrics(confirmation, confirmation_probability, threshold)
    observed_confirmation_utility = confirmation_result["risk_utility_corrected_minus_2x_introduced"]

    rng = np.random.default_rng(args.seed + 991)
    null_results = []
    for index in range(args.permutation_controls):
        permuted = rng.permutation(target)
        null_oof, null_model = crossfit_probability(route_discovery, permuted, args.folds)
        null_threshold, _ = select_threshold(route_discovery, null_oof)
        null_probability = null_model.predict_proba(confirmation[list(FEATURES)])[:, 1]
        null_metric = route_metrics(confirmation, null_probability, null_threshold)
        null_results.append({
            "threshold": null_threshold,
            "route_activated": null_metric["route_activated"],
            "corrected": null_metric["corrected"],
            "introduced": null_metric["introduced"],
            "utility": null_metric["risk_utility_corrected_minus_2x_introduced"],
        })
    discovery_result["formula_cluster_bootstrap_delta_ci95"] = bootstrap_delta(
        discovery, discovery_result["new_correct"], args.bootstrap, args.seed
    )
    confirmation_result["formula_cluster_bootstrap_delta_ci95"] = bootstrap_delta(
        confirmation, confirmation_result["new_correct"], args.bootstrap, args.seed + 1
    )
    for result in (discovery_result, confirmation_result):
        result.pop("active_mask")
        result.pop("new_correct")
    discovery_ledger = discovery.copy()
    discovery_ledger["gate_probability"] = discovery_probability
    discovery_ledger["route_activated"] = (
        discovery_ledger["route_candidate"] & (discovery_probability >= threshold)
    )
    confirmation_ledger = confirmation.copy()
    confirmation_ledger["gate_probability"] = confirmation_probability
    confirmation_ledger["route_activated"] = (
        confirmation_ledger["route_candidate"] & (confirmation_probability >= threshold)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    discovery_ledger.to_csv(args.output.parent / "discovery_gate_ledger.csv.gz", index=False)
    confirmation_ledger.to_csv(args.output.parent / "confirmation_gate_ledger.csv.gz", index=False)
    coef = final_model.named_steps["logisticregression"].coef_[0]
    scaler = final_model.named_steps["standardscaler"]
    logistic = final_model.named_steps["logisticregression"]
    frozen_gate = {
        "format": "chemaware_spectral_consensus_gate_v1",
        "features": list(FEATURES),
        "raw_views": list(RAW_VIEWS),
        "route_candidate_rule": "at least 3 unique-top raw-view votes and consensus differs from DreaMS winner",
        "raw_view_exact_top_ties": "abstain",
        "official_metric_tie_rule": "truth must score strictly greater than every negative",
        "threshold": threshold,
        "standard_scaler_mean": [float(value) for value in scaler.mean_],
        "standard_scaler_scale": [float(value) for value in scaler.scale_],
        "logistic_coefficient": [float(value) for value in logistic.coef_[0]],
        "logistic_intercept": float(logistic.intercept_[0]),
        "training_scope": "discovery route candidates only; formula-grouped OOF threshold selection",
        "discovery_formula_ledger_sha256": hashlib.sha256(
            "\n".join(sorted(set(route_discovery["formula"].astype(str)))).encode("utf-8")
        ).hexdigest(),
        "input_sha256": {name: sha256(path) for name, path in files.items()},
        "test_split_seen": False,
        "frozen_before_test": True,
    }
    frozen_gate_path = args.output.parent / "frozen_gate.json"
    frozen_gate_path.write_text(json.dumps(frozen_gate, ensure_ascii=False, indent=2), encoding="utf-8")
    output = {
        "status": "chemaware_spectral_consensus_applicability_audited",
        "training_was_run": False,
        "teacher_gate_was_fit": True,
        "features_are_identity_label_free_at_application": True,
        "features": list(FEATURES),
        "discovery_route_candidates": len(route_discovery),
        "discovery_beneficial_route_prevalence": float(route_discovery["beneficial_route"].mean()),
        "threshold_selection": {
            "protocol": "5-fold formula-grouped OOF; maximize corrected-2*introduced; frozen once",
            "selected": oof_selection,
        },
        "standardized_logistic_coefficients": {
            feature: float(value) for feature, value in zip(FEATURES, coef)
        },
        "evaluation": {
            "discovery_oof": discovery_result,
            "formula_disjoint_confirmation": confirmation_result,
        },
        "permutation_control": {
            "count": args.permutation_controls,
            "protocol": (
                "globally permute discovery beneficial-route labels; repeat the same formula-grouped "
                "cross-fit and actual-utility threshold search; evaluate frozen random-label gate on confirmation"
            ),
            "confirmation_utility_histogram": {
                str(value): int(sum(row["utility"] == value for row in null_results))
                for value in sorted({row["utility"] for row in null_results})
            },
            "confirmation_utility_mean": float(np.mean([row["utility"] for row in null_results])),
            "confirmation_utility_max": int(max(row["utility"] for row in null_results)),
            "empirical_p_utility_ge_observed": float(
                (1 + sum(row["utility"] >= observed_confirmation_utility for row in null_results))
                / (1 + len(null_results))
            ),
        },
        "ledgers": {
            "discovery": str(args.output.parent / "discovery_gate_ledger.csv.gz"),
            "confirmation": str(args.output.parent / "confirmation_gate_ledger.csv.gz"),
        },
        "frozen_gate": {
            "path": str(frozen_gate_path),
            "sha256": sha256(frozen_gate_path),
            "frozen_before_test": True,
        },
        "provenance": {name: {"path": str(path), "sha256": sha256(path)} for name, path in files.items()},
        "claim_limit": (
            "The gate uses candidate reference spectra and is only a training-teacher applicability "
            "test. It is not a shared-embedding result, not P3, and does not authorize model training."
        ),
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
