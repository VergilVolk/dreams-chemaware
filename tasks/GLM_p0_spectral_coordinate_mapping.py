#!/usr/bin/env python
"""P0-L: hidden-name, mass-conditioned spectral-coordinate identifiability.

This gate does not build a biological atlas. It asks a smaller prerequisite:
given a frozen precursor-mass candidate neighbourhood and a frozen reference
library, can an MS/MS scorer assign an independently acquired query to the
coordinate representing the same molecule, including when every positive
reference was acquired on another instrument family?

The program consumes the already frozen GNPS Gold/Silver panels and the
15-method article score bundle. Molecule names/formulas are used only after
scoring, to evaluate the hidden-name assignment. No model is trained and no
method is selected here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PANELS = ("identity_disjoint", "formula_disjoint")
COVERAGES = (0.2, 0.4, 0.6, 0.8, 1.0)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "data/validation/gnps_gold_silver_10ppm_benchmark_v1",
    )
    parser.add_argument(
        "--identity-panel",
        type=Path,
        default=ROOT
        / "data/validation/GLM_gnps_identity_panel_reconstruction/"
        "panel_identity_disjoint.npz",
        help="Certified identity panel; server copies may use benchmark/panel_identity_disjoint.npz.",
    )
    parser.add_argument(
        "--score-bundle",
        type=Path,
        default=ROOT
        / "data/validation/GLM_gnps_article_benchmark_s2v26/run15/bundle/"
        "method_scores.npz",
    )
    parser.add_argument("--primary-method", default="noise_v1")
    parser.add_argument("--baseline-method", default="official_dreams")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261007)
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as body:
        return {key: np.asarray(body[key]) for key in body.files}


def instrument_family(value: object) -> str:
    text = str(value).lower().replace("-", "").replace(" ", "")
    if "orbitrap" in text or "qexactive" in text or "hybridft" in text:
        return "orbitrap_ft"
    if "qtof" in text or "tof" in text or "maxis" in text:
        return "qtof"
    return "unknown"


def validate_panel(panel: dict[str, np.ndarray], pair_count: int, name: str) -> None:
    required = {
        "query_row", "query_ik14", "query_formula", "near_query", "query_ptr",
        "molecule_ptr", "molecule_label", "molecule_formula", "candidate_row",
        "independent_positive",
    }
    missing = required - set(panel)
    if missing:
        raise RuntimeError(f"{name} panel fields missing: {sorted(missing)}")
    n_queries = len(panel["query_row"])
    if len(panel["query_ptr"]) != n_queries + 1:
        raise RuntimeError(f"{name}: query_ptr length mismatch")
    if len(panel["molecule_ptr"]) != len(panel["molecule_label"]) + 1:
        raise RuntimeError(f"{name}: molecule_ptr length mismatch")
    if len(panel["candidate_row"]) != pair_count:
        raise RuntimeError(f"{name}: score/panel edge count mismatch")
    if int(panel["query_ptr"][0]) != 0 or int(panel["query_ptr"][-1]) != len(
        panel["molecule_label"]
    ):
        raise RuntimeError(f"{name}: invalid query_ptr boundaries")
    if int(panel["molecule_ptr"][0]) != 0 or int(panel["molecule_ptr"][-1]) != pair_count:
        raise RuntimeError(f"{name}: invalid molecule_ptr boundaries")
    positives = np.add.reduceat(
        panel["molecule_label"].astype(np.int64), panel["query_ptr"][:-1]
    )
    if not np.all(positives == 1):
        raise RuntimeError(f"{name}: every query must have exactly one positive coordinate")
    if not np.all(panel["independent_positive"]):
        raise RuntimeError(f"{name}: independent positive requirement is not universal")


def cross_instrument_masks(
    panel: dict[str, np.ndarray], instrument_by_row: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n_queries = len(panel["query_row"])
    available = np.zeros(n_queries, dtype=bool)
    strict = np.zeros(n_queries, dtype=bool)
    labels = panel["molecule_label"].astype(bool)
    for query in range(n_queries):
        q_lo, q_hi = panel["query_ptr"][query : query + 2]
        positive_molecule = np.flatnonzero(labels[q_lo:q_hi])
        if len(positive_molecule) != 1:
            raise RuntimeError("positive-coordinate uniqueness changed after validation")
        molecule = int(q_lo + positive_molecule[0])
        c_lo, c_hi = panel["molecule_ptr"][molecule : molecule + 2]
        rows = panel["candidate_row"][c_lo:c_hi]
        query_instrument = instrument_by_row[int(panel["query_row"][query])]
        reference_instruments = instrument_by_row[rows]
        known_query = query_instrument != "unknown"
        different = known_query & (reference_instruments != "unknown") & (
            reference_instruments != query_instrument
        )
        same = known_query & (reference_instruments == query_instrument)
        available[query] = bool(np.any(different))
        strict[query] = bool(np.any(different) and not np.any(same))
    return available, strict


def evaluate_method(
    panel: dict[str, np.ndarray], pair_scores: np.ndarray
) -> dict[str, np.ndarray]:
    pair_scores = np.asarray(pair_scores, dtype=np.float64)
    if pair_scores.ndim != 1 or len(pair_scores) != len(panel["candidate_row"]):
        raise RuntimeError("pair-score vector does not align to the frozen panel")
    if not np.all(np.isfinite(pair_scores)):
        raise RuntimeError("pair scores contain non-finite values")
    molecule_scores = np.maximum.reduceat(pair_scores, panel["molecule_ptr"][:-1])
    n_queries = len(panel["query_row"])
    rank = np.empty(n_queries, dtype=np.int64)
    confidence = np.empty(n_queries, dtype=np.float64)
    same_formula_error = np.zeros(n_queries, dtype=bool)
    cross_formula_error = np.zeros(n_queries, dtype=bool)
    positive_score = np.empty(n_queries, dtype=np.float64)
    best_negative_score = np.empty(n_queries, dtype=np.float64)
    labels = panel["molecule_label"].astype(bool)
    same_formula = panel["molecule_formula"].astype(str) == np.repeat(
        panel["query_formula"].astype(str), np.diff(panel["query_ptr"])
    )
    for query in range(n_queries):
        lo, hi = panel["query_ptr"][query : query + 2]
        scores = molecule_scores[lo:hi]
        local_labels = labels[lo:hi]
        positive = int(np.flatnonzero(local_labels)[0])
        negative = ~local_labels
        true_score = float(scores[positive])
        best_negative = float(np.max(scores[negative]))
        # Frozen benchmark policy: every negative tied with the positive ranks ahead.
        rank[query] = 1 + int(np.sum(scores[negative] >= true_score))
        ordered = np.partition(scores, len(scores) - 2)
        confidence[query] = float(ordered[-1] - ordered[-2])
        positive_score[query] = true_score
        best_negative_score[query] = best_negative
        if best_negative >= true_score:
            top_negative = negative & np.isclose(scores, best_negative, rtol=0.0, atol=0.0)
            has_same = bool(np.any(same_formula[lo:hi] & top_negative))
            same_formula_error[query] = has_same
            cross_formula_error[query] = not has_same
    return {
        "rank": rank,
        "correct": rank == 1,
        "confidence": confidence,
        "same_formula_error": same_formula_error,
        "cross_formula_error": cross_formula_error,
        "positive_score": positive_score,
        "best_negative_score": best_negative_score,
    }


def selective_metrics(correct: np.ndarray, confidence: np.ndarray) -> dict[str, float]:
    if len(correct) == 0:
        return {
            "aurc": float("nan"),
            **{f"risk_at_{int(c*100)}pct": float("nan") for c in COVERAGES},
        }
    order = np.argsort(-confidence, kind="stable")
    errors = (~correct[order]).astype(np.float64)
    cumulative = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    result = {"aurc": float(np.mean(cumulative))}
    for coverage in COVERAGES:
        count = max(1, int(np.ceil(coverage * len(errors))))
        result[f"risk_at_{int(coverage*100)}pct"] = float(cumulative[count - 1])
    return result


def summarize(outcome: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, object]:
    rank = outcome["rank"][mask]
    correct = outcome["correct"][mask]
    confidence = outcome["confidence"][mask]
    n = int(np.sum(mask))
    if n == 0:
        return {"queries": 0}
    result: dict[str, object] = {
        "queries": n,
        "recall@1": float(np.mean(rank <= 1)),
        "recall@5": float(np.mean(rank <= 5)),
        "recall@10": float(np.mean(rank <= 10)),
        "recall@20": float(np.mean(rank <= 20)),
        "mrr": float(np.mean(1.0 / rank)),
        "same_formula_false_merge_rate": float(np.mean(outcome["same_formula_error"][mask])),
        "cross_formula_error_rate": float(np.mean(outcome["cross_formula_error"][mask])),
        "mean_top1_top2_gap": float(np.mean(confidence)),
    }
    result["selective_assignment"] = selective_metrics(correct, confidence)
    return result


def cluster_bootstrap_delta(
    candidate: np.ndarray,
    baseline: np.ndarray,
    clusters: np.ndarray,
    mask: np.ndarray,
    resamples: int,
    seed: int,
) -> dict[str, float]:
    candidate = np.asarray(candidate, dtype=np.float64)[mask]
    baseline = np.asarray(baseline, dtype=np.float64)[mask]
    clusters = np.asarray(clusters).astype(str)[mask]
    delta = candidate - baseline
    unique, inverse = np.unique(clusters, return_inverse=True)
    cluster_sum = np.bincount(inverse, weights=delta, minlength=len(unique))
    cluster_n = np.bincount(inverse, minlength=len(unique))
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, 256):
        stop = min(resamples, start + 256)
        sampled = rng.integers(0, len(unique), size=(stop - start, len(unique)))
        values[start:stop] = cluster_sum[sampled].sum(axis=1) / cluster_n[sampled].sum(axis=1)
    return {
        "delta_pp": float(np.mean(delta) * 100.0),
        "ci95_low_pp": float(np.quantile(values, 0.025) * 100.0),
        "ci95_high_pp": float(np.quantile(values, 0.975) * 100.0),
        "clusters": int(len(unique)),
        "queries": int(len(delta)),
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.bootstrap_resamples < 100:
        raise ValueError("bootstrap-resamples must be at least 100")
    manifest_path = args.benchmark / "manifest.csv.gz"
    formula_panel_path = args.benchmark / "panel_formula_disjoint.npz"
    required = [manifest_path, formula_panel_path, args.identity_panel, args.score_bundle]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"P0-L inputs missing: {missing}")

    manifest = pd.read_csv(manifest_path, usecols=["row", "instrument", "filename"])
    if not np.array_equal(manifest["row"].to_numpy(np.int64), np.arange(len(manifest))):
        raise RuntimeError("manifest row registry is not contiguous")
    instrument_by_row = manifest["instrument"].map(instrument_family).to_numpy()

    panels = {
        "identity_disjoint": load_npz(args.identity_panel),
        "formula_disjoint": load_npz(formula_panel_path),
    }
    with np.load(args.score_bundle, allow_pickle=False) as body:
        methods = body["method_names"].astype(str).tolist()
        score_arrays = {
            name: np.asarray(body[f"scores_{name}"], dtype=np.float32)
            for name in PANELS
        }
    if args.primary_method not in methods or args.baseline_method not in methods:
        raise RuntimeError("primary and baseline methods must both exist in the frozen bundle")
    for name in PANELS:
        if score_arrays[name].shape[0] != len(methods):
            raise RuntimeError(f"{name}: method axis does not match method_names")
        validate_panel(panels[name], score_arrays[name].shape[1], name)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    report: dict[str, object] = {
        "status": "P0_L_SPECTRAL_COORDINATE_IDENTIFIABILITY_COMPLETE",
        "stage": "local_mass_conditioned_coordinate_gate",
        "primary_method": args.primary_method,
        "baseline_method": args.baseline_method,
        "methods": methods,
        "hidden_name_boundary": (
            "IK14 and formula labels are never score inputs; they are read only after the frozen "
            "pair scores have been aggregated to reference-molecule coordinates."
        ),
        "benchmark_reuse_disclosure": (
            "The GNPS panels have already been used for article-level method evaluation. This "
            "P0-L analysis can test coordinate viability and the previously unreported strict "
            "cross-instrument stratum, but it is not an independent external confirmation or a "
            "new checkpoint-selection panel."
        ),
        "inputs": {
            "manifest_sha256": sha256_file(manifest_path),
            "identity_panel_sha256": sha256_file(args.identity_panel),
            "formula_panel_sha256": sha256_file(formula_panel_path),
            "score_bundle_sha256": sha256_file(args.score_bundle),
        },
        "panels": {},
    }
    try:
        for panel_index, name in enumerate(PANELS):
            panel = panels[name]
            cross_available, strict_cross = cross_instrument_masks(panel, instrument_by_row)
            masks = {
                "all": np.ones(len(panel["query_row"]), dtype=bool),
                "near_structure": panel["near_query"].astype(bool),
                "cross_instrument_available": cross_available,
                "strict_cross_instrument": strict_cross,
            }
            panel_report: dict[str, object] = {
                "queries": int(len(panel["query_row"])),
                "coordinate_semantics": (
                    "one candidate molecule is one local reference coordinate; replicate reference "
                    "spectra are aggregated by maximum score within the frozen 10-ppm neighbourhood"
                ),
                "subset_sizes": {key: int(np.sum(value)) for key, value in masks.items()},
                "methods": {},
            }
            table = pd.DataFrame(
                {
                    "query_index": np.arange(len(panel["query_row"])),
                    "query_row": panel["query_row"],
                    "query_ik14_evaluation_only": panel["query_ik14"],
                    "query_formula_evaluation_only": panel["query_formula"],
                    "near_structure": panel["near_query"].astype(bool),
                    "cross_instrument_available": cross_available,
                    "strict_cross_instrument": strict_cross,
                }
            )
            outcomes: dict[str, dict[str, np.ndarray]] = {}
            for method_index, method in enumerate(methods):
                outcome = evaluate_method(panel, score_arrays[name][method_index])
                outcomes[method] = outcome
                panel_report["methods"][method] = {
                    subset: summarize(outcome, mask) for subset, mask in masks.items()
                }
                table[f"{method}__rank"] = outcome["rank"]
                table[f"{method}__confidence_gap"] = outcome["confidence"]
                table[f"{method}__same_formula_error"] = outcome["same_formula_error"]
            primary = outcomes[args.primary_method]
            baseline = outcomes[args.baseline_method]
            comparisons = {}
            for subset_index, (subset, mask) in enumerate(masks.items()):
                comparisons[subset] = {
                    "recall1": cluster_bootstrap_delta(
                        primary["correct"], baseline["correct"], panel["query_formula"], mask,
                        args.bootstrap_resamples,
                        args.bootstrap_seed + panel_index * 1000 + subset_index * 100,
                    ),
                    "same_formula_false_merge": cluster_bootstrap_delta(
                        primary["same_formula_error"], baseline["same_formula_error"],
                        panel["query_formula"], mask, args.bootstrap_resamples,
                        args.bootstrap_seed + panel_index * 1000 + subset_index * 100 + 1,
                    ),
                }
            panel_report["primary_vs_baseline"] = comparisons
            report["panels"][name] = panel_report
            table.to_csv(staging / f"per_query_{name}.csv.gz", index=False, compression="gzip")

        identity = report["panels"]["identity_disjoint"]
        strict_metrics = identity["methods"][args.primary_method]["strict_cross_instrument"]
        strict_delta = identity["primary_vs_baseline"]["strict_cross_instrument"]["recall1"]
        near_false_merge = identity["primary_vs_baseline"]["near_structure"][
            "same_formula_false_merge"
        ]
        formula_delta = report["panels"]["formula_disjoint"]["primary_vs_baseline"]["all"][
            "recall1"
        ]
        gates = {
            "strict_cross_instrument_queries_ge_1000": strict_metrics["queries"] >= 1000,
            "strict_cross_instrument_recall1_ge_0p80": strict_metrics["recall@1"] >= 0.80,
            "strict_cross_instrument_risk_at_60pct_le_0p02": strict_metrics[
                "selective_assignment"
            ]["risk_at_60pct"] <= 0.02,
            "primary_strict_cross_instrument_delta_ci_low_gt_0": strict_delta[
                "ci95_low_pp"
            ] > 0.0,
            "primary_near_false_merge_delta_ci_high_le_0": near_false_merge[
                "ci95_high_pp"
            ] <= 0.0,
            "primary_formula_disjoint_delta_ci_low_gt_0": formula_delta["ci95_low_pp"] > 0.0,
        }
        viability_keys = [
            "strict_cross_instrument_queries_ge_1000",
            "strict_cross_instrument_recall1_ge_0p80",
            "strict_cross_instrument_risk_at_60pct_le_0p02",
        ]
        advancement_keys = [
            "primary_strict_cross_instrument_delta_ci_low_gt_0",
            "primary_near_false_merge_delta_ci_high_le_0",
            "primary_formula_disjoint_delta_ci_low_gt_0",
        ]
        report["preregistered_gates"] = gates
        report["decision"] = {
            "coordinate_viability_pass": all(gates[key] for key in viability_keys),
            "project_mapper_advancement_pass": all(gates[key] for key in advancement_keys),
            "interpretation": (
                "Viability authorizes a later global reference-coordinate benchmark; advancement "
                "requires the project mapper to add value over official DreaMS without increasing "
                "near-structure false merges. Failure does not erase ordinary retrieval results."
            ),
        }
        report["claim_limit"] = (
            "P0-L evaluates local, precursor-mass-conditioned coordinate identifiability on a "
            "labelled library. It is not a cross-study feature-alignment result, a biological "
            "association, a global atlas, an independent model-selection result, or evidence that "
            "unknown entities improve phenotyping."
        )
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (staging / "README.txt").write_text(
            "P0-L hidden-name spectral-coordinate identifiability. Read report.json before using "
            "per-query tables. Labels in those tables are evaluation-only.\n",
            encoding="utf-8",
        )
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report["decision"], indent=2), flush=True)
    print(f"P0-L complete: {args.output}", flush=True)


if __name__ == "__main__":
    main()
