#!/usr/bin/env python
"""Evaluate frozen BioAware B30 on a complete real-library retrieval ranking.

The primary analysis has one vote per physical query spectrum.  Official
DreaMS ranks every molecule in the strict 10-ppm, same-adduct candidate set by
the maximum cosine over that molecule's reference spectra.  Frozen BioAware
B30 is an action router rather than a calibrated all-candidate scorer, so its
complete ranking is defined conservatively: when B30 intervenes, its selected
candidate is promoted to rank one and every other DreaMS ordering relation is
preserved.  Positive-ion queries are an explicit abstention stratum because
B30 was qualified only for negative ion mode.

This opened, cross-fitted multicohort benchmark is the closest currently
available deployment-like evaluation.  It is not an independent blind test
and its candidate-level AUROC is not the NIST20 spectrum-pair ledger used for
the approximately 0.85 AUROC in the DreaMS paper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_bioaware_b12_multicohort_catalog_action import build_universe  # noqa: E402


EXPECTED_NEGATIVE_ROWS = 860
EXPECTED_NEGATIVE_PHYSICAL = 753
EXPECTED_POSITIVE_ROWS = 878
EXPECTED_ALL_ROWS = 1738
EXPECTED_LIBRARY_SPECTRA = 259176
EXPECTED_LIBRARY_IDENTITIES = 30984
TOP_K = (1, 2, 5, 10, 20)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--internal-candidates", type=Path,
        default=ROOT / "data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--st-candidates", type=Path,
        default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--st-queries", type=Path,
        default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/per_query.csv.gz",
    )
    parser.add_argument(
        "--kgmn-candidates", type=Path,
        default=ROOT / "data/validation/bioaware_kgmn200std_hidden_seed_v1/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--kgmn-seeds", type=Path,
        default=ROOT / "data/validation/bioaware_kgmn200std_confirmation_manifest_v2/seed_features.csv.gz",
    )
    parser.add_argument(
        "--b30-transitions", type=Path,
        default=ROOT / "data/validation/bioaware_b30_cross_source_sink_veto_localcheck_20260907_v1/nested_sink_veto_transitions.csv.gz",
    )
    parser.add_argument(
        "--b30-report", type=Path,
        default=ROOT / "data/validation/bioaware_b30_cross_source_sink_veto_localcheck_20260907_v1/report.json",
    )
    parser.add_argument(
        "--b33-graph", type=Path,
        default=ROOT / "data/validation/bioaware_b33_full_reference_graph_frozen_20260907_v2/full_candidate_graph.npz",
    )
    parser.add_argument(
        "--b33-report", type=Path,
        default=ROOT / "data/validation/bioaware_b33_full_reference_graph_frozen_20260907_v2/report.json",
    )
    parser.add_argument(
        "--library-report", type=Path,
        default=ROOT / "data/validation/bioaware_full16_action_support_m0_v7_20260906/report.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260912)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, body: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(body, handle, indent=2, sort_keys=True, allow_nan=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def pooled_binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Standard pooled ROC-AUC with half credit for score ties."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    positive = scores[labels]
    negative = scores[~labels]
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("AUROC requires positive and negative observations")
    combined = np.concatenate([positive, negative])
    ranks = pd.Series(combined).rank(method="average").to_numpy(float)
    rank_sum_positive = float(ranks[: len(positive)].sum())
    u = rank_sum_positive - len(positive) * (len(positive) + 1) / 2
    return float(u / (len(positive) * len(negative)))


def strict_ranks(
    candidate_ids: np.ndarray,
    scores: np.ndarray,
    truth: str,
    promoted: str | None = None,
) -> tuple[int, float, float, np.ndarray]:
    """Return strict rank, query AUROC, pair credits and canonical scores.

    Ties count against the true candidate for ranks.  AUROC follows the
    standard convention and gives ties half credit.  If ``promoted`` is set,
    it is made the unique top candidate while all other scores remain exactly
    unchanged.  The returned canonical score vector is used only for the
    explicitly labelled candidate-level pooled AUROC.
    """
    candidate_ids = np.asarray(candidate_ids).astype(str)
    scores = np.asarray(scores, dtype=float)
    truth_mask = candidate_ids == str(truth)
    if int(truth_mask.sum()) != 1:
        raise RuntimeError(f"truth multiplicity is not one for {truth}")
    if len(candidate_ids) < 2 or not np.isfinite(scores).all():
        raise RuntimeError("candidate group is invalid")
    canonical = scores.copy()
    if promoted is not None:
        promoted_mask = candidate_ids == str(promoted)
        if int(promoted_mask.sum()) != 1:
            raise RuntimeError(f"promoted candidate missing or duplicated: {promoted}")
        maximum = float(scores.max())
        canonical[promoted_mask] = np.nextafter(maximum, math.inf)
    truth_score = float(canonical[truth_mask][0])
    negative_scores = canonical[~truth_mask]
    rank = 1 + int(np.sum(negative_scores >= truth_score))
    credits = (truth_score > negative_scores).astype(float)
    credits += 0.5 * (truth_score == negative_scores)
    return rank, float(credits.mean()), float(credits.sum()), canonical


def build_rank_ledger(
    candidates: pd.DataFrame, transitions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "query_id", "candidate_id", "truth_candidate_id", "truth_formula",
        "spectral_score", "source", "polarity", "baseline_candidate_id",
        "baseline_unique", "baseline_correct", "reference_spectra",
    }
    missing = required - set(candidates.columns)
    if missing:
        raise RuntimeError(f"candidate universe missing columns: {sorted(missing)}")
    if candidates.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("duplicate query/candidate rows")
    negative = candidates.loc[candidates["polarity"].eq("negative")]
    positive = candidates.loc[candidates["polarity"].eq("positive")]
    if negative["query_id"].nunique() != EXPECTED_NEGATIVE_ROWS:
        raise RuntimeError("negative query coverage changed")
    if positive["query_id"].nunique() != EXPECTED_POSITIVE_ROWS:
        raise RuntimeError("positive query coverage changed")

    transition_required = {
        "query_id", "source", "physical_query_id", "truth_candidate_id",
        "truth_formula", "baseline_candidate_id", "final_candidate_id",
        "baseline_correct", "final_correct", "intervene", "corrected",
        "introduced", "b30_veto",
    }
    missing = transition_required - set(transitions.columns)
    if missing:
        raise RuntimeError(f"B30 transitions missing columns: {sorted(missing)}")
    if len(transitions) != EXPECTED_NEGATIVE_ROWS or transitions["query_id"].nunique() != len(transitions):
        raise RuntimeError("B30 transition coverage changed")
    decisions = transitions.set_index("query_id")
    if not decisions.index.is_unique:
        raise RuntimeError("B30 transition query IDs are not unique")

    rows: list[dict] = []
    candidate_parts: list[pd.DataFrame] = []
    for query_id, group in candidates.groupby("query_id", sort=False):
        query_id = str(query_id)
        group = group.copy()
        truth_values = group["truth_candidate_id"].astype(str).unique()
        baseline_values = group["baseline_candidate_id"].astype(str).unique()
        if len(truth_values) != 1 or len(baseline_values) != 1:
            raise RuntimeError(f"{query_id}: inconsistent truth or baseline")
        truth = truth_values[0]
        baseline = baseline_values[0]
        polarity = str(group["polarity"].iloc[0])
        source = str(group["source"].iloc[0])
        if polarity == "negative":
            if query_id not in decisions.index:
                raise RuntimeError(f"{query_id}: missing B30 decision")
            decision = decisions.loc[query_id]
            if source != str(decision["source"]):
                raise RuntimeError(f"{query_id}: source mismatch")
            for column, value in (
                ("truth_candidate_id", truth),
                ("baseline_candidate_id", baseline),
            ):
                if str(decision[column]) != value:
                    raise RuntimeError(f"{query_id}: {column} mismatch")
            final = str(decision["final_candidate_id"])
            physical_query_id = str(decision["physical_query_id"])
            intervene = bool(decision["intervene"])
            b30_veto = bool(decision["b30_veto"])
            expected_final_correct = bool(decision["final_correct"])
            evaluation_stratum = "negative_module_scope"
        else:
            final = baseline
            physical_query_id = query_id
            intervene = False
            b30_veto = False
            expected_final_correct = bool(group["baseline_correct"].iloc[0])
            evaluation_stratum = "positive_explicit_abstention"

        candidate_ids = group["candidate_id"].astype(str).to_numpy()
        scores = group["spectral_score"].to_numpy(float)
        official_rank, official_auc, official_credit, official_scores = strict_ranks(
            candidate_ids, scores, truth, promoted=None
        )
        final_rank, final_auc, final_credit, final_scores = strict_ranks(
            candidate_ids, scores, truth, promoted=final if intervene else None
        )
        baseline_score_rows = scores[candidate_ids == baseline]
        if len(baseline_score_rows) != 1 or not np.isclose(
            float(baseline_score_rows[0]), float(scores.max()), rtol=0, atol=1e-12
        ):
            raise RuntimeError(f"{query_id}: declared baseline candidate is not a maximum")
        baseline_unique_recomputed = int(np.sum(np.isclose(
            scores, float(scores.max()), rtol=0, atol=1e-12
        ))) == 1
        if baseline_unique_recomputed != bool(group["baseline_unique"].iloc[0]):
            raise RuntimeError(f"{query_id}: baseline uniqueness replay mismatch")
        if (official_rank == 1) != bool(group["baseline_correct"].iloc[0]):
            raise RuntimeError(f"{query_id}: official rank replay mismatch")
        if (final_rank == 1) != expected_final_correct:
            raise RuntimeError(f"{query_id}: BioAware promotion replay mismatch")

        physical_key = source + "::" + physical_query_id
        rows.append({
            "query_id": query_id,
            "physical_key": physical_key,
            "source": source,
            "polarity": polarity,
            "evaluation_stratum": evaluation_stratum,
            "truth_candidate_id": truth,
            "truth_formula": str(group["truth_formula"].iloc[0]),
            "baseline_candidate_id": baseline,
            "bioaware_candidate_id": final,
            "intervene": intervene,
            "b30_veto": b30_veto,
            "candidate_count": int(len(group)),
            "reference_spectra_in_candidate_group": int(group["reference_spectra"].astype(float).sum()),
            "official_rank": official_rank,
            "bioaware_rank": final_rank,
            "official_reciprocal_rank": 1.0 / official_rank,
            "bioaware_reciprocal_rank": 1.0 / final_rank,
            "official_query_auroc": official_auc,
            "bioaware_query_auroc": final_auc,
            "official_pair_credit_sum": official_credit,
            "bioaware_pair_credit_sum": final_credit,
            "within_query_negative_pairs": int(len(group) - 1),
        })
        local_candidates = group[["query_id", "candidate_id", "truth_candidate_id"]].copy()
        local_candidates["is_truth"] = candidate_ids == truth
        local_candidates["official_score"] = official_scores
        local_candidates["bioaware_minimal_promotion_score"] = final_scores
        candidate_parts.append(local_candidates)

    query_ledger = pd.DataFrame(rows)
    candidate_ledger = pd.concat(candidate_parts, ignore_index=True)
    if len(query_ledger) != EXPECTED_ALL_ROWS or query_ledger["query_id"].nunique() != EXPECTED_ALL_ROWS:
        raise RuntimeError("complete query coverage changed")
    return query_ledger, candidate_ledger


def metric_summary(query_frame: pd.DataFrame, candidate_frame: pd.DataFrame) -> dict:
    output: dict[str, dict | float | int] = {}
    for model, rank_column, rr_column, auc_column, credit_column, score_column in (
        (
            "official_dreams", "official_rank", "official_reciprocal_rank",
            "official_query_auroc", "official_pair_credit_sum", "official_score",
        ),
        (
            "dreams_plus_bioaware_b30", "bioaware_rank", "bioaware_reciprocal_rank",
            "bioaware_query_auroc", "bioaware_pair_credit_sum",
            "bioaware_minimal_promotion_score",
        ),
    ):
        ranks = query_frame[rank_column].to_numpy(int)
        metrics = {f"recall_at_{k}": float(np.mean(ranks <= k)) for k in TOP_K}
        metrics.update({
            "mrr": float(query_frame[rr_column].mean()),
            "mean_rank": float(np.mean(ranks)),
            "median_rank": float(np.median(ranks)),
            "macro_query_auroc": float(query_frame[auc_column].mean()),
            "micro_within_query_auroc": float(
                query_frame[credit_column].sum()
                / query_frame["within_query_negative_pairs"].sum()
            ),
            "pooled_candidate_auroc": pooled_binary_auroc(
                candidate_frame[score_column].to_numpy(float),
                candidate_frame["is_truth"].to_numpy(bool),
            ),
        })
        output[model] = metrics

    delta = {
        key: float(output["dreams_plus_bioaware_b30"][key] - output["official_dreams"][key])
        for key in output["official_dreams"]
    }
    transitions: dict[str, dict[str, int]] = {}
    for k in TOP_K:
        before = query_frame["official_rank"].le(k)
        after = query_frame["bioaware_rank"].le(k)
        transitions[f"recall_at_{k}"] = {
            "corrected": int((~before & after).sum()),
            "introduced": int((before & ~after).sum()),
            "net": int(after.sum() - before.sum()),
        }
    output["delta_bioaware_minus_dreams"] = delta
    output["rank_transitions"] = transitions
    return output


def cluster_bootstrap_deltas(
    frame: pd.DataFrame, cluster: str, repeats: int, seed: int,
) -> dict:
    columns: dict[str, np.ndarray] = {}
    for k in TOP_K:
        columns[f"recall_at_{k}"] = (
            frame["bioaware_rank"].le(k).astype(float)
            - frame["official_rank"].le(k).astype(float)
        ).to_numpy(float)
    columns["mrr"] = (
        frame["bioaware_reciprocal_rank"] - frame["official_reciprocal_rank"]
    ).to_numpy(float)
    columns["macro_query_auroc"] = (
        frame["bioaware_query_auroc"] - frame["official_query_auroc"]
    ).to_numpy(float)
    work = frame[[cluster]].copy()
    grouped_indices = [np.asarray(index, dtype=int) for index in work.groupby(cluster, sort=False).indices.values()]
    if len(grouped_indices) < 2:
        raise RuntimeError(f"too few {cluster} clusters")
    rng = np.random.default_rng(seed)
    draws = {metric: np.empty(repeats, dtype=float) for metric in columns}
    for draw_index in range(repeats):
        selected = rng.integers(0, len(grouped_indices), len(grouped_indices))
        row_index = np.concatenate([grouped_indices[position] for position in selected])
        for metric, values in columns.items():
            draws[metric][draw_index] = float(values[row_index].mean())
    return {
        "cluster": cluster,
        "clusters": int(len(grouped_indices)),
        "resamples": int(repeats),
        "metrics": {
            metric: {
                "mean_delta": float(values.mean()),
                "ci_low": float(np.quantile(draws[metric], 0.025)),
                "ci_high": float(np.quantile(draws[metric], 0.975)),
            }
            for metric, values in columns.items()
        },
    }


def mcnemar_exact(corrected: int, introduced: int) -> float:
    discordant = int(corrected + introduced)
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(corrected, introduced) + 1))
    return float(min(1.0, 2.0 * tail / (2**discordant)))


def physical_primary(frame: pd.DataFrame) -> pd.DataFrame:
    immutable = (
        "truth_candidate_id", "truth_formula", "baseline_candidate_id",
        "bioaware_candidate_id", "candidate_count", "official_rank",
        "bioaware_rank", "official_query_auroc", "bioaware_query_auroc",
    )
    for column in immutable:
        maximum = int(frame.groupby("physical_key")[column].nunique(dropna=False).max())
        if maximum != 1:
            raise RuntimeError(f"physical duplicate mismatch: {column}")
    return frame.drop_duplicates("physical_key", keep="first").reset_index(drop=True)


def candidate_subset(candidates: pd.DataFrame, query_ids: set[str]) -> pd.DataFrame:
    return candidates.loc[candidates["query_id"].astype(str).isin(query_ids)].copy()


def panel_report(
    query_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    repeats: int,
    seed: int,
) -> dict:
    query_ids = set(query_frame["query_id"].astype(str))
    candidates = candidate_subset(candidate_frame, query_ids)
    metrics = metric_summary(query_frame, candidates)
    r1 = metrics["rank_transitions"]["recall_at_1"]
    return {
        "n_queries": int(len(query_frame)),
        "n_truth_identities": int(query_frame["truth_candidate_id"].nunique()),
        "n_truth_formulas": int(query_frame["truth_formula"].nunique()),
        "n_candidate_rows": int(len(candidates)),
        "candidate_count": {
            "minimum": int(query_frame["candidate_count"].min()),
            "median": float(query_frame["candidate_count"].median()),
            "p90": float(query_frame["candidate_count"].quantile(0.9)),
            "maximum": int(query_frame["candidate_count"].max()),
        },
        "interventions": int(query_frame["intervene"].sum()),
        "intervention_rate": float(query_frame["intervene"].mean()),
        "metrics": metrics,
        "recall_at_1_mcnemar_exact_p": mcnemar_exact(r1["corrected"], r1["introduced"]),
        "formula_cluster_bootstrap": cluster_bootstrap_deltas(
            query_frame, "truth_formula", repeats, seed
        ),
        "identity_cluster_bootstrap": cluster_bootstrap_deltas(
            query_frame, "truth_candidate_id", repeats, seed + 1
        ),
    }


def validate_frozen_inputs(
    args: argparse.Namespace, candidates: pd.DataFrame, query_ledger: pd.DataFrame,
) -> dict:
    b30 = json.loads(args.b30_report.read_text(encoding="utf-8"))
    b33 = json.loads(args.b33_report.read_text(encoding="utf-8"))
    library = json.loads(args.library_report.read_text(encoding="utf-8"))
    if b30.get("status") != "bioaware_b30_cross_source_sink_veto_complete":
        raise RuntimeError("B30 report status changed")
    if b33.get("status") != "bioaware_b33_full_candidate_graph_complete":
        raise RuntimeError("B33 report status changed")
    if library.get("status") != "bioaware_full16_action_support_m0_complete":
        raise RuntimeError("library report status changed")
    if not b30.get("strictly_better_action_than_B17"):
        raise RuntimeError("B30 no longer passes its frozen action gate")
    library_stats = library["candidate_library"]
    if int(library_stats["reference_rows"]) != EXPECTED_LIBRARY_SPECTRA:
        raise RuntimeError("reference library spectrum count changed")
    if int(library_stats["identities"]) != EXPECTED_LIBRARY_IDENTITIES:
        raise RuntimeError("reference library identity count changed")
    if int(b33["queries"]) != EXPECTED_NEGATIVE_ROWS or int(b33["reference_spectra"]) != 3515:
        raise RuntimeError("B33 full-reference graph size changed")

    graph = np.load(args.b33_graph, allow_pickle=False)
    required = {
        "query_id", "candidate_ptr", "candidate_id", "candidate_official_score",
        "deployed_final_candidate_id", "baseline_rank_full_graph",
    }
    if not required.issubset(graph.files):
        raise RuntimeError("B33 graph arrays changed")
    negative = candidates.loc[candidates["polarity"].eq("negative")].copy()
    negative_index = negative.set_index(["query_id", "candidate_id"])
    if not negative_index.index.is_unique:
        raise RuntimeError("negative candidate universe index is not unique")
    score_errors: list[float] = []
    for query_position, query_id in enumerate(graph["query_id"].astype(str)):
        start = int(graph["candidate_ptr"][query_position])
        stop = int(graph["candidate_ptr"][query_position + 1])
        graph_ids = graph["candidate_id"][start:stop].astype(str)
        graph_scores = graph["candidate_official_score"][start:stop].astype(float)
        for candidate_id, graph_score in zip(graph_ids, graph_scores, strict=True):
            try:
                local_score = float(negative_index.loc[(query_id, candidate_id), "spectral_score"])
            except KeyError as error:
                raise RuntimeError(f"B33 candidate absent from universe: {query_id}/{candidate_id}") from error
            score_errors.append(abs(local_score - float(graph_score)))
    if len(score_errors) != int(b33["candidate_rows"]):
        raise RuntimeError("B33 candidate-row replay coverage changed")
    maximum_score_error = max(score_errors, default=0.0)
    if maximum_score_error > 2e-6:
        raise RuntimeError(f"B33 official score replay failed: {maximum_score_error}")

    negative_ledger = query_ledger.loc[query_ledger["polarity"].eq("negative")]
    official_r1 = float(negative_ledger["official_rank"].eq(1).mean())
    bioaware_r1 = float(negative_ledger["bioaware_rank"].eq(1).mean())
    if not math.isclose(official_r1, 0.6604651162790698, abs_tol=1e-15):
        raise RuntimeError(f"B30 official Recall@1 replay failed: {official_r1}")
    if not math.isclose(bioaware_r1, 0.7197674418604652, abs_tol=1e-15):
        raise RuntimeError(f"B30 final Recall@1 replay failed: {bioaware_r1}")
    if int(b33["official"]["errors"]) != int((negative_ledger["official_rank"] != 1).sum()):
        raise RuntimeError("B33 official error count replay failed")
    return {
        "B30_negative_recall1_reproduced": True,
        "B33_complete_candidate_graph_reproduced": True,
        "B33_maximum_candidate_score_error": maximum_score_error,
        "library_reference_spectra": int(library_stats["reference_rows"]),
        "library_identities": int(library_stats["identities"]),
        "library_formula_values_available": int(library_stats["formula_values_available"]),
        "full_reference_graph_spectra": int(b33["reference_spectra"]),
        "full_reference_links": int(b33["candidate_reference_links"]),
    }


def main() -> None:
    args = arguments()
    if args.bootstrap_resamples < 1000:
        raise ValueError("formal B35 requires at least 1,000 bootstrap resamples")
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {args.output_dir}")
    dependencies = (
        args.internal_candidates, args.st_candidates, args.st_queries,
        args.kgmn_candidates, args.kgmn_seeds, args.b30_transitions,
        args.b30_report, args.b33_graph, args.b33_report, args.library_report,
    )
    for path in dependencies:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    candidates, universe_provenance = build_universe(args)
    transitions = pd.read_csv(args.b30_transitions)
    query_ledger, candidate_ledger = build_rank_ledger(candidates, transitions)
    physical = physical_primary(query_ledger)
    expected_physical = EXPECTED_NEGATIVE_PHYSICAL + EXPECTED_POSITIVE_ROWS
    if len(physical) != expected_physical:
        raise RuntimeError(f"physical-query coverage changed: {len(physical)} != {expected_physical}")
    input_gate = validate_frozen_inputs(args, candidates, query_ledger)

    panels: dict[str, dict] = {}
    selections = {
        "primary_all_real_physical_queries": physical,
        "negative_module_scope_physical_queries": physical.loc[physical["polarity"].eq("negative")],
        "positive_abstention_queries": physical.loc[physical["polarity"].eq("positive")],
        "row_weighted_historical_replay": query_ledger,
    }
    for index, (name, frame) in enumerate(selections.items()):
        panels[name] = panel_report(
            frame.reset_index(drop=True), candidate_ledger,
            args.bootstrap_resamples, args.seed + 10 * index,
        )
        print(
            f"[B35 {name}] n={len(frame):,} "
            f"R1={panels[name]['metrics']['official_dreams']['recall_at_1']:.4f}->"
            f"{panels[name]['metrics']['dreams_plus_bioaware_b30']['recall_at_1']:.4f}",
            flush=True,
        )

    by_source = {
        source: panel_report(
            group.reset_index(drop=True), candidate_ledger,
            args.bootstrap_resamples, args.seed + 100 + index,
        )
        for index, (source, group) in enumerate(physical.groupby("source", sort=True))
    }
    primary = panels["primary_all_real_physical_queries"]
    primary_r1 = primary["metrics"]["rank_transitions"]["recall_at_1"]
    formula_r1_ci = primary["formula_cluster_bootstrap"]["metrics"]["recall_at_1"]
    gates = {
        "all_1738_rows_and_expected_physical_queries": (
            len(query_ledger) == EXPECTED_ALL_ROWS and len(physical) == expected_physical
        ),
        "negative_B30_recall1_exact": input_gate["B30_negative_recall1_reproduced"],
        "B33_full_candidate_scores_exact": input_gate["B33_complete_candidate_graph_reproduced"],
        "positive_polarity_is_strict_abstention": bool(
            (physical.loc[physical["polarity"].eq("positive"), "official_rank"].to_numpy()
             == physical.loc[physical["polarity"].eq("positive"), "bioaware_rank"].to_numpy()).all()
        ),
        "primary_recall1_positive": primary_r1["net"] > 0,
        "primary_corrected_gt_introduced": primary_r1["corrected"] > primary_r1["introduced"],
        "primary_formula_cluster_recall1_ci_low_positive": formula_r1_ci["ci_low"] > 0,
        "primary_recall_at_2_5_10_20_nonnegative": all(
            primary["metrics"]["delta_bioaware_minus_dreams"][f"recall_at_{k}"] >= 0
            for k in (2, 5, 10, 20)
        ),
        "primary_mrr_nonnegative": primary["metrics"]["delta_bioaware_minus_dreams"]["mrr"] >= 0,
        "primary_macro_query_auroc_nonnegative": (
            primary["metrics"]["delta_bioaware_minus_dreams"]["macro_query_auroc"] >= 0
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    per_query_path = args.output_dir / "per_query.csv.gz"
    query_ledger.to_csv(per_query_path, index=False, compression="gzip")
    report = {
        "status": "bioaware_b35_real_library_retrieval_complete",
        "formal": True,
        "primary_estimand": (
            "one vote per physical real MS/MS query across six cohorts and both ionisation "
            "polarities; B30 acts only in qualified negative mode and abstains in positive mode"
        ),
        "retrieval_protocol": {
            "search_library": (
                "259,176 reference spectra / 30,984 molecular identities before query-local filtering"
            ),
            "candidate_filter": "strict 10 ppm, same adduct, at least two candidate molecules",
            "molecule_score": "maximum official DreaMS cosine over all reference spectra of the molecule",
            "tie_policy": "every negative tied with the truth counts ahead of the truth",
            "bioaware_full_ranking": (
                "promote the frozen B30-selected molecule to unique rank 1; preserve every "
                "DreaMS ordering relation among all other candidates"
            ),
            "positive_mode_policy": "explicit abstention; official DreaMS ranking is unchanged",
        },
        "auc_definitions": {
            "macro_query_auroc": (
                "standard one-positive-vs-candidate-negatives AUROC computed within each query, "
                "then averaged with equal query weight"
            ),
            "micro_within_query_auroc": (
                "all within-query truth-vs-negative ordering comparisons pooled; queries with "
                "more candidates contribute more comparisons"
            ),
            "pooled_candidate_auroc": (
                "ROC-AUC after flattening candidate rows; BioAware is scalarised only by the "
                "minimal score promotion needed to realise its frozen top-1 action"
            ),
            "DreaMS_paper_approximately_0_85": (
                "pooled ROC-AUC on NIST20 individual spectrum pairs under the paper's pair ledger"
            ),
            "exactly_comparable_to_DreaMS_paper_0_85": False,
            "reason_not_exactly_comparable": (
                "B35 ranks molecule candidates by max-over-reference spectra on different real "
                "cohorts; B30 is a candidate action router, not a calibrated spectrum-pair cosine"
            ),
        },
        "input_reproduction_gates": input_gate,
        "panels": panels,
        "by_source_physical_queries": by_source,
        "gates": gates,
        "pass_opened_real_library_benchmark": bool(all(gates.values())),
        "contracts": {
            "frozen_B30_only": True,
            "model_fitted_during_evaluation": False,
            "positive_outcomes_used_to_tune_B30": False,
            "nonpromoted_candidate_order_changed": False,
            "P2b_used": False,
            "phenotype_used": False,
            "shared_embedding_changed": False,
            "opened_development_queries": True,
            "independent_blind_test": False,
        },
        "provenance": {
            **universe_provenance["provenance"],
            "B30_transitions_sha256": sha256(args.b30_transitions),
            "B30_report_sha256": sha256(args.b30_report),
            "B33_graph_sha256": sha256(args.b33_graph),
            "B33_report_sha256": sha256(args.b33_report),
            "library_report_sha256": sha256(args.library_report),
            "per_query_sha256": sha256(per_query_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "B35 is a large-reference-library, real-query, opened multicohort cross-fitted "
            "benchmark. A positive result supports a deployment-like development comparison "
            "against official DreaMS; it does not establish independent external validation, "
            "the exact NIST20 paper AUROC, reaction causality, shared-embedding improvement, "
            "or SOTA."
        ),
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if not report["pass_opened_real_library_benchmark"]:
        raise RuntimeError(f"B35 scientific gate failed: {gates}")


if __name__ == "__main__":
    main()
