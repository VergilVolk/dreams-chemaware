#!/usr/bin/env python3
"""Describe where frozen CALMS v1.0 changes WSE and DreaMS decisions.

This is a read-only, post hoc analysis of the consumed GNPS development
panels.  It does not fit a model, select a threshold, or create a deployment
router.  Every stratum is fixed in this file, and the only data-derived bins
are quartiles of each comparator's truth-blind Top-1--Top-2 score gap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CALMS_METHOD = "full23_v132"
BASELINE_METHODS = ("weighted_spectral_entropy", "official_dreams")
REQUIRED_METHODS = (*BASELINE_METHODS, CALMS_METHOD)
MARGIN_LABELS = ("Q1_low", "Q2", "Q3", "Q4_high")
EVENT_LABELS = ("corrected", "introduced", "retained", "both_wrong")


@dataclass(frozen=True)
class PanelSpec:
    name: str
    evaluation_directory: str
    score_filename: str
    panel_path: Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        default=Path("data/validation/gnps_run17_mega_evaluation_v1_20261102"),
    )
    parser.add_argument(
        "--score-root",
        type=Path,
        default=Path("data/validation/gnps_run17_mega_evaluation_inputs_20261102"),
    )
    parser.add_argument(
        "--identity-panel",
        type=Path,
        default=Path(
            "data/validation/GLM_gnps_identity_panel_reconstruction/"
            "panel_identity_disjoint.npz"
        ),
    )
    parser.add_argument(
        "--formula-panel",
        type=Path,
        default=Path(
            "data/validation/gnps_gold_silver_10ppm_benchmark_v1/"
            "panel_formula_disjoint.npz"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/validation/gnps_gold_silver_10ppm_benchmark_v1/manifest.csv.gz"
        ),
    )
    parser.add_argument(
        "--panels",
        nargs="+",
        choices=("identity_disjoint", "formula_disjoint"),
        default=("identity_disjoint", "formula_disjoint"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261105)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(array: np.ndarray, name: str) -> object:
    if np.asarray(array).shape != ():
        raise RuntimeError(f"{name} must be a scalar")
    return np.asarray(array).item()


def string_array(values: Iterable[object]) -> np.ndarray:
    return np.char.strip(np.asarray(list(values)).astype("U"))


def boolean_array(values: pd.Series, name: str) -> np.ndarray:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.to_numpy(dtype=bool)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all() and set(numeric.unique()).issubset({0, 1}):
        return numeric.to_numpy(dtype=np.int8).astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not normalized.isin(mapping).all():
        raise RuntimeError(f"{name} is not a strict Boolean column")
    return normalized.map(mapping).to_numpy(dtype=bool)


def panel_specs(args: argparse.Namespace) -> dict[str, PanelSpec]:
    return {
        "identity_disjoint": PanelSpec(
            "identity_disjoint",
            "identity",
            "scores_identity_disjoint.npz",
            args.identity_panel,
        ),
        "formula_disjoint": PanelSpec(
            "formula_disjoint",
            "formula",
            "scores_formula_disjoint.npz",
            args.formula_panel,
        ),
    }


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "row",
        "ik14",
        "formula",
        "precursor_mz",
        "library_quality",
        "n_peaks",
        "instrument",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"manifest is missing columns: {missing}")
    if frame["row"].isna().any() or frame["row"].duplicated().any():
        raise RuntimeError("manifest row must be unique and non-missing")
    frame["row"] = pd.to_numeric(frame["row"], errors="raise").astype(np.int64)
    return frame.set_index("row", drop=False)


def load_panel(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    required = {
        "query_row",
        "query_ik14",
        "query_formula",
        "query_precursor_mz",
        "query_ptr",
        "molecule_ik14",
        "molecule_formula",
        "molecule_label",
        "molecule_ptr",
        "candidate_row",
        "near_query",
    }
    with np.load(path, allow_pickle=False) as body:
        missing = sorted(required - set(body.files))
        if missing:
            raise RuntimeError(f"panel is missing arrays: {missing}")
        panel = {key: np.asarray(body[key]) for key in body.files}

    query_ptr = np.asarray(panel["query_ptr"], dtype=np.int64)
    molecule_ptr = np.asarray(panel["molecule_ptr"], dtype=np.int64)
    labels = np.asarray(panel["molecule_label"], dtype=bool)
    n_queries = len(panel["query_row"])
    n_edges = len(labels)
    n_pairs = len(panel["candidate_row"])
    if len(query_ptr) != n_queries + 1 or len(molecule_ptr) != n_edges + 1:
        raise RuntimeError("panel pointer lengths are inconsistent")
    if (
        query_ptr[0] != 0
        or query_ptr[-1] != n_edges
        or molecule_ptr[0] != 0
        or molecule_ptr[-1] != n_pairs
        or np.any(np.diff(query_ptr) < 2)
        or np.any(np.diff(molecule_ptr) < 1)
    ):
        raise RuntimeError("panel pointers violate the closed-candidate contract")
    if len(panel["near_query"]) != n_queries:
        raise RuntimeError("near_query length does not match the query axis")
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        if int(np.count_nonzero(labels[int(left) : int(right)])) != 1:
            raise RuntimeError(f"query {query} does not have exactly one positive molecule")
    return panel


def load_score_bundle(
    path: Path, panel_path: Path, panel: dict[str, np.ndarray]
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    required = {
        "schema_version",
        "N_Q",
        "N_E",
        "N_P",
        "panel_sha256",
        "method_names",
        "molecule_scores",
    }
    with np.load(path, allow_pickle=False) as body:
        missing = sorted(required - set(body.files))
        if missing:
            raise RuntimeError(f"score bundle is missing arrays: {missing}")
        names = [str(value) for value in np.asarray(body["method_names"]).tolist()]
        matrix = np.asarray(body["molecule_scores"], dtype=np.float64)
        reported = {
            "schema_version": str(scalar(body["schema_version"], "schema_version")),
            "N_Q": int(scalar(body["N_Q"], "N_Q")),
            "N_E": int(scalar(body["N_E"], "N_E")),
            "N_P": int(scalar(body["N_P"], "N_P")),
            "panel_sha256": str(scalar(body["panel_sha256"], "panel_sha256")),
        }
    if len(names) != len(set(names)):
        raise RuntimeError("score bundle method names are not unique")
    missing_methods = sorted(set(REQUIRED_METHODS) - set(names))
    if missing_methods:
        raise RuntimeError(f"score bundle is missing required methods: {missing_methods}")
    actual_panel_sha = sha256_file(panel_path)
    if reported["panel_sha256"] != actual_panel_sha:
        raise RuntimeError(
            "score bundle panel SHA-256 does not match the selected panel: "
            f"{reported['panel_sha256']} != {actual_panel_sha}"
        )
    expected_counts = {
        "N_Q": len(panel["query_row"]),
        "N_E": len(panel["molecule_label"]),
        "N_P": len(panel["candidate_row"]),
    }
    if any(reported[key] != value for key, value in expected_counts.items()):
        raise RuntimeError(
            f"score bundle counts disagree with panel: {reported} vs {expected_counts}"
        )
    if matrix.shape != (len(names), expected_counts["N_E"]):
        raise RuntimeError("score matrix shape disagrees with method and molecule axes")
    by_method = {name: matrix[index] for index, name in enumerate(names)}
    for name in REQUIRED_METHODS:
        if not np.all(np.isfinite(by_method[name])):
            raise RuntimeError(f"{name} has non-finite molecule scores")
    provenance = {
        **reported,
        "score_bundle_sha256": sha256_file(path),
        "panel_file_sha256": actual_panel_sha,
        "method_names": names,
    }
    return by_method, provenance


def strict_rank(values: np.ndarray, positive_index: int) -> int:
    positive = float(values[positive_index])
    negative = np.delete(values, positive_index)
    return 1 + int(np.count_nonzero(negative >= positive))


def top_two(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 2 or not np.all(np.isfinite(values)):
        raise RuntimeError("Top-1--Top-2 gap requires at least two finite scores")
    pair = np.partition(np.asarray(values, dtype=np.float64), -2)[-2:]
    return float(np.max(pair)), float(np.min(pair))


def validate_query_metrics(
    path: Path,
    panel: dict[str, np.ndarray],
    scores: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required_columns = {
        "method",
        "query_index",
        "query_row",
        "query_ik14",
        "query_formula",
        "near",
        "candidate_molecule_edges",
        "candidate_pair_edges",
        "positive_molecule_edge",
        "positive_score",
        "best_negative_score",
        "rank",
        "reciprocal_rank",
        "macro_query_auroc",
    }
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise RuntimeError(f"query metrics are missing columns: {missing}")
    selected = frame[frame["method"].isin(REQUIRED_METHODS)].copy()
    n_queries = len(panel["query_row"])
    query_ptr = np.asarray(panel["query_ptr"], dtype=np.int64)
    molecule_ptr = np.asarray(panel["molecule_ptr"], dtype=np.int64)
    labels = np.asarray(panel["molecule_label"], dtype=bool)
    expected_index = np.arange(n_queries, dtype=np.int64)
    expected_candidate_count = np.diff(query_ptr)
    expected_pair_count = np.asarray(
        [molecule_ptr[int(right)] - molecule_ptr[int(left)] for left, right in zip(query_ptr[:-1], query_ptr[1:])],
        dtype=np.int64,
    )
    positive_edges = np.empty(n_queries, dtype=np.int64)
    outcomes: dict[str, np.ndarray] = {}
    canonical: pd.DataFrame | None = None

    for method in REQUIRED_METHODS:
        method_frame = selected[selected["method"] == method].sort_values("query_index")
        if len(method_frame) != n_queries or method_frame["query_index"].duplicated().any():
            raise RuntimeError(f"{method} query metrics do not cover the panel exactly once")
        index = pd.to_numeric(method_frame["query_index"], errors="raise").to_numpy(np.int64)
        if not np.array_equal(index, expected_index):
            raise RuntimeError(f"{method} query_index is not the frozen panel order")

        observed_rows = pd.to_numeric(method_frame["query_row"], errors="raise").to_numpy(np.int64)
        observed_ik14 = string_array(method_frame["query_ik14"])
        observed_formula = string_array(method_frame["query_formula"])
        observed_near = boolean_array(method_frame["near"], f"{method}.near")
        observed_candidates = pd.to_numeric(
            method_frame["candidate_molecule_edges"], errors="raise"
        ).to_numpy(np.int64)
        observed_pairs = pd.to_numeric(
            method_frame["candidate_pair_edges"], errors="raise"
        ).to_numpy(np.int64)
        if not (
            np.array_equal(observed_rows, np.asarray(panel["query_row"], dtype=np.int64))
            and np.array_equal(observed_ik14, string_array(panel["query_ik14"]))
            and np.array_equal(observed_formula, string_array(panel["query_formula"]))
            and np.array_equal(observed_near, np.asarray(panel["near_query"], dtype=bool))
            and np.array_equal(observed_candidates, expected_candidate_count)
            and np.array_equal(observed_pairs, expected_pair_count)
        ):
            raise RuntimeError(f"{method} query metrics drifted from the frozen panel axis")

        ranks = np.empty(n_queries, dtype=np.int64)
        positive_scores = np.empty(n_queries, dtype=np.float64)
        best_negative_scores = np.empty(n_queries, dtype=np.float64)
        macro_aurocs = np.empty(n_queries, dtype=np.float64)
        top1_scores = np.empty(n_queries, dtype=np.float64)
        top2_scores = np.empty(n_queries, dtype=np.float64)
        method_scores = np.asarray(scores[method], dtype=np.float64)
        for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
            left, right = int(left), int(right)
            values = method_scores[left:right]
            local_positive = np.flatnonzero(labels[left:right])
            if len(local_positive) != 1:
                raise RuntimeError(f"query {query} positive label drifted")
            positive_index = int(local_positive[0])
            positive_edges[query] = left + positive_index
            ranks[query] = strict_rank(values, positive_index)
            positive_scores[query] = values[positive_index]
            negative = np.delete(values, positive_index)
            best_negative_scores[query] = float(np.max(negative))
            macro_aurocs[query] = (
                float(np.count_nonzero(values[positive_index] > negative))
                + 0.5 * float(np.count_nonzero(values[positive_index] == negative))
            ) / len(negative)
            top1_scores[query], top2_scores[query] = top_two(values)

        observed_rank = pd.to_numeric(method_frame["rank"], errors="raise").to_numpy(np.int64)
        observed_rr = pd.to_numeric(
            method_frame["reciprocal_rank"], errors="raise"
        ).to_numpy(np.float64)
        observed_positive_edge = pd.to_numeric(
            method_frame["positive_molecule_edge"], errors="raise"
        ).to_numpy(np.int64)
        if not (
            np.array_equal(observed_rank, ranks)
            and np.allclose(observed_rr, 1.0 / ranks, atol=1e-12, rtol=0)
            and np.array_equal(observed_positive_edge, positive_edges)
            and np.allclose(
                pd.to_numeric(method_frame["positive_score"], errors="raise"),
                positive_scores,
                atol=2e-6,
                rtol=0,
            )
            and np.allclose(
                pd.to_numeric(method_frame["best_negative_score"], errors="raise"),
                best_negative_scores,
                atol=2e-6,
                rtol=0,
            )
            and np.allclose(
                pd.to_numeric(method_frame["macro_query_auroc"], errors="raise"),
                macro_aurocs,
                atol=2e-6,
                rtol=0,
            )
        ):
            raise RuntimeError(f"{method} query outcomes disagree with molecule scores")
        outcomes[method] = ranks
        outcomes[f"{method}__top1"] = top1_scores
        outcomes[f"{method}__top2"] = top2_scores
        if canonical is None:
            canonical = method_frame.reset_index(drop=True)

    assert canonical is not None
    return canonical, outcomes


def instrument_family(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    if "orbitrap" in text or "hybrid ft" in text or "q-exactive" in text:
        return "Orbitrap_FT"
    if "tof" in text:
        return "TOF_qTOF"
    return "Other_unknown"


def candidate_count_bin(count: int) -> str:
    if 2 <= count <= 3:
        return "2_3"
    if 4 <= count <= 7:
        return "4_7"
    if count >= 8:
        return "8_plus"
    raise RuntimeError(f"invalid closed-set candidate count: {count}")


def peak_count_bin(count: int) -> str:
    if count < 0:
        raise RuntimeError(f"invalid peak count: {count}")
    if count < 50:
        return "lt_50"
    if count < 100:
        return "50_99"
    return "100_plus"


def margin_quartiles(gaps: np.ndarray) -> tuple[np.ndarray, list[float]]:
    gaps = np.asarray(gaps, dtype=np.float64)
    if len(gaps) < 4 or not np.all(np.isfinite(gaps)) or np.any(gaps < -1e-12):
        raise RuntimeError("baseline Top-1--Top-2 gaps cannot define quartiles")
    edges = np.quantile(gaps, [0.0, 0.25, 0.5, 0.75, 1.0]).astype(np.float64)
    if np.any(np.diff(edges) < 0):
        raise AssertionError("quantile edges are not monotone")
    codes = np.searchsorted(edges[1:-1], gaps, side="right")
    labels = np.asarray(MARGIN_LABELS, dtype="U16")[codes]
    return labels, [float(value) for value in edges]


def stable_seed(seed: int, *parts: object) -> int:
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def cluster_bootstrap_delta(
    delta: np.ndarray,
    clusters: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    delta = np.asarray(delta, dtype=np.float64)
    clusters = string_array(clusters)
    point = float(np.mean(delta)) if len(delta) else None
    if len(delta) == 0:
        return {
            "clusters": 0,
            "point": None,
            "ci_low": None,
            "ci_high": None,
            "estimable": False,
        }
    unique, inverse = np.unique(clusters, return_inverse=True)
    counts = np.bincount(inverse).astype(np.int64)
    sums = np.bincount(inverse, weights=delta).astype(np.float64)
    if len(unique) < 2:
        return {
            "clusters": int(len(unique)),
            "point": point,
            "ci_low": None,
            "ci_high": None,
            "estimable": False,
        }
    rng = np.random.default_rng(seed)
    samples = np.empty(repeats, dtype=np.float64)
    batch_size = min(256, repeats)
    offset = 0
    while offset < repeats:
        size = min(batch_size, repeats - offset)
        take = rng.integers(0, len(unique), size=(size, len(unique)))
        samples[offset : offset + size] = (
            sums[take].sum(axis=1) / counts[take].sum(axis=1)
        )
        offset += size
    low, high = np.quantile(samples, [0.025, 0.975])
    return {
        "clusters": int(len(unique)),
        "point": point,
        "ci_low": float(low),
        "ci_high": float(high),
        "estimable": True,
    }


def build_ledger(
    panel_name: str,
    baseline: str,
    canonical: pd.DataFrame,
    outcomes: dict[str, np.ndarray],
    panel: dict[str, np.ndarray],
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, list[float]]:
    query_rows = np.asarray(panel["query_row"], dtype=np.int64)
    try:
        metadata = manifest.loc[query_rows].copy()
    except KeyError as error:
        raise RuntimeError("one or more panel query rows are absent from the manifest") from error
    if len(metadata) != len(query_rows):
        raise RuntimeError("manifest join changed the query denominator")
    if not (
        np.array_equal(string_array(metadata["ik14"]), string_array(panel["query_ik14"]))
        and np.array_equal(
            string_array(metadata["formula"]), string_array(panel["query_formula"])
        )
        and np.allclose(
            pd.to_numeric(metadata["precursor_mz"], errors="raise"),
            np.asarray(panel["query_precursor_mz"], dtype=np.float64),
            atol=1e-9,
            rtol=0,
        )
    ):
        raise RuntimeError("manifest chemistry drifted from the panel query axis")

    baseline_rank = np.asarray(outcomes[baseline], dtype=np.int64)
    calms_rank = np.asarray(outcomes[CALMS_METHOD], dtype=np.int64)
    baseline_correct = baseline_rank == 1
    calms_correct = calms_rank == 1
    event = np.select(
        (
            calms_correct & ~baseline_correct,
            ~calms_correct & baseline_correct,
            calms_correct & baseline_correct,
        ),
        EVENT_LABELS[:3],
        default=EVENT_LABELS[3],
    )
    baseline_top1 = np.asarray(outcomes[f"{baseline}__top1"], dtype=np.float64)
    baseline_top2 = np.asarray(outcomes[f"{baseline}__top2"], dtype=np.float64)
    calms_top1 = np.asarray(outcomes[f"{CALMS_METHOD}__top1"], dtype=np.float64)
    calms_top2 = np.asarray(outcomes[f"{CALMS_METHOD}__top2"], dtype=np.float64)
    gap = baseline_top1 - baseline_top2
    quartile, quartile_edges = margin_quartiles(gap)
    candidate_counts = pd.to_numeric(
        canonical["candidate_molecule_edges"], errors="raise"
    ).to_numpy(np.int64)
    n_peaks = pd.to_numeric(metadata["n_peaks"], errors="raise").to_numpy(np.int64)

    ledger = pd.DataFrame(
        {
            "panel": panel_name,
            "baseline": baseline,
            "query_index": np.arange(len(query_rows), dtype=np.int64),
            "query_row": query_rows,
            "query_ik14": string_array(panel["query_ik14"]),
            "query_formula": string_array(panel["query_formula"]),
            "near": np.asarray(panel["near_query"], dtype=bool),
            "candidate_molecule_edges": candidate_counts,
            "candidate_pair_edges": pd.to_numeric(
                canonical["candidate_pair_edges"], errors="raise"
            ).to_numpy(np.int64),
            "precursor_mz": np.asarray(panel["query_precursor_mz"], dtype=np.float64),
            "library_quality": metadata["library_quality"].to_numpy(),
            "n_peaks": n_peaks,
            "instrument": metadata["instrument"].fillna("").astype(str).to_numpy(),
            "instrument_family": [instrument_family(value) for value in metadata["instrument"]],
            "candidate_count_bin": [candidate_count_bin(int(value)) for value in candidate_counts],
            "peak_count_bin": [peak_count_bin(int(value)) for value in n_peaks],
            "baseline_top1_score": baseline_top1,
            "baseline_top2_score": baseline_top2,
            "baseline_top1_top2_gap": gap,
            "baseline_margin_quartile": quartile,
            "calms_top1_score": calms_top1,
            "calms_top2_score": calms_top2,
            "calms_top1_top2_gap": calms_top1 - calms_top2,
            "baseline_rank": baseline_rank,
            "calms_rank": calms_rank,
            "baseline_reciprocal_rank": 1.0 / baseline_rank,
            "calms_reciprocal_rank": 1.0 / calms_rank,
            "baseline_correct": baseline_correct,
            "calms_correct": calms_correct,
            "event": event,
            "recall1_delta": calms_correct.astype(np.int8) - baseline_correct.astype(np.int8),
            "mrr_delta": 1.0 / calms_rank - 1.0 / baseline_rank,
            "rank_improvement": baseline_rank - calms_rank,
        }
    )
    if not set(ledger["event"].unique()).issubset(EVENT_LABELS):
        raise AssertionError("unexpected exchange event")
    return ledger, quartile_edges


def summarize_subset(
    ledger: pd.DataFrame,
    dimension: str,
    value: str,
    mask: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, object]:
    subset = ledger.loc[np.asarray(mask, dtype=bool)]
    n = len(subset)
    delta = subset["recall1_delta"].to_numpy(np.float64)
    formula_ci = cluster_bootstrap_delta(
        delta,
        subset["query_formula"].to_numpy(),
        repeats,
        stable_seed(seed, ledger["panel"].iat[0], ledger["baseline"].iat[0], dimension, value, "formula"),
    )
    identity_ci = cluster_bootstrap_delta(
        delta,
        subset["query_ik14"].to_numpy(),
        repeats,
        stable_seed(seed, ledger["panel"].iat[0], ledger["baseline"].iat[0], dimension, value, "identity"),
    )
    event_counts = subset["event"].value_counts()

    def mean(column: str) -> float | None:
        return float(subset[column].mean()) if n else None

    return {
        "panel": str(ledger["panel"].iat[0]),
        "baseline": str(ledger["baseline"].iat[0]),
        "stratum_dimension": dimension,
        "stratum": value,
        "queries": n,
        "identities": int(subset["query_ik14"].nunique()) if n else 0,
        "formulas": int(subset["query_formula"].nunique()) if n else 0,
        "baseline_recall1": mean("baseline_correct"),
        "calms_recall1": mean("calms_correct"),
        "delta_recall1": mean("recall1_delta"),
        "delta_recall1_pp": 100.0 * mean("recall1_delta") if n else None,
        "corrected": int(event_counts.get("corrected", 0)),
        "introduced": int(event_counts.get("introduced", 0)),
        "net_correct": int(event_counts.get("corrected", 0) - event_counts.get("introduced", 0)),
        "retained": int(event_counts.get("retained", 0)),
        "both_wrong": int(event_counts.get("both_wrong", 0)),
        "baseline_mrr": mean("baseline_reciprocal_rank"),
        "calms_mrr": mean("calms_reciprocal_rank"),
        "mrr_delta": mean("mrr_delta"),
        "formula_clusters": formula_ci["clusters"],
        "formula_ci_low_pp": (
            100.0 * float(formula_ci["ci_low"])
            if formula_ci["ci_low"] is not None
            else None
        ),
        "formula_ci_high_pp": (
            100.0 * float(formula_ci["ci_high"])
            if formula_ci["ci_high"] is not None
            else None
        ),
        "identity_clusters": identity_ci["clusters"],
        "identity_ci_low_pp": (
            100.0 * float(identity_ci["ci_low"])
            if identity_ci["ci_low"] is not None
            else None
        ),
        "identity_ci_high_pp": (
            100.0 * float(identity_ci["ci_high"])
            if identity_ci["ci_high"] is not None
            else None
        ),
    }


def summarize_ledger(ledger: pd.DataFrame, repeats: int, seed: int) -> list[dict[str, object]]:
    groups: list[tuple[str, str, np.ndarray]] = [
        ("all", "all", np.ones(len(ledger), dtype=bool)),
        ("near", "non_near", ~ledger["near"].to_numpy(bool)),
        ("near", "near", ledger["near"].to_numpy(bool)),
    ]
    fixed = (
        ("candidate_count", "candidate_count_bin", ("2_3", "4_7", "8_plus")),
        ("instrument_family", "instrument_family", ("Orbitrap_FT", "TOF_qTOF", "Other_unknown")),
        ("peak_count", "peak_count_bin", ("lt_50", "50_99", "100_plus")),
        ("baseline_margin_quartile", "baseline_margin_quartile", MARGIN_LABELS),
    )
    for dimension, column, levels in fixed:
        observed = ledger[column].astype(str).to_numpy()
        for level in levels:
            groups.append((dimension, level, observed == level))
    return [
        summarize_subset(ledger, dimension, value, mask, repeats, seed)
        for dimension, value, mask in groups
    ]


def json_ready(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def run(args: argparse.Namespace) -> Path:
    if args.bootstrap_resamples < 100:
        raise ValueError("bootstrap-resamples must be at least 100")
    if args.output.exists():
        raise FileExistsError(args.output)
    selected_panels = list(dict.fromkeys(args.panels))
    specs = panel_specs(args)
    manifest = load_manifest(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{args.output.name}.staging-", dir=args.output.parent)
    )
    report: dict[str, object] = {
        "schema": "calms_correction_landscape_v1",
        "status": "CALMS_CORRECTION_LANDSCAPE_COMPLETE",
        "method": {"public_name": "CALMS v1.0", "internal_name": CALMS_METHOD},
        "baselines": list(BASELINE_METHODS),
        "evaluation_role": "consumed GNPS development mechanism analysis",
        "truth_status": "consumed",
        "allow_model_selection": False,
        "allow_routing": False,
        "claim_boundary": (
            "Descriptive paired correction landscape on consumed GNPS panels; "
            "not an unopened external confirmation and not a routing rule."
        ),
        "strata": {
            "near": ["non_near", "near"],
            "candidate_count": {"2_3": [2, 3], "4_7": [4, 7], "8_plus": [8, None]},
            "instrument_family": {
                "Orbitrap_FT": "contains orbitrap, hybrid ft, or q-exactive",
                "TOF_qTOF": "contains tof",
                "Other_unknown": "all remaining or missing labels",
            },
            "peak_count": {"lt_50": [0, 49], "50_99": [50, 99], "100_plus": [100, None]},
            "baseline_margin_quartile": (
                "Panel- and comparator-specific quartiles computed only from the "
                "truth-blind baseline Top-1--Top-2 candidate-score gap; ties stay together."
            ),
        },
        "bootstrap": {
            "resamples": int(args.bootstrap_resamples),
            "seed": int(args.seed),
            "confidence_level": 0.95,
            "cluster_units": ["query_formula", "query_ik14"],
            "multiplicity_adjustment": None,
            "interpretation": "descriptive subgroup intervals",
        },
        "inputs": {
            "manifest": {
                "path": str(args.manifest),
                "sha256": sha256_file(args.manifest),
                "rows": int(len(manifest)),
            },
            "evaluation_root": str(args.evaluation_root),
            "score_root": str(args.score_root),
        },
        "panels": {},
        "artifacts": {},
    }
    summaries: list[dict[str, object]] = []
    try:
        for panel_name in selected_panels:
            spec = specs[panel_name]
            panel = load_panel(spec.panel_path)
            score_path = args.score_root / spec.score_filename
            scores, score_provenance = load_score_bundle(score_path, spec.panel_path, panel)
            query_metrics_path = (
                args.evaluation_root / spec.evaluation_directory / "query_metrics.csv.gz"
            )
            canonical, outcomes = validate_query_metrics(
                query_metrics_path, panel, scores
            )
            panel_report: dict[str, object] = {
                "queries": int(len(panel["query_row"])),
                "molecule_edges": int(len(panel["molecule_label"])),
                "spectrum_pair_edges": int(len(panel["candidate_row"])),
                "panel": {
                    "path": str(spec.panel_path),
                    "sha256": sha256_file(spec.panel_path),
                },
                "score_bundle": {"path": str(score_path), **score_provenance},
                "query_metrics": {
                    "path": str(query_metrics_path),
                    "sha256": sha256_file(query_metrics_path),
                },
                "baselines": {},
            }
            for baseline in BASELINE_METHODS:
                ledger, quartile_edges = build_ledger(
                    panel_name, baseline, canonical, outcomes, panel, manifest
                )
                ledger_name = f"query_ledger_{panel_name}_{baseline}.csv.gz"
                ledger.to_csv(staging / ledger_name, index=False, compression="gzip")
                panel_summaries = summarize_ledger(
                    ledger, args.bootstrap_resamples, args.seed
                )
                summaries.extend(panel_summaries)
                global_summary = next(
                    row
                    for row in panel_summaries
                    if row["stratum_dimension"] == "all"
                )
                panel_report["baselines"][baseline] = {
                    "margin_quartile_edges": quartile_edges,
                    "global": global_summary,
                    "query_ledger": ledger_name,
                }
            report["panels"][panel_name] = panel_report

        summary = pd.DataFrame.from_records(summaries)
        summary.to_csv(staging / "strata_summary.tsv", sep="\t", index=False)
        report["artifacts"] = {
            "strata_summary": "strata_summary.tsv",
            "query_ledgers": sorted(path.name for path in staging.glob("query_ledger_*.csv.gz")),
        }
        (staging / "report.json").write_text(
            json.dumps(json_ready(report), indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return args.output


def main() -> None:
    args = arguments()
    output = run(args)
    print(
        json.dumps(
            {"status": "CALMS_CORRECTION_LANDSCAPE_COMPLETE", "output": str(output)},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
