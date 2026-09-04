#!/usr/bin/env python
"""Locked BioAware v2-0 comparison on the MTBLS13729 pilot.

The script keeps the original 21-query pseudo-truth benchmark frozen while
separating two concepts that v1 accidentally coupled:

1. every eligible high-confidence feature may act as a reaction seed;
2. a seed may support a query only when the phenotype-blind experimental
   feature graph connects their MS1 features.

The comparison therefore isolates baseline DreaMS, archived-v1 seed coverage,
expanded one-hop Rhea coverage, and the experimental+biochemical two-layer
gate.  No phenotype or differential-abundance field is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from annotation.bioaware import (  # noqa: E402
    BioAwareConfig,
    aggregate_query_support,
    build_one_hop_evidence,
    fuse_candidates,
    top1_transition_table,
)


ALLOWED_BEST_COLUMNS = [
    "feature_id",
    "best_ik14",
    "max_cosine",
    "n_support_spectra",
    "structure_agreement_fraction",
    "annotation_evidence_tier",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def noisy_or(values: np.ndarray) -> float:
    values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    return float(1.0 - np.prod(1.0 - values)) if values.size else 0.0


def build_expanded_seeds(
    link_dir: Path,
    panels: list[str],
    graph_compounds: set[str],
    *,
    minimum_cosine: float,
    minimum_support_spectra: int,
    minimum_agreement: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for panel in panels:
        path = link_dir / f"{panel}__feature_best_annotations.csv.gz"
        best = pd.read_csv(path, usecols=ALLOWED_BEST_COLUMNS)
        best["best_ik14"] = best["best_ik14"].fillna("").astype(str)
        eligible = best[
            best["best_ik14"].isin(graph_compounds)
            & (pd.to_numeric(best["max_cosine"], errors="coerce") >= minimum_cosine)
            & (pd.to_numeric(best["n_support_spectra"], errors="coerce") >= minimum_support_spectra)
            & (
                pd.to_numeric(best["structure_agreement_fraction"], errors="coerce")
                >= minimum_agreement
            )
            & (best["annotation_evidence_tier"] == "Level 2a-supported")
        ].copy()
        eligible["seed_query_id"] = panel + ":" + eligible["feature_id"].astype(str)
        eligible["seed_compound_id"] = eligible["best_ik14"]
        eligible["seed_score"] = eligible["max_cosine"].clip(0, 1).astype(float)
        eligible["reference_kind"] = "frozen_level2a_supported_spectral_annotation"
        rows.append(
            eligible[
                ["seed_query_id", "seed_compound_id", "seed_score", "reference_kind"]
            ]
        )
    if not rows:
        raise RuntimeError("no expanded BioAware seeds")
    seeds = pd.concat(rows, ignore_index=True).drop_duplicates(
        ["seed_query_id", "seed_compound_id"]
    )
    if seeds.empty:
        raise RuntimeError("expanded BioAware seed table is empty")
    return seeds


def load_experimental_edges(graph_dir: Path, panels: list[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for panel in panels:
        path = graph_dir / f"{panel}__edges.csv.gz"
        edge = pd.read_csv(
            path,
            usecols=[
                "feature_id_a",
                "feature_id_b",
                "dreams_cosine",
                "metdna_spectral_edge",
                "dual_data_support",
            ],
        )
        edge = edge[edge["metdna_spectral_edge"].astype(bool)].copy()
        forward = pd.DataFrame(
            {
                "query_id": panel + ":" + edge["feature_id_a"].astype(str),
                "seed_query_id": panel + ":" + edge["feature_id_b"].astype(str),
                "experimental_similarity": edge["dreams_cosine"].astype(float),
                "dual_data_support": edge["dual_data_support"].astype(bool),
            }
        )
        reverse = forward.rename(
            columns={"query_id": "seed_query_id", "seed_query_id": "query_id"}
        )[["query_id", "seed_query_id", "experimental_similarity", "dual_data_support"]]
        rows.extend([forward, reverse])
    if not rows:
        raise RuntimeError("experimental feature graph contains no eligible edge")
    result = pd.concat(rows, ignore_index=True)
    return result.sort_values("experimental_similarity", ascending=False).drop_duplicates(
        ["query_id", "seed_query_id"]
    )


def attach_two_layer_support(
    candidates: pd.DataFrame,
    paths: pd.DataFrame,
    experimental_edges: pd.DataFrame,
    *,
    truth_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Require query-feature -> seed-feature -> reaction -> candidate evidence."""

    edge_by_query = {
        query: group for query, group in experimental_edges.groupby("query_id", sort=False)
    }
    path_by_candidate = {
        candidate: group for candidate, group in paths.groupby("candidate_id", sort=False)
    }
    support_rows: list[dict] = []
    explanation_rows: list[pd.DataFrame] = []
    for query_id, group in candidates.groupby("query_id", sort=False):
        truths = group[truth_col].dropna().astype(str).unique()
        if len(truths) != 1:
            raise RuntimeError(f"query {query_id} has {len(truths)} truth identities")
        truth = truths[0]
        edge = edge_by_query.get(query_id)
        edge_lookup = (
            edge.set_index("seed_query_id")
            if edge is not None
            else pd.DataFrame(
                columns=["experimental_similarity", "dual_data_support"]
            )
        )
        for candidate_id in group["candidate_id"].astype(str):
            selected = path_by_candidate.get(candidate_id)
            if selected is None or edge is None:
                support_rows.append(
                    {
                        "query_id": query_id,
                        "candidate_id": candidate_id,
                        "network_support": 0.0,
                        "network_path_count": 0,
                        "dual_supported_path_count": 0,
                    }
                )
                continue
            selected = selected[
                (selected["seed_query_id"].astype(str) != str(query_id))
                & (selected["seed_compound_id"].astype(str) != truth)
                & (selected["seed_query_id"].astype(str).isin(edge_lookup.index.astype(str)))
            ].copy()
            if not selected.empty:
                selected = selected.join(
                    edge_lookup[["experimental_similarity", "dual_data_support"]],
                    on="seed_query_id",
                    validate="many_to_one",
                )
                selected["two_layer_contribution"] = (
                    selected["contribution"].astype(float)
                    * selected["experimental_similarity"].clip(0, 1).astype(float)
                )
                # Multiple feature instances of one biochemical seed must not
                # inflate evidence. Keep its strongest feature/reaction path.
                selected = selected.sort_values(
                    "two_layer_contribution", ascending=False
                ).drop_duplicates(["seed_compound_id", "reaction_id"])
            support_rows.append(
                {
                    "query_id": query_id,
                    "candidate_id": candidate_id,
                    "network_support": noisy_or(
                        selected["two_layer_contribution"].to_numpy(float)
                    ) if len(selected) else 0.0,
                    "network_path_count": int(len(selected)),
                    "dual_supported_path_count": int(
                        selected["dual_data_support"].sum()
                    ) if len(selected) else 0,
                }
            )
            if len(selected):
                selected.insert(0, "query_id", query_id)
                selected.insert(1, "query_candidate_id", candidate_id)
                explanation_rows.append(selected)
    support = pd.DataFrame(support_rows)
    scored = candidates.merge(
        support, on=["query_id", "candidate_id"], how="left", validate="one_to_one"
    )
    explanations = (
        pd.concat(explanation_rows, ignore_index=True)
        if explanation_rows
        else pd.DataFrame()
    )
    return scored, explanations


def evaluate_supported(
    supported: pd.DataFrame,
    config: BioAwareConfig,
    truth_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    scored, decisions = fuse_candidates(supported, config)
    per_query, summary = top1_transition_table(scored, truth_col=truth_col)
    summary["intervention_rate"] = float(decisions["bioaware_applied"].mean())
    summary["queries_with_network_evidence"] = int(
        decisions["network_available"].sum()
    )
    c = int(summary["corrected"])
    i = int(summary["introduced"])
    summary["mcnemar_exact_p"] = float(
        binomtest(min(c, i), c + i, 0.5).pvalue
    ) if c + i else 1.0
    return scored, decisions, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=Path("data/mtbls13729/bioaware_v1_input/candidates.csv.gz"))
    parser.add_argument("--archived-seeds", type=Path, default=Path("data/mtbls13729/bioaware_v1_input/seeds.csv.gz"))
    parser.add_argument("--participants", type=Path, default=Path("data/reference/bioaware_rhea_offline_20260827/rhea_participants.csv.gz"))
    parser.add_argument("--link-dir", type=Path, default=Path("data/mtbls13729/ms1_ms2_link"))
    parser.add_argument("--feature-graph", type=Path, default=Path("data/mtbls13729/bioaware_v2_feature_graph"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/bioaware_v2_eval"))
    parser.add_argument("--panels", nargs="+", default=["neg_rp", "pos_rp"])
    parser.add_argument("--minimum-cosine", type=float, default=0.80)
    parser.add_argument("--minimum-support-spectra", type=int, default=2)
    parser.add_argument("--minimum-agreement", type=float, default=0.60)
    args = parser.parse_args()

    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(args.candidates)
    archived_seeds = pd.read_csv(args.archived_seeds)
    participants = pd.read_csv(args.participants)
    feature_graph_report = json.loads(
        (args.feature_graph / "report.json").read_text(encoding="utf-8")
    )
    formal = bool(feature_graph_report.get("formal")) and set(args.panels) == {
        "neg_rp",
        "pos_rp",
    }
    graph_compounds = set(participants["compound_id"].dropna().astype(str))
    expanded_seeds = build_expanded_seeds(
        args.link_dir,
        args.panels,
        graph_compounds,
        minimum_cosine=args.minimum_cosine,
        minimum_support_spectra=args.minimum_support_spectra,
        minimum_agreement=args.minimum_agreement,
    )
    edges = load_experimental_edges(args.feature_graph, args.panels)
    config = BioAwareConfig()

    results: dict[str, dict] = {}
    artifacts: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for name, seeds in [("archived_v1", archived_seeds), ("expanded_rhea_only", expanded_seeds)]:
        paths = build_one_hop_evidence(participants, seeds, config)
        supported, explanations = aggregate_query_support(
            candidates,
            paths,
            truth_col="truth_candidate_id",
            exclude_same_query=True,
            exclude_truth_identity=True,
        )
        scored, decisions, summary = evaluate_supported(
            supported, config, "truth_candidate_id"
        )
        results[name] = {**summary, "evidence_paths": int(len(explanations))}
        artifacts[name] = (scored, decisions)

    expanded_paths = build_one_hop_evidence(participants, expanded_seeds, config)
    two_layer_supported, two_layer_explanations = attach_two_layer_support(
        candidates,
        expanded_paths,
        edges,
        truth_col="truth_candidate_id",
    )
    two_layer_scored, two_layer_decisions, two_layer_summary = evaluate_supported(
        two_layer_supported, config, "truth_candidate_id"
    )
    results["two_layer"] = {
        **two_layer_summary,
        "evidence_paths": int(len(two_layer_explanations)),
        "dual_supported_paths": int(
            two_layer_supported["dual_supported_path_count"].sum()
        ),
    }

    baseline = results["archived_v1"]["baseline_recall1"]
    report = {
        "status": "mtbls13729_bioaware_v2_two_layer_comparison_complete",
        "formal": formal,
        "reference_truth": "frozen Level 2a-supported spectral pseudo-truth; not standard-confirmed identity",
        "baseline_recall1": baseline,
        "queries": int(candidates["query_id"].nunique()),
        "archived_seed_rows": int(len(archived_seeds)),
        "expanded_seed_rows": int(len(expanded_seeds)),
        "expanded_seed_compounds": int(expanded_seeds["seed_compound_id"].nunique()),
        "experimental_edges": int(len(edges)),
        "results": results,
        "configuration": config.to_dict(),
        "gates": {
            "expanded_seed_coverage_increased": len(expanded_seeds) > len(archived_seeds),
            "two_layer_corrected_gt_introduced": (
                results["two_layer"]["corrected"] > results["two_layer"]["introduced"]
            ),
            "two_layer_not_worse_than_baseline": (
                results["two_layer"]["bioaware_recall1"] >= baseline
            ),
            "two_layer_intervenes": results["two_layer"]["intervention_rate"] > 0,
        },
        "provenance": {
            "candidates_sha256": sha256(args.candidates),
            "archived_seeds_sha256": sha256(args.archived_seeds),
            "participants_sha256": sha256(args.participants),
            "feature_graph_report_sha256": sha256(args.feature_graph / "report.json"),
        },
        "claim_limit": (
            "This saturated 21-query pseudo-truth pilot is a mechanism and safety audit. "
            "It cannot establish external annotation accuracy, biological causality, or MSI Level 1/2 identity."
        ),
    }
    report["gates"]["pass"] = all(report["gates"].values())
    expanded_seeds.to_csv(out / "expanded_seeds.csv.gz", index=False)
    two_layer_scored.to_csv(out / "two_layer_candidate_scores.csv.gz", index=False)
    two_layer_decisions.to_csv(out / "two_layer_query_decisions.csv", index=False)
    two_layer_explanations.to_csv(out / "two_layer_evidence_paths.csv.gz", index=False)
    for name, (scored, decisions) in artifacts.items():
        scored.to_csv(out / f"{name}_candidate_scores.csv.gz", index=False)
        decisions.to_csv(out / f"{name}_query_decisions.csv", index=False)
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
