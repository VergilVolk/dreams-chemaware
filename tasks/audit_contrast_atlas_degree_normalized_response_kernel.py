#!/usr/bin/env python
"""Falsify a degree-normalized, structure-agnostic response kernel.

This is a bounded development audit, not a model-selection loop.  It reuses
two already consumed cross-study panels and their frozen spectral candidate
graphs.  All qualified spectral edges are retained, but their WSE weights are
symmetrically normalized by the weighted degree of both incident anchors.

The response score is a weighted cosine of beta/sqrt(beta^2 + SE^2).  Its null
permutes complete candidate-anchor response profiles among precursor-mass
strata.  The permutation occurs at the anchor (not edge-row) level, preserving
the frozen graph, candidate response covariance, and graph degree structure.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Panel:
    name: str
    edges: Path
    query_effects: Path
    candidate_effects: Path
    library_partition: Path
    query_contrasts: tuple[str, ...]
    candidate_contrasts: tuple[str, ...]
    targets: frozenset[str]
    distractors: frozenset[str]
    query_column: str
    candidate_column: str
    query_study: str
    candidate_study: str
    excluded_anchor_mz: tuple[float, ...] = ()


PANELS = (
    Panel(
        name="msv097426_mouse_hfd_to_msv084112_human_obesity",
        edges=ROOT / "data/validation/msv097426_to_msv084112_hfd_obesity_query_20261014_v2/spectral_edge_candidates.csv.gz",
        query_effects=ROOT / "data/validation/msv000097426_hfd_contrast_effects_20261014_v1/contrast_effects.csv",
        candidate_effects=ROOT / "data/validation/msv000084112_matched_contrast_effects_20261014_v1/contrast_effects.csv",
        library_partition=ROOT / "data/validation/msv097426_to_msv084112_hfd_obesity_query_20261014_v2/library_partition.csv.gz",
        query_contrasts=("mouse_hfd_vs_nfd",),
        candidate_contrasts=(
            "diabetes_normal_weight",
            "diabetes_obese",
            "obesity_with_diabetes",
            "obesity_without_diabetes",
        ),
        targets=frozenset({"obesity_with_diabetes", "obesity_without_diabetes"}),
        distractors=frozenset({"diabetes_normal_weight", "diabetes_obese"}),
        query_column="mouse_anchor_id",
        candidate_column="human_anchor_id",
        query_study="mouse",
        candidate_study="human",
        excluded_anchor_mz=(396.1684, 220.1366),
    ),
    Panel(
        name="msv084112_human_urine_to_msv083077_human_saliva_obesity",
        edges=ROOT / "data/validation/msv084112_to_msv083077_obesity_query_20261014_v1/spectral_edge_candidates.csv.gz",
        query_effects=ROOT / "data/validation/msv000084112_matched_contrast_effects_20261014_v1/contrast_effects.csv",
        candidate_effects=ROOT / "data/validation/msv000083077_saliva_contrast_effects_20261014_v1/contrast_effects.csv",
        library_partition=ROOT / "data/validation/msv084112_to_msv083077_obesity_query_20261014_v1/library_partition.csv.gz",
        query_contrasts=("obesity_without_diabetes", "obesity_with_diabetes"),
        candidate_contrasts=(
            "depression_nonobese",
            "depression_obese",
            "obesity_high_depression",
            "obesity_low_depression",
        ),
        targets=frozenset({"obesity_high_depression", "obesity_low_depression"}),
        distractors=frozenset({"depression_nonobese", "depression_obese"}),
        query_column="mouse_anchor_id",
        candidate_column="human_anchor_id",
        query_study="query",
        candidate_study="candidate",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signed_reliability(beta: np.ndarray, se: np.ndarray) -> np.ndarray:
    denominator = np.sqrt(beta * beta + se * se)
    return np.divide(beta, denominator, out=np.zeros_like(beta), where=denominator > 0)


def weighted_cosine(left: np.ndarray, right: np.ndarray, weight: np.ndarray) -> float:
    denominator = float(np.sqrt(np.sum(weight * left * left) * np.sum(weight * right * right)))
    return float(np.sum(weight * left * right) / denominator) if denominator > 0 else float("nan")


def filter_reported_anchors(panel: Panel, edges: pd.DataFrame) -> pd.DataFrame:
    keep = np.ones(len(edges), dtype=bool)
    query_mz_column = panel.query_column.replace("anchor_id", "precursor_mz")
    candidate_mz_column = panel.candidate_column.replace("anchor_id", "precursor_mz")
    for anchor_mz in panel.excluded_anchor_mz:
        delta = anchor_mz * 10e-6
        keep &= np.abs(edges[query_mz_column].to_numpy(float) - anchor_mz) > delta
        keep &= np.abs(edges[candidate_mz_column].to_numpy(float) - anchor_mz) > delta
    return edges.loc[keep].copy()


def degree_normalized_weight(panel: Panel, edges: pd.DataFrame) -> np.ndarray:
    query_degree = edges.groupby(panel.query_column)["wse"].transform("sum").to_numpy(float)
    candidate_degree = edges.groupby(panel.candidate_column)["wse"].transform("sum").to_numpy(float)
    return edges["wse"].to_numpy(float) / np.sqrt(query_degree * candidate_degree)


def evaluate_panel(panel: Panel, permutations: int, seed: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    edges = filter_reported_anchors(panel, pd.read_csv(panel.edges))
    query_effects = pd.read_csv(panel.query_effects)
    candidate_effects = pd.read_csv(panel.candidate_effects)
    partition = pd.read_csv(panel.library_partition)
    query_support = partition[partition.study == panel.query_study].set_index("anchor_id")["library_supported"]
    candidate_support = partition[partition.study == panel.candidate_study].set_index("anchor_id")["library_supported"]

    query_beta = query_effects.pivot(index="anchor_id", columns="contrast_id", values="beta_log2")
    query_se = query_effects.pivot(index="anchor_id", columns="contrast_id", values="se")
    candidate_beta = candidate_effects.pivot(index="anchor_id", columns="contrast_id", values="beta_log2")
    candidate_se = candidate_effects.pivot(index="anchor_id", columns="contrast_id", values="se")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    worlds = {
        "all": edges,
        "unresolved_both": edges[
            ~edges[panel.query_column].map(query_support).fillna(False)
            & ~edges[panel.candidate_column].map(candidate_support).fillna(False)
        ].copy(),
    }

    for world, world_edges in worlds.items():
        weight = degree_normalized_weight(panel, world_edges)
        query_ids = world_edges[panel.query_column].to_numpy(str)
        candidate_ids = world_edges[panel.candidate_column].to_numpy(str)
        query_vectors = {
            contrast: signed_reliability(
                query_beta.loc[query_ids, contrast].to_numpy(float),
                query_se.loc[query_ids, contrast].to_numpy(float),
            )
            for contrast in panel.query_contrasts
        }

        def score(mapped_candidate_ids: np.ndarray) -> dict[str, float]:
            result: dict[str, float] = {}
            for contrast in panel.candidate_contrasts:
                candidate_vector = signed_reliability(
                    candidate_beta.loc[mapped_candidate_ids, contrast].to_numpy(float),
                    candidate_se.loc[mapped_candidate_ids, contrast].to_numpy(float),
                )
                result[contrast] = float(np.mean([
                    weighted_cosine(query_vectors[query_contrast], candidate_vector, weight)
                    for query_contrast in panel.query_contrasts
                ]))
            return result

        observed = score(candidate_ids)
        ordered = sorted(observed, key=lambda key: (-observed[key], key))
        margin = max(observed[key] for key in panel.targets) - max(
            observed[key] for key in panel.distractors
        )
        candidate_mz_column = panel.candidate_column.replace("anchor_id", "precursor_mz")
        nodes = world_edges[[panel.candidate_column, candidate_mz_column]].drop_duplicates(
            panel.candidate_column
        ).set_index(panel.candidate_column)
        node_ids = nodes.index.to_numpy(str)
        strata = pd.qcut(
            nodes[candidate_mz_column],
            q=min(10, max(2, len(nodes) // 10)),
            labels=False,
            duplicates="drop",
        ).to_numpy()
        null = np.empty(permutations, dtype=float)
        for iteration in range(permutations):
            mapping: dict[str, str] = {}
            for stratum in np.unique(strata):
                indices = np.flatnonzero(strata == stratum)
                mapping.update(zip(node_ids[indices], rng.permutation(node_ids[indices])))
            shuffled = np.asarray([mapping[key] for key in candidate_ids], dtype=object)
            shuffled_score = score(shuffled)
            null[iteration] = max(shuffled_score[key] for key in panel.targets) - max(
                shuffled_score[key] for key in panel.distractors
            )
        p_value = float((1 + np.sum(null >= margin)) / (permutations + 1))
        for contrast, value in observed.items():
            rows.append({
                "panel": panel.name,
                "world": world,
                "candidate_contrast": contrast,
                "score": value,
                "is_target": contrast in panel.targets,
            })
        rows.append({
            "panel": panel.name,
            "world": world,
            "candidate_contrast": "__SUMMARY__",
            "score": margin,
            "is_target": True,
            "n_edges": int(len(world_edges)),
            "n_query_anchors": int(world_edges[panel.query_column].nunique()),
            "n_candidate_anchors": int(world_edges[panel.candidate_column].nunique()),
            "top1": ordered[0],
            "top1_is_target": ordered[0] in panel.targets,
            "target_minus_distractor_margin": margin,
            "mass_stratified_node_response_permutation_p": p_value,
            "null_margin_q95": float(np.quantile(null, 0.95)),
        })

    provenance = {
        "spectral_edges": {"path": str(panel.edges.relative_to(ROOT)), "sha256": sha256(panel.edges)},
        "query_effects": {
            "path": str(panel.query_effects.relative_to(ROOT)),
            "sha256": sha256(panel.query_effects),
        },
        "candidate_effects": {
            "path": str(panel.candidate_effects.relative_to(ROOT)),
            "sha256": sha256(panel.candidate_effects),
        },
        "library_partition": {
            "path": str(panel.library_partition.relative_to(ROOT)),
            "sha256": sha256(panel.library_partition),
        },
    }
    return rows, provenance


def main() -> None:
    output_dir = ROOT / "data/validation/contrast_atlas_degree_normalized_response_kernel_20261014_v2"
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    permutations = 4999
    seed = 20261014
    all_rows: list[dict[str, object]] = []
    provenance: dict[str, object] = {}
    for panel_index, panel in enumerate(PANELS):
        rows, inputs = evaluate_panel(panel, permutations, seed + panel_index)
        all_rows.extend(rows)
        provenance[panel.name] = inputs
    scores = pd.DataFrame(all_rows)
    scores.to_csv(output_dir / "scores.csv", index=False)
    summaries = scores[scores.candidate_contrast == "__SUMMARY__"].copy()
    positive = summaries[summaries.panel == PANELS[0].name]
    report = {
        "status": "development_consumed_posthoc_method_falsification",
        "method": "all-qualified-edge symmetric-degree-normalized WSE response kernel",
        "primary_positive_panel": PANELS[0].name,
        "primary_pass_rule": (
            "top1_is_target and node-level mass-stratified response-link permutation p<=0.05 "
            "in both all and unresolved_both worlds"
        ),
        "pass_primary_positive_panel": bool(
            positive.top1_is_target.all()
            and (positive.mass_stratified_node_response_permutation_p <= 0.05).all()
        ),
        "summaries": summaries.replace({np.nan: None}).to_dict(orient="records"),
        "parameters": {
            "permutations": permutations,
            "seed": seed,
            "edge_thresholds": "inherited unchanged from consumed source graphs: 10 ppm, WSE>=0.70, >=6 matched peaks",
            "effect_transform": "beta/sqrt(beta^2+SE^2)",
            "edge_weight": "WSE/sqrt(weighted_query_degree*weighted_candidate_degree)",
            "null_unit": "complete candidate-anchor response profile within precursor-mass decile",
        },
        "claim_boundary": {
            "role": "development/consumed; post-hoc after reciprocal-edge failures were known",
            "structure_agnostic": "library identity is used only to define the unresolved_both audit world, never to build or weight edges",
            "not_external": True,
            "no_threshold_or_weight_selection": True,
        },
        "input_sha256": provenance,
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
