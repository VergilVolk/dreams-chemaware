#!/usr/bin/env python3
"""Open study-response retrieval from locally replicated MS/MS response edges.

The chemical graph is frozen upstream without phenotype effects.  This script
uses that graph to ask a different question from a global response centroid:
does a candidate study contain an excess of *local* spectral correspondences
whose two endpoints are independently response-qualified in the same (or the
opposite) direction?

The unit of retrieval is a study-level contrast.  Distinct entities are never
merged and their beta/SE values are never meta-analysed.  Same-precursor
mutual-neighbour edges are reported as identity hypotheses; mass-incompatible
mutual-neighbour edges are reported separately as chemical-relation
hypotheses.  Whole entity response profiles are reassigned within study and
common identity stratum to obtain candidate-specific nulls.  A query-level
max-T distribution controls selection across the full source-disjoint
candidate universe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from run_contrast_atlas_sparse_fragment_measure_baseline import (
    iter_mgf,
    normalise_entity_id,
    sha256,
    spectrum_id,
)


WORLDS = ("all", "known", "unresolved")
TRACKS = (
    "combined_mutual",
    "same_precursor_10ppm_identity_hypothesis",
    "nonmass_chemical_relation_hypothesis",
)
FACES = ("similar", "opposite")
EDGE_KEY_COLUMNS = [
    "left_study_id",
    "right_study_id",
    "evidence_world",
    "left_entity_id",
    "right_entity_id",
]


@dataclass
class StudyProfiles:
    entities: np.ndarray
    contrasts: list[str]
    status: np.ndarray
    positive: np.ndarray
    negative: np.ndarray
    weight: np.ndarray
    beta: np.ndarray
    se: np.ndarray
    q: np.ndarray


def bh_adjust(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    output = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    selected = values[finite]
    if not len(selected):
        return output
    order = np.argsort(selected, kind="stable")
    ranked = selected[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty(len(selected), dtype=float)
    restored[order] = np.minimum(adjusted, 1.0)
    output[finite] = restored
    return output


def add_effect_statistics(frame: pd.DataFrame, q_threshold: float) -> pd.DataFrame:
    result = frame.copy()
    beta = pd.to_numeric(result["beta"], errors="coerce").to_numpy(float)
    se = pd.to_numeric(result["se"], errors="coerce").to_numpy(float)
    valid = np.isfinite(beta) & np.isfinite(se) & (se > 0)
    z = np.full(len(result), np.nan, dtype=float)
    z[valid] = beta[valid] / se[valid]
    p = np.full(len(result), np.nan, dtype=float)
    p[valid] = 2.0 * norm.sf(np.abs(z[valid]))
    result["effect_z"] = z
    result["effect_p"] = p
    result["effect_q"] = np.nan
    for indices in result.groupby(["study_id", "contrast_id"], sort=False).groups.values():
        result.loc[indices, "effect_q"] = bh_adjust(
            result.loc[indices, "effect_p"].to_numpy(float)
        )
    denominator = np.sqrt(beta * beta + se * se)
    result["response_weight"] = np.divide(
        beta,
        denominator,
        out=np.zeros_like(beta),
        where=valid & (denominator > 0),
    )
    result["response_qualified"] = result["effect_q"].le(q_threshold)
    return result


def build_profiles(
    effects: pd.DataFrame,
    contrast_manifest: pd.DataFrame,
    q_threshold: float,
) -> dict[str, StudyProfiles]:
    output: dict[str, StudyProfiles] = {}
    for study in contrast_manifest["study_id"].astype(str).drop_duplicates():
        block = effects.loc[effects["study_id"].eq(study)].copy()
        entities = np.asarray(sorted(block["entity_id"].astype(str).unique()), dtype=str)
        contrasts = contrast_manifest.loc[
            contrast_manifest["study_id"].astype(str).eq(study), "contrast_id"
        ].astype(str).tolist()
        erow = {value: index for index, value in enumerate(entities)}
        ccol = {value: index for index, value in enumerate(contrasts)}
        shape = (len(entities), len(contrasts))
        beta = np.full(shape, np.nan, dtype=np.float32)
        se = np.full(shape, np.nan, dtype=np.float32)
        q = np.full(shape, np.nan, dtype=np.float32)
        weight = np.zeros(shape, dtype=np.float32)
        status_by_entity = block.drop_duplicates("entity_id").set_index("entity_id")[
            "identity_status"
        ].astype(str)
        status = np.asarray([status_by_entity.loc[value] for value in entities], dtype=str)
        if block.duplicated(["entity_id", "contrast_id"]).any():
            raise RuntimeError(f"duplicate entity/contrast effects in {study}")
        for row in block.itertuples(index=False):
            i = erow[str(row.entity_id)]
            j = ccol[str(row.contrast_id)]
            beta[i, j] = float(row.beta)
            se[i, j] = float(row.se)
            q[i, j] = float(row.effect_q)
            weight[i, j] = float(row.response_weight)
        qualified = np.isfinite(q) & (q <= q_threshold)
        output[study] = StudyProfiles(
            entities=entities,
            contrasts=contrasts,
            status=status,
            positive=qualified & (beta > 0),
            negative=qualified & (beta < 0),
            weight=weight,
            beta=beta,
            se=se,
            q=q,
        )
    return output


def stable_study_seed(study: str) -> int:
    return int(hashlib.sha256(study.encode("utf-8")).hexdigest()[:8], 16)


def build_profile_reassignments(
    profiles: dict[str, StudyProfiles],
    permutations: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Reassign whole multi-contrast profiles within common identity stratum."""
    output: dict[str, np.ndarray] = {}
    for study in sorted(profiles):
        profile = profiles[study]
        donors = np.tile(
            np.arange(len(profile.entities), dtype=np.int32), (permutations, 1)
        )
        study_seed = stable_study_seed(study)
        for replicate in range(permutations):
            rng = np.random.default_rng(
                np.random.SeedSequence([seed, replicate, study_seed])
            )
            for identity_status in sorted(np.unique(profile.status)):
                indices = np.flatnonzero(profile.status == identity_status)
                donors[replicate, indices] = rng.permutation(indices)
        output[study] = donors
    return output


def load_precursors(
    alignment_report: dict[str, object],
    registry: pd.DataFrame,
) -> tuple[dict[tuple[str, str], float], list[dict[str, object]]]:
    output: dict[tuple[str, str], float] = {}
    summaries: list[dict[str, object]] = []
    for source in alignment_report["sources"]:
        study = str(source["study_id"])
        block = registry.loc[registry["study_id"].eq(study)]
        raw_to_entity = dict(zip(block["raw_entity_id"], block["entity_id"]))
        path = Path(str(source["mgf_path"]))
        found = 0
        for metadata, _peaks in iter_mgf(path):
            raw_id = spectrum_id(metadata)
            entity = raw_to_entity.get(raw_id)
            if entity is None:
                continue
            fields = metadata.get("PEPMASS", "").split()
            if not fields:
                continue
            try:
                value = float(fields[0])
            except ValueError:
                continue
            if np.isfinite(value) and value > 0:
                output[(study, str(entity))] = value
                found += 1
        summaries.append({
            "study_id": study,
            "response_entities": int(len(block)),
            "entities_with_valid_precursor": int(found),
            "mgf_path": str(path),
        })
    return output, summaries


def canonical_mutual_edges(
    edge_path: Path,
    precursors: dict[tuple[str, str], float],
    ppm_threshold: float,
) -> pd.DataFrame:
    edges = pd.read_csv(edge_path, sep="\t", dtype=str)
    edges["chemical_similarity"] = pd.to_numeric(edges["chemical_similarity"])
    table = (
        edges.groupby(
            [
                "left_study_id",
                "right_study_id",
                "evidence_world",
                "left_entity_id",
                "right_entity_id",
            ],
            sort=False,
        )
        .agg(
            chemical_similarity=("chemical_similarity", "max"),
            directional_support=("direction", "nunique"),
        )
        .reset_index()
    )
    table = table.loc[table["directional_support"].eq(2)].copy()
    left_mz = np.asarray(
        [
            precursors.get((str(s), str(e)), np.nan)
            for s, e in zip(table["left_study_id"], table["left_entity_id"])
        ],
        dtype=float,
    )
    right_mz = np.asarray(
        [
            precursors.get((str(s), str(e)), np.nan)
            for s, e in zip(table["right_study_id"], table["right_entity_id"])
        ],
        dtype=float,
    )
    valid = np.isfinite(left_mz) & np.isfinite(right_mz) & (left_mz > 0) & (right_mz > 0)
    ppm = np.full(len(table), np.nan, dtype=float)
    ppm[valid] = (
        np.abs(left_mz[valid] - right_mz[valid])
        / ((left_mz[valid] + right_mz[valid]) / 2.0)
        * 1e6
    )
    table["left_precursor_mz"] = left_mz
    table["right_precursor_mz"] = right_mz
    table["precursor_ppm_difference"] = ppm
    table["precursor_compatible_10ppm"] = valid & (ppm <= ppm_threshold)
    table["precursor_incompatible_over_10ppm"] = valid & (ppm > ppm_threshold)
    return table.reset_index(drop=True)


def apply_algorithm_edge_scores(
    mutual_edges: pd.DataFrame,
    score_path: Path,
    score_column: str,
    missing_policy: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Replace edge strength without changing the exact frozen pair axis."""
    usecols = [*EDGE_KEY_COLUMNS, score_column]
    scores = pd.read_csv(
        score_path, sep="\t", dtype={key: str for key in EDGE_KEY_COLUMNS}
    )
    missing_columns = set(usecols) - set(scores.columns)
    if missing_columns:
        raise RuntimeError(f"algorithm score table lacks columns: {sorted(missing_columns)}")
    scores = scores[usecols].copy()
    if scores.duplicated(EDGE_KEY_COLUMNS).any():
        raise RuntimeError("algorithm score table contains duplicate frozen edge keys")
    if mutual_edges.duplicated(EDGE_KEY_COLUMNS).any():
        raise RuntimeError("canonical mutual edge table contains duplicate keys")

    base_index = pd.MultiIndex.from_frame(mutual_edges[EDGE_KEY_COLUMNS])
    score_index = pd.MultiIndex.from_frame(scores[EDGE_KEY_COLUMNS])
    if len(base_index) != len(score_index) or set(base_index) != set(score_index):
        absent = len(set(base_index) - set(score_index))
        outside = len(set(score_index) - set(base_index))
        raise RuntimeError(
            f"algorithm score axis differs from frozen edges: absent={absent}, outside={outside}"
        )
    ordered = scores.set_index(EDGE_KEY_COLUMNS).loc[base_index, score_column]
    raw = pd.to_numeric(ordered, errors="coerce").to_numpy(float)
    available = np.isfinite(raw)
    if missing_policy == "error" and not available.all():
        raise RuntimeError(f"algorithm score is missing for {int((~available).sum())} edge rows")
    if missing_policy == "base":
        replacement = np.where(
            available,
            raw,
            mutual_edges["chemical_similarity"].to_numpy(float),
        )
    elif missing_policy in {"zero", "error"}:
        replacement = np.where(available, raw, 0.0)
    else:
        raise ValueError(missing_policy)

    result = mutual_edges.copy()
    result["base_chemical_similarity"] = result["chemical_similarity"].to_numpy(float)
    result["algorithm_score_raw"] = raw
    result["algorithm_score_available"] = available
    # Response strength is non-negative. Negative embedding cosine therefore
    # means no support on this fixed edge, not evidence for reverse biology.
    result["chemical_similarity"] = np.clip(replacement, 0.0, 1.0)
    provenance = {
        "path": str(score_path),
        "sha256": sha256(score_path),
        "score_column": score_column,
        "score_transform": "clip_to_[0,1]; negative cosine supplies zero support",
        "missing_policy": missing_policy,
        "edge_rows": len(result),
        "edge_rows_available": int(available.sum()),
        "edge_rows_unavailable": int((~available).sum()),
        "all_world_unique_edges_available": int(
            (available & result["evidence_world"].eq("all").to_numpy()).sum()
        ),
    }
    return result, provenance


def track_mask(block: pd.DataFrame, track: str) -> np.ndarray:
    if track == "combined_mutual":
        return np.ones(len(block), dtype=bool)
    if track == "same_precursor_10ppm_identity_hypothesis":
        return block["precursor_compatible_10ppm"].to_numpy(bool)
    if track == "nonmass_chemical_relation_hypothesis":
        return block["precursor_incompatible_over_10ppm"].to_numpy(bool)
    raise ValueError(track)


def count_and_strength(
    left: StudyProfiles,
    right: StudyProfiles,
    left_rows: np.ndarray,
    right_rows: np.ndarray,
    similarity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lp = left.positive[left_rows].astype(np.int32)
    ln = left.negative[left_rows].astype(np.int32)
    rp = right.positive[right_rows].astype(np.int32)
    rn = right.negative[right_rows].astype(np.int32)
    same = lp.T @ rp + ln.T @ rn
    opposite = lp.T @ rn + ln.T @ rp

    left_weight = left.weight[left_rows].astype(np.float64)
    right_weight = right.weight[right_rows].astype(np.float64)
    # Continuous response evidence is primary.  It uses uncertainty-shrunk
    # beta/SE information without treating a normal-approximation feature p
    # value as exact for small-n sources.
    left_positive_weight = np.clip(left_weight, 0.0, None)
    left_negative_weight = np.clip(-left_weight, 0.0, None)
    right_positive_weight = np.clip(right_weight, 0.0, None)
    right_negative_weight = np.clip(-right_weight, 0.0, None)
    weighted_lp = similarity[:, None] * left_positive_weight
    weighted_ln = similarity[:, None] * left_negative_weight
    weighted_rp = right_positive_weight
    weighted_rn = right_negative_weight
    same_strength = weighted_lp.T @ weighted_rp + weighted_ln.T @ weighted_rn
    opposite_strength = weighted_lp.T @ weighted_rn + weighted_ln.T @ weighted_rp
    return same, opposite, same_strength, opposite_strength


def null_statistics(
    left: StudyProfiles,
    right: StudyProfiles,
    left_rows: np.ndarray,
    right_rows: np.ndarray,
    similarity: np.ndarray,
    left_donors: np.ndarray,
    right_donors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    permutations = left_donors.shape[0]
    shape = (permutations, len(left.contrasts), len(right.contrasts))
    same = np.zeros(shape, dtype=np.int32)
    opposite = np.zeros(shape, dtype=np.int32)
    same_strength = np.zeros(shape, dtype=np.float32)
    opposite_strength = np.zeros(shape, dtype=np.float32)
    for replicate in range(permutations):
        lr = left_donors[replicate, left_rows]
        rr = right_donors[replicate, right_rows]
        lp = left.positive[lr].astype(np.int32)
        ln = left.negative[lr].astype(np.int32)
        rp = right.positive[rr].astype(np.int32)
        rn = right.negative[rr].astype(np.int32)
        same[replicate] = lp.T @ rp + ln.T @ rn
        opposite[replicate] = lp.T @ rn + ln.T @ rp
        lw = left.weight[lr].astype(np.float64)
        rw = right.weight[rr].astype(np.float64)
        lwp = np.clip(lw, 0.0, None)
        lwn = np.clip(-lw, 0.0, None)
        rwp = np.clip(rw, 0.0, None)
        rwn = np.clip(-rw, 0.0, None)
        same_strength[replicate] = (
            (similarity[:, None] * lwp).T @ rwp
            + (similarity[:, None] * lwn).T @ rwn
        )
        opposite_strength[replicate] = (
            (similarity[:, None] * lwp).T @ rwn
            + (similarity[:, None] * lwn).T @ rwp
        )
    return same, opposite, same_strength, opposite_strength


def stabilised_z(observed: np.ndarray, null: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = null.mean(axis=0)
    sd = null.std(axis=0, ddof=1)
    z = (observed - mean) / np.sqrt(sd * sd + 0.5)
    return z, mean, sd


def empirical_upper_p(observed: np.ndarray, null: np.ndarray) -> np.ndarray:
    return (1.0 + (null >= observed[None, ...]).sum(axis=0)) / (null.shape[0] + 1.0)


def score_open_candidates(
    profiles: dict[str, StudyProfiles],
    donors: dict[str, np.ndarray],
    mutual_edges: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, tuple[np.ndarray, np.ndarray]]]:
    rows: list[dict[str, object]] = []
    null_by_undirected: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    row_id = 0
    edge_blocks = {
        (str(left), str(right), str(world)): block
        for (left, right, world), block in mutual_edges.groupby(
            ["left_study_id", "right_study_id", "evidence_world"], sort=False
        )
    }
    empty = mutual_edges.iloc[0:0].copy()
    for left_study, right_study in combinations(sorted(profiles), 2):
        left = profiles[left_study]
        right = profiles[right_study]
        left_lookup = {value: index for index, value in enumerate(left.entities)}
        right_lookup = {value: index for index, value in enumerate(right.entities)}
        for world in WORLDS:
            base = edge_blocks.get((left_study, right_study, world), empty)
            for track in TRACKS:
                block = base.loc[track_mask(base, track)].reset_index(drop=True)
                if len(block):
                    left_rows = np.asarray(
                        [left_lookup[str(value)] for value in block["left_entity_id"]],
                        dtype=np.int64,
                    )
                    right_rows = np.asarray(
                        [right_lookup[str(value)] for value in block["right_entity_id"]],
                        dtype=np.int64,
                    )
                    similarity = block["chemical_similarity"].to_numpy(float)
                    same, opposite, same_strength, opposite_strength = count_and_strength(
                        left, right, left_rows, right_rows, similarity
                    )
                    (
                        null_same_count,
                        null_opposite_count,
                        null_same_strength,
                        null_opposite_strength,
                    ) = null_statistics(
                        left,
                        right,
                        left_rows,
                        right_rows,
                        similarity,
                        donors[str(left_study)],
                        donors[str(right_study)],
                    )
                else:
                    matrix_shape = (len(left.contrasts), len(right.contrasts))
                    null_shape = (donors[str(left_study)].shape[0], *matrix_shape)
                    same = np.zeros(matrix_shape, dtype=np.int32)
                    opposite = np.zeros(matrix_shape, dtype=np.int32)
                    same_strength = np.zeros(matrix_shape, dtype=float)
                    opposite_strength = np.zeros(matrix_shape, dtype=float)
                    null_same_count = np.zeros(null_shape, dtype=np.int32)
                    null_opposite_count = np.zeros(null_shape, dtype=np.int32)
                    null_same_strength = np.zeros(null_shape, dtype=np.float32)
                    null_opposite_strength = np.zeros(null_shape, dtype=np.float32)
                same_z, same_mean, same_sd = stabilised_z(
                    same_strength, null_same_strength
                )
                opposite_z, opposite_mean, opposite_sd = stabilised_z(
                    opposite_strength, null_opposite_strength
                )
                same_p = empirical_upper_p(same_strength, null_same_strength)
                opposite_p = empirical_upper_p(opposite_strength, null_opposite_strength)
                for li, left_contrast in enumerate(left.contrasts):
                    for ri, right_contrast in enumerate(right.contrasts):
                        undirected_id = row_id
                        null_by_undirected[undirected_id] = (
                            (null_same_strength[:, li, ri] - same_mean[li, ri])
                            / np.sqrt(same_sd[li, ri] ** 2 + 0.5),
                            (null_opposite_strength[:, li, ri] - opposite_mean[li, ri])
                            / np.sqrt(opposite_sd[li, ri] ** 2 + 0.5),
                        )
                        rows.append({
                            "undirected_id": undirected_id,
                            "left_study_id": str(left_study),
                            "left_contrast_id": left_contrast,
                            "right_study_id": str(right_study),
                            "right_contrast_id": right_contrast,
                            "evidence_world": str(world),
                            "evidence_track": track,
                            "mutual_edges": int(len(block)),
                            "weighted_evidence_edges": int(
                                np.count_nonzero(
                                    block["chemical_similarity"].to_numpy(float) > 0
                                )
                            ),
                            "same_direction_dual_fdr_edges": int(same[li, ri]),
                            "opposite_direction_dual_fdr_edges": int(opposite[li, ri]),
                            "same_direction_strength": float(same_strength[li, ri]),
                            "opposite_direction_strength": float(opposite_strength[li, ri]),
                            "same_strength_null_mean": float(same_mean[li, ri]),
                            "same_strength_null_sd": float(same_sd[li, ri]),
                            "opposite_strength_null_mean": float(opposite_mean[li, ri]),
                            "opposite_strength_null_sd": float(opposite_sd[li, ri]),
                            "same_count_null_mean": float(null_same_count[:, li, ri].mean()),
                            "opposite_count_null_mean": float(null_opposite_count[:, li, ri].mean()),
                            "similar_z": float(same_z[li, ri]),
                            "opposite_z": float(opposite_z[li, ri]),
                            "similar_candidate_p": float(same_p[li, ri]),
                            "opposite_candidate_p": float(opposite_p[li, ri]),
                        })
                        row_id += 1
    return pd.DataFrame(rows), null_by_undirected


def make_directed_ledger(
    undirected: pd.DataFrame,
    null_by_undirected: dict[int, tuple[np.ndarray, np.ndarray]],
    contrast_manifest: pd.DataFrame,
) -> pd.DataFrame:
    source_lookup = dict(
        zip(
            contrast_manifest["study_id"].astype(str),
            contrast_manifest["source_family"].astype(str),
        )
    )
    frames = []
    for reverse in (False, True):
        block = undirected.copy()
        if reverse:
            block = block.rename(columns={
                "left_study_id": "right_study_id",
                "left_contrast_id": "right_contrast_id",
                "right_study_id": "left_study_id",
                "right_contrast_id": "left_contrast_id",
            })
        block = block.rename(columns={
            "left_study_id": "query_study_id",
            "left_contrast_id": "query_contrast_id",
            "right_study_id": "candidate_study_id",
            "right_contrast_id": "candidate_contrast_id",
        })
        frames.append(block)
    ledger = pd.concat(frames, ignore_index=True)
    ledger.insert(
        2,
        "query_source_family",
        ledger["query_study_id"].map(source_lookup),
    )
    ledger.insert(
        5,
        "candidate_source_family",
        ledger["candidate_study_id"].map(source_lookup),
    )
    group_columns = [
        "query_study_id",
        "query_contrast_id",
        "evidence_world",
        "evidence_track",
    ]
    ledger["candidate_universe_size"] = ledger.groupby(group_columns)[
        "candidate_study_id"
    ].transform("size").astype(int)
    ledger["score_available"] = ledger["mutual_edges"].gt(0)
    for face, score_column in (("similar", "similar_z"), ("opposite", "opposite_z")):
        rank_column = f"{face}_rank"
        p_column = f"{face}_query_maxT_p"
        rank_values = ledger[score_column].where(ledger["score_available"])
        ledger[rank_column] = rank_values.groupby(
            [ledger[column] for column in group_columns]
        ).rank(method="min", ascending=False).astype("Int64")
        ledger[p_column] = np.nan
        null_index = 0 if face == "similar" else 1
        for _key, indices in ledger.groupby(group_columns, sort=False).groups.items():
            ordered = np.asarray(
                [i for i in indices if bool(ledger.loc[i, "score_available"])],
                dtype=np.int64,
            )
            if not len(ordered):
                ledger.loc[list(indices), p_column] = 1.0
                continue
            null_stack = np.vstack(
                [null_by_undirected[int(ledger.loc[i, "undirected_id"])][null_index] for i in ordered]
            )
            max_null = null_stack.max(axis=0)
            observed = ledger.loc[ordered, score_column].to_numpy(float)
            adjusted = np.asarray(
                [(1.0 + np.sum(max_null >= value)) / (len(max_null) + 1.0) for value in observed]
            )
            ledger.loc[ordered, p_column] = adjusted

    expected = contrast_manifest.set_index(["study_id", "contrast_id"])[
        "source_disjoint_candidate_contrasts"
    ].astype(int)
    for row in ledger.drop_duplicates(group_columns).itertuples(index=False):
        target = int(expected.loc[(str(row.query_study_id), str(row.query_contrast_id))])
        if int(row.candidate_universe_size) != target:
            raise RuntimeError(
                f"candidate denominator drift for {row.query_study_id}/{row.query_contrast_id}: "
                f"{row.candidate_universe_size} != {target}"
            )
    return ledger.sort_values(
        [*group_columns, "similar_rank", "candidate_study_id", "candidate_contrast_id"]
    ).reset_index(drop=True)


def top_hits(
    ledger: pd.DataFrame,
    minimum_support: int,
    query_alpha: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = [
        "query_study_id",
        "query_source_family",
        "query_contrast_id",
        "evidence_world",
        "evidence_track",
    ]
    for key, block in ledger.groupby(group_columns, sort=True):
        for face in FACES:
            rank_column = f"{face}_rank"
            available = block.loc[block["score_available"]]
            ranked = available if len(available) else block
            hit = ranked.sort_values(
                [rank_column, "candidate_study_id", "candidate_contrast_id"]
            ).iloc[0]
            support_column = (
                "same_direction_dual_fdr_edges"
                if face == "similar"
                else "opposite_direction_dual_fdr_edges"
            )
            p_column = f"{face}_query_maxT_p"
            score_column = f"{face}_z"
            support = int(hit[support_column])
            if (
                not bool(hit["score_available"])
                or int(hit["weighted_evidence_edges"]) < minimum_support
            ):
                call = "abstain_insufficient_evidence"
            elif float(hit[p_column]) <= query_alpha and float(hit[score_column]) > 0:
                call = f"{face}_response"
            else:
                call = "no_calibrated_match"
            record = {column: value for column, value in zip(group_columns, key)}
            record.update({
                "response_face": face,
                "result_class": call,
                "adjudication_status": "unadjudicated_hypothesis",
                "candidate_study_id": hit["candidate_study_id"],
                "candidate_source_family": hit["candidate_source_family"],
                "candidate_contrast_id": hit["candidate_contrast_id"],
                "candidate_universe_size": int(hit["candidate_universe_size"]),
                "mutual_edges": int(hit["mutual_edges"]),
                "weighted_evidence_edges": int(hit["weighted_evidence_edges"]),
                "dual_fdr_support_edges": support,
                "enrichment_z": float(hit[score_column]),
                "candidate_empirical_p": float(hit[f"{face}_candidate_p"]),
                "query_maxT_p": float(hit[p_column]),
            })
            rows.append(record)
    return pd.DataFrame(rows)


def audit_registered_relations(
    ledger: pd.DataFrame,
    relation_path: Path,
) -> pd.DataFrame:
    relations = pd.read_csv(relation_path, sep="\t", dtype=str)
    records: list[dict[str, object]] = []
    for relation in relations.to_dict("records"):
        matched = ledger.loc[
            ledger["query_study_id"].eq(str(relation["left_study_id"]))
            & ledger["query_contrast_id"].eq(str(relation["left_contrast_id"]))
            & ledger["candidate_study_id"].eq(str(relation["right_study_id"]))
            & ledger["candidate_contrast_id"].eq(str(relation["right_contrast_id"]))
        ]
        if len(matched) != len(WORLDS) * len(TRACKS):
            raise RuntimeError(f"registered relation did not match the full score matrix: {relation}")
        for row in matched.itertuples(index=False):
            record = dict(relation)
            for column in (
                "evidence_world",
                "evidence_track",
                "mutual_edges",
                "weighted_evidence_edges",
                "same_direction_dual_fdr_edges",
                "opposite_direction_dual_fdr_edges",
                "similar_z",
                "opposite_z",
                "similar_candidate_p",
                "opposite_candidate_p",
                "similar_query_maxT_p",
                "opposite_query_maxT_p",
                "similar_rank",
                "opposite_rank",
                "candidate_universe_size",
            ):
                record[column] = getattr(row, column)
            records.append(record)
    result = pd.DataFrame(records)
    result["best_face_rank"] = result[["similar_rank", "opposite_rank"]].min(axis=1)
    result["best_face_query_maxT_p"] = result[
        ["similar_query_maxT_p", "opposite_query_maxT_p"]
    ].min(axis=1)
    return result


def supporting_edge_details(
    top: pd.DataFrame,
    mutual_edges: pd.DataFrame,
    profiles: dict[str, StudyProfiles],
    top_n: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    selected = top.loc[
        top["evidence_world"].isin(["all", "unresolved"])
        & top["evidence_track"].isin(
            [
                "same_precursor_10ppm_identity_hypothesis",
                "nonmass_chemical_relation_hypothesis",
            ]
        )
    ]
    for hit in selected.itertuples(index=False):
        qstudy = str(hit.query_study_id)
        cstudy = str(hit.candidate_study_id)
        low, high = sorted((qstudy, cstudy))
        block = mutual_edges.loc[
            mutual_edges["left_study_id"].eq(low)
            & mutual_edges["right_study_id"].eq(high)
            & mutual_edges["evidence_world"].eq(str(hit.evidence_world))
        ].copy()
        block = block.loc[track_mask(block, str(hit.evidence_track))]
        if not len(block):
            continue
        if qstudy == low:
            query_entities = block["left_entity_id"].astype(str).to_numpy()
            candidate_entities = block["right_entity_id"].astype(str).to_numpy()
            query_mz = block["left_precursor_mz"].to_numpy(float)
            candidate_mz = block["right_precursor_mz"].to_numpy(float)
        else:
            query_entities = block["right_entity_id"].astype(str).to_numpy()
            candidate_entities = block["left_entity_id"].astype(str).to_numpy()
            query_mz = block["right_precursor_mz"].to_numpy(float)
            candidate_mz = block["left_precursor_mz"].to_numpy(float)
        qp = profiles[qstudy]
        cp = profiles[cstudy]
        qerow = {value: index for index, value in enumerate(qp.entities)}
        cerow = {value: index for index, value in enumerate(cp.entities)}
        qcol = qp.contrasts.index(str(hit.query_contrast_id))
        ccol = cp.contrasts.index(str(hit.candidate_contrast_id))
        qrows = np.asarray([qerow[x] for x in query_entities], dtype=np.int64)
        crows = np.asarray([cerow[x] for x in candidate_entities], dtype=np.int64)
        qsign = np.sign(qp.beta[qrows, qcol])
        csign = np.sign(cp.beta[crows, ccol])
        if str(hit.response_face) == "similar":
            supported = (qsign != 0) & (qsign == csign)
        else:
            supported = (qsign != 0) & (qsign == -csign)
        if not supported.any():
            continue
        contribution = (
            block["chemical_similarity"].to_numpy(float)
            * np.abs(qp.weight[qrows, qcol])
            * np.abs(cp.weight[crows, ccol])
        )
        order = np.flatnonzero(supported)
        order = order[np.argsort(-contribution[order], kind="stable")[:top_n]]
        for evidence_rank, index in enumerate(order, start=1):
            records.append({
                "query_study_id": qstudy,
                "query_contrast_id": str(hit.query_contrast_id),
                "candidate_study_id": cstudy,
                "candidate_contrast_id": str(hit.candidate_contrast_id),
                "evidence_world": str(hit.evidence_world),
                "evidence_track": str(hit.evidence_track),
                "response_face": str(hit.response_face),
                "result_class": str(hit.result_class),
                "evidence_rank": evidence_rank,
                "query_entity_id": query_entities[index],
                "candidate_entity_id": candidate_entities[index],
                "query_precursor_mz": float(query_mz[index]),
                "candidate_precursor_mz": float(candidate_mz[index]),
                "precursor_ppm_difference": float(
                    block.iloc[index]["precursor_ppm_difference"]
                ),
                "chemical_similarity": float(
                    block.iloc[index]["chemical_similarity"]
                ),
                "query_beta": float(qp.beta[qrows[index], qcol]),
                "query_se": float(qp.se[qrows[index], qcol]),
                "query_fdr": float(qp.q[qrows[index], qcol]),
                "candidate_beta": float(cp.beta[crows[index], ccol]),
                "candidate_se": float(cp.se[crows[index], ccol]),
                "candidate_fdr": float(cp.q[crows[index], ccol]),
                "weighted_edge_contribution": float(contribution[index]),
            })
    return pd.DataFrame(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--identity-partition", required=True, type=Path)
    parser.add_argument("--alignment-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--permutations", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--effect-fdr", type=float, default=0.05)
    parser.add_argument("--query-alpha", type=float, default=0.05)
    parser.add_argument("--minimum-support", type=int, default=3)
    parser.add_argument("--precursor-ppm", type=float, default=10.0)
    parser.add_argument("--top-evidence-per-hit", type=int, default=10)
    parser.add_argument(
        "--algorithm-edge-scores",
        type=Path,
        help="Optional exact-axis score table; it may reweight but not add/drop edges",
    )
    parser.add_argument("--algorithm-score-column")
    parser.add_argument("--algorithm-method-name", default="base_fragment_cosine")
    parser.add_argument(
        "--missing-algorithm-score",
        choices=["zero", "base", "error"],
        default="zero",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.permutations < 19:
        raise ValueError("at least 19 permutations are required for alpha 0.05")
    if (args.algorithm_edge_scores is None) != (args.algorithm_score_column is None):
        raise ValueError(
            "--algorithm-edge-scores and --algorithm-score-column must be supplied together"
        )

    paths = {
        "entities": args.collection / "response_entities.tsv.gz",
        "contrasts": args.collection / "contrast_manifest.tsv",
        "relations": args.collection / "registered_relations.tsv",
        "edges": args.alignment_dir / "source_pair_frozen_edges.tsv.gz",
        "alignment_report": args.alignment_dir / "report.json",
        "identity_report": args.identity_partition.parent / "report.json",
    }
    for path in (*paths.values(), args.identity_partition):
        if not path.is_file():
            raise FileNotFoundError(path)
    alignment_report = json.loads(paths["alignment_report"].read_text(encoding="utf-8"))
    identity_report = json.loads(paths["identity_report"].read_text(encoding="utf-8"))
    if alignment_report["provenance"]["edges"] != sha256(paths["edges"]):
        raise RuntimeError("frozen spectral edge checksum drift")
    if alignment_report["provenance"]["response_entities_sha256"] != sha256(paths["entities"]):
        raise RuntimeError("response collection checksum drift")
    if identity_report["provenance"]["combined_partition_sha256"] != sha256(
        args.identity_partition
    ):
        raise RuntimeError("common identity partition checksum drift")

    effects = pd.read_csv(paths["entities"], sep="\t", low_memory=False)
    effects["study_id"] = effects["study_id"].astype(str)
    effects["entity_id"] = effects["entity_id"].astype(str)
    raw_registry = effects[["study_id", "raw_entity_id", "entity_id"]].drop_duplicates()
    raw_registry["raw_entity_id"] = raw_registry["raw_entity_id"].map(normalise_entity_id)
    identity = pd.read_csv(args.identity_partition, sep="\t", dtype=str)
    effects = effects.drop(columns=["identity_status"], errors="ignore").merge(
        identity[["study_id", "entity_id", "identity_status"]],
        on=["study_id", "entity_id"],
        how="left",
        validate="many_to_one",
    )
    if effects["identity_status"].isna().any():
        raise RuntimeError("common identity partition does not cover all response rows")
    effects = add_effect_statistics(effects, args.effect_fdr)
    contrasts = pd.read_csv(paths["contrasts"], sep="\t")
    contrasts["study_id"] = contrasts["study_id"].astype(str)
    contrasts["contrast_id"] = contrasts["contrast_id"].astype(str)
    profiles = build_profiles(effects, contrasts, args.effect_fdr)
    donors = build_profile_reassignments(profiles, args.permutations, args.seed)
    precursors, precursor_summary = load_precursors(
        alignment_report, raw_registry
    )
    mutual_edges = canonical_mutual_edges(
        paths["edges"], precursors, args.precursor_ppm
    )
    algorithm_score_provenance: dict[str, object] = {
        "method_name": args.algorithm_method_name,
        "score_source": "upstream fragment cosine frozen before response scoring",
        "edge_rows": len(mutual_edges),
        "edge_rows_available": len(mutual_edges),
        "edge_rows_unavailable": 0,
    }
    if args.algorithm_edge_scores is not None:
        mutual_edges, replacement = apply_algorithm_edge_scores(
            mutual_edges,
            args.algorithm_edge_scores,
            args.algorithm_score_column,
            args.missing_algorithm_score,
        )
        algorithm_score_provenance.update(replacement)

    undirected, nulls = score_open_candidates(profiles, donors, mutual_edges)
    ledger = make_directed_ledger(undirected, nulls, contrasts)

    args.output_dir.mkdir(parents=True)
    outputs = {
        "edge_handoff": args.output_dir / "frozen_mutual_edge_algorithm_handoff.tsv.gz",
        "scores": args.output_dir / "open_local_edge_candidate_scores.tsv.gz",
        "top_hits": args.output_dir / "open_local_edge_top_hits.tsv",
        "registered": args.output_dir / "registered_relation_ranks.tsv",
        "evidence": args.output_dir / "top_hit_contributing_edges.tsv.gz",
    }
    compression = {"method": "gzip", "mtime": 0}
    mutual_edges.to_csv(
        outputs["edge_handoff"], sep="\t", index=False, compression=compression
    )
    # Scores are frozen before the registered relation table is opened.
    ledger.to_csv(outputs["scores"], sep="\t", index=False, compression=compression)
    score_hash_before_truth = sha256(outputs["scores"])
    top = top_hits(ledger, args.minimum_support, args.query_alpha)
    top.to_csv(outputs["top_hits"], sep="\t", index=False)
    registered = audit_registered_relations(ledger, paths["relations"])
    registered.to_csv(outputs["registered"], sep="\t", index=False)
    evidence = supporting_edge_details(
        top, mutual_edges, profiles, args.top_evidence_per_hit
    )
    evidence.to_csv(outputs["evidence"], sep="\t", index=False, compression=compression)

    call_summary = (
        top.groupby(["evidence_world", "evidence_track", "response_face", "result_class"])
        .size()
        .rename("queries")
        .reset_index()
        .to_dict("records")
    )
    relation_summary = []
    primary_registered = registered.loc[
        registered["eligible_primary_relevance"].astype(str).str.lower().eq("true")
    ]
    for (world, track), block in primary_registered.groupby(
        ["evidence_world", "evidence_track"], sort=True
    ):
        relation_summary.append({
            "evidence_world": world,
            "evidence_track": track,
            "registered_directional_relations": int(len(block)),
            "best_face_top1": int(block["best_face_rank"].eq(1).sum()),
            "best_face_top5": int(block["best_face_rank"].le(5).sum()),
            "median_best_face_rank": float(block["best_face_rank"].median()),
            "query_maxT_005": int(block["best_face_query_maxT_p"].le(args.query_alpha).sum()),
        })

    report = {
        "schema": "contrast_atlas_open_local_edge_enrichment_v2_continuous",
        "status": "DEVELOPMENT_CONSUMED_OPEN_LOCAL_EDGE_CONTINUOUS_RETRIEVAL_COMPLETE",
        "data_role": "development_consumed",
        "method": {
            "unit": "study-level directed response contrast",
            "chemical_graph": (
                "upstream phenotype-blind mutual top-1 fragment-cosine graph; "
                "the exact edge set is fixed while the named algorithm may replace edge strength"
            ),
            "edge_strength_method": args.algorithm_method_name,
            "response_evidence": "continuous beta/sqrt(beta^2+SE^2) on each endpoint; same and opposite sign energies kept separate",
            "local_support": "mutual spectral edges contribute locally; absent edges remain zero and distinct entities are never merged",
            "null": "whole multi-contrast entity response profiles reassigned within study and common identity stratum",
            "candidate_calibration": "candidate-specific permutation z and empirical upper-tail p",
            "open_query_calibration": "query-level max-T across the complete source-disjoint candidate universe",
            "identity_track": f"mutual top-1 and precursor compatible within {args.precursor_ppm} ppm; identity hypothesis, not proof",
            "relation_track": f"mutual top-1 and valid precursor difference greater than {args.precursor_ppm} ppm; distinct-response association, never effect meta-analysis",
            "minimum_support_edges_for_answer": args.minimum_support,
            "query_alpha": args.query_alpha,
            "endpoint_fdr_role": (
                f"two-sided normal-approximation BH FDR <= {args.effect_fdr} edge counts are diagnostic only and do not enter ranking, "
                "because several small-n source summaries do not retain enough variance components to reconstruct exact t degrees of freedom"
            ),
        },
        "algorithm_handoff": {
            "pair_manifest": outputs["edge_handoff"].name,
            "rows": int(len(mutual_edges)),
            "join_key": [
                "left_study_id", "right_study_id", "evidence_world",
                "left_entity_id", "right_entity_id",
            ],
            "contract": (
                "a replacement spectral model may update score/evidence columns on these exact keys; "
                "it may not add or remove pairs when testing algorithm contribution to the fixed response task"
            ),
            "applied_score": algorithm_score_provenance,
        },
        "denominators": {
            "independent_studies": int(contrasts["study_id"].nunique()),
            "source_families": int(contrasts["source_family"].nunique()),
            "study_level_contrasts": int(len(contrasts)),
            "response_rows": int(len(effects)),
            "unique_study_entities": int(effects[["study_id", "entity_id"]].drop_duplicates().shape[0]),
            "frozen_directional_edges": int(alignment_report["denominators"]["frozen_directional_edges"]),
            "canonical_mutual_edges": int(len(mutual_edges)),
            "mutual_identity_hypothesis_edges": int(mutual_edges["precursor_compatible_10ppm"].sum()),
            "mutual_nonmass_relation_edges": int(mutual_edges["precursor_incompatible_over_10ppm"].sum()),
            "directed_query_candidate_pairs": int(
                ledger[[
                    "query_study_id", "query_contrast_id",
                    "candidate_study_id", "candidate_contrast_id",
                ]].drop_duplicates().shape[0]
            ),
            "candidate_score_rows": int(len(ledger)),
            "top_hit_rows": int(len(top)),
            "permutations": args.permutations,
            "registered_directional_relations": int(len(pd.read_csv(paths["relations"], sep="\t"))),
            "registered_signed_truth_relations": int(
                (~pd.read_csv(paths["relations"], sep="\t")["expected_signed_direction"].astype(str).str.lower().eq("unspecified")).sum()
            ),
        },
        "open_query_calls": call_summary,
        "registered_relation_descriptive_performance": relation_summary,
        "precursor_coverage": precursor_summary,
        "leakage_status": (
            "candidate spectral edges, common identity strata and the complete candidate universe were frozen upstream; "
            "registered biological relations were opened only after the complete score ledger was written and hashed"
        ),
        "claim_boundary": [
            "All 15 sources and all 51 contrasts are development/consumed; this is not sealed external validation.",
            "No registered relation supplies signed metabolite-level truth; similar/opposite calls are calibrated hypotheses, not confirmed mechanisms.",
            "A learned embedding score reweights a graph admitted by the upstream fragment-cosine mutual-neighbour rule; this run does not test learned-model candidate recall.",
            "No calibrated match means no candidate passed this finite-power rule; it is not evidence that no biological relation exists.",
            "Unresolved means lacking support under the common strict library policy, not a novel or unique molecule.",
            "The 10-ppm track is a same-precursor identity hypothesis; isomers and ion-family redundancy remain possible.",
            "The nonmass track relates distinct observations and never merges their beta or standard errors.",
            "Every open top hit remains an unadjudicated hypothesis unless it is separately matched to a pre-existing eligible relation or independent experiment.",
        ],
        "provenance": {
            "response_entities_sha256": sha256(paths["entities"]),
            "contrast_manifest_sha256": sha256(paths["contrasts"]),
            "registered_relations_sha256": sha256(paths["relations"]),
            "identity_partition_sha256": sha256(args.identity_partition),
            "frozen_edges_sha256": sha256(paths["edges"]),
            **(
                {"algorithm_edge_scores_sha256": sha256(args.algorithm_edge_scores)}
                if args.algorithm_edge_scores is not None
                else {}
            ),
            "score_sha256_before_registered_relations": score_hash_before_truth,
            **{name: sha256(path) for name, path in outputs.items()},
            "implementation_sha256": sha256(Path(__file__)),
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "checksums.sha256").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in [*outputs.values(), args.output_dir / "report.json"]),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
