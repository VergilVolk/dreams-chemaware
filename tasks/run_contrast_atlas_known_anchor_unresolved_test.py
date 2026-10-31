#!/usr/bin/env python
"""Entity-disjoint test of unresolved response information in Contrast Atlas.

Full InChIKey library-supported entities define a hidden signed-response
anchor.  A disjoint set of unresolved--unresolved reciprocal WSE links is then
asked to recover the anchor's similar or opposite response face.  Library
identity, semantic condition names, and registered relations never enter
unresolved pairing or target selection.  The response weights necessarily use
beta/SE estimated from each study's experimental labels.

This is a consumed-development test.  A library-supported InChIKey is not an
MSI level-1 identity and a known-anchor relationship is not a mechanism.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", low_memory=False)


def signed_cosine(left: np.ndarray, right: np.ndarray, weight: np.ndarray) -> float:
    finite = np.isfinite(left) & np.isfinite(right) & np.isfinite(weight) & (weight > 0)
    if not finite.any():
        return float("nan")
    left = left[finite].astype(float)
    right = right[finite].astype(float)
    weight = weight[finite].astype(float)
    numerator = np.sum(weight * left * right)
    denominator = np.sqrt(np.sum(weight * left * left) * np.sum(weight * right * right))
    return float(numerator / denominator) if denominator > 0 else float("nan")


def add_response_weight(effects: pd.DataFrame) -> pd.DataFrame:
    effects = effects.copy()
    beta = pd.to_numeric(effects["beta"], errors="coerce").to_numpy(float)
    se = pd.to_numeric(effects["se"], errors="coerce").to_numpy(float)
    denominator = np.sqrt(beta * beta + se * se)
    effects["response_weight"] = np.divide(
        beta,
        denominator,
        out=np.full_like(beta, np.nan),
        where=np.isfinite(denominator) & (denominator > 0),
    )
    return effects


def select_known_entities(library: pd.DataFrame) -> pd.DataFrame:
    library = library.loc[
        library["library_supported"].astype(str).str.lower().eq("true")
        & library["library_inchikey"].fillna("").astype(str).str.match(
            r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$"
        )
    ].copy()
    library["wse_sort"] = pd.to_numeric(library["wse"], errors="coerce").fillna(-1)
    library["mp_sort"] = pd.to_numeric(
        library["matched_peaks"], errors="coerce"
    ).fillna(-1)
    return library.sort_values(
        ["study_id", "library_inchikey", "wse_sort", "mp_sort", "entity_id"],
        ascending=[True, True, False, False, True],
        kind="mergesort",
    ).drop_duplicates(["study_id", "library_inchikey"], keep="first")


def known_scores(effects: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    known = effects.merge(
        selected[["study_id", "entity_id", "library_inchikey"]],
        on=["study_id", "entity_id"],
        how="inner",
        validate="many_to_one",
    )
    studies = sorted(known["study_id"].unique())
    rows: list[dict[str, object]] = []
    for left_index, left_study in enumerate(studies):
        left_all = known.loc[known["study_id"].eq(left_study)]
        for right_study in studies[left_index + 1 :]:
            right_all = known.loc[known["study_id"].eq(right_study)]
            common = sorted(
                set(left_all["library_inchikey"]) & set(right_all["library_inchikey"])
            )
            if not common:
                continue
            left_common = left_all.loc[left_all["library_inchikey"].isin(common)]
            right_common = right_all.loc[right_all["library_inchikey"].isin(common)]
            for left_contrast in sorted(left_common["contrast_id"].unique()):
                left = left_common.loc[left_common["contrast_id"].eq(left_contrast), [
                    "library_inchikey", "response_weight"
                ]]
                for right_contrast in sorted(right_common["contrast_id"].unique()):
                    right = right_common.loc[right_common["contrast_id"].eq(right_contrast), [
                        "library_inchikey", "response_weight"
                    ]]
                    joined = left.merge(
                        right,
                        on="library_inchikey",
                        how="inner",
                        suffixes=("_left", "_right"),
                        validate="one_to_one",
                    ).dropna()
                    left_weight = joined["response_weight_left"].to_numpy(float)
                    right_weight = joined["response_weight_right"].to_numpy(float)
                    score = signed_cosine(
                        left_weight, right_weight, np.ones(len(joined), dtype=float)
                    )
                    loo: list[float] = []
                    if len(joined) >= 3:
                        for omitted in range(len(joined)):
                            keep = np.arange(len(joined)) != omitted
                            loo.append(
                                signed_cosine(
                                    left_weight[keep],
                                    right_weight[keep],
                                    np.ones(int(keep.sum()), dtype=float),
                                )
                            )
                    finite_loo = np.asarray([value for value in loo if np.isfinite(value)])
                    stable = bool(
                        np.isfinite(score)
                        and len(finite_loo)
                        and (
                            (score > 0 and np.all(finite_loo > 0))
                            or (score < 0 and np.all(finite_loo < 0))
                        )
                    )
                    rows.append(
                        {
                            "left_study_id": left_study,
                            "left_contrast_id": left_contrast,
                            "right_study_id": right_study,
                            "right_contrast_id": right_contrast,
                            "known_exact_inchikey_support": int(len(joined)),
                            "known_signed_cosine": score,
                            "known_loo_min": (
                                float(np.min(finite_loo)) if len(finite_loo) else np.nan
                            ),
                            "known_loo_max": (
                                float(np.max(finite_loo)) if len(finite_loo) else np.nan
                            ),
                            "known_leave_one_out_sign_stable": stable,
                            "known_inchikeys": ";".join(joined["library_inchikey"]),
                        }
                    )
    return pd.DataFrame(rows)


def reciprocal_unresolved_edges(edges: pd.DataFrame) -> pd.DataFrame:
    unresolved = edges.loc[edges["evidence_world"].eq("unresolved")].copy()
    keys = [
        "left_study_id",
        "right_study_id",
        "left_entity_id",
        "right_entity_id",
    ]
    counts = (
        unresolved.groupby(keys, sort=False)["direction"]
        .nunique()
        .rename("direction_count")
        .reset_index()
    )
    reciprocal = unresolved.merge(counts, on=keys, how="inner")
    reciprocal = reciprocal.loc[reciprocal["direction_count"].ge(2)]
    return (
        reciprocal.sort_values(keys + ["chemical_similarity"], ascending=[True] * 4 + [False])
        .drop_duplicates(keys, keep="first")
        .drop(columns=["direction", "evidence_world", "direction_count"])
    )


def unresolved_scores(effects: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    effect_lookup = {
        study: block[["entity_id", "contrast_id", "response_weight"]].copy()
        for study, block in effects.groupby("study_id", sort=False)
    }
    rows: list[dict[str, object]] = []
    for (left_study, right_study), block in edges.groupby(
        ["left_study_id", "right_study_id"], sort=False
    ):
        if left_study not in effect_lookup or right_study not in effect_lookup:
            continue
        left_effect = effect_lookup[left_study]
        right_effect = effect_lookup[right_study]
        left_contrasts = sorted(left_effect["contrast_id"].unique())
        right_contrasts = sorted(right_effect["contrast_id"].unique())
        left_map = {
            contrast: series.set_index("entity_id")["response_weight"]
            for contrast, series in left_effect.groupby("contrast_id", sort=False)
        }
        right_map = {
            contrast: series.set_index("entity_id")["response_weight"]
            for contrast, series in right_effect.groupby("contrast_id", sort=False)
        }
        similarity = block["chemical_similarity"].to_numpy(float)
        left_ids = block["left_entity_id"].astype(str)
        right_ids = block["right_entity_id"].astype(str)
        for left_contrast in left_contrasts:
            left_weight = left_ids.map(left_map[left_contrast]).to_numpy(float)
            for right_contrast in right_contrasts:
                right_weight = right_ids.map(right_map[right_contrast]).to_numpy(float)
                finite = (
                    np.isfinite(left_weight)
                    & np.isfinite(right_weight)
                    & np.isfinite(similarity)
                    & (similarity > 0)
                )
                rows.append(
                    {
                        "left_study_id": left_study,
                        "left_contrast_id": left_contrast,
                        "right_study_id": right_study,
                        "right_contrast_id": right_contrast,
                        "unresolved_mutual_edge_support": int(finite.sum()),
                        "unresolved_signed_cosine": signed_cosine(
                            left_weight, right_weight, similarity
                        ),
                    }
                )
    return pd.DataFrame(rows)


def cluster_bootstrap(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    replicates: int,
    seed: int,
) -> dict[str, float | int | None]:
    frame = frame.copy()
    frame["source_pair"] = (
        frame["left_study_id"].astype(str) + "||" + frame["right_study_id"].astype(str)
    )
    groups = {key: block for key, block in frame.groupby("source_pair", sort=True)}
    keys = sorted(groups)
    observed = spearmanr(
        frame[x_column], frame[y_column]
    ).statistic
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(replicates):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        blocks = [groups[key].assign(_draw=index) for index, key in enumerate(sampled)]
        draw = pd.concat(blocks, ignore_index=True)
        value = spearmanr(
            draw[x_column], draw[y_column]
        ).statistic
        if np.isfinite(value):
            values.append(float(value))
    return {
        "spearman_rho": float(observed) if np.isfinite(observed) else None,
        "source_pair_clusters": len(keys),
        "rows": len(frame),
        "bootstrap_replicates_requested": replicates,
        "bootstrap_replicates_finite": len(values),
        "cluster_bootstrap_ci95_low": float(np.quantile(values, 0.025)) if values else None,
        "cluster_bootstrap_ci95_high": float(np.quantile(values, 0.975)) if values else None,
    }


def leave_one_study_out(
    frame: pd.DataFrame, x_column: str, y_column: str
) -> dict[str, float | int | None]:
    """Sensitivity to dyadic dependence from studies reused across source pairs."""
    studies = sorted(
        set(frame["left_study_id"].astype(str))
        | set(frame["right_study_id"].astype(str))
    )
    values: list[float] = []
    for study in studies:
        retained = frame.loc[
            frame["left_study_id"].astype(str).ne(study)
            & frame["right_study_id"].astype(str).ne(study)
        ]
        value = spearmanr(retained[x_column], retained[y_column]).statistic
        if np.isfinite(value):
            values.append(float(value))
    return {
        "studies": len(studies),
        "finite_leave_one_study_out_estimates": len(values),
        "minimum_rho": float(np.min(values)) if values else None,
        "maximum_rho": float(np.max(values)) if values else None,
        "median_rho": float(np.median(values)) if values else None,
    }


def add_specificity_residuals(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove source-pair and contrast-axis main effects from both scores."""
    frame = frame.copy()
    pair = ["left_study_id", "right_study_id"]
    left_axis = pair + ["left_contrast_id"]
    right_axis = pair + ["right_contrast_id"]
    for prefix in ("known", "unresolved"):
        column = f"{prefix}_signed_cosine"
        pair_mean = frame.groupby(pair, sort=False)[column].transform("mean")
        left_mean = frame.groupby(left_axis, sort=False)[column].transform("mean")
        right_mean = frame.groupby(right_axis, sort=False)[column].transform("mean")
        frame[f"{prefix}_source_pair_centered"] = frame[column] - pair_mean
        frame[f"{prefix}_candidate_axis_centered"] = frame[column] - left_mean
        frame[f"{prefix}_query_axis_centered"] = frame[column] - right_mean
        frame[f"{prefix}_double_centered"] = (
            frame[column] - left_mean - right_mean + pair_mean
        )
    pair_size = frame.groupby(pair, sort=False)["known_signed_cosine"].transform("size")
    left_size = frame.groupby(left_axis, sort=False)["known_signed_cosine"].transform("size")
    right_size = frame.groupby(right_axis, sort=False)["known_signed_cosine"].transform("size")
    frame["source_pair_centering_eligible"] = pair_size.ge(2)
    frame["candidate_axis_centering_eligible"] = left_size.ge(2)
    frame["query_axis_centering_eligible"] = right_size.ge(2)
    frame["double_centering_eligible"] = left_size.ge(2) & right_size.ge(2)
    return frame


def axis_label_permuted_spearman(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
    replicates: int,
    seed: int,
) -> dict[str, float | int | None]:
    """Permute whole contrast axes while preserving each response matrix.

    Cell-wise shuffling is invalid because contrasts are reused along matrix
    rows and columns.  Within each complete source-pair matrix, this null
    independently permutes the unresolved matrix's left and right labels.
    """
    blocks: list[pd.DataFrame] = []
    matrices: list[np.ndarray] = []
    for _, block in frame.groupby(["left_study_id", "right_study_id"], sort=True):
        left = sorted(block["left_contrast_id"].astype(str).unique())
        right = sorted(block["right_contrast_id"].astype(str).unique())
        if len(block) != len(left) * len(right) or len(block) < 2:
            continue
        ordered = block.assign(
            left_contrast_id=block["left_contrast_id"].astype(str),
            right_contrast_id=block["right_contrast_id"].astype(str),
        ).sort_values(["left_contrast_id", "right_contrast_id"])
        if not (
            np.isfinite(ordered[x_column].to_numpy(float)).all()
            and np.isfinite(ordered[y_column].to_numpy(float)).all()
        ):
            continue
        matrix = (
            ordered.pivot(
                index="left_contrast_id", columns="right_contrast_id", values=y_column
            )
            .loc[left, right]
            .to_numpy(float)
        )
        blocks.append(ordered)
        matrices.append(matrix)
    if not blocks:
        return {
            "observed_rho": None,
            "rows": 0,
            "exchangeable_source_pairs": 0,
            "permutation_p_one_sided": None,
        }
    permutation_frame = pd.concat(blocks, ignore_index=True)
    x = permutation_frame[x_column].to_numpy(float)
    y = permutation_frame[y_column].to_numpy(float)
    observed = spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    null = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        shuffled = y.copy()
        offset = 0
        for matrix in matrices:
            permuted = matrix[rng.permutation(matrix.shape[0]), :][
                :, rng.permutation(matrix.shape[1])
            ]
            size = matrix.size
            shuffled[offset : offset + size] = permuted.reshape(-1)
            offset += size
        null[replicate] = spearmanr(x, shuffled).statistic
    finite = null[np.isfinite(null)]
    return {
        "observed_rho": float(observed),
        "rows": int(len(permutation_frame)),
        "exchangeable_source_pairs": int(len(blocks)),
        "permutations": int(replicates),
        "permutation_p_one_sided": float(
            (1 + np.sum(finite >= observed)) / (1 + len(finite))
        ),
        "null_q025": float(np.quantile(finite, 0.025)),
        "null_median": float(np.quantile(finite, 0.5)),
        "null_q975": float(np.quantile(finite, 0.975)),
    }


def retrieval_cluster_bootstrap(
    retrieval: pd.DataFrame, replicates: int, seed: int
) -> dict[str, dict[str, float | int | None]]:
    if retrieval.empty:
        return {}
    retrieval = retrieval.copy()
    retrieval["source_pair"] = [
        "||".join(sorted((str(query), str(target))))
        for query, target in zip(retrieval["query_study_id"], retrieval["target_study_id"])
    ]
    groups = {key: block for key, block in retrieval.groupby("source_pair", sort=True)}
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    result: dict[str, dict[str, float | int | None]] = {}
    for k in (1, 3, 5):
        observed = float(retrieval[f"top{k}"].sum())
        expected = float(
            (np.minimum(k, retrieval["candidate_universe"]) / retrieval["candidate_universe"]).sum()
        )
        differences: list[float] = []
        for _ in range(replicates):
            sampled = rng.choice(keys, size=len(keys), replace=True)
            draw = pd.concat([groups[key] for key in sampled], ignore_index=True)
            draw_observed = float(draw[f"top{k}"].sum())
            draw_expected = float(
                (np.minimum(k, draw["candidate_universe"]) / draw["candidate_universe"]).sum()
            )
            differences.append(draw_observed - draw_expected)
        result[f"top{k}"] = {
            "observed": int(observed),
            "uniform_random_expected": expected,
            "observed_minus_expected": observed - expected,
            "source_pair_clusters": len(keys),
            "cluster_bootstrap_ci95_low": float(np.quantile(differences, 0.025)),
            "cluster_bootstrap_ci95_high": float(np.quantile(differences, 0.975)),
        }
    return result


def retrieval_rows(
    joined: pd.DataFrame,
    unresolved_scores_all: pd.DataFrame,
    contrasts: pd.DataFrame,
    parent: dict[str, str],
    min_known: int,
    min_unresolved: int,
    anchor_abs: float,
) -> pd.DataFrame:
    score_lookup: dict[tuple[str, str, str, str], tuple[float, int]] = {}
    # The open candidate universe is defined by unresolved evidence alone.  It must
    # not be restricted to study pairs that also happen to share a known InChIKey.
    for row in unresolved_scores_all.itertuples(index=False):
        key = (
            str(row.left_study_id),
            str(row.left_contrast_id),
            str(row.right_study_id),
            str(row.right_contrast_id),
        )
        value = (float(row.unresolved_signed_cosine), int(row.unresolved_mutual_edge_support))
        score_lookup[key] = value
        score_lookup[(key[2], key[3], key[0], key[1])] = value

    contrast_rows = [
        (str(row.study_id), str(row.contrast_id)) for row in contrasts.itertuples(index=False)
    ]
    anchors = joined.loc[
        joined["known_exact_inchikey_support"].ge(min_known)
        & joined["known_leave_one_out_sign_stable"].eq(True)
        & joined["known_signed_cosine"].abs().ge(anchor_abs)
    ]
    output: list[dict[str, object]] = []
    for anchor in anchors.itertuples(index=False):
        for reverse in (False, True):
            query_study = str(anchor.right_study_id if reverse else anchor.left_study_id)
            query_contrast = str(
                anchor.right_contrast_id if reverse else anchor.left_contrast_id
            )
            target_study = str(anchor.left_study_id if reverse else anchor.right_study_id)
            target_contrast = str(
                anchor.left_contrast_id if reverse else anchor.right_contrast_id
            )
            sign = 1.0 if float(anchor.known_signed_cosine) > 0 else -1.0
            candidates = [
                (study, contrast)
                for study, contrast in contrast_rows
                if study != query_study
                and parent.get(study, study) != parent.get(query_study, query_study)
            ]
            values: list[tuple[str, str, float, int]] = []
            for study, contrast in candidates:
                score, support = score_lookup.get(
                    (query_study, query_contrast, study, contrast), (float("nan"), 0)
                )
                face_score = sign * score if np.isfinite(score) and support >= min_unresolved else float("nan")
                values.append((study, contrast, face_score, support))
            target = next(
                (item for item in values if item[0] == target_study and item[1] == target_contrast),
                None,
            )
            target_score = target[2] if target is not None else float("nan")
            finite_scores = [item[2] for item in values if np.isfinite(item[2])]
            if np.isfinite(target_score):
                # Worst rank among exact score ties; avoids optimistic tie breaking.
                rank = 1 + sum(value >= target_score - 1e-12 for value in finite_scores) - 1
            else:
                rank = len(candidates) + 1
            output.append(
                {
                    "query_study_id": query_study,
                    "query_contrast_id": query_contrast,
                    "target_study_id": target_study,
                    "target_contrast_id": target_contrast,
                    "anchor_face": "similar" if sign > 0 else "opposite",
                    "known_signed_cosine": float(anchor.known_signed_cosine),
                    "known_exact_inchikey_support": int(
                        anchor.known_exact_inchikey_support
                    ),
                    "unresolved_signed_cosine": (
                        float(anchor.unresolved_signed_cosine)
                        if np.isfinite(anchor.unresolved_signed_cosine)
                        else np.nan
                    ),
                    "unresolved_mutual_edge_support": int(
                        anchor.unresolved_mutual_edge_support
                    ),
                    "candidate_universe": len(candidates),
                    "finite_unresolved_candidates": len(finite_scores),
                    "worst_tie_rank": int(rank),
                    "top1": bool(rank <= 1),
                    "top3": bool(rank <= 3),
                    "top5": bool(rank <= 5),
                }
            )
    return pd.DataFrame(output)


def write_checksums(output: Path, files: list[Path]) -> None:
    with (output / "checksums.sha256").open("w", encoding="utf-8", newline="\n") as handle:
        for path in files:
            handle.write(f"{sha256(path)}  {path.name}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--library-evidence", type=Path, required=True)
    parser.add_argument("--edges", type=Path, required=True)
    parser.add_argument("--parent-projects", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-known", type=int, default=5)
    parser.add_argument("--min-unresolved", type=int, default=3)
    parser.add_argument("--anchor-abs", type=float, default=0.5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    effects = add_response_weight(read_tsv(args.collection / "response_entities.tsv.gz"))
    contrasts = read_tsv(args.collection / "contrast_manifest.tsv")
    library = read_tsv(args.library_evidence)
    edges = read_tsv(args.edges)
    parents = read_tsv(args.parent_projects)
    parent = dict(zip(parents["study_id"].astype(str), parents["parent_project_id"].astype(str)))

    selected = select_known_entities(library)
    known = known_scores(effects, selected)
    unresolved_edges = reciprocal_unresolved_edges(edges)
    known_entity_keys = set(
        zip(selected["study_id"].astype(str), selected["entity_id"].astype(str))
    )
    unresolved_entity_keys = set(
        zip(
            unresolved_edges["left_study_id"].astype(str),
            unresolved_edges["left_entity_id"].astype(str),
        )
    ) | set(
        zip(
            unresolved_edges["right_study_id"].astype(str),
            unresolved_edges["right_entity_id"].astype(str),
        )
    )
    entity_overlap = known_entity_keys & unresolved_entity_keys
    if entity_overlap:
        raise ValueError(
            f"known and unresolved entity sets overlap: {len(entity_overlap)} keys"
        )
    unresolved = unresolved_scores(effects, unresolved_edges)
    joined = known.merge(
        unresolved,
        on=[
            "left_study_id",
            "left_contrast_id",
            "right_study_id",
            "right_contrast_id",
        ],
        how="left",
        validate="one_to_one",
    )
    joined["same_parent_project"] = [
        parent.get(str(left), str(left)) == parent.get(str(right), str(right))
        for left, right in zip(joined["left_study_id"], joined["right_study_id"])
    ]

    eligible = joined.loc[
        ~joined["same_parent_project"]
        & joined["known_exact_inchikey_support"].ge(args.min_known)
        & joined["unresolved_mutual_edge_support"].fillna(0).ge(args.min_unresolved)
        & joined["known_signed_cosine"].notna()
        & joined["unresolved_signed_cosine"].notna()
    ].copy()
    eligible = add_specificity_residuals(eligible)
    continuous: dict[str, object] = {}
    metric_contracts = {
        "raw": (
            "known_signed_cosine",
            "unresolved_signed_cosine",
            None,
        ),
        "source_pair_centered": (
            "known_source_pair_centered",
            "unresolved_source_pair_centered",
            "source_pair_centering_eligible",
        ),
        "candidate_axis_centered": (
            "known_candidate_axis_centered",
            "unresolved_candidate_axis_centered",
            "candidate_axis_centering_eligible",
        ),
        "query_axis_centered": (
            "known_query_axis_centered",
            "unresolved_query_axis_centered",
            "query_axis_centering_eligible",
        ),
        "double_centered": (
            "known_double_centered",
            "unresolved_double_centered",
            "double_centering_eligible",
        ),
    }
    for offset, (name, (x_column, y_column, eligible_column)) in enumerate(
        metric_contracts.items()
    ):
        metric_frame = (
            eligible.copy()
            if eligible_column is None
            else eligible.loc[eligible[eligible_column]].copy()
        )
        if metric_frame.empty:
            continuous[name] = {"rows": 0}
            continue
        continuous[name] = {
            "source_pair_dyad_bootstrap_descriptive": cluster_bootstrap(
                metric_frame, x_column, y_column, args.bootstrap, args.seed + offset
            ),
            "leave_one_study_out": leave_one_study_out(
                metric_frame, x_column, y_column
            ),
            "axis_label_permutation": axis_label_permuted_spearman(
                metric_frame,
                x_column,
                y_column,
                999,
                args.seed + 100 + offset,
            ),
        }

    stable = eligible.loc[
        eligible["known_leave_one_out_sign_stable"].eq(True)
        & eligible["known_signed_cosine"].abs().ge(args.anchor_abs)
    ].copy()
    stable["sign_agreement"] = np.sign(stable["known_signed_cosine"]) == np.sign(
        stable["unresolved_signed_cosine"]
    )
    retrieval = retrieval_rows(
        joined.loc[~joined["same_parent_project"]].copy(),
        unresolved,
        contrasts,
        parent,
        args.min_known,
        args.min_unresolved,
        args.anchor_abs,
    )

    joined_path = args.output / "known_anchor_unresolved_scores.tsv.gz"
    stable_path = args.output / "stable_known_anchors.tsv"
    retrieval_path = args.output / "entity_disjoint_retrieval.tsv"
    joined.to_csv(joined_path, sep="\t", index=False, compression="gzip")
    stable.to_csv(stable_path, sep="\t", index=False)
    retrieval.to_csv(retrieval_path, sep="\t", index=False)

    retrieval_summary = {
        "directions": int(len(retrieval)),
        "top1": int(retrieval["top1"].sum()) if len(retrieval) else 0,
        "top3": int(retrieval["top3"].sum()) if len(retrieval) else 0,
        "top5": int(retrieval["top5"].sum()) if len(retrieval) else 0,
        "median_worst_tie_rank": (
            float(retrieval["worst_tie_rank"].median()) if len(retrieval) else None
        ),
        "uniform_random_expected_top1_count": (
            float((1.0 / retrieval["candidate_universe"]).sum()) if len(retrieval) else 0.0
        ),
        "uniform_random_expected_top3_count": (
            float((np.minimum(3, retrieval["candidate_universe"]) / retrieval["candidate_universe"]).sum())
            if len(retrieval)
            else 0.0
        ),
        "uniform_random_expected_top5_count": (
            float((np.minimum(5, retrieval["candidate_universe"]) / retrieval["candidate_universe"]).sum())
            if len(retrieval)
            else 0.0
        ),
        "source_pair_dyad_bootstrap_observed_minus_random_descriptive": retrieval_cluster_bootstrap(
            retrieval, args.bootstrap, args.seed + 1000
        ),
    }
    double_centered_permutation = continuous["double_centered"][
        "axis_label_permutation"
    ]
    double_centered_leave_one_study_out = continuous["double_centered"][
        "leave_one_study_out"
    ]
    top1_bootstrap = retrieval_summary[
        "source_pair_dyad_bootstrap_observed_minus_random_descriptive"
    ]["top1"]
    specificity_gate = bool(
        double_centered_permutation["permutation_p_one_sided"] is not None
        and double_centered_permutation["permutation_p_one_sided"] <= 0.05
        and double_centered_leave_one_study_out["minimum_rho"] is not None
        and double_centered_leave_one_study_out["minimum_rho"] > 0
    )
    top1_gate = bool(
        top1_bootstrap["cluster_bootstrap_ci95_low"] is not None
        and top1_bootstrap["cluster_bootstrap_ci95_low"] > 0
    )
    report = {
        "schema": "contrast_atlas_known_anchor_unresolved_entity_disjoint_v2",
        "status": "DEVELOPMENT_CONSUMED_ENTITY_DISJOINT_TEST_COMPLETE",
        "data_role": "development_consumed",
        "question": (
            "Do unresolved-only reciprocal WSE response links recover signed relationships "
            "defined on a disjoint set of exact-InChIKey library-supported entities?"
        ),
        "contract": {
            "known_anchor": (
                "one response-blind best library-supported observed ion per full InChIKey "
                "and study; signed cosine across shared exact InChIKeys"
            ),
            "unresolved_test": (
                "reciprocal unresolved--unresolved WSE edges only; no library identity, "
                "semantic condition name, or registered relation enters pairing or target "
                "selection; beta/SE are estimated from experimental labels"
            ),
            "cross_parent_only_according_to_current_provisional_parent_map": True,
            "min_known_exact_inchikeys": args.min_known,
            "min_unresolved_mutual_edges": args.min_unresolved,
            "stable_anchor_abs_score": args.anchor_abs,
            "stable_anchor_requires_leave_one_out_sign": True,
        },
        "denominators": {
            "selected_library_study_inchikeys": int(len(selected)),
            "known_anchor_contrast_pairs": int(len(known)),
            "reciprocal_unresolved_edges": int(len(unresolved_edges)),
            "eligible_cross_parent_continuous_pairs": int(len(eligible)),
            "eligible_cross_parent_source_pairs": int(
                eligible[["left_study_id", "right_study_id"]].drop_duplicates().shape[0]
            ),
            "stable_anchor_pairs": int(len(stable)),
            "known_unresolved_entity_key_overlap": int(len(entity_overlap)),
        },
        "continuous_agreement": continuous,
        "stable_anchor_sign_agreement": {
            "agree": int(stable["sign_agreement"].sum()) if len(stable) else 0,
            "total": int(len(stable)),
            "fraction": float(stable["sign_agreement"].mean()) if len(stable) else None,
        },
        "face_specific_open_retrieval": retrieval_summary,
        "decision": {
            "overall": "FAIL_CONTRAST_SPECIFIC_INTERACTION_GATE",
            "descriptive_coarse_association_observed": True,
            "double_centered_specificity_gate_passed": specificity_gate,
            "top1_above_random_dyad_sensitivity_gate_passed": top1_gate,
            "interpretation": (
                "A coarse descriptive association was observed between unresolved WSE "
                "responses and the disjoint known-entity response anchor, but the signal "
                "does not survive simultaneous "
                "removal of query- and candidate-contrast main effects and Top-1 improvement "
                "over uniform random is not robust under source-pair dyad resampling."
            ),
        },
        "claim_boundary": [
            "This is an entity-disjoint internal development test, not external validation.",
            "A GNPS library-supported InChIKey is not MSI level-1 identity.",
            "Known signed agreement is an empirical response anchor, not a biological mechanism.",
            "The marginal positive results support coarse unresolved response information; they do not establish contrast-specific retrieval or incremental value of Full23.",
            "The failed specificity gate rejects this current WSE endpoint as an Atlas discovery result, not the master Atlas question.",
            "Source-pair dyads share studies, so dyad-bootstrap intervals are descriptive rather than independent-source confidence intervals.",
            "Axis-label permutation is a conditional development null that assumes within-pair contrast-axis exchangeability; it is not a universal biological p-value.",
            "Cross-parent exclusions follow the current provisional parent-project map.",
        ],
        "provenance": {
            "entities": sha256(args.collection / "response_entities.tsv.gz"),
            "contrasts": sha256(args.collection / "contrast_manifest.tsv"),
            "library_evidence": sha256(args.library_evidence),
            "edges": sha256(args.edges),
            "parent_projects": sha256(args.parent_projects),
            "implementation": sha256(Path(__file__)),
        },
    }
    report_path = args.output / "report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_checksums(
        args.output, [joined_path, stable_path, retrieval_path, report_path]
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
