#!/usr/bin/env python3
"""Run direct known-metabolite study-response baselines on Contrast Atlas.

The candidate universe is every source-disjoint target contrast.  Strict
Gold/Silver full-InChIKey representatives are selected without reading any
phenotype or effect.  Scores are written before registered biological
relations are opened.  These are development/consumed baselines; they do not
provide biological truth or validate an unresolved-information increment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2, fisher_exact


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLLECTION = (
    ROOT / "data/validation/contrast_atlas_open_response_collection_v9_frozenbase_chemprop_20261023"
)
DEFAULT_IDENTITIES = (
    ROOT / "data/validation/contrast_atlas_open_v9_frozenbase_common_identity_shards_20261023"
)
DEFAULT_OUTPUT = ROOT / "data/validation/contrast_atlas_known_response_baselines_v2_20261029"
EXPECTED_COLLECTION_STATUS = "DEVELOPMENT_CONSUMED_FROZEN_BASE_EXTENSION_COMPLETE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_full_inchikey(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]")


def load_identity_representatives(identity_root: Path, studies: list[str]) -> pd.DataFrame:
    """Choose one response-blind feature per study and exact full InChIKey."""
    frames: list[pd.DataFrame] = []
    for study in studies:
        path = identity_root / study / "identity_partition.tsv.gz"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, sep="\t")
        required = {
            "study_id", "entity_id", "library_supported", "library_inchikey",
            "wse", "matched_peaks", "common_identity_policy",
        }
        if not required.issubset(frame.columns):
            raise RuntimeError(f"identity shard missing columns: {path}")
        supported = frame["library_supported"].astype(str).str.lower().eq("true")
        frame = frame.loc[supported & valid_full_inchikey(frame["library_inchikey"])].copy()
        frame["wse"] = pd.to_numeric(frame["wse"], errors="coerce").fillna(-np.inf)
        frame["matched_peaks"] = pd.to_numeric(frame["matched_peaks"], errors="coerce").fillna(-1)
        frame = frame.sort_values(
            ["study_id", "library_inchikey", "wse", "matched_peaks", "entity_id"],
            ascending=[True, True, False, False, True],
            kind="stable",
        )
        frame["same_identity_feature_count"] = frame.groupby(
            ["study_id", "library_inchikey"]
        )["entity_id"].transform("size")
        frame = frame.drop_duplicates(["study_id", "library_inchikey"], keep="first")
        frames.append(frame[[
            "common_identity_policy", "study_id", "entity_id", "library_inchikey",
            "wse", "matched_peaks", "same_identity_feature_count",
        ]])
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["study_id", "library_inchikey"]).any():
        raise RuntimeError("identity representative selection is not unique")
    if result["common_identity_policy"].nunique() != 1:
        raise RuntimeError("common identity policy drift across shards")
    return result


def response_vectors(response: pd.DataFrame, representatives: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    required = {"study_id", "contrast_id", "entity_id", "beta", "se"}
    if not required.issubset(response.columns):
        raise RuntimeError(f"response ledger missing columns: {sorted(required - set(response.columns))}")
    selected = response.merge(
        representatives[["study_id", "entity_id", "library_inchikey"]],
        on=["study_id", "entity_id"], how="inner", validate="many_to_one",
    )
    selected["beta"] = pd.to_numeric(selected["beta"], errors="raise")
    selected["se"] = pd.to_numeric(selected["se"], errors="raise")
    if (~np.isfinite(selected["beta"])).any() or (~np.isfinite(selected["se"])).any():
        raise RuntimeError("non-finite beta/SE in selected known responses")
    if (selected["se"] <= 0).any():
        raise RuntimeError("non-positive response SE")
    if selected.duplicated(["study_id", "contrast_id", "library_inchikey"]).any():
        raise RuntimeError("representative identity still duplicated within a contrast")
    selected["z"] = selected["beta"] / selected["se"]
    selected["shrunk_response"] = selected["beta"] / np.sqrt(
        np.square(selected["beta"]) + np.square(selected["se"])
    )
    return {
        key: block[["library_inchikey", "beta", "se", "z", "shrunk_response"]].copy()
        for key, block in selected.groupby(["study_id", "contrast_id"], sort=False)
    }


def overlap_pvalue(left: np.ndarray, right: np.ndarray, universe_size: int) -> tuple[float, float]:
    a = int(np.sum(left & right))
    b = int(np.sum(left)) - a
    c = int(np.sum(right)) - a
    d = int(universe_size) - a - b - c
    if min(a, b, c, d) < 0:
        raise RuntimeError("invalid overlap contingency table")
    odds, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater")
    if not math.isfinite(float(odds)):
        odds = np.inf if a > 0 else 0.0
    return float(odds), float(pvalue)


def fisher_combine_supported(parts: list[tuple[float, float]]) -> float:
    pvalues = [max(min(p, 1.0), 1e-300) if odds > 1 else 1.0 for odds, p in parts]
    statistic = -2.0 * sum(math.log(p) for p in pvalues)
    return float(chi2.sf(statistic, 2 * len(pvalues)))


def pair_metrics(
    query: pd.DataFrame,
    target: pd.DataFrame,
    minimum_shared: int,
    active_z: float,
    minimum_double_active: int,
) -> dict[str, Any]:
    aligned = query.merge(
        target, on="library_inchikey", how="inner", suffixes=("_query", "_target"),
        validate="one_to_one",
    )
    n = int(len(aligned))
    result: dict[str, Any] = {
        "shared_known_identity_count": n,
        "continuous_available": False,
        "continuous_unavailability_reason": "",
        "continuous_similar_score": np.nan,
        "continuous_opposite_score": np.nan,
        "continuous_absolute_score": np.nan,
        "idmet_available": False,
        "idmet_unavailability_reason": "",
        "double_active_identity_count": 0,
        "active_pp": 0, "active_pn": 0, "active_np": 0, "active_nn": 0,
        "idmet_signed_log_odds": np.nan,
        "idmet_two_sided_pvalue": np.nan,
        "idmet_absolute_score": np.nan,
        "ora_available": False,
        "ora_unavailability_reason": "",
        "ora_similar_pvalue": np.nan,
        "ora_opposite_pvalue": np.nan,
        "ora_similar_score": np.nan,
        "ora_opposite_score": np.nan,
        "ora_best_score": np.nan,
    }
    if n < minimum_shared:
        reason = "no_shared_known_identity" if n == 0 else "fewer_than_minimum_shared_known_identities"
        result["continuous_unavailability_reason"] = reason
        result["idmet_unavailability_reason"] = reason
        result["ora_unavailability_reason"] = reason
        return result

    q = aligned["shrunk_response_query"].to_numpy(float)
    t = aligned["shrunk_response_target"].to_numpy(float)
    denominator = math.sqrt(float(np.dot(q, q)) * float(np.dot(t, t)))
    if denominator > 0 and math.isfinite(denominator):
        signed = float(np.dot(q, t) / denominator)
        result.update({
            "continuous_available": True,
            "continuous_similar_score": signed,
            "continuous_opposite_score": -signed,
            "continuous_absolute_score": abs(signed),
        })
    else:
        result["continuous_unavailability_reason"] = "zero_response_norm"

    zq = aligned["z_query"].to_numpy(float)
    zt = aligned["z_target"].to_numpy(float)
    q_pos, q_neg = zq >= active_z, zq <= -active_z
    t_pos, t_neg = zt >= active_z, zt <= -active_z
    both = (q_pos | q_neg) & (t_pos | t_neg)
    result["double_active_identity_count"] = int(np.sum(both))
    pp = int(np.sum(q_pos & t_pos)); pn = int(np.sum(q_pos & t_neg))
    np_ = int(np.sum(q_neg & t_pos)); nn = int(np.sum(q_neg & t_neg))
    result.update({"active_pp": pp, "active_pn": pn, "active_np": np_, "active_nn": nn})
    if int(np.sum(both)) >= minimum_double_active:
        table = [[pp, pn], [np_, nn]]
        odds, pvalue = fisher_exact(table, alternative="two-sided")
        log_odds = math.log((pp + 0.5) * (nn + 0.5) / ((pn + 0.5) * (np_ + 0.5)))
        result.update({
            "idmet_available": True,
            "idmet_signed_log_odds": float(log_odds),
            "idmet_two_sided_pvalue": float(pvalue),
            "idmet_absolute_score": abs(float(log_odds)),
        })
    else:
        result["idmet_unavailability_reason"] = "fewer_than_minimum_double_active_identities"

    if np.any(q_pos | q_neg) and np.any(t_pos | t_neg):
        same_p = fisher_combine_supported([
            overlap_pvalue(q_pos, t_pos, n), overlap_pvalue(q_neg, t_neg, n),
        ])
        opposite_p = fisher_combine_supported([
            overlap_pvalue(q_pos, t_neg, n), overlap_pvalue(q_neg, t_pos, n),
        ])
        same_score = -math.log10(max(same_p, 1e-300))
        opposite_score = -math.log10(max(opposite_p, 1e-300))
        result.update({
            "ora_available": True,
            "ora_similar_pvalue": same_p,
            "ora_opposite_pvalue": opposite_p,
            "ora_similar_score": same_score,
            "ora_opposite_score": opposite_score,
            "ora_best_score": max(same_score, opposite_score),
        })
    else:
        result["ora_unavailability_reason"] = "query_or_target_has_no_active_known_identity"
    return result


def add_ranks(scores: pd.DataFrame) -> pd.DataFrame:
    specifications = {
        "continuous_similar_score": ("continuous_available", "continuous_similar_rank"),
        "continuous_opposite_score": ("continuous_available", "continuous_opposite_rank"),
        "continuous_absolute_score": ("continuous_available", "continuous_absolute_rank"),
        "idmet_signed_log_odds": ("idmet_available", "idmet_similar_rank"),
        "idmet_absolute_score": ("idmet_available", "idmet_absolute_rank"),
        "ora_similar_score": ("ora_available", "ora_similar_rank"),
        "ora_opposite_score": ("ora_available", "ora_opposite_rank"),
        "ora_best_score": ("ora_available", "ora_best_rank"),
    }
    keys = ["query_study_id", "query_contrast_id"]
    for score_column, (available_column, rank_column) in specifications.items():
        scores[rank_column] = np.nan
        for _, index in scores.groupby(keys, sort=False).groups.items():
            idx = list(index)
            eligible = scores.loc[idx, available_column] & scores.loc[idx, score_column].notna()
            eligible_idx = scores.loc[idx].index[eligible]
            scores.loc[eligible_idx, rank_column] = scores.loc[eligible_idx, score_column].rank(
                method="min", ascending=False
            )
    return scores


def fixed_scores(
    vectors: dict[tuple[str, str], pd.DataFrame],
    contrasts: pd.DataFrame,
    minimum_shared: int,
    active_z: float,
    minimum_double_active: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    records = contrasts.to_dict("records")
    empty = pd.DataFrame(columns=["library_inchikey", "beta", "se", "z", "shrunk_response"])
    for query in records:
        query_vector = vectors.get((query["study_id"], query["contrast_id"]), empty)
        for target in records:
            if query["source_family"] == target["source_family"]:
                continue
            row = {
                "query_study_id": query["study_id"],
                "query_source_family": query["source_family"],
                "query_contrast_id": query["contrast_id"],
                "target_study_id": target["study_id"],
                "target_source_family": target["source_family"],
                "target_contrast_id": target["contrast_id"],
            }
            row.update(pair_metrics(
                query_vector,
                vectors.get((target["study_id"], target["contrast_id"]), empty),
                minimum_shared, active_z, minimum_double_active,
            ))
            rows.append(row)
    scores = add_ranks(pd.DataFrame(rows))
    expected = contrasts.set_index(["study_id", "contrast_id"])[
        "source_disjoint_candidate_contrasts"
    ].astype(int).to_dict()
    observed = scores.groupby(["query_study_id", "query_contrast_id"]).size().to_dict()
    if observed != expected:
        raise RuntimeError("fixed source-disjoint target universe drift")
    return scores.sort_values([
        "query_study_id", "query_contrast_id", "target_study_id", "target_contrast_id"
    ], kind="stable").reset_index(drop=True)


def registered_summary(joined: pd.DataFrame) -> dict[str, Any]:
    rank_columns = ["continuous_absolute_rank", "idmet_absolute_rank", "ora_best_rank"]
    result: dict[str, Any] = {}
    for column in rank_columns:
        values = pd.to_numeric(joined[column], errors="coerce")
        result[column] = {
            "available": int(values.notna().sum()),
            "top1": int(values.le(1).sum()),
            "top5": int(values.le(5).sum()),
            "median_rank_among_available": None if not values.notna().any() else float(values.median()),
        }
    return result


def run(
    collection_dir: Path,
    identity_root: Path,
    output_dir: Path,
    minimum_shared: int,
    active_z: float,
    minimum_double_active: int,
) -> dict[str, Any]:
    collection_dir, identity_root, output_dir = (
        collection_dir.resolve(), identity_root.resolve(), output_dir.resolve()
    )
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    report_path = collection_dir / "report.json"
    collection_report = json.loads(report_path.read_text(encoding="utf-8"))
    if collection_report.get("status") != EXPECTED_COLLECTION_STATUS:
        raise RuntimeError("collection status is not the frozen development collection")
    response_path = collection_dir / "response_entities.tsv.gz"
    contrast_path = collection_dir / "contrast_manifest.tsv"
    relation_path = collection_dir / "registered_relations.tsv"
    contrasts = pd.read_csv(contrast_path, sep="\t")
    if len(contrasts) != 51 or contrasts["source_family"].nunique() != 15:
        raise RuntimeError("expected frozen 15-source / 51-contrast collection")

    studies = sorted(contrasts["study_id"].unique())
    identity_paths = [identity_root / study / "identity_partition.tsv.gz" for study in studies]
    representatives = load_identity_representatives(identity_root, studies)
    response = pd.read_csv(response_path, sep="\t", low_memory=False)
    vectors = response_vectors(response, representatives)
    scores = fixed_scores(vectors, contrasts, minimum_shared, active_z, minimum_double_active)

    output_dir.mkdir(parents=True)
    representatives_path = output_dir / "strict_known_identity_representatives.tsv.gz"
    scores_path = output_dir / "fixed_candidate_scores.tsv.gz"
    representatives.to_csv(
        representatives_path, sep="\t", index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    scores.to_csv(scores_path, sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
    score_hash_before_truth = sha256(scores_path)

    # Registered relations are opened only after all candidate scores and ranks exist.
    relations = pd.read_csv(relation_path, sep="\t")
    joined = relations.merge(
        scores,
        left_on=["left_study_id", "left_contrast_id", "right_study_id", "right_contrast_id"],
        right_on=["query_study_id", "query_contrast_id", "target_study_id", "target_contrast_id"],
        how="left", validate="one_to_one",
    )
    if joined["shared_known_identity_count"].isna().any():
        raise RuntimeError("registered relation absent from fixed candidate universe")
    joined["adjudication"] = "DESCRIPTIVE_RELEVANCE_RANK_ONLY_NO_SIGNED_TRUTH"
    registered_path = output_dir / "registered_relation_ranks.tsv"
    joined.to_csv(registered_path, sep="\t", index=False)
    if sha256(scores_path) != score_hash_before_truth:
        raise RuntimeError("scores changed after registered relations were opened")

    primary = joined.loc[joined["eligible_primary_relevance"].astype(str).str.lower().eq("true")]
    exact_drug = primary.loc[primary["relation_class"].eq("consumed_exact_drug_cross_design")]
    report = {
        "schema": "contrast_atlas_known_response_baselines_v2",
        "status": "DEVELOPMENT_CONSUMED_KNOWN_RESPONSE_BASELINES_COMPLETE",
        "data_role": "development_consumed",
        "methods": {
            "continuous_strict_known": "cosine of full-InChIKey-aligned beta/sqrt(beta^2+SE^2)",
            "idmet_like_signed_overlap": "z-thresholded up/down 2x2 signed log odds; not an exact software reproduction",
            "idmet_plus_like_ora": "one-sided same/opposite up/down overlap enrichment on the pairwise detected background; not an exact software reproduction",
        },
        "parameters": {
            "minimum_shared_known_identities": int(minimum_shared),
            "active_absolute_z": float(active_z),
            "minimum_double_active_identities_for_signed_overlap": int(minimum_double_active),
            "identity_key": "full InChIKey",
            "duplicate_identity_representative": "highest WSE, then matched peaks, then entity_id; selected before effects",
        },
        "denominators": {
            "source_families": int(contrasts["source_family"].nunique()),
            "study_level_contrasts": int(len(contrasts)),
            "response_rows": int(len(response)),
            "strict_known_study_identity_representatives": int(len(representatives)),
            "unique_full_inchikeys": int(representatives["library_inchikey"].nunique()),
            "source_disjoint_directed_candidate_pairs": int(len(scores)),
            "continuous_available_pairs": int(scores["continuous_available"].sum()),
            "idmet_like_available_pairs": int(scores["idmet_available"].sum()),
            "ora_available_pairs": int(scores["ora_available"].sum()),
            "registered_directed_relations": int(len(joined)),
            "primary_eligible_registered_relations": int(len(primary)),
            "exact_drug_registered_relations": int(len(exact_drug)),
        },
        "registered_relevance_rank_summary": {
            "primary_eligible_all": registered_summary(primary),
            "consumed_exact_drug_cross_design": registered_summary(exact_drug),
        },
        "truth_boundary": {
            "score_sha256_before_registered_relations": score_hash_before_truth,
            "registered_relations_opened_after_score_write": True,
            "signed_entity_truth_count": 0,
            "pass_fail_adjudicated": False,
        },
        "next_required_comparison": {
            "unknown_increment": "all versus the draw-wise strongest of known_only, measurement_matched_known and unresolved_shuffled_ms2",
            "algorithm_increment_identity_track": "WSE versus Official versus Noise V1 versus Full23 on identical complete identity blocks",
            "algorithm_increment_relation_track": "plain fragment/WSE versus separately validated relation scorers; Full23 is not currently eligible",
        },
        "claim_boundary": [
            "These baselines quantify direct study-response competitors; they do not establish that any registered relation is biologically correct.",
            "Registered relations have no signed truth, so same/opposite accuracy is not reported.",
            "The exact-drug relations are consumed and belong to one broader ChemProp project, not an external validation.",
            "Known-only evidence shortage causes abstention while retaining the original query-target denominator.",
            "No unresolved-information or Full23 increment is evaluated by this run.",
        ],
        "input_sha256": {
            str(path.relative_to(ROOT).as_posix()): sha256(path)
            for path in [report_path, response_path, contrast_path, relation_path] + identity_paths
        },
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "output_sha256": {
            representatives_path.name: sha256(representatives_path),
            scores_path.name: sha256(scores_path),
            registered_path.name: sha256(registered_path),
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-dir", type=Path, default=DEFAULT_COLLECTION)
    parser.add_argument("--identity-root", type=Path, default=DEFAULT_IDENTITIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-shared-known", type=int, default=3)
    parser.add_argument("--active-z", type=float, default=1.96)
    parser.add_argument("--minimum-double-active", type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(
        args.collection_dir, args.identity_root, args.output_dir,
        args.minimum_shared_known, args.active_z, args.minimum_double_active,
    ), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
