"""Turn the real-error atlas into auditable error mechanisms and cohorts.

The analysis deliberately separates the two sides of a retrieval violation:

1. positive deficit: spectra of the same molecule are less similar than the
   matched-correct reference distribution;
2. negative excess: a different molecule is more similar than the
   matched-correct reference distribution.

Those axes are not mutually exclusive and must not be collapsed into one
"error type".  Mechanism labels produced here are screening hypotheses for a
matched-control peak-occlusion experiment, not chemical ground truth.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ATLAS = ROOT / "data/validation/g8r_real_error_atlas"
DEFAULT_OUTPUT = ROOT / "data/validation/g8r_real_error_analysis"
RAW_FEATURES = (
    "sqrt_cosine", "entropy_similarity", "neutral_loss_sqrt_cosine",
    "intensity_coverage_min", "top10_match_fraction",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dir", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-group-identities", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def candidate_bin(values: pd.Series) -> pd.Series:
    return pd.cut(
        values.astype(float), bins=[1, 2, 4, 8, np.inf],
        labels=["2", "3-4", "5-8", "9+"], include_lowest=True,
    ).astype(str)


def robust_reference(
    frame: pd.DataFrame, value: str, prefix: str,
) -> pd.DataFrame:
    """Reference medians/IQRs from official-DreaMS correct queries only."""
    reference = frame.loc[frame["dreams_correct"]].copy()
    grouped = reference.groupby(["has_near_candidate", "candidate_bin"], observed=True)[value]
    output = grouped.agg(["median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75), "count"])
    output.columns = [f"{prefix}_median", f"{prefix}_q25", f"{prefix}_q75", f"{prefix}_n"]
    output[f"{prefix}_scale"] = np.maximum(
        output[f"{prefix}_q75"] - output[f"{prefix}_q25"], 0.02,
    )
    return output.reset_index()


def bh_qvalues(pvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def identity_group_enrichment(
    query: pd.DataFrame, column: str, minimum: int,
) -> pd.DataFrame:
    identity = query.groupby("query_ik14", sort=False).agg(
        group=(column, "first"),
        error=("dreams_correct", lambda x: not bool(np.all(x))),
        introduced=("transition", lambda x: bool(np.any(x == "introduced"))),
        corrected=("transition", lambda x: bool(np.any(x == "corrected"))),
        n_queries=("query_index", "size"),
    ).reset_index()
    total_error = int(identity["error"].sum())
    total_correct = int(len(identity) - total_error)
    rows = []
    for group, body in identity.groupby("group", dropna=False, sort=False):
        if not str(group) or str(group) == "nan" or len(body) < minimum:
            continue
        a = int(body["error"].sum())
        b = int(len(body) - a)
        c = total_error - a
        d = total_correct - b
        odds, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater")
        rows.append({
            "group": str(group),
            "n_identities": int(len(body)),
            "error_identities": a,
            "error_rate": float(a / len(body)),
            "corrected_identities": int(body["corrected"].sum()),
            "introduced_identities": int(body["introduced"].sum()),
            "odds_ratio": float(odds),
            "fisher_p": float(pvalue),
        })
    result = pd.DataFrame(rows)
    if len(result):
        result["bh_q"] = bh_qvalues(result["fisher_p"].to_numpy(float))
        result = result.sort_values(
            ["bh_q", "error_rate", "n_identities"], ascending=[True, False, False]
        )
    return result


def get_candidate_roles(candidate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for query_index, group in candidate.groupby("query_index", sort=False):
        positive = group.loc[group["label"] == 1]
        if len(positive) != 1:
            raise RuntimeError(f"query {query_index} does not have one positive molecule")
        positive = positive.iloc[0]
        negatives = group.loc[group["label"] == 0]
        if negatives.empty:
            raise RuntimeError(f"query {query_index} has no negative molecule")
        hard = negatives.sort_values(
            ["dreams_score", "candidate_ik14"], ascending=[False, True], kind="mergesort"
        ).iloc[0]
        record = {
            "query_index": int(query_index),
            "positive_ik14": positive["candidate_ik14"],
            "positive_smiles": positive["candidate_smiles"],
            "positive_scaffold": positive["candidate_scaffold"],
            "positive_dreams_score": float(positive["dreams_score"]),
            "positive_p2b_score": float(positive["p2b_score"]),
            "hard_negative_ik14": hard["candidate_ik14"],
            "hard_negative_formula": hard["candidate_formula"],
            "hard_negative_smiles": hard["candidate_smiles"],
            "hard_negative_scaffold": hard["candidate_scaffold"],
            "hard_negative_grade": hard["mces_grade"],
            "hard_negative_dreams_score": float(hard["dreams_score"]),
            "hard_negative_p2b_score": float(hard["p2b_score"]),
            "hard_negative_dreams_pair_row": int(hard["dreams_winning_pair_row"]),
        }
        raw_votes = 0
        for name in RAW_FEATURES:
            pos_value = float(positive[f"dreams_pair_{name}"])
            neg_value = float(hard[f"dreams_pair_{name}"])
            record[f"positive_{name}"] = pos_value
            record[f"hard_negative_{name}"] = neg_value
            record[f"delta_positive_minus_negative_{name}"] = pos_value - neg_value
            raw_votes += int(pos_value > neg_value + 1e-12)
        record["raw_features_favoring_positive"] = raw_votes
        rows.append(record)
    return pd.DataFrame(rows)


def assign_screening_hypotheses(frame: pd.DataFrame) -> pd.DataFrame:
    wrong = ~frame["dreams_correct"]
    frame["positive_deficit"] = wrong & (frame["positive_score_reference_z"] <= -1.0)
    frame["negative_excess"] = wrong & (frame["negative_score_reference_z"] >= 1.0)
    frame["both_score_arms"] = frame["positive_deficit"] & frame["negative_excess"]
    frame["comparative_boundary_error"] = wrong & ~(
        frame["positive_deficit"] | frame["negative_excess"]
    )

    correct_negative = frame.loc[frame["dreams_correct"], "hard_negative_top10_match_fraction"]
    shared_threshold = float(correct_negative.quantile(0.75))
    coverage_threshold = float(
        frame.loc[frame["dreams_correct"], "hard_negative_intensity_coverage_min"].quantile(0.75)
    )
    neutral_threshold = float(
        frame.loc[frame["dreams_correct"], "hard_negative_neutral_loss_sqrt_cosine"].quantile(0.75)
    )
    frame["shared_major_peak_screen"] = (
        wrong
        & (frame["hard_negative_top10_match_fraction"] >= shared_threshold)
        & (frame["hard_negative_intensity_coverage_min"] >= coverage_threshold)
    )
    frame["neutral_loss_convergence_screen"] = (
        wrong & (frame["hard_negative_neutral_loss_sqrt_cosine"] >= neutral_threshold)
    )
    frame["cross_condition_positive_screen"] = (
        wrong
        & (frame["positive_cross_instrument"] | (frame["positive_collision_energy_delta"] >= 10.0))
    )
    frame["raw_evidence_can_rescue"] = wrong & (frame["raw_features_favoring_positive"] >= 3)
    if "dreams_rule_jaccard" in frame and "positive_dreams_rule_jaccard" in frame:
        frame["rules_favor_positive"] = (
            wrong
            & frame["positive_dreams_rule_jaccard"].notna()
            & frame["dreams_rule_jaccard"].notna()
            & (frame["positive_dreams_rule_jaccard"] > frame["dreams_rule_jaccard"])
        )
        frame["rules_favor_wrong"] = (
            wrong
            & frame["positive_dreams_rule_jaccard"].notna()
            & frame["dreams_rule_jaccard"].notna()
            & (frame["positive_dreams_rule_jaccard"] < frame["dreams_rule_jaccard"])
        )
    else:
        frame["rules_favor_positive"] = False
        frame["rules_favor_wrong"] = False

    labels = []
    for row in frame.itertuples(index=False):
        if bool(row.dreams_correct):
            label = "official_correct"
        elif bool(row.both_score_arms):
            label = "positive_deficit_and_negative_excess"
        elif bool(row.positive_deficit):
            label = "positive_deficit_only"
        elif bool(row.negative_excess):
            label = "negative_excess_only"
        else:
            label = "comparative_boundary_error"
        labels.append(label)
    frame["score_error_family"] = labels
    return frame, {
        "shared_major_peak_q75": shared_threshold,
        "negative_intensity_coverage_q75": coverage_threshold,
        "negative_neutral_loss_q75": neutral_threshold,
        "positive_deficit_z": -1.0,
        "negative_excess_z": 1.0,
    }


def summarize_boolean(frame: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    return {column: int(frame[column].fillna(False).sum()) for column in columns}


def main() -> None:
    args = parse_args()
    if args.min_group_identities < 2:
        raise ValueError("--min-group-identities must be >=2")
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    required = {
        "query": args.atlas_dir / "query_summary.csv.gz",
        "candidate": args.atlas_dir / "candidate_edges.csv.gz",
        "report": args.atlas_dir / "report.json",
    }
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    atlas_report = json.loads(required["report"].read_text(encoding="utf-8"))
    if atlas_report.get("status") != "g8r_real_error_atlas_complete":
        raise RuntimeError("input atlas is incomplete")
    query = pd.read_csv(required["query"])
    candidate = pd.read_csv(required["candidate"])
    if query["query_index"].duplicated().any():
        raise RuntimeError("query table contains duplicates")
    roles = get_candidate_roles(candidate)
    overlapping = (set(query.columns) & set(roles.columns)) - {"query_index"}
    if overlapping:
        raise RuntimeError(
            f"query/candidate-role columns would be silently suffixed: {sorted(overlapping)}"
        )
    frame = query.merge(roles, on="query_index", validate="one_to_one")
    if len(frame) != len(query):
        raise RuntimeError("candidate roles did not cover every query")
    frame["candidate_bin"] = candidate_bin(frame["n_candidate_molecules"])
    positive_reference = robust_reference(frame, "positive_dreams_score", "positive_ref")
    negative_reference = robust_reference(frame, "hard_negative_dreams_score", "negative_ref")
    frame = frame.merge(
        positive_reference, on=["has_near_candidate", "candidate_bin"], how="left",
        validate="many_to_one",
    ).merge(
        negative_reference, on=["has_near_candidate", "candidate_bin"], how="left",
        validate="many_to_one",
    )
    if frame[["positive_ref_median", "negative_ref_median"]].isna().any().any():
        raise RuntimeError("a difficulty stratum has no correct-query reference")
    frame["positive_score_reference_z"] = (
        frame["positive_dreams_score"] - frame["positive_ref_median"]
    ) / frame["positive_ref_scale"]
    frame["negative_score_reference_z"] = (
        frame["hard_negative_dreams_score"] - frame["negative_ref_median"]
    ) / frame["negative_ref_scale"]
    frame, thresholds = assign_screening_hypotheses(frame)

    formula = identity_group_enrichment(frame, "query_formula", args.min_group_identities)
    scaffold_frame = identity_group_enrichment(frame, "query_scaffold", args.min_group_identities)
    wrong = frame.loc[~frame["dreams_correct"]].copy()
    priority_columns = [
        "query_index", "query_row", "query_ik14", "query_formula", "query_smiles",
        "query_instrument", "query_collision_energy", "transition",
        "score_error_family", "positive_dreams_score", "hard_negative_dreams_score",
        "dreams_margin", "p2b_margin", "positive_ik14", "positive_smiles",
        "hard_negative_ik14", "hard_negative_formula", "hard_negative_smiles",
        "hard_negative_grade", "positive_dreams_pair_row", "hard_negative_dreams_pair_row",
        "positive_cross_instrument", "positive_collision_energy_delta",
        "shared_major_peak_screen", "neutral_loss_convergence_screen",
        "cross_condition_positive_screen", "raw_evidence_can_rescue",
        "rules_favor_positive", "rules_favor_wrong",
    ]
    priority = wrong[priority_columns].copy()
    priority["occlusion_priority"] = (
        3 * priority["shared_major_peak_screen"].astype(int)
        + 3 * priority["cross_condition_positive_screen"].astype(int)
        + 2 * priority["raw_evidence_can_rescue"].astype(int)
        + priority["neutral_loss_convergence_screen"].astype(int)
    )
    priority = priority.sort_values(
        ["occlusion_priority", "dreams_margin"], ascending=[False, True], kind="mergesort"
    )

    screen_columns = [
        "positive_deficit", "negative_excess", "both_score_arms",
        "comparative_boundary_error", "shared_major_peak_screen",
        "neutral_loss_convergence_screen", "cross_condition_positive_screen",
        "raw_evidence_can_rescue", "rules_favor_positive", "rules_favor_wrong",
    ]
    report = {
        "status": "g8r_real_error_analysis_complete",
        "n_queries": int(len(frame)),
        "n_identities": int(frame["query_ik14"].nunique()),
        "official_errors": int((~frame["dreams_correct"]).sum()),
        "transitions": {
            str(key): int(value) for key, value in frame["transition"].value_counts().items()
        },
        "score_error_families": {
            str(key): int(value)
            for key, value in wrong["score_error_family"].value_counts().items()
        },
        "screen_counts_among_official_errors": summarize_boolean(wrong, screen_columns),
        "thresholds_fitted_on_this_training_atlas": thresholds,
        "formula_groups_tested": int(len(formula)),
        "scaffold_groups_tested": int(len(scaffold_frame)),
        "causal_next_step": (
            "Run paired targeted-vs-count/intensity/mz-matched-random peak deletion separately "
            "for the positive-deficit and negative-excess arms. Train only on directionally "
            "replicated cases; do not use these screening labels as ground truth."
        ),
        "claim_limit": (
            "The robust score decomposition is descriptive relative to matched correct-query "
            "strata. Peak/rule labels are hypotheses until causal occlusion replicates them."
        ),
        "atlas_provenance": atlas_report.get("provenance", {}),
    }

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{args.output_dir.name}.building-", dir=args.output_dir.parent,
    ))
    try:
        frame.to_csv(staging / "query_error_signatures.csv.gz", index=False, compression="gzip")
        priority.to_csv(staging / "occlusion_priority_cases.csv.gz", index=False, compression="gzip")
        formula.to_csv(staging / "formula_error_enrichment.csv", index=False)
        scaffold_frame.to_csv(staging / "scaffold_error_enrichment.csv", index=False)
        positive_reference.to_csv(staging / "positive_score_reference.csv", index=False)
        negative_reference.to_csv(staging / "negative_score_reference.csv", index=False)
        (staging / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        if args.output_dir.exists():
            if not args.overwrite:
                raise FileExistsError(args.output_dir)
            backup = args.output_dir.with_name(args.output_dir.name + ".previous")
            if backup.exists():
                raise FileExistsError(f"refusing overwrite because backup exists: {backup}")
            args.output_dir.replace(backup)
        staging.replace(args.output_dir)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
