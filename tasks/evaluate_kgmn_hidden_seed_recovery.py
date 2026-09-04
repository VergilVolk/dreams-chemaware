#!/usr/bin/env python3
"""Evaluate matched KGMN hidden-seed predictions on one frozen denominator.

Both the author and experimental runner must export the same long table with
one row per candidate identity.  Candidate rank is recomputed here from scores;
ties at the truth score count against the truth.  Missing predictions remain in
the denominator as failures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


REQUIRED_PREDICTION_COLUMNS = {
    "repeat",
    "truth_inchikey1",
    "candidate_inchikey1",
    "candidate_score",
    "propagation_depth",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_ik14(series: pd.Series, label: str) -> pd.Series:
    values = series.fillna("").astype(str).str.strip().str.slice(0, 14)
    if values.eq("").any():
        raise RuntimeError(f"{label} contains empty identities")
    return values


def load_denominator(contract_dir: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    report_path = contract_dir / "report.json"
    split_path = contract_dir / "hidden_seed_splits.csv.gz"
    if not report_path.is_file() or not split_path.is_file():
        raise FileNotFoundError("external-validation contract is incomplete")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "kgmn_external_validation_contract_frozen":
        raise RuntimeError("external-validation contract status mismatch")
    expected_hash = report.get("provenance", {}).get("outputs_sha256", {}).get(
        "hidden_seed_splits.csv.gz"
    )
    if expected_hash != sha256(split_path):
        raise RuntimeError("hidden-seed split hash mismatch")
    splits = pd.read_csv(split_path)
    required = {"repeat", "inchikey1", "polarity_presence", "role"}
    if not required.issubset(splits.columns):
        raise RuntimeError(f"hidden-seed split misses columns: {sorted(required - set(splits.columns))}")
    splits["inchikey1"] = _clean_ik14(splits["inchikey1"], "split")
    denominator = splits.loc[
        splits["role"].eq("hidden_validation"),
        ["repeat", "inchikey1", "polarity_presence"],
    ].rename(columns={"inchikey1": "truth_inchikey1"})
    if denominator.duplicated(["repeat", "truth_inchikey1"]).any():
        raise RuntimeError("hidden-seed denominator contains duplicate repeat/identity pairs")
    expected_repeats = int(report["primary_protocol"]["repeats"])
    if denominator["repeat"].nunique() != expected_repeats:
        raise RuntimeError("hidden-seed denominator lost repeats")
    if denominator.groupby("repeat").size().nunique() != 1:
        raise RuntimeError("hidden-seed repeats have different denominator sizes")
    return denominator.sort_values(["repeat", "truth_inchikey1"]).reset_index(drop=True), report


def load_predictions(path: Path, denominator: pd.DataFrame, arm: str) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = REQUIRED_PREDICTION_COLUMNS.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{arm} predictions miss columns: {sorted(missing)}")
    frame = frame.copy()
    frame["truth_inchikey1"] = _clean_ik14(frame["truth_inchikey1"], f"{arm} truth")
    frame["candidate_inchikey1"] = _clean_ik14(frame["candidate_inchikey1"], f"{arm} candidate")
    frame["repeat"] = pd.to_numeric(frame["repeat"], errors="raise").astype(int)
    frame["candidate_score"] = pd.to_numeric(frame["candidate_score"], errors="raise").astype(float)
    frame["propagation_depth"] = pd.to_numeric(
        frame["propagation_depth"], errors="raise"
    ).astype(int)
    if np.any(~np.isfinite(frame["candidate_score"])):
        raise RuntimeError(f"{arm} predictions contain non-finite scores")
    if (frame["propagation_depth"] < 0).any():
        raise RuntimeError(f"{arm} predictions contain negative propagation depth")
    allowed = set(map(tuple, denominator[["repeat", "truth_inchikey1"]].to_numpy()))
    observed = set(map(tuple, frame[["repeat", "truth_inchikey1"]].to_numpy()))
    extras = observed.difference(allowed)
    if extras:
        raise RuntimeError(f"{arm} predictions contain non-hidden queries: {sorted(extras)[:5]}")
    # Multiple paths, ion forms or polarities can yield the same identity.  The
    # runner may emit all of them; identity scoring uses its strongest path.
    frame = (
        frame.sort_values(
            ["repeat", "truth_inchikey1", "candidate_inchikey1", "candidate_score", "propagation_depth"],
            ascending=[True, True, True, False, True],
            kind="mergesort",
        )
        .drop_duplicates(["repeat", "truth_inchikey1", "candidate_inchikey1"], keep="first")
        .reset_index(drop=True)
    )
    return frame


def score_arm(denominator: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    grouped = {
        key: group
        for key, group in predictions.groupby(["repeat", "truth_inchikey1"], sort=False)
    }
    rows: list[dict[str, object]] = []
    for record in denominator.itertuples(index=False):
        key = (int(record.repeat), str(record.truth_inchikey1))
        candidates = grouped.get(key)
        annotated = candidates is not None and not candidates.empty
        rank: int | None = None
        truth_depth: int | None = None
        top_score_depth: int | None = None
        if annotated:
            maximum = float(candidates["candidate_score"].max())
            top_score_depth = int(
                candidates.loc[candidates["candidate_score"].eq(maximum), "propagation_depth"].min()
            )
            truth = candidates.loc[candidates["candidate_inchikey1"].eq(record.truth_inchikey1)]
            if not truth.empty:
                truth_score = float(truth["candidate_score"].max())
                truth_depth = int(
                    truth.loc[truth["candidate_score"].eq(truth_score), "propagation_depth"].min()
                )
                # Strict rank: every candidate tied with the truth counts ahead.
                rank = int(1 + candidates.loc[
                    candidates["candidate_score"].ge(truth_score), "candidate_inchikey1"
                ].nunique() - 1)
        rows.append(
            {
                "repeat": key[0],
                "truth_inchikey1": key[1],
                "polarity_presence": record.polarity_presence,
                "annotated": bool(annotated),
                "truth_recovered": rank is not None,
                "truth_rank": rank,
                "top1_correct": rank == 1,
                "top3_correct": rank is not None and rank <= 3,
                "truth_propagation_depth": truth_depth,
                "top_score_propagation_depth": top_score_depth,
            }
        )
    return pd.DataFrame(rows)


def identity_cluster_bootstrap(
    candidate: pd.Series,
    author: pd.Series,
    identities: pd.Series,
    *,
    resamples: int,
    seed: int,
) -> dict[str, object]:
    frame = pd.DataFrame(
        {
            "candidate": candidate.astype(float),
            "author": author.astype(float),
            "identity": identities.astype(str),
        }
    )
    per_identity = frame.groupby("identity")[["candidate", "author"]].mean()
    deltas = (per_identity["candidate"] - per_identity["author"]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=float)
    for index in range(resamples):
        draws[index] = float(rng.choice(deltas, size=len(deltas), replace=True).mean())
    return {
        "mean_delta": float(deltas.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "identity_clusters": int(len(deltas)),
        "resamples": int(resamples),
    }


def summarize_arm(scored: pd.DataFrame) -> dict[str, object]:
    depth = (
        scored.loc[scored["truth_recovered"]]
        .groupby("truth_propagation_depth", dropna=False)
        .agg(instances=("top1_correct", "size"), recall1=("top1_correct", "mean"))
    )
    return {
        "instances": int(len(scored)),
        "identities": int(scored["truth_inchikey1"].nunique()),
        "annotation_coverage": float(scored["annotated"].mean()),
        "truth_recovery": float(scored["truth_recovered"].mean()),
        "recall1": float(scored["top1_correct"].mean()),
        "recall3": float(scored["top3_correct"].mean()),
        "truth_recovery_by_propagation_depth": {
            str(int(index)): {
                "instances": int(row["instances"]),
                "recall1": float(row["recall1"]),
            }
            for index, row in depth.iterrows()
            if pd.notna(index)
        },
    }


def transition_by_depth(author: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, object]:
    corrected = (~author["top1_correct"] & candidate["top1_correct"])
    introduced = (author["top1_correct"] & ~candidate["top1_correct"])
    depth = candidate["top_score_propagation_depth"].fillna(-1).astype(int)
    rows: dict[str, object] = {}
    for value in sorted(depth.unique()):
        mask = depth.eq(value)
        rows["missing" if value < 0 else str(value)] = {
            "instances": int(mask.sum()),
            "corrected": int((corrected & mask).sum()),
            "introduced": int((introduced & mask).sum()),
            "net": int((corrected & mask).sum() - (introduced & mask).sum()),
        }
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-dir", type=Path, required=True)
    parser.add_argument("--author-predictions", type=Path, required=True)
    parser.add_argument("--candidate-predictions", type=Path, required=True)
    parser.add_argument("--candidate-arm", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite output directory: {args.output_dir}")
    if args.bootstrap_resamples < 1000:
        raise ValueError("formal evaluation requires at least 1000 bootstrap resamples")

    denominator, contract = load_denominator(args.contract_dir)
    author_predictions = load_predictions(args.author_predictions, denominator, "author")
    candidate_predictions = load_predictions(
        args.candidate_predictions, denominator, args.candidate_arm
    )
    author = score_arm(denominator, author_predictions)
    candidate = score_arm(denominator, candidate_predictions)
    keys = ["repeat", "truth_inchikey1"]
    if not author[keys].equals(candidate[keys]):
        raise RuntimeError("author and candidate denominators differ")

    corrected = (~author["top1_correct"] & candidate["top1_correct"])
    introduced = (author["top1_correct"] & ~candidate["top1_correct"])
    discordant = int(corrected.sum() + introduced.sum())
    mcnemar_p = (
        float(binomtest(int(corrected.sum()), discordant, p=0.5, alternative="greater").pvalue)
        if discordant > 0 else 1.0
    )
    recall1_ci = identity_cluster_bootstrap(
        candidate["top1_correct"],
        author["top1_correct"],
        denominator["truth_inchikey1"],
        resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    recall3_ci = identity_cluster_bootstrap(
        candidate["top3_correct"],
        author["top3_correct"],
        denominator["truth_inchikey1"],
        resamples=args.bootstrap_resamples,
        seed=args.seed + 1,
    )
    coverage_ci = identity_cluster_bootstrap(
        candidate["annotated"],
        author["annotated"],
        denominator["truth_inchikey1"],
        resamples=args.bootstrap_resamples,
        seed=args.seed + 2,
    )
    gates = {
        "recall1_identity_cluster_ci_positive": recall1_ci["ci_low"] > 0,
        "recall3_identity_cluster_ci_nonnegative": recall3_ci["ci_low"] >= 0,
        "annotation_coverage_identity_cluster_ci_nonnegative": coverage_ci["ci_low"] >= 0,
        "corrected_gt_introduced": int(corrected.sum()) > int(introduced.sum()),
        "mcnemar_one_sided_p_le_0_05": mcnemar_p <= 0.05,
    }
    args.output_dir.mkdir(parents=True)
    per_query = denominator.copy()
    for prefix, scored in (("author", author), (args.candidate_arm, candidate)):
        for column in (
            "annotated",
            "truth_recovered",
            "truth_rank",
            "top1_correct",
            "top3_correct",
            "truth_propagation_depth",
            "top_score_propagation_depth",
        ):
            per_query[f"{prefix}_{column}"] = scored[column]
    per_query["corrected"] = corrected
    per_query["introduced"] = introduced
    per_query_path = args.output_dir / "per_identity_repeat.csv.gz"
    per_query.to_csv(per_query_path, index=False)

    report = {
        "status": "kgmn_hidden_seed_recovery_evaluation_complete",
        "formal": True,
        "protocol": (
            "matched 10-repeat identity-level hidden-seed recovery; missing predictions stay in denominator; "
            "candidate identity uses maximum path score; score ties count against truth"
        ),
        "candidate_arm": args.candidate_arm,
        "author": summarize_arm(author),
        "candidate": summarize_arm(candidate),
        "comparison": {
            "corrected": int(corrected.sum()),
            "introduced": int(introduced.sum()),
            "mcnemar_one_sided_p": mcnemar_p,
            "recall1_identity_cluster_bootstrap": recall1_ci,
            "recall3_identity_cluster_bootstrap": recall3_ci,
            "annotation_coverage_identity_cluster_bootstrap": coverage_ci,
            "transitions_by_candidate_top_score_propagation_depth": transition_by_depth(author, candidate),
        },
        "gates": gates,
        "pass": all(gates.values()),
        "provenance": {
            "contract_report_sha256": sha256(args.contract_dir / "report.json"),
            "hidden_seed_splits_sha256": sha256(args.contract_dir / "hidden_seed_splits.csv.gz"),
            "author_predictions_sha256": sha256(args.author_predictions),
            "candidate_predictions_sha256": sha256(args.candidate_predictions),
            "per_identity_repeat_sha256": sha256(per_query_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "Passing establishes incremental hidden-seed recovery on the frozen KGMN external protocol. "
            "The 46STD test_evaluation mode is a closed-world author candidate universe; passing does not "
            "establish open-world annotation, MSI Level 1 identity, phenotype mechanism, shared-embedding "
            "improvement, or SOTA."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
