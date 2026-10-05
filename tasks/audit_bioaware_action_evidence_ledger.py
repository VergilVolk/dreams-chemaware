#!/usr/bin/env python
"""Build a single, fail-closed evidence ledger for BioAware action discovery.

This audit does not fit a model, tune a threshold, evaluate P2b, or change an
embedding.  It reconciles every already-produced BioAware ranking action on the
chemically filtered 548-query negative-ion benchmark and makes the actual
independent evidence count explicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


NETWORK_FEATURES = [
    "known_mass_candidate_fraction",
    "known_path_fraction",
    "known_inverse_depth_mean",
    "known_log_seed_support_mean",
    "known_log_degree",
    "edge0_complete_fraction",
    "edge0_bottleneck_mean",
    "edge1_complete_fraction",
    "edge1_bottleneck_mean",
    "predicted_edge_increment",
]

REQUIRED_TRANSITION_COLUMNS = {
    "query_id",
    "truth_candidate_id",
    "truth_formula",
    "baseline_candidate_id",
    "proposed_candidate_id",
    "final_candidate_id",
    "baseline_correct",
    "final_correct",
    "corrected",
    "introduced",
    "delta",
    "intervene",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    mapping = {"true": True, "false": False, "1": True, "0": False}
    values = series.astype(str).str.strip().str.lower().map(mapping)
    if values.isna().any():
        bad = sorted(series.loc[values.isna()].astype(str).unique())[:5]
        raise RuntimeError(f"cannot interpret boolean values: {bad}")
    return values.astype(bool)


def biological_source(unit_id: str) -> str:
    text = str(unit_id)
    return text.rsplit("__", 1)[0] if "__" in text else text


def formula_cluster_bootstrap(
    frame: pd.DataFrame, repeats: int = 5000, seed: int = 20260906
) -> dict:
    grouped = frame.groupby("truth_formula", sort=False)["delta"].agg(["sum", "count"])
    if grouped.empty:
        raise RuntimeError("cannot bootstrap an empty transition table")
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=float)
    for index in range(repeats):
        draw = rng.integers(0, len(grouped), len(grouped))
        values[index] = sums[draw].sum() / counts[draw].sum()
    return {
        "mean": float(frame["delta"].mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "formulas": int(len(grouped)),
        "resamples": int(repeats),
    }


def validate_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    required = {
        "query_id", "candidate_id", "spectral_score", "best_library_row",
        "truth_candidate_id", "truth_formula", "unit_id", "baseline_correct",
        "top_candidate_id", "is_positive", *NETWORK_FEATURES,
    }
    missing = required - set(candidates.columns)
    if missing:
        raise RuntimeError(f"candidate table missing columns: {sorted(missing)}")
    if candidates.duplicated(["query_id", "candidate_id"]).any():
        raise RuntimeError("candidate identities are not unique within query")
    candidates = candidates.copy()
    candidates["is_positive"] = as_bool(candidates["is_positive"])
    candidates["baseline_correct"] = as_bool(candidates["baseline_correct"])
    invariant_columns = [
        "truth_candidate_id", "truth_formula", "unit_id", "baseline_correct",
        "top_candidate_id",
    ]
    for column in invariant_columns:
        if int(candidates.groupby("query_id")[column].nunique(dropna=False).max()) != 1:
            raise RuntimeError(f"query-level column is not invariant: {column}")
    positive_count = candidates.groupby("query_id")["is_positive"].sum()
    if not bool((positive_count == 1).all()):
        raise RuntimeError("every query must have exactly one truth candidate")
    for query_id, group in candidates.groupby("query_id", sort=False):
        truth = str(group.loc[group["is_positive"], "candidate_id"].iloc[0])
        declared_truth = str(group["truth_candidate_id"].iloc[0])
        if truth != declared_truth:
            raise RuntimeError(f"truth-candidate mismatch for {query_id}")
        top = str(group["top_candidate_id"].iloc[0])
        if top not in set(group["candidate_id"].astype(str)):
            raise RuntimeError(f"baseline top candidate absent for {query_id}")
        declared_correct = bool(group["baseline_correct"].iloc[0])
        if declared_correct != (top == truth):
            raise RuntimeError(f"baseline correctness mismatch for {query_id}")
    return candidates


def query_metadata(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for query_id, group in candidates.groupby("query_id", sort=False):
        first = group.iloc[0]
        spectral = group["spectral_score"].to_numpy(float)
        order = np.sort(spectral)[::-1]
        rows.append({
            "query_id": str(query_id),
            "truth_candidate_id": str(first["truth_candidate_id"]),
            "truth_formula": str(first["truth_formula"]),
            "acquisition_unit": str(first["unit_id"]),
            "biological_source": biological_source(first["unit_id"]),
            "adduct": str(first.get("adduct", "")),
            "baseline_candidate_id": str(first["top_candidate_id"]),
            "baseline_correct": bool(first["baseline_correct"]),
            "candidate_count": int(len(group)),
            "baseline_margin_recomputed": float(order[0] - order[1]),
        })
    return pd.DataFrame(rows)


def validate_transitions(
    frame: pd.DataFrame, metadata: pd.DataFrame, label: str
) -> pd.DataFrame:
    missing = REQUIRED_TRANSITION_COLUMNS - set(frame.columns)
    if missing:
        raise RuntimeError(f"{label}: transition table missing {sorted(missing)}")
    if frame["query_id"].duplicated().any():
        raise RuntimeError(f"{label}: duplicate query rows")
    expected = set(metadata["query_id"].astype(str))
    observed = set(frame["query_id"].astype(str))
    if observed != expected:
        raise RuntimeError(
            f"{label}: query-set mismatch missing={len(expected-observed)} extra={len(observed-expected)}"
        )
    frame = frame.copy()
    for column in ("baseline_correct", "final_correct", "corrected", "introduced", "intervene"):
        frame[column] = as_bool(frame[column])
    frame["query_id"] = frame["query_id"].astype(str)
    joined = frame.merge(
        metadata[
            ["query_id", "truth_candidate_id", "truth_formula", "baseline_candidate_id",
             "baseline_correct", "acquisition_unit", "biological_source", "candidate_count",
             "baseline_margin_recomputed"]
        ],
        on="query_id",
        suffixes=("", "__expected"),
        validate="one_to_one",
    )
    for column in ("truth_candidate_id", "truth_formula", "baseline_candidate_id"):
        if not bool(
            joined[column].astype(str).eq(joined[f"{column}__expected"].astype(str)).all()
        ):
            raise RuntimeError(f"{label}: {column} does not reproduce frozen candidates")
    if not bool(joined["baseline_correct"].eq(joined["baseline_correct__expected"]).all()):
        raise RuntimeError(f"{label}: baseline correctness changed")
    expected_final = joined["final_candidate_id"].astype(str).eq(
        joined["truth_candidate_id"].astype(str)
    )
    if not bool(joined["final_correct"].eq(expected_final).all()):
        raise RuntimeError(f"{label}: final correctness inconsistent with final candidate")
    expected_corrected = (~joined["baseline_correct"]) & joined["final_correct"]
    expected_introduced = joined["baseline_correct"] & (~joined["final_correct"])
    if not bool(joined["corrected"].eq(expected_corrected).all()):
        raise RuntimeError(f"{label}: corrected flags are inconsistent")
    if not bool(joined["introduced"].eq(expected_introduced).all()):
        raise RuntimeError(f"{label}: introduced flags are inconsistent")
    expected_delta = joined["final_correct"].astype(int) - joined["baseline_correct"].astype(int)
    if not bool(joined["delta"].astype(int).eq(expected_delta).all()):
        raise RuntimeError(f"{label}: delta flags are inconsistent")
    joined["action_outcome"] = np.select(
        [
            joined["corrected"], joined["introduced"], joined["intervene"],
            ~joined["baseline_correct"], joined["baseline_correct"],
        ],
        ["corrected", "introduced", "intervened_neutral", "abstained_wrong", "protected_correct"],
        default="inconsistent",
    )
    if bool(joined["action_outcome"].eq("inconsistent").any()):
        raise RuntimeError(f"{label}: unclassified transition state")
    return joined.drop(columns=[column for column in joined if column.endswith("__expected")])


def action_summary(frame: pd.DataFrame, repeats: int, seed: int) -> dict:
    corrected = frame.loc[frame["corrected"]]
    introduced = frame.loc[frame["introduced"]]
    interventions = frame.loc[frame["intervene"]]
    pairs = corrected[
        ["truth_candidate_id", "baseline_candidate_id"]
    ].drop_duplicates()
    by_pair = corrected.groupby(
        ["truth_candidate_id", "baseline_candidate_id"], sort=False
    ).agg(
        queries=("query_id", "size"),
        sources=("biological_source", "nunique"),
    ).reset_index()
    return {
        "queries": int(len(frame)),
        "baseline_recall1": float(frame["baseline_correct"].mean()),
        "final_recall1": float(frame["final_correct"].mean()),
        "delta_recall1": float(frame["delta"].mean()),
        "delta_pp": float(100.0 * frame["delta"].mean()),
        "corrected": int(frame["corrected"].sum()),
        "introduced": int(frame["introduced"].sum()),
        "risk_weighted_net_lambda2": int(frame["corrected"].sum() - 2 * frame["introduced"].sum()),
        "interventions": int(frame["intervene"].sum()),
        "intervention_precision": (
            float(frame["corrected"].sum() / len(interventions)) if len(interventions) else None
        ),
        "baseline_errors_recovered_fraction": float(
            frame["corrected"].sum() / max(1, (~frame["baseline_correct"]).sum())
        ),
        "corrected_truth_identities": int(corrected["truth_candidate_id"].nunique()),
        "corrected_truth_formulas": int(corrected["truth_formula"].nunique()),
        "corrected_truth_baseline_pairs": int(len(pairs)),
        "corrected_pairs_replicated_across_sources": int((by_pair["sources"] >= 2).sum()),
        "maximum_queries_per_corrected_pair": int(by_pair["queries"].max()) if len(by_pair) else 0,
        "introduced_truth_identities": int(introduced["truth_candidate_id"].nunique()),
        "formula_cluster_bootstrap": formula_cluster_bootstrap(frame, repeats, seed),
    }


def candidate_role_record(
    indexed: pd.DataFrame, query_id: str, candidate_id: str, role: str
) -> dict:
    key = (str(query_id), str(candidate_id))
    if key not in indexed.index:
        raise RuntimeError(f"missing {role} candidate row: {key}")
    row = indexed.loc[key]
    result = {
        f"{role}_candidate_id": str(candidate_id),
        f"{role}_best_library_row": int(row["best_library_row"]),
        f"{role}_spectral_score": float(row["spectral_score"]),
    }
    for feature in NETWORK_FEATURES:
        result[f"{role}_{feature}"] = float(row[feature])
    return result


def changed_action_table(
    strict_frames: dict[str, pd.DataFrame], candidates: pd.DataFrame, manifest: pd.DataFrame
) -> pd.DataFrame:
    indexed = candidates.set_index(["query_id", "candidate_id"])
    if not indexed.index.is_unique:
        raise RuntimeError("candidate identities are not unique within query")
    rows: list[dict] = []
    for recipe, frame in strict_frames.items():
        for transition in frame.loc[frame["corrected"] | frame["introduced"]].itertuples(index=False):
            record = transition._asdict()
            record["recipe"] = recipe
            for role, candidate_id in (
                ("truth", transition.truth_candidate_id),
                ("baseline", transition.baseline_candidate_id),
                ("proposal", transition.proposed_candidate_id),
            ):
                record.update(candidate_role_record(indexed, transition.query_id, candidate_id, role))
            for feature in NETWORK_FEATURES:
                record[f"proposal_minus_baseline__{feature}"] = (
                    record[f"proposal_{feature}"] - record[f"baseline_{feature}"]
                )
            rows.append(record)
    changed = pd.DataFrame(rows)
    for role in ("truth", "baseline", "proposal"):
        row_column = f"{role}_best_library_row"
        lookup = manifest[
            ["library_row", "name", "smiles", "inchikey", "precursor_mz"]
        ].rename(columns={
            "library_row": row_column,
            "name": f"{role}_name",
            "smiles": f"{role}_smiles",
            "inchikey": f"{role}_manifest_inchikey",
            "precursor_mz": f"{role}_library_precursor_mz",
        })
        changed = changed.merge(lookup, on=row_column, how="left", validate="many_to_one")
        if changed[f"{role}_name"].isna().any():
            raise RuntimeError(f"manifest lookup failed for {role}")
        if not bool(
            changed[f"{role}_manifest_inchikey"].astype(str).str[:14].eq(
                changed[f"{role}_candidate_id"].astype(str)
            ).all()
        ):
            raise RuntimeError(f"manifest identity mismatch for {role}")
    return changed


def baseline_error_headroom(
    candidates: pd.DataFrame, strict_frames: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    indexed = candidates.set_index(["query_id", "candidate_id"])
    if not indexed.index.is_unique:
        raise RuntimeError("candidate identities are not unique within query")
    safe_corrected = set(
        strict_frames["strict_safe"].loc[strict_frames["strict_safe"]["corrected"], "query_id"]
    )
    high_corrected = set(
        strict_frames["strict_high_recall"].loc[
            strict_frames["strict_high_recall"]["corrected"], "query_id"
        ]
    )
    rows: list[dict] = []
    for query_id, group in candidates.groupby("query_id", sort=False):
        if bool(group["baseline_correct"].iloc[0]):
            continue
        truth_id = str(group["truth_candidate_id"].iloc[0])
        baseline_id = str(group["top_candidate_id"].iloc[0])
        truth = indexed.loc[(str(query_id), truth_id)]
        baseline = indexed.loc[(str(query_id), baseline_id)]
        record = {
            "query_id": str(query_id),
            "acquisition_unit": str(group["unit_id"].iloc[0]),
            "biological_source": biological_source(group["unit_id"].iloc[0]),
            "truth_candidate_id": truth_id,
            "baseline_candidate_id": baseline_id,
            "truth_formula": str(group["truth_formula"].iloc[0]),
            "candidate_count": int(len(group)),
            "safe_corrected": str(query_id) in safe_corrected,
            "high_recall_corrected": str(query_id) in high_corrected,
        }
        any_truth_advantage = False
        any_unique_top = False
        for feature in NETWORK_FEATURES:
            truth_value = float(truth[feature])
            baseline_value = float(baseline[feature])
            nontruth = group.loc[~as_bool(group["is_positive"]), feature].to_numpy(float)
            record[f"truth_minus_baseline__{feature}"] = truth_value - baseline_value
            record[f"truth_unique_top__{feature}"] = bool(
                len(nontruth) and truth_value > float(np.max(nontruth))
            )
            any_truth_advantage |= truth_value > baseline_value
            any_unique_top |= record[f"truth_unique_top__{feature}"]
        record["any_truth_network_feature_gt_baseline"] = any_truth_advantage
        record["truth_unique_top_in_any_network_feature"] = any_unique_top
        record["truth_path_while_baseline_none"] = bool(
            float(truth["known_path_fraction"]) > 0
            and float(baseline["known_path_fraction"]) == 0
        )
        record["truth_edge0_while_baseline_none"] = bool(
            float(truth["edge0_complete_fraction"]) > 0
            and float(baseline["edge0_complete_fraction"]) == 0
        )
        record["truth_edge1_while_baseline_none"] = bool(
            float(truth["edge1_complete_fraction"]) > 0
            and float(baseline["edge1_complete_fraction"]) == 0
        )
        rows.append(record)
    return pd.DataFrame(rows)


def set_overlap_matrix(frames: dict[str, pd.DataFrame], column: str) -> pd.DataFrame:
    names = list(frames)
    sets = {name: set(frame.loc[frame[column], "query_id"].astype(str)) for name, frame in frames.items()}
    values = np.zeros((len(names), len(names)), dtype=float)
    for i, left in enumerate(names):
        for j, right in enumerate(names):
            union = sets[left] | sets[right]
            values[i, j] = len(sets[left] & sets[right]) / len(union) if union else 1.0
    return pd.DataFrame(values, index=names, columns=names)


def external_result(path: Path, name: str) -> dict:
    report = read_json(path)
    if "overall" in report:
        result = report["overall"]
    else:
        result = report
    return {
        "name": name,
        "queries": int(result.get("queries", result.get("query_rotations", 0))),
        "unique_queries": int(result.get("unique_queries", result.get("queries", 0))),
        "baseline_recall1": float(result.get("baseline_recall1", np.nan)),
        "bioaware_recall1": float(result.get("bioaware_recall1", np.nan)),
        "delta_recall1": float(result.get("delta_recall1", np.nan)),
        "corrected": int(result.get("corrected", 0)),
        "introduced": int(result.get("introduced", 0)),
        "interventions": int(result.get("interventions", 0)),
        "confirmatory_pass": bool(report.get("confirmatory_pass", report.get("pass", False))),
        "report_sha256": sha256(path),
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-features", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_negative_loso_ranker_v4_chemically_filtered/candidate_features.csv.gz"),
    )
    parser.add_argument(
        "--ablation-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_negative_loso_ablation_v3_chemically_filtered"),
    )
    parser.add_argument(
        "--strict-safe-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_negative_source_loso_v2_full_bioaware"),
    )
    parser.add_argument(
        "--strict-high-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_negative_source_loso_v2_full_no_edge_gate"),
    )
    parser.add_argument(
        "--candidate-permutation-report", type=Path,
        default=Path("data/validation/bioaware_metdna3_external_negative_candidate_permutation_v2_full_bioaware/report.json"),
    )
    parser.add_argument(
        "--same-formula-candidate-features", type=Path,
        default=Path(
            "data/validation/bioaware_same_formula_network_expert_v3_development/"
            "candidate_features.csv.gz"
        ),
    )
    parser.add_argument(
        "--same-formula-action-dir", type=Path,
        default=Path(
            "data/validation/bioaware_same_formula_uncertainty_expert_v4_development"
        ),
    )
    parser.add_argument(
        "--same-formula-permutation-report", type=Path,
        default=Path(
            "data/validation/bioaware_same_formula_uncertainty_expert_v4_permutation/"
            "report.json"
        ),
    )
    parser.add_argument(
        "--mona-manifest", type=Path,
        default=Path("data/models/mona_neg_dreams_emb/manifest.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_action_evidence_ledger_v2_20260906"),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output directory is non-empty: {args.output_dir}")
    required_paths: Iterable[Path] = [
        args.candidate_features, args.mona_manifest, args.candidate_permutation_report,
        args.strict_safe_dir / "report.json",
        args.strict_high_dir / "report.json",
        args.strict_safe_dir / "source_identity_formula_purged__transitions.csv.gz",
        args.strict_high_dir / "source_identity_formula_purged__transitions.csv.gz",
        args.same_formula_candidate_features,
        args.same_formula_action_dir / "report.json",
        args.same_formula_action_dir / "query_transitions.csv.gz",
        args.same_formula_permutation_report,
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    candidates = validate_candidates(pd.read_csv(args.candidate_features))
    metadata = query_metadata(candidates)
    if len(candidates) != 2003 or len(metadata) != 548:
        raise RuntimeError(
            f"chemically filtered protocol changed: candidates={len(candidates)} queries={len(metadata)}"
        )

    frames: dict[str, pd.DataFrame] = {}
    for path in sorted(args.ablation_dir.glob("*__transitions.csv.gz")):
        name = path.name.removesuffix("__transitions.csv.gz")
        frame = validate_transitions(pd.read_csv(path), metadata, f"ablation:{name}")
        frame["protocol"] = "eight_unit_identity_purged"
        frame["recipe"] = name
        frames[f"ablation::{name}"] = frame
    if len(frames) != 12:
        raise RuntimeError(f"expected 12 ablation recipes, found {len(frames)}")

    strict_specs = {
        "strict_safe": args.strict_safe_dir,
        "strict_high_recall": args.strict_high_dir,
    }
    strict_frames: dict[str, pd.DataFrame] = {}
    for name, directory in strict_specs.items():
        path = directory / "source_identity_formula_purged__transitions.csv.gz"
        frame = validate_transitions(pd.read_csv(path), metadata, f"strict:{name}")
        frame["protocol"] = "source_identity_formula_purged"
        frame["recipe"] = name
        frames[f"strict::{name}"] = frame
        strict_frames[name] = frame

    same_formula_candidates = validate_candidates(
        pd.read_csv(args.same_formula_candidate_features)
    )
    same_formula_metadata = query_metadata(same_formula_candidates)
    if len(same_formula_candidates) != 1800 or len(same_formula_metadata) != 482:
        raise RuntimeError(
            "same-formula protocol changed: "
            f"candidates={len(same_formula_candidates)} "
            f"queries={len(same_formula_metadata)}"
        )
    same_formula = validate_transitions(
        pd.read_csv(args.same_formula_action_dir / "query_transitions.csv.gz"),
        same_formula_metadata,
        "same_formula_uncertainty",
    )
    same_formula["protocol"] = "eight_unit_identity_formula_purged_same_formula"
    same_formula["recipe"] = "same_formula_uncertainty"
    frames["same_formula::uncertainty"] = same_formula

    safe_interventions = set(
        strict_frames["strict_safe"].loc[strict_frames["strict_safe"]["intervene"], "query_id"]
    )
    high_interventions = set(
        strict_frames["strict_high_recall"].loc[
            strict_frames["strict_high_recall"]["intervene"], "query_id"
        ]
    )

    summaries: list[dict] = []
    source_rows: list[dict] = []
    for index, (key, frame) in enumerate(frames.items()):
        summary = action_summary(frame, args.bootstrap_resamples, 20260906 + index)
        summary.update({"key": key, "protocol": frame["protocol"].iloc[0], "recipe": frame["recipe"].iloc[0]})
        summaries.append(summary)
        for source, local in frame.groupby("biological_source", sort=True):
            source_rows.append({
                "key": key,
                "protocol": frame["protocol"].iloc[0],
                "recipe": frame["recipe"].iloc[0],
                "biological_source": source,
                "queries": int(len(local)),
                "baseline_recall1": float(local["baseline_correct"].mean()),
                "final_recall1": float(local["final_correct"].mean()),
                "delta_pp": float(100 * local["delta"].mean()),
                "corrected": int(local["corrected"].sum()),
                "introduced": int(local["introduced"].sum()),
                "interventions": int(local["intervene"].sum()),
            })
    summary_frame = pd.DataFrame(summaries).sort_values("delta_pp", ascending=False)

    manifest = pd.read_csv(args.mona_manifest).reset_index(names="library_row")
    changed = changed_action_table(strict_frames, candidates, manifest)
    same_formula_changed = changed_action_table(
        {"same_formula_uncertainty": same_formula},
        same_formula_candidates,
        manifest,
    )
    headroom = baseline_error_headroom(candidates, strict_frames)
    all_ledger = pd.concat(frames.values(), ignore_index=True, sort=False)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    ledger_path = args.output_dir / "query_action_ledger.csv.gz"
    summary_path = args.output_dir / "action_summary.csv"
    source_path = args.output_dir / "action_by_source.csv"
    changed_path = args.output_dir / "strict_changed_actions.csv.gz"
    same_formula_changed_path = args.output_dir / "same_formula_changed_actions.csv.gz"
    headroom_path = args.output_dir / "baseline_error_headroom.csv.gz"
    corrected_overlap_path = args.output_dir / "corrected_action_jaccard.csv"
    introduced_overlap_path = args.output_dir / "introduced_action_jaccard.csv"
    all_ledger.to_csv(ledger_path, index=False, compression="gzip")
    summary_frame.to_csv(summary_path, index=False)
    pd.DataFrame(source_rows).to_csv(source_path, index=False)
    changed.to_csv(changed_path, index=False, compression="gzip")
    same_formula_changed.to_csv(
        same_formula_changed_path, index=False, compression="gzip"
    )
    headroom.to_csv(headroom_path, index=False, compression="gzip")
    set_overlap_matrix(frames, "corrected").to_csv(corrected_overlap_path)
    set_overlap_matrix(frames, "introduced").to_csv(introduced_overlap_path)

    safe_summary = next(item for item in summaries if item["key"] == "strict::strict_safe")
    high_summary = next(item for item in summaries if item["key"] == "strict::strict_high_recall")
    same_formula_summary = next(
        item for item in summaries if item["key"] == "same_formula::uncertainty"
    )
    permutation = read_json(args.candidate_permutation_report)
    same_formula_permutation = read_json(args.same_formula_permutation_report)
    external_paths = [
        (Path("data/validation/bioaware_st001154_hilic_frozen_evaluation_v1/report.json"), "ST001154_hilic"),
        (Path("data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/report.json"), "ST001154_hilic_extension"),
        (Path("data/validation/bioaware_kgmn200std_hidden_seed_v1/report.json"), "KGMN_200STD_hidden_seed"),
    ]
    external = [external_result(path, name) for path, name in external_paths]

    headroom_counts = {
        "official_errors": int(len(headroom)),
        "any_truth_network_feature_gt_baseline": int(headroom["any_truth_network_feature_gt_baseline"].sum()),
        "truth_unique_top_in_any_network_feature": int(headroom["truth_unique_top_in_any_network_feature"].sum()),
        "truth_path_while_baseline_none": int(headroom["truth_path_while_baseline_none"].sum()),
        "truth_edge0_while_baseline_none": int(headroom["truth_edge0_while_baseline_none"].sum()),
        "truth_edge1_while_baseline_none": int(headroom["truth_edge1_while_baseline_none"].sum()),
        "strict_safe_corrected": int(headroom["safe_corrected"].sum()),
        "strict_high_recall_corrected": int(headroom["high_recall_corrected"].sum()),
        "warning": "These truth-conditioned counts are retrospective oracle headroom, never deployable actions.",
    }

    development_action_gate = {
        "safe_delta_ge_3pp": bool(safe_summary["delta_pp"] >= 3.0),
        "safe_formula_cluster_ci_low_positive": bool(safe_summary["formula_cluster_bootstrap"]["ci_low"] > 0),
        "safe_corrected_gt_2x_introduced": bool(safe_summary["corrected"] > 2 * safe_summary["introduced"]),
        "safe_corrected_identities_ge_20": bool(safe_summary["corrected_truth_identities"] >= 20),
        "safe_corrected_formulas_ge_15": bool(safe_summary["corrected_truth_formulas"] >= 15),
        "same_formula_delta_ge_3pp": bool(same_formula_summary["delta_pp"] >= 3.0),
        "same_formula_formula_cluster_ci_low_positive": bool(
            same_formula_summary["formula_cluster_bootstrap"]["ci_low"] > 0
        ),
        "same_formula_corrected_identities_ge_20": bool(
            same_formula_summary["corrected_truth_identities"] >= 20
        ),
        "candidate_assignment_permutation_pass": bool(permutation.get("pass", False)),
        "degree_path_matched_null_pass": False,
        "independent_external_positive_confirmation": bool(any(item["confirmatory_pass"] for item in external)),
    }
    action_ready_for_embedding = bool(all(development_action_gate.values()))

    report = {
        "status": "bioaware_action_evidence_ledger_complete",
        "formal": True,
        "purpose": "Action-first reconciliation; no fitting, threshold search, P2b, phenotype or embedding update.",
        "candidate_protocol": {
            "queries": int(metadata.shape[0]),
            "candidate_pairs": int(len(candidates)),
            "truth_identities": int(metadata["truth_candidate_id"].nunique()),
            "truth_formulas": int(metadata["truth_formula"].nunique()),
            "official_errors": int((~metadata["baseline_correct"]).sum()),
        },
        "strict_safe": safe_summary,
        "strict_high_recall": high_summary,
        "same_formula_uncertainty": same_formula_summary,
        "strict_safe_interventions_subset_of_high_recall": bool(safe_interventions <= high_interventions),
        "strict_safe_intervention_overlap": int(len(safe_interventions & high_interventions)),
        "candidate_assignment_permutation": {
            "observed_delta_pp": 100 * float(permutation["observed"]["delta_recall1"]),
            "null_mean_pp": 100 * float(permutation["null_metrics"]["delta_recall1"]["null_mean"]),
            "null_p95_pp": 100 * float(permutation["null_metrics"]["delta_recall1"]["null_p95"]),
            "empirical_p": float(permutation["null_metrics"]["delta_recall1"]["empirical_one_sided_p_ge_observed"]),
            "repeats": int(permutation["repeats"]),
            "limit": "Candidate-assignment falsification on opened development data; not a degree/path-matched null or external confirmation.",
        },
        "same_formula_candidate_assignment_permutation": {
            "observed_delta_pp": 100 * float(
                same_formula_permutation["observed"]["delta_recall1"]
            ),
            "null_mean_pp": 100 * float(
                same_formula_permutation["null_metrics"]["delta_recall1"]["null_mean"]
            ),
            "null_p95_pp": 100 * float(
                same_formula_permutation["null_metrics"]["delta_recall1"]["null_p95"]
            ),
            "empirical_p": float(
                same_formula_permutation["null_metrics"]["delta_recall1"]
                ["empirical_one_sided_p_ge_observed"]
            ),
            "repeats": int(same_formula_permutation["repeats"]),
            "limit": (
                "Opened same-formula development falsification; not external confirmation "
                "or a degree/path-matched null."
            ),
        },
        "retrospective_action_headroom": headroom_counts,
        "external_frozen_evaluations": external,
        "development_action_gate": development_action_gate,
        "action_ready_for_embedding": action_ready_for_embedding,
        "decision": (
            "FREEZE_ACTION_THEN_TEST_EMBEDDING" if action_ready_for_embedding
            else "EXPAND_AND_FALSIFY_ACTION_BEFORE_ANY_BIOAWARE_EMBEDDING_TRAINING"
        ),
        "next_action_families": [
            "degree- and path-availability-matched candidate counterfactual",
            "reaction-mass-difference versus fragment/neutral-loss transformation consistency",
            "sample-level RT/coelution/correlation and ion-family conflict resolution",
            "recursive seed propagation with cross-fit seed removal and bounded depth",
            "network-coverage-aware abstention calibrated for nonzero external activation",
        ],
        "contracts": {
            "models_fitted": False,
            "thresholds_tuned": False,
            "P2b_used": False,
            "phenotype_used": False,
            "shared_embedding_changed": False,
            "opened_548_query_data_are_confirmation": False,
            "truth_conditioned_headroom_is_deployable": False,
        },
        "provenance": {
            "candidate_features_sha256": sha256(args.candidate_features),
            "same_formula_candidate_features_sha256": sha256(
                args.same_formula_candidate_features
            ),
            "same_formula_action_report_sha256": sha256(
                args.same_formula_action_dir / "report.json"
            ),
            "same_formula_permutation_report_sha256": sha256(
                args.same_formula_permutation_report
            ),
            "mona_manifest_sha256": sha256(args.mona_manifest),
            "candidate_permutation_report_sha256": sha256(args.candidate_permutation_report),
            "query_action_ledger_sha256": sha256(ledger_path),
            "action_summary_sha256": sha256(summary_path),
            "action_by_source_sha256": sha256(source_path),
            "strict_changed_actions_sha256": sha256(changed_path),
            "same_formula_changed_actions_sha256": sha256(
                same_formula_changed_path
            ),
            "baseline_error_headroom_sha256": sha256(headroom_path),
            "corrected_action_jaccard_sha256": sha256(corrected_overlap_path),
            "introduced_action_jaccard_sha256": sha256(introduced_overlap_path),
        },
        "claim_limit": (
            "Two current network actions are development-qualified at 3-6 pp, but the "
            "same-formula action corrects only 19 independent truth identities and the stricter "
            "four-source action only 8-12. Neither is externally confirmed, SOTA, or ready to "
            "supervise a shared encoder."
        ),
    }
    report_path = args.output_dir / "report.json"
    write_json(report_path, report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
