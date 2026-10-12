#!/usr/bin/env python
"""ChemProp2 M1: does an unresolved-MS/MS response reproduce across replicates?

This is deliberately a within-source development experiment.  It estimates a
microbiome-specific, time-resolved response for each drug and biological
replicate, then asks whether a held-out replicate retrieves the same drug from
prototypes built from the other replicates.  It does not test cross-study or
cross-instrument generalization and it does not infer reaction direction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


SAMPLE_RE = re.compile(
    r"^[Tt](?P<time>\d+)_(?P<drug>[^_]+)_(?P<bac>noBac|Bac)_rep(?P<rep>\d+).*\.mzML Peak area$",
    re.IGNORECASE,
)
CONTROL_DRUGS = {"dmso", "medium"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_samples(columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in columns:
        match = SAMPLE_RE.match(column)
        if match is None:
            continue
        rows.append(
            {
                "column": column,
                "time": int(match.group("time")),
                "drug": match.group("drug").lower(),
                "bacteria": match.group("bac").lower() == "bac",
                "replicate": int(match.group("rep")),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("no ChemProp2 study samples were parsed")
    if result.duplicated(["time", "drug", "bacteria", "replicate"]).any():
        raise RuntimeError("duplicate time/drug/bacteria/replicate sample key")
    return result.sort_values(["drug", "replicate", "time", "bacteria"]).reset_index(drop=True)


def nonempty_mgf_scans(path: Path) -> set[int]:
    scans: set[int] = set()
    current: int | None = None
    peak_count = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line == "BEGIN IONS":
                current = None
                peak_count = 0
            elif line.startswith("SCANS="):
                current = int(line.split("=", 1)[1])
            elif line == "END IONS":
                if current is not None and peak_count >= 3:
                    scans.add(current)
            elif line and "=" not in line:
                fields = line.split()
                if len(fields) >= 2:
                    try:
                        float(fields[0])
                        float(fields[1])
                    except ValueError:
                        continue
                    peak_count += 1
    return scans


def robust_log2(values: np.ndarray) -> np.ndarray:
    positive = np.where(values > 0, values, np.inf)
    minima = np.min(positive, axis=0)
    minima = np.where(np.isfinite(minima) & (minima > 0), minima, 1.0)
    pseudo = 0.5 * minima
    return np.log2(values + pseudo[None, :])


def median_center(values: np.ndarray) -> np.ndarray:
    center = np.nanmedian(values, axis=1, keepdims=True)
    return values - center


def feature_qc_mask(
    quant: pd.DataFrame,
    samples: pd.DataFrame,
    scan_ids: set[int],
    blank_ratio: float,
    qc_cv_max: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    sample_columns = samples["column"].tolist()
    study = quant[sample_columns].to_numpy(float).T
    area_columns = [column for column in quant.columns if column.endswith(" Peak area")]
    blank_columns = [column for column in area_columns if "blank" in column.lower()]
    qc_columns = [
        column for column in area_columns
        if any(token in column.lower() for token in ("pooled_qc", "polled_qc", "polled_qcs", "qc-mix"))
    ]
    if not blank_columns or not qc_columns:
        raise RuntimeError(f"blank/QC columns missing: blank={len(blank_columns)} qc={len(qc_columns)}")
    blank = quant[blank_columns].to_numpy(float).T
    qc = quant[qc_columns].to_numpy(float).T

    study_q95 = np.quantile(study, 0.95, axis=0)
    blank_q95 = np.quantile(blank, 0.95, axis=0)
    blank_pass = (study_q95 > 0) & (study_q95 >= blank_ratio * np.maximum(blank_q95, 1.0))

    qc_detect = np.mean(qc > 0, axis=0)
    qc_mean = np.mean(qc, axis=0)
    qc_sd = np.std(qc, axis=0, ddof=1)
    qc_cv = np.divide(qc_sd, qc_mean, out=np.full_like(qc_mean, np.inf), where=qc_mean > 0)
    qc_pass = (qc_detect >= 0.50) & (qc_cv <= qc_cv_max)

    # Preserve drug-specific features if one experimental cell is reproducibly detected.
    reproducible_cell = np.zeros(len(quant), dtype=bool)
    indexed = samples.set_index(["drug", "time", "bacteria"])
    for _, cell in indexed.groupby(level=[0, 1, 2], sort=False):
        columns = cell["column"].tolist()
        if len(columns) >= 3:
            reproducible_cell |= np.sum(quant[columns].to_numpy(float) > 0, axis=1) >= 2

    has_ms2 = quant["row ID"].astype(int).isin(scan_ids).to_numpy()
    mask = blank_pass & (qc_pass | reproducible_cell) & has_ms2
    audit = pd.DataFrame(
        {
            "row_id": quant["row ID"].astype(int),
            "blank_pass": blank_pass,
            "qc_pass": qc_pass,
            "reproducible_cell": reproducible_cell,
            "has_nonempty_ms2": has_ms2,
            "qc_detection_rate": qc_detect,
            "qc_cv": qc_cv,
            "study_q95": study_q95,
            "blank_q95": blank_q95,
            "analysis_pass": mask,
        }
    )
    return mask, audit


def response_profiles(
    log_values: np.ndarray,
    samples: pd.DataFrame,
    feature_mask: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    sample_index = {row.column: idx for idx, row in samples.iterrows()}
    drugs = sorted(set(samples["drug"]) - CONTROL_DRUGS)
    times = sorted(set(samples["time"]))
    replicates = sorted(set(samples["replicate"]))
    if 0 not in times or len(times) < 3 or not {"dmso", "medium"}.issubset(set(samples["drug"])):
        raise RuntimeError("response construction requires T0, later times, DMSO+Bac, and Medium+noBac")

    keyed = {
        (row.drug, int(row.time), bool(row.bacteria), int(row.replicate)): sample_index[row.column]
        for row in samples.itertuples(index=False)
    }
    later = [time for time in times if time > 0]
    rows: list[np.ndarray] = []
    meta: list[dict[str, object]] = []
    for drug in drugs:
        for replicate in replicates:
            # Some deposited T7 files are missing. Require T0 and at least six later times.
            available_later = [
                time for time in later
                if all((drug, time, bac, replicate) in keyed for bac in (False, True))
                and ("dmso", time, True, replicate) in keyed
                and ("medium", time, False, replicate) in keyed
            ]
            if not all((drug, 0, bac, replicate) in keyed for bac in (False, True)):
                continue
            if ("dmso", 0, True, replicate) not in keyed or ("medium", 0, False, replicate) not in keyed:
                continue
            if len(available_later) < 6:
                continue

            def bac_difference(name: str, time: int) -> np.ndarray:
                return (
                    log_values[keyed[(name, time, True, replicate)]]
                    - log_values[keyed[(name, time, False, replicate)]]
                )

            drug0 = bac_difference(drug, 0)
            def vehicle_difference(time: int) -> np.ndarray:
                return (
                    log_values[keyed[("dmso", time, True, replicate)]]
                    - log_values[keyed[("medium", time, False, replicate)]]
                )

            vehicle0 = vehicle_difference(0)
            trajectory = np.stack(
                [
                    (bac_difference(drug, time) - drug0)
                    - (vehicle_difference(time) - vehicle0)
                    for time in available_later
                ]
            )
            if mode == "trajectory_auc":
                x = np.asarray(available_later, dtype=float)
                profile = np.trapezoid(trajectory, x=x, axis=0) / (x[-1] - x[0])
            elif mode == "endpoint":
                profile = trajectory[-1]
            else:
                raise RuntimeError(f"unknown response mode: {mode}")
            rows.append(profile[feature_mask])
            meta.append(
                {
                    "drug": drug,
                    "replicate": replicate,
                    "n_later_times": len(available_later),
                    "later_times": ",".join(map(str, available_later)),
                }
            )
    return np.stack(rows), pd.DataFrame(meta)


def cosine_rows(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    qnorm = np.linalg.norm(query)
    cnorm = np.linalg.norm(candidates, axis=1)
    denominator = qnorm * cnorm
    return np.divide(candidates @ query, denominator, out=np.full(len(candidates), -1.0), where=denominator > 0)


def retrieval(profile: np.ndarray, meta: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    if mask.sum() == 0:
        raise RuntimeError("retrieval mask selects zero features")
    selected = profile[:, mask].copy()
    scale = np.nanmedian(np.abs(selected - np.nanmedian(selected, axis=0)), axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    selected = np.clip(selected / scale[None, :], -10.0, 10.0)
    drugs = sorted(meta["drug"].unique())
    rows: list[dict[str, object]] = []
    for idx, item in meta.iterrows():
        candidate_profiles: list[np.ndarray] = []
        for drug in drugs:
            train = (meta["drug"].eq(drug) & ~meta["replicate"].eq(item.replicate)).to_numpy()
            candidate_profiles.append(np.mean(selected[train], axis=0))
        scores = cosine_rows(selected[idx], np.stack(candidate_profiles))
        order = np.argsort(-scores, kind="mergesort")
        truth = drugs.index(str(item.drug))
        rank = int(np.flatnonzero(order == truth)[0] + 1)
        rows.append(
            {
                "drug": item.drug,
                "replicate": int(item.replicate),
                "rank": rank,
                "recall1": rank == 1,
                "reciprocal_rank": 1.0 / rank,
                "truth_similarity": float(scores[truth]),
                "top_wrong_similarity": float(np.max(np.delete(scores, truth))),
                "margin": float(scores[truth] - np.max(np.delete(scores, truth))),
            }
        )
    return pd.DataFrame(rows)


def cluster_bootstrap_delta(
    left: pd.DataFrame, right: pd.DataFrame, column: str, seed: int, draws: int = 10000,
) -> dict[str, float]:
    key = ["drug", "replicate"]
    merged = left[key + [column]].merge(right[key + [column]], on=key, suffixes=("_left", "_right"))
    drugs = sorted(merged["drug"].unique())
    merged[f"{column}_left"] = merged[f"{column}_left"].astype(float)
    merged[f"{column}_right"] = merged[f"{column}_right"].astype(float)
    by_drug = merged.groupby("drug").apply(
        lambda frame: float((frame[f"{column}_left"] - frame[f"{column}_right"]).mean()),
        include_groups=False,
    )
    rng = np.random.default_rng(seed)
    samples = np.asarray(
        [np.mean([by_drug.loc[drug] for drug in rng.choice(drugs, len(drugs), replace=True)]) for _ in range(draws)]
    )
    estimate = float((merged[f"{column}_left"] - merged[f"{column}_right"]).mean())
    return {
        "estimate": estimate,
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "clusters": len(drugs),
        "queries": len(merged),
    }


def summarize(frame: pd.DataFrame) -> dict[str, float | int]:
    return {
        "queries": int(len(frame)),
        "drugs": int(frame["drug"].nunique()),
        "recall1": float(frame["recall1"].mean()),
        "mrr": float(frame["reciprocal_rank"].mean()),
        "median_rank": float(frame["rank"].median()),
        "mean_margin": float(frame["margin"].mean()),
    }


def permutation_controls(
    profiles: np.ndarray,
    meta: pd.DataFrame,
    supported_mask: np.ndarray,
    unresolved_mask: np.ndarray,
    seed: int,
    draws: int = 200,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    unresolved_indices = np.flatnonzero(unresolved_mask)
    n_supported = int(supported_mask.sum())
    if n_supported == 0 or len(unresolved_indices) < n_supported:
        raise RuntimeError("insufficient supported/unresolved features for count-matched controls")
    subset_metrics: list[tuple[float, float]] = []
    permuted_metrics: list[tuple[float, float]] = []
    unresolved_profiles = profiles[:, unresolved_mask]
    for _ in range(draws):
        subset = rng.choice(unresolved_indices, n_supported, replace=False)
        subset_result = retrieval(profiles, meta, np.isin(np.arange(profiles.shape[1]), subset))
        subset_metrics.append(
            (float(subset_result["recall1"].mean()), float(subset_result["reciprocal_rank"].mean()))
        )

        shuffled = np.empty_like(unresolved_profiles)
        for row_idx in range(len(shuffled)):
            shuffled[row_idx] = unresolved_profiles[row_idx, rng.permutation(unresolved_profiles.shape[1])]
        shuffled_result = retrieval(
            shuffled, meta, np.ones(shuffled.shape[1], dtype=bool),
        )
        permuted_metrics.append(
            (float(shuffled_result["recall1"].mean()), float(shuffled_result["reciprocal_rank"].mean()))
        )

    def distribution(values: np.ndarray) -> dict[str, float | int]:
        return {
            "draws": int(len(values)),
            "mean": float(np.mean(values)),
            "q025": float(np.quantile(values, 0.025)),
            "median": float(np.median(values)),
            "q975": float(np.quantile(values, 0.975)),
        }

    subset_array = np.asarray(subset_metrics)
    permuted_array = np.asarray(permuted_metrics)
    observed = retrieval(profiles, meta, unresolved_mask)
    observed_recall1 = float(observed["recall1"].mean())
    observed_mrr = float(observed["reciprocal_rank"].mean())
    return {
        "count_matched_unresolved": {
            "features_per_draw": n_supported,
            "recall1": distribution(subset_array[:, 0]),
            "mrr": distribution(subset_array[:, 1]),
        },
        "independent_coordinate_permutation": {
            "features_per_draw": int(unresolved_mask.sum()),
            "recall1": distribution(permuted_array[:, 0]),
            "mrr": distribution(permuted_array[:, 1]),
            "empirical_p_recall1_ge_observed": float(
                (1 + np.sum(permuted_array[:, 0] >= observed_recall1)) / (draws + 1)
            ),
            "empirical_p_mrr_ge_observed": float(
                (1 + np.sum(permuted_array[:, 1] >= observed_mrr)) / (draws + 1)
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("data/reference/chemprop2_m1_20261011"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/chemprop2_response_m1_v2_20261011"))
    parser.add_argument("--library-score", type=float, default=0.80)
    parser.add_argument("--blank-ratio", type=float, default=3.0)
    parser.add_argument("--qc-cv-max", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20261011)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite output: {args.output_dir}")
    paths = {
        "quantification": args.source_dir / "quantification.csv",
        "mgf": args.source_dir / "consensus.mgf",
        "library": args.source_dir / "library_matches.tsv",
        "metadata": args.source_dir / "metadata.tsv",
        "network": args.source_dir / "network_singletons.graphml",
    }
    for path in paths.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing source asset: {path}")

    quant = pd.read_csv(paths["quantification"])
    samples = parse_samples(list(quant.columns))
    drug_samples = samples.loc[~samples["drug"].isin(CONTROL_DRUGS)].copy()
    if drug_samples["drug"].nunique() != 12 or set(drug_samples["replicate"]) != {1, 2, 3}:
        raise RuntimeError("expected 12 drugs and three replicates")

    scans = nonempty_mgf_scans(paths["mgf"])
    mask, audit = feature_qc_mask(
        quant, samples, scans, args.blank_ratio, args.qc_cv_max,
    )
    libraries = pd.read_csv(paths["library"], sep="\t")
    libraries["#Scan#"] = pd.to_numeric(libraries["#Scan#"], errors="raise").astype(int)
    libraries["MQScore"] = pd.to_numeric(libraries["MQScore"], errors="raise")
    supported_scans = set(libraries.loc[libraries["MQScore"] >= args.library_score, "#Scan#"])
    supported = quant["row ID"].astype(int).isin(supported_scans).to_numpy() & mask
    unresolved = mask & ~supported

    values = quant[samples["column"].tolist()].to_numpy(float).T
    log_raw = robust_log2(values)
    normalizations = {
        "log2_feature_pseudocount": log_raw,
        "log2_feature_pseudocount_median_centered": median_center(log_raw),
    }
    results: dict[str, pd.DataFrame] = {}
    profile_sets: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    profile_meta: pd.DataFrame | None = None
    for normalization, log_values in normalizations.items():
        for mode in ("trajectory_auc", "endpoint"):
            profiles, meta = response_profiles(log_values, samples, mask, mode)
            profile_sets[f"{normalization}|{mode}"] = (profiles, meta)
            if profile_meta is None:
                profile_meta = meta
            for universe, universe_mask in {
                "all_ms2_qc": np.ones(mask.sum(), dtype=bool),
                "library_supported": supported[mask],
                "unresolved": unresolved[mask],
            }.items():
                key = f"{normalization}|{mode}|{universe}"
                results[key] = retrieval(profiles, meta, universe_mask)

    primary_prefix = "log2_feature_pseudocount_median_centered|trajectory_auc|"
    all_result = results[primary_prefix + "all_ms2_qc"]
    known_result = results[primary_prefix + "library_supported"]
    unknown_result = results[primary_prefix + "unresolved"]
    primary_profiles, primary_meta = profile_sets[
        "log2_feature_pseudocount_median_centered|trajectory_auc"
    ]
    controls = permutation_controls(
        primary_profiles,
        primary_meta,
        supported[mask],
        unresolved[mask],
        args.seed + 100,
    )
    deltas = {
        "all_minus_library_supported_recall1": cluster_bootstrap_delta(
            all_result, known_result, "recall1", args.seed,
        ),
        "all_minus_library_supported_mrr": cluster_bootstrap_delta(
            all_result, known_result, "reciprocal_rank", args.seed + 1,
        ),
        "unresolved_minus_library_supported_recall1": cluster_bootstrap_delta(
            unknown_result, known_result, "recall1", args.seed + 2,
        ),
        "unresolved_minus_library_supported_mrr": cluster_bootstrap_delta(
            unknown_result, known_result, "reciprocal_rank", args.seed + 3,
        ),
    }

    args.output_dir.mkdir(parents=True)
    audit.to_csv(args.output_dir / "feature_qc_audit.csv.gz", index=False, compression="gzip")
    assert profile_meta is not None
    profile_meta.to_csv(args.output_dir / "response_profile_manifest.csv", index=False)
    result_rows: list[pd.DataFrame] = []
    summaries: dict[str, dict[str, float | int]] = {}
    for key, frame in results.items():
        normalization, mode, universe = key.split("|")
        tagged = frame.copy()
        tagged.insert(0, "universe", universe)
        tagged.insert(0, "response_mode", mode)
        tagged.insert(0, "normalization", normalization)
        result_rows.append(tagged)
        summaries[key] = summarize(frame)
    pd.concat(result_rows, ignore_index=True).to_csv(args.output_dir / "retrieval_results.csv", index=False)

    report = {
        "status": "chemprop2_response_reproducibility_m1_complete",
        "formal": False,
        "evidence_role": "development_consumed",
        "leakage_status": (
            "within-source replicate retrieval; drug labels used only as retrieval truth; "
            "not cross-study, cross-source, or cross-instrument"
        ),
        "scientific_question": (
            "Does a microbiome-specific time-resolved response reproduce across held-out "
            "biological replicates, and do unresolved MS/MS entities carry repeatable drug information?"
        ),
        "estimand": (
            "drug Bac-minus-noBac time response, baseline-corrected at T0 and differenced "
            "against the matched DMSO Bac-minus-noBac trajectory"
        ),
        "dataset": {
            "accession": "MSV000094899",
            "gnps2_task": "1b5b94b4191d4223a5f57afb2aaaf0b0",
            "features_total": int(len(quant)),
            "parsed_study_injections": int(len(samples)),
            "drug_injections": int(len(drug_samples)),
            "drugs": int(drug_samples["drug"].nunique()),
            "replicates": sorted(map(int, drug_samples["replicate"].unique())),
            "time_counts": {str(k): int(v) for k, v in samples["time"].value_counts().sort_index().items()},
            "nonempty_ms2_scans": int(len(scans)),
            "library_score_threshold": args.library_score,
            "source_hashes": {name: sha256(path) for name, path in paths.items()},
        },
        "feature_universes": {
            "qc_blank_ms2_pass": int(mask.sum()),
            "library_supported": int(supported.sum()),
            "unresolved": int(unresolved.sum()),
            "library_supported_definition": (
                f"GNPS library match MQScore >= {args.library_score}; not authentic-standard confirmed"
            ),
        },
        "retrieval": summaries,
        "primary_deltas_cluster_bootstrap_by_drug": deltas,
        "primary_negative_controls": controls,
        "interpretation_gate": {
            "pass_to_cross_design_m1b": bool(
                deltas["unresolved_minus_library_supported_mrr"]["ci95_low"] > 0
                and summarize(unknown_result)["recall1"] > (1.0 / 12.0)
                and controls["independent_coordinate_permutation"]["empirical_p_mrr_ge_observed"] <= 0.05
                and controls["count_matched_unresolved"]["mrr"]["q025"]
                > summarize(known_result)["mrr"]
            ),
            "does_not_establish": [
                "unknown incremental mechanism retrieval",
                "cross-study transfer",
                "cross-instrument transfer",
                "reaction direction or exact structure",
                "superiority of a frozen unified ranking algorithm",
            ],
        },
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "feature_universes": report["feature_universes"],
        "primary": {
            "all": summarize(all_result),
            "library_supported": summarize(known_result),
            "unresolved": summarize(unknown_result),
        },
        "deltas": deltas,
        "controls": controls,
        "gate": report["interpretation_gate"],
    }, indent=2))


if __name__ == "__main__":
    main()
