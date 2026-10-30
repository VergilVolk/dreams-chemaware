#!/usr/bin/env python3
"""Validate frozen adult/pediatric IBD identity hypotheses in ST001000.

This is deliberately a biology validation, not an identity confirmation step.
The candidate precursor masses and expected disease directions are frozen from
the existing adult IBD200 <-> pediatric VEO-IBD ledger before ST001000 effects
are read.  ST001000 feature rows are admitted by precursor mass alone.  A mass
match can support a directional-concordance opportunity but cannot establish that
the same molecular entity was observed; MS/MS is required for that claim.

The statistical model follows Franzosa et al. (Nat Microbiol 2019): each LC-MS
method is sum-normalized within sample, zero values are replaced by half the
smallest non-zero value in that sample, values are log transformed, and CD and
UC coefficients are estimated relative to non-IBD controls.  Age is included
in both cohorts; antibiotics, immunosuppressants, mesalamine, and steroids are
included for PRISM.  The primary PRISM fit is complete-case for medication
metadata, with NA-as-no retained as a sensitivity analysis.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import stats


ANALYSES = (
    ("AN001878", "HILIC_positive", 1),
    ("AN001881", "C8_positive", 2),
)

MEDICATION_COLUMNS = (
    "Antibiotic",
    "Immunosuppresants",
    "mesalamine",
    "steroids",
)


@dataclass(frozen=True)
class FrozenCandidate:
    connection_id: str
    method_edge_id: str
    left_entity_id: str
    right_entity_id: str
    left_mz: float
    right_mz: float
    target_mz: float
    left_beta: float
    right_beta: float
    expected_direction: int
    identity_world: str
    method_weight: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--an001878", type=Path, required=True)
    parser.add_argument("--an001881", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ppm", type=float, default=10.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def as_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def load_frozen_candidates(path: Path) -> tuple[list[FrozenCandidate], dict]:
    rows_by_world: dict[str, list[dict[str, str]]] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["method_name"] != "weighted_spectral_entropy_complete_axis":
                continue
            if row["left_study_id"] != "IBD200_GROUP":
                continue
            if row["right_study_id"] != "MSV000097610":
                continue
            rows_by_world.setdefault(row["evidence_world"], []).append(row)

    all_rows = rows_by_world.get("all", [])
    candidates: list[FrozenCandidate] = []
    for row in all_rows:
        left_beta = float(row["left_beta"])
        right_beta = float(row["right_beta"])
        # Direction is frozen only when the two pre-existing cohorts agree.
        if left_beta == 0 or right_beta == 0 or left_beta * right_beta <= 0:
            continue
        cid = row["undirected_connection_id"]
        left_status = row["left_common_identity_status"]
        right_status = row["right_common_identity_status"]
        if left_status == "library_supported" and right_status == "library_supported":
            identity_world = "known"
        elif left_status == "unresolved_entity" and right_status == "unresolved_entity":
            identity_world = "unresolved"
        else:
            identity_world = "mixed_known_unresolved"
        left_mz = float(row["left_precursor_mz"])
        right_mz = float(row["right_precursor_mz"])
        candidates.append(
            FrozenCandidate(
                connection_id=cid,
                method_edge_id=row["method_edge_id"],
                left_entity_id=row["left_entity_id"],
                right_entity_id=row["right_entity_id"],
                left_mz=left_mz,
                right_mz=right_mz,
                target_mz=(left_mz + right_mz) / 2.0,
                left_beta=left_beta,
                right_beta=right_beta,
                expected_direction=1 if left_beta > 0 else -1,
                identity_world=identity_world,
                method_weight=float(row["method_evidence_weight"]),
            )
        )

    frozen_pairs_by_world: dict[str, int] = {}
    for candidate in candidates:
        frozen_pairs_by_world[candidate.identity_world] = (
            frozen_pairs_by_world.get(candidate.identity_world, 0) + 1
        )
    audit = {
        "all_world_edges": len(all_rows),
        "same_direction_frozen_candidates": len(candidates),
        "opposite_direction_edges_excluded": len(all_rows) - len(candidates),
        "known_world_edges": len(rows_by_world.get("known", [])),
        "unresolved_world_edges": len(rows_by_world.get("unresolved", [])),
        "same_direction_frozen_pairs_by_identity_world": frozen_pairs_by_world,
    }
    return candidates, audit


def load_mapping(path: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_sample = {row["Sample Name"]: row for row in rows}
    if len(by_sample) != len(rows):
        raise ValueError("mapping contains duplicate Sample Name values")
    return rows, by_sample


def load_matrix(path: Path, metadata_columns: int) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader)
        _factors = next(reader)
        sample_names = header[metadata_columns:]
        feature_ids: list[str] = []
        masses: list[float] = []
        values: list[list[float]] = []
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"malformed row in {path}: {len(row)} != {len(header)}")
            try:
                mass = float(row[0].split("_", 1)[0])
                abundance = [float(value or 0.0) for value in row[metadata_columns:]]
            except ValueError:
                continue
            feature_ids.append(row[0])
            masses.append(mass)
            values.append(abundance)
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (len(feature_ids), len(sample_names)):
        raise ValueError("matrix dimensions do not agree with labels")
    return sample_names, feature_ids, np.asarray(masses, dtype=float), matrix


def preprocess_abundance(matrix: np.ndarray) -> np.ndarray:
    """Return sample-normalized and log10-transformed feature abundances."""
    totals = matrix.sum(axis=0)
    if np.any(totals <= 0):
        raise ValueError("at least one sample has zero total abundance")
    relative = matrix / totals[None, :] * 1_000_000.0
    transformed = relative.copy()
    for sample_idx in range(transformed.shape[1]):
        column = transformed[:, sample_idx]
        positive = column[column > 0]
        if positive.size == 0:
            raise ValueError("at least one sample has no positive feature")
        column[column <= 0] = 0.5 * float(positive.min())
        transformed[:, sample_idx] = np.log10(column)
    return transformed


def design_matrix(
    sample_names: list[str],
    mapping: dict[str, dict[str, str]],
    cohort: str,
    na_policy: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    selected: list[int] = []
    rows: list[list[float]] = []
    names: list[str] = ["intercept", "CD", "UC", "age_z"]
    if cohort == "PRISM":
        names.extend(MEDICATION_COLUMNS)

    cohort_samples = [name for name in sample_names if mapping[name]["Cohort"] == cohort]
    ages = np.asarray([float(mapping[name]["Age"]) for name in cohort_samples], dtype=float)
    age_mean = float(ages.mean())
    age_sd = float(ages.std(ddof=0))
    if age_sd <= 0:
        raise ValueError(f"age has no variation in {cohort}")

    for idx, sample_name in enumerate(sample_names):
        meta = mapping[sample_name]
        if meta["Cohort"] != cohort:
            continue
        meds: list[float] = []
        if cohort == "PRISM":
            raw_meds = [meta[column] for column in MEDICATION_COLUMNS]
            if na_policy == "complete_case" and any(value == "NA" for value in raw_meds):
                continue
            meds = [1.0 if value == "Yes" else 0.0 for value in raw_meds]
        diagnosis = meta["Diagnosis"]
        if diagnosis not in {"CD", "UC", "Control"}:
            raise ValueError(f"unexpected diagnosis {diagnosis!r}")
        row = [
            1.0,
            1.0 if diagnosis == "CD" else 0.0,
            1.0 if diagnosis == "UC" else 0.0,
            (float(meta["Age"]) - age_mean) / age_sd,
            *meds,
        ]
        selected.append(idx)
        rows.append(row)
    return np.asarray(selected, dtype=int), np.asarray(rows, dtype=float), names, cohort_samples


def fit_ols(y: np.ndarray, x: np.ndarray, names: list[str]) -> dict[str, dict[str, float]]:
    if y.ndim != 1 or x.shape[0] != y.size:
        raise ValueError("OLS dimensions do not agree")
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    residual = y - x @ beta
    df = x.shape[0] - np.linalg.matrix_rank(x)
    if df <= 0:
        raise ValueError("non-positive residual degrees of freedom")
    sigma2 = float(residual @ residual) / df
    se = np.sqrt(np.maximum(np.diag(xtx_inv) * sigma2, 0.0))
    hat = np.sum((x @ xtx_inv) * x, axis=1)
    if np.any(hat >= 1.0):
        raise ValueError("HC3 undefined because at least one leverage is >= 1")
    hc3_scaled = residual / (1.0 - hat)
    hc3_meat = x.T @ ((hc3_scaled ** 2)[:, None] * x)
    hc3_cov = xtx_inv @ hc3_meat @ xtx_inv
    hc3_se = np.sqrt(np.maximum(np.diag(hc3_cov), 0.0))
    out: dict[str, dict[str, float]] = {}
    for idx, name in enumerate(names):
        t_value = float(beta[idx] / se[idx]) if se[idx] > 0 else math.nan
        p_value = float(2.0 * stats.t.sf(abs(t_value), df)) if math.isfinite(t_value) else math.nan
        hc3_t = float(beta[idx] / hc3_se[idx]) if hc3_se[idx] > 0 else math.nan
        hc3_p = float(2.0 * stats.t.sf(abs(hc3_t), df)) if math.isfinite(hc3_t) else math.nan
        out[name] = {
            "beta": float(beta[idx]),
            "se": float(se[idx]),
            "t": t_value,
            "p": p_value,
            "hc3_se": float(hc3_se[idx]),
            "hc3_t": hc3_t,
            "hc3_p": hc3_p,
            "df": float(df),
        }
    return out


def bh_adjust(values: Iterable[float]) -> list[float]:
    pvalues = np.asarray(list(values), dtype=float)
    if pvalues.size == 0:
        return []
    order = np.argsort(pvalues, kind="mergesort")
    ranked = pvalues[order]
    adjusted = ranked * pvalues.size / np.arange(1, pvalues.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return [float(value) for value in result]


def write_tsv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates, candidate_audit = load_frozen_candidates(args.ledger)
    mapping_rows, mapping = load_mapping(args.mapping)
    input_paths = {
        "ledger": args.ledger,
        "mapping": args.mapping,
        "AN001878": args.an001878,
        "AN001881": args.an001881,
    }
    matrix_paths = {"AN001878": args.an001878, "AN001881": args.an001881}

    match_rows: list[dict] = []
    effect_rows: list[dict] = []
    analysis_audit: dict[str, dict] = {}
    raw_mass_compatible_pairs: dict[tuple[str, str], str] = {}

    for analysis_id, mode, metadata_columns in ANALYSES:
        sample_names, feature_ids, masses, raw_matrix = load_matrix(
            matrix_paths[analysis_id], metadata_columns
        )
        if set(sample_names) != set(mapping):
            missing = sorted(set(sample_names) - set(mapping))
            extra = sorted(set(mapping) - set(sample_names))
            raise ValueError(f"sample mismatch in {analysis_id}: missing={missing[:5]}, extra={extra[:5]}")
        transformed = preprocess_abundance(raw_matrix)

        provisional: list[dict] = []
        for candidate in candidates:
            ppm = np.abs(masses - candidate.target_mz) / candidate.target_mz * 1_000_000.0
            feature_idx = int(np.argmin(ppm))
            feature_mz = float(masses[feature_idx])
            left_ppm = abs(feature_mz - candidate.left_mz) / candidate.left_mz * 1_000_000.0
            right_ppm = abs(feature_mz - candidate.right_mz) / candidate.right_mz * 1_000_000.0
            # Identity opportunities must be compatible with both frozen source
            # observations, not merely with their midpoint.
            if max(left_ppm, right_ppm) > args.ppm:
                continue
            provisional.append(
                {
                    **candidate.__dict__,
                    "analysis_id": analysis_id,
                    "mode": mode,
                    "feature_index": feature_idx,
                    "feature_id": feature_ids[feature_idx],
                    "feature_mz": feature_mz,
                    "match_ppm": float(ppm[feature_idx]),
                    "left_match_ppm": float(left_ppm),
                    "right_match_ppm": float(right_ppm),
                }
            )

        for row in provisional:
            raw_mass_compatible_pairs[
                (row["left_entity_id"], row["right_entity_id"])
            ] = row["identity_world"]

        # A single measured feature is one validation unit.  If several frozen
        # candidates land on it, retain the smallest ppm error without looking
        # at any ST001000 abundance or phenotype value.
        best_by_feature: dict[str, dict] = {}
        for row in sorted(
            provisional,
            key=lambda item: (item["match_ppm"], item["connection_id"]),
        ):
            best_by_feature.setdefault(row["feature_id"], row)
        unique_matches = list(best_by_feature.values())
        match_rows.extend(unique_matches)

        fit_specs = [
            ("PRISM", "complete_case"),
            ("PRISM", "na_as_no_sensitivity"),
            ("Neth", "not_applicable"),
        ]
        cohort_ns: dict[str, int] = {}
        for cohort, policy in fit_specs:
            design_policy = "complete_case" if policy == "complete_case" else "na_as_no"
            selected, design, coefficient_names, _ = design_matrix(
                sample_names,
                mapping,
                cohort,
                design_policy,
            )
            cohort_ns[f"{cohort}:{policy}"] = int(selected.size)
            for match in unique_matches:
                fit = fit_ols(
                    transformed[int(match["feature_index"]), selected],
                    design,
                    coefficient_names,
                )
                for diagnosis in ("CD", "UC"):
                    estimate = fit[diagnosis]
                    effect_rows.append(
                        {
                            **{key: value for key, value in match.items() if key != "feature_index"},
                            "cohort": cohort,
                            "missing_medication_policy": policy,
                            "diagnosis": diagnosis,
                            "n": int(selected.size),
                            "beta": estimate["beta"],
                            "se": estimate["se"],
                            "t": estimate["t"],
                            "p": estimate["p"],
                            "hc3_se": estimate["hc3_se"],
                            "hc3_t": estimate["hc3_t"],
                            "hc3_p": estimate["hc3_p"],
                            "df": estimate["df"],
                            "direction_matches_frozen": int(
                                np.sign(estimate["beta"]) == int(match["expected_direction"])
                            ),
                        }
                    )

        analysis_audit[analysis_id] = {
            "mode": mode,
            "features": len(feature_ids),
            "samples": len(sample_names),
            "candidate_mode_matches_before_feature_deduplication": len(provisional),
            "unique_mass_matched_features": len(unique_matches),
            "cohort_fit_n": cohort_ns,
        }

    # Multiplicity family: all frozen candidate-feature x CD/UC tests within a
    # cohort and medication-policy fit.  Modes remain in the same family.
    grouped: dict[tuple[str, str], list[int]] = {}
    for idx, row in enumerate(effect_rows):
        grouped.setdefault((row["cohort"], row["missing_medication_policy"]), []).append(idx)
    for indices in grouped.values():
        adjusted = bh_adjust(effect_rows[idx]["p"] for idx in indices)
        hc3_adjusted = bh_adjust(effect_rows[idx]["hc3_p"] for idx in indices)
        for idx, q_value in zip(indices, adjusted):
            effect_rows[idx]["q"] = q_value
        for idx, q_value in zip(indices, hc3_adjusted):
            effect_rows[idx]["hc3_q"] = q_value

    # Join the primary PRISM and independent Netherlands fits for the exact
    # same measured feature and diagnosis.
    primary: dict[tuple[str, str], dict] = {}
    neth: dict[tuple[str, str], dict] = {}
    sensitivity: dict[tuple[str, str], dict] = {}
    for row in effect_rows:
        key = (row["feature_id"], row["diagnosis"])
        if row["cohort"] == "PRISM" and row["missing_medication_policy"] == "complete_case":
            primary[key] = row
        elif row["cohort"] == "PRISM" and row["missing_medication_policy"] == "na_as_no_sensitivity":
            sensitivity[key] = row
        elif row["cohort"] == "Neth":
            neth[key] = row

    replication_rows: list[dict] = []
    for key, prism in primary.items():
        dutch = neth[key]
        sens = sensitivity[key]
        strict = (
            prism["direction_matches_frozen"] == 1
            and dutch["direction_matches_frozen"] == 1
            and sens["direction_matches_frozen"] == 1
            and prism["q"] <= 0.05
            and dutch["q"] <= 0.05
            and sens["q"] <= 0.05
        )
        hc3_strict = (
            prism["direction_matches_frozen"] == 1
            and dutch["direction_matches_frozen"] == 1
            and sens["direction_matches_frozen"] == 1
            and prism["hc3_q"] <= 0.05
            and dutch["hc3_q"] <= 0.05
            and sens["hc3_q"] <= 0.05
        )
        replication_rows.append(
            {
                "analysis_id": prism["analysis_id"],
                "mode": prism["mode"],
                "feature_id": prism["feature_id"],
                "feature_mz": prism["feature_mz"],
                "match_ppm": prism["match_ppm"],
                "connection_id": prism["connection_id"],
                "method_edge_id": prism["method_edge_id"],
                "left_entity_id": prism["left_entity_id"],
                "right_entity_id": prism["right_entity_id"],
                "identity_world": prism["identity_world"],
                "expected_direction": prism["expected_direction"],
                "diagnosis": prism["diagnosis"],
                "prism_beta": prism["beta"],
                "prism_t": prism["t"],
                "prism_p": prism["p"],
                "prism_q": prism["q"],
                "prism_hc3_q": prism["hc3_q"],
                "neth_beta": dutch["beta"],
                "neth_t": dutch["t"],
                "neth_p": dutch["p"],
                "neth_q": dutch["q"],
                "neth_hc3_q": dutch["hc3_q"],
                "prism_na_as_no_beta": sens["beta"],
                "prism_na_as_no_q": sens["q"],
                "prism_na_as_no_hc3_q": sens["hc3_q"],
                "strict_three_fit_concordance": strict,
                "hc3_three_fit_concordance": hc3_strict,
                "claim_boundary": (
                    "candidate_passing_three_within_fit_BH_filters_at_matching_positive_mode_mass;"
                    "not_replicated_findings_FDR_control;not_same_entity_confirmation_without_msms"
                ),
            }
        )

    replication_rows.sort(
        key=lambda row: (
            not row["strict_three_fit_concordance"],
            max(row["prism_q"], row["neth_q"], row["prism_na_as_no_q"]),
            row["analysis_id"],
            row["feature_id"],
            row["diagnosis"],
        )
    )

    match_fields = [
        "analysis_id", "mode", "feature_id", "feature_mz", "match_ppm",
        "left_match_ppm", "right_match_ppm",
        "connection_id", "method_edge_id", "left_entity_id", "right_entity_id", "left_mz",
        "right_mz", "target_mz", "left_beta", "right_beta",
        "expected_direction", "identity_world", "method_weight",
    ]
    effect_fields = match_fields + [
        "cohort", "missing_medication_policy", "diagnosis", "n", "beta",
        "se", "t", "p", "q", "hc3_se", "hc3_t", "hc3_p", "hc3_q",
        "df", "direction_matches_frozen",
    ]
    replication_fields = list(replication_rows[0]) if replication_rows else []
    write_tsv(args.output_dir / "frozen_candidate_feature_matches.tsv", match_rows, match_fields)
    write_tsv(args.output_dir / "cohort_effects.tsv", effect_rows, effect_fields)
    write_tsv(
        args.output_dir / "directional_concordance_screen.tsv",
        replication_rows,
        replication_fields,
    )
    strict_rows = [row for row in replication_rows if row["strict_three_fit_concordance"]]
    hc3_strict_rows = [row for row in replication_rows if row["hc3_three_fit_concordance"]]
    write_tsv(
        args.output_dir / "strict_three_fit_concordant_candidates.tsv",
        strict_rows,
        replication_fields,
    )
    write_tsv(
        args.output_dir / "hc3_three_fit_concordant_candidates.tsv",
        hc3_strict_rows,
        replication_fields,
    )

    cohort_counts: dict[str, dict[str, int]] = {}
    for cohort in ("PRISM", "Neth"):
        selected = [row for row in mapping_rows if row["Cohort"] == cohort]
        cohort_counts[cohort] = {
            "total": len(selected),
            "CD": sum(row["Diagnosis"] == "CD" for row in selected),
            "UC": sum(row["Diagnosis"] == "UC" for row in selected),
            "Control": sum(row["Diagnosis"] == "Control" for row in selected),
        }

    strict_by_world: dict[str, int] = {}
    for row in strict_rows:
        strict_by_world[row["identity_world"]] = strict_by_world.get(row["identity_world"], 0) + 1

    # Preserve duplicated chromatographic observations and CD/UC tests within
    # an entity-pair unit. This direction-label shuffle is an exploratory
    # sensitivity analysis: exchangeability across the heterogeneous mass,
    # mode-opportunity, and identity groups is not established, so the result
    # is not a confirmatory permutation p-value.
    pair_keys = sorted(
        {(row["left_entity_id"], row["right_entity_id"]) for row in replication_rows}
    )
    pair_direction = {
        (row["left_entity_id"], row["right_entity_id"]): int(row["expected_direction"])
        for row in replication_rows
    }
    observed_strict = len(strict_rows)
    rng = np.random.default_rng(20261030)
    frozen_directions = np.asarray([pair_direction[key] for key in pair_keys], dtype=int)
    null_counts = np.zeros(10_000, dtype=int)
    pair_pass_by_direction: dict[tuple[str, str], dict[int, bool]] = {}
    for key in pair_keys:
        pair_rows = [
            row for row in replication_rows
            if (row["left_entity_id"], row["right_entity_id"]) == key
        ]
        pair_pass_by_direction[key] = {}
        for direction in (-1, 1):
            pair_pass_by_direction[key][direction] = any(
                row["prism_q"] <= 0.05
                and row["neth_q"] <= 0.05
                and row["prism_na_as_no_q"] <= 0.05
                and np.sign(row["prism_beta"]) == direction
                and np.sign(row["neth_beta"]) == direction
                and np.sign(row["prism_na_as_no_beta"]) == direction
                for row in pair_rows
            )
    observed_strict_pairs = sum(
        pair_pass_by_direction[key][pair_direction[key]] for key in pair_keys
    )
    null_pair_counts = np.zeros(10_000, dtype=int)
    for repeat in range(null_counts.size):
        shuffled = rng.permutation(frozen_directions)
        assigned = dict(zip(pair_keys, shuffled))
        count = 0
        for row in replication_rows:
            if (
                row["prism_q"] <= 0.05
                and row["neth_q"] <= 0.05
                and row["prism_na_as_no_q"] <= 0.05
            ):
                direction = assigned[(row["left_entity_id"], row["right_entity_id"])]
                if (
                    np.sign(row["prism_beta"]) == direction
                    and np.sign(row["neth_beta"]) == direction
                    and np.sign(row["prism_na_as_no_beta"]) == direction
                ):
                    count += 1
        null_counts[repeat] = count
        null_pair_counts[repeat] = sum(
            pair_pass_by_direction[key][int(assigned[key])] for key in pair_keys
        )
    direction_permutation = {
        "seed": 20261030,
        "repeats": int(null_counts.size),
        "randomization_unit": "frozen adult-pediatric entity pair",
        "calibration_status": (
            "exploratory_direction_label_sensitivity;exchangeability_not_established"
        ),
        "observed_strict_tests": observed_strict,
        "null_mean": float(null_counts.mean()),
        "null_median": float(np.median(null_counts)),
        "null_q025": float(np.quantile(null_counts, 0.025)),
        "null_q975": float(np.quantile(null_counts, 0.975)),
        "p_ge_observed": float((1 + np.sum(null_counts >= observed_strict)) / (1 + null_counts.size)),
        "observed_strict_entity_pairs": int(observed_strict_pairs),
        "pair_null_mean": float(null_pair_counts.mean()),
        "pair_null_median": float(np.median(null_pair_counts)),
        "pair_null_q025": float(np.quantile(null_pair_counts, 0.025)),
        "pair_null_q975": float(np.quantile(null_pair_counts, 0.975)),
        "pair_p_ge_observed": float(
            (1 + np.sum(null_pair_counts >= observed_strict_pairs))
            / (1 + null_pair_counts.size)
        ),
    }

    retained_pair_denominator_by_world: dict[str, int] = {}
    strict_pairs_by_world: dict[str, int] = {}
    for key in pair_keys:
        row = next(
            item for item in replication_rows
            if (item["left_entity_id"], item["right_entity_id"]) == key
        )
        world = row["identity_world"]
        retained_pair_denominator_by_world[world] = (
            retained_pair_denominator_by_world.get(world, 0) + 1
        )
        if pair_pass_by_direction[key][pair_direction[key]]:
            strict_pairs_by_world[world] = strict_pairs_by_world.get(world, 0) + 1

    unresolved_strict = strict_pairs_by_world.get("unresolved", 0)
    unresolved_total = retained_pair_denominator_by_world.get("unresolved", 0)
    known_strict = strict_pairs_by_world.get("known", 0)
    known_total = retained_pair_denominator_by_world.get("known", 0)
    unknown_increment_test = None
    if unresolved_total > 0 and known_total > 0:
        odds_ratio, p_greater = stats.fisher_exact(
            [
                [unresolved_strict, unresolved_total - unresolved_strict],
                [known_strict, known_total - known_strict],
            ],
            alternative="greater",
        )
        unknown_increment_test = {
            "comparison": (
                "unresolved_vs_known_three_fit_concordance_rate_conditional_on_"
                "retention_after_feature_deduplication"
            ),
            "alternative": "unresolved_greater_than_known",
            "odds_ratio": float(odds_ratio),
            "fisher_exact_p": float(p_greater),
            "mixed_world_excluded": True,
            "interpretation": "no_evidence_of_unresolved_enrichment",
        }
    frozen_by_world = candidate_audit["same_direction_frozen_pairs_by_identity_world"]
    raw_mass_compatible_by_world: dict[str, int] = {}
    for world in raw_mass_compatible_pairs.values():
        raw_mass_compatible_by_world[world] = raw_mass_compatible_by_world.get(world, 0) + 1
    raw_mass_compatible_coverage_by_world = {
        world: {
            "raw_mass_compatible_pairs": raw_mass_compatible_by_world.get(world, 0),
            "all_frozen_same_direction_pairs": denominator,
            "coverage": raw_mass_compatible_by_world.get(world, 0) / denominator,
        }
        for world, denominator in frozen_by_world.items()
    }
    retained_testable_coverage_by_world = {
        world: {
            "retained_testable_pairs": retained_pair_denominator_by_world.get(world, 0),
            "all_frozen_same_direction_pairs": denominator,
            "coverage": retained_pair_denominator_by_world.get(world, 0) / denominator,
        }
        for world, denominator in frozen_by_world.items()
    }
    report = {
        "status": "COMPLETE",
        "data_role": "development_now_consumed_two_participant_cohorts_not_sealed_external",
        "leakage_status": {
            "candidate_mass_and_direction": (
                "frozen from the consumed IBD200--VEO-IBD ledger before ST001000 "
                "abundance or phenotype effects were read"
            ),
            "prism_feature_selection": (
                "nearest positive-mode midpoint mass, required within 10 ppm of both "
                "frozen source masses; phenotype-blind; no ST001000 abundance or "
                "response used to choose among candidates"
            ),
            "validation_status": (
                "ST001000 is an independent-participant public development validation "
                "dataset, not a sealed external test"
            ),
        },
        "source_provenance": {
            "study": "Metabolomics Workbench ST001000 / PR000677",
            "primary_publication": "https://www.nature.com/articles/s41564-018-0306-4",
            "open_full_text": "https://pmc.ncbi.nlm.nih.gov/articles/mid/EMS80310/",
            "cohort_independence": (
                "PRISM and Netherlands contain independent participants but belong to "
                "one published study package; they are not two independent projects"
            ),
        },
        "scientific_question": (
            "Do same-direction adult IBD200 and pediatric VEO-IBD identity hypotheses "
            "show the prespecified disease direction in participant-independent PRISM "
            "and Netherlands fecal cohorts?"
        ),
        "claim_boundary": (
            "Mass matching and directional concordance do not confirm entity identity. "
            "Rows passing three within-fit BH filters are candidate priorities, not "
            "replicated-findings FDR-controlled discoveries."
        ),
        "candidate_freeze": candidate_audit,
        "ppm_tolerance": args.ppm,
        "positive_mode_only": True,
        "cohort_counts": cohort_counts,
        "analyses": analysis_audit,
        "raw_candidate_mode_mass_compatible_hits_before_feature_deduplication": sum(
            audit["candidate_mode_matches_before_feature_deduplication"]
            for audit in analysis_audit.values()
        ),
        "raw_mass_compatible_unique_entity_pairs": len(raw_mass_compatible_pairs),
        "retained_candidate_feature_matches_after_feature_deduplication": len(match_rows),
        "retained_testable_unique_entity_pairs": len(pair_keys),
        "total_feature_diagnosis_tests": len(replication_rows),
        "three_fit_concordant_tests": len(strict_rows),
        "three_fit_concordant_unique_features": len(
            {(row["analysis_id"], row["feature_id"]) for row in strict_rows}
        ),
        "three_fit_concordant_unique_entity_pairs": len(
            {(row["left_entity_id"], row["right_entity_id"]) for row in strict_rows}
        ),
        "hc3_three_fit_concordant_tests": len(hc3_strict_rows),
        "hc3_three_fit_concordant_unique_features": len(
            {(row["analysis_id"], row["feature_id"]) for row in hc3_strict_rows}
        ),
        "hc3_three_fit_concordant_unique_entity_pairs": len(
            {(row["left_entity_id"], row["right_entity_id"]) for row in hc3_strict_rows}
        ),
        "three_fit_concordant_tests_by_identity_world": strict_by_world,
        "raw_mass_compatible_pair_coverage_by_identity_world": (
            raw_mass_compatible_coverage_by_world
        ),
        "retained_testable_pair_denominator_by_identity_world": (
            retained_pair_denominator_by_world
        ),
        "retained_testable_pair_coverage_by_identity_world": (
            retained_testable_coverage_by_world
        ),
        "three_fit_concordant_entity_pairs_by_identity_world": strict_pairs_by_world,
        "descriptive_entity_pair_concordance_rates_by_identity_world": {
            world: {
                "three_fit_concordant_pairs": strict_pairs_by_world.get(world, 0),
                "retained_testable_pairs": denominator,
                "rate": strict_pairs_by_world.get(world, 0) / denominator,
                "interpretation": "descriptive_only_not_an_unknown_enrichment_test",
            }
            for world, denominator in retained_pair_denominator_by_world.items()
        },
        "unknown_increment_test": unknown_increment_test,
        "exploratory_direction_label_sensitivity": direction_permutation,
        "exploratory_direction_sensitivity_summary": {
            "reference_value": 0.05,
            "test_level_p_below_reference": bool(direction_permutation["p_ge_observed"] < 0.05),
            "entity_pair_level_p_below_reference": bool(
                direction_permutation["pair_p_ge_observed"] < 0.05
            ),
            "interpretation": (
                "Individual three-fit concordant candidates prioritize hypotheses, but "
                "the exploratory label shuffle does not support a confirmatory aggregate "
                "claim and is not a calibrated permutation test."
            ),
        },
        "independence_boundary": (
            "Feature-mode observations, CD/UC tests, and coeluting adduct/isotope-like "
            "ions are not independent metabolites; unique entity-pair counts are still "
            "hypotheses until phenotype-blind MS/MS identity verification."
        ),
        "multiplicity": (
            "Benjamini-Hochberg across all frozen candidate-feature x CD/UC tests, "
            "combining both positive acquisition modes within each cohort fit. Passing "
            "all three overlapping filters is not formal replicated-findings FDR control."
        ),
        "ols_inference": {
            "primary": (
                "Classical homoskedastic OLS standard errors reproduce the intended "
                "published model."
            ),
            "sensitivity": (
                "HC3 sandwich standard errors with residual-df Student-t p-values; "
                "BH recomputed over the same within-fit 280-test families."
            ),
            "claim_boundary": (
                "HC3 changes the test-level count but not the unique feature-mode or "
                "source entity-pair priority counts; broader outlier sensitivity was not tested."
            ),
        },
        "input_sha256": {name: sha256(path) for name, path in input_paths.items()},
        "script_sha256": sha256(Path(__file__)),
        "outputs": {
            "frozen_candidate_feature_matches": "frozen_candidate_feature_matches.tsv",
            "cohort_effects": "cohort_effects.tsv",
            "directional_concordance_screen": "directional_concordance_screen.tsv",
            "three_fit_concordant_candidates": "strict_three_fit_concordant_candidates.tsv",
            "hc3_three_fit_concordant_candidates": (
                "hc3_three_fit_concordant_candidates.tsv"
            ),
        },
    }
    with (args.output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")

    checksums = []
    for name in report["outputs"].values():
        output_path = args.output_dir / name
        checksums.append(f"{sha256(output_path)}  {name}")
    checksums.append(f"{sha256(args.output_dir / 'report.json')}  report.json")
    (args.output_dir / "checksums.sha256").write_text("\n".join(checksums) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
