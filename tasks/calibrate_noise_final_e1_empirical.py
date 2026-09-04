#!/usr/bin/env python
"""E1: calibrate realistic MS/MS noise from repeated spectra of one molecule.

This stage uses only P3-disjoint real training spectra.  Identity is used to
form replicate groups, never as a peak/action outcome label.  The output is a
frozen empirical recipe for E2; it does not train or evaluate an embedding.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/models/MassSpecGym_MurckoHist_split.hdf5"))
    parser.add_argument("--p3-dir", type=Path, default=Path("data/validation/g8r_p3_test"))
    parser.add_argument(
        "--e0-dir", type=Path, default=Path("data/validation/g8r_noise_final_e0_unified_matrix")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/validation/g8r_noise_final_e1_empirical_calibration")
    )
    parser.add_argument("--minimum-spectra", type=int, default=3)
    parser.add_argument("--maximum-spectra", type=int, default=12)
    parser.add_argument("--fragment-tolerance", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--allow-partial-e0-for-smoke", action="store_true")
    parser.add_argument("--smoke-use-all-real-train", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode() if isinstance(value, (bytes, bytearray)) else str(value) for value in values],
        dtype=object,
    )


def stable_hash(*values: Any) -> int:
    body = "|".join(map(str, values)).encode()
    return int.from_bytes(hashlib.blake2b(body, digest_size=8).digest(), "little")


def valid_peaks(spectrum: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mz = np.asarray(spectrum[0], dtype=np.float64)
    intensity = np.asarray(spectrum[1], dtype=np.float64)
    keep = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
    mz = mz[keep]
    intensity = intensity[keep]
    if intensity.size and intensity.max() > 0:
        intensity = intensity / intensity.max()
    order = np.argsort(mz, kind="mergesort")
    return mz[order], intensity[order]


def ce_bin(value: float) -> str:
    return "unknown" if not np.isfinite(value) else f"{int(round(value / 10.0) * 10)}"


def clean_instrument(value: str) -> str:
    return "unknown" if value.lower() in {"", "nan", "none"} else value


def condition_relation(inst_a: str, ce_a: float, inst_b: str, ce_b: float) -> str:
    inst_a, inst_b = clean_instrument(inst_a), clean_instrument(inst_b)
    if inst_a == "unknown" or inst_b == "unknown":
        return "unknown_instrument"
    if inst_a != inst_b:
        return "cross_instrument"
    if not np.isfinite(ce_a) or not np.isfinite(ce_b):
        return "same_instrument_unknown_ce"
    if abs(ce_a - ce_b) < 5.0:
        return "same_instrument_same_ce"
    return "same_instrument_cross_ce"


def mz_band(value: float) -> str:
    if value < 100:
        return "000-100"
    if value < 250:
        return "100-250"
    if value < 500:
        return "250-500"
    return "500+"


def intensity_band(value: float) -> str:
    if value <= 0.02:
        return "low<=0.02"
    if value <= 0.10:
        return "mid<=0.10"
    return "high>0.10"


def choose_rows(rows: list[int], conditions: dict[int, str], maximum: int, seed: int) -> list[int]:
    if len(rows) <= maximum:
        return sorted(rows)
    buckets: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        buckets[conditions[row]].append(row)
    for name in buckets:
        buckets[name].sort(key=lambda row: stable_hash(seed, row))
    selected: list[int] = []
    names = sorted(buckets)
    while len(selected) < maximum:
        progress = False
        for name in names:
            if buckets[name] and len(selected) < maximum:
                selected.append(buckets[name].pop(0))
                progress = True
        if not progress:
            break
    return sorted(selected)


def cluster_group_peaks(
    spectra: dict[int, tuple[np.ndarray, np.ndarray]], tolerance: float
) -> tuple[list[dict[int, tuple[float, float]]], int]:
    """Complete-linkage-like 1D clusters with at most one peak per spectrum.

    A cluster diameter never exceeds ``tolerance``.  If one spectrum contributes
    more than one peak to a cluster, only its more intense peak is retained and
    the duplicate count is reported for auditing.
    """
    records: list[tuple[float, float, int]] = []
    for row, (mz, intensity) in spectra.items():
        records.extend((float(m), float(i), int(row)) for m, i in zip(mz, intensity))
    records.sort(key=lambda value: (value[0], value[2], -value[1]))
    raw_clusters: list[list[tuple[float, float, int]]] = []
    current: list[tuple[float, float, int]] = []
    minimum = np.nan
    for record in records:
        if not current or record[0] - minimum <= tolerance:
            if not current:
                minimum = record[0]
            current.append(record)
        else:
            raw_clusters.append(current)
            current = [record]
            minimum = record[0]
    if current:
        raw_clusters.append(current)

    duplicate_peaks = 0
    clusters: list[dict[int, tuple[float, float]]] = []
    for cluster in raw_clusters:
        by_row: dict[int, tuple[float, float]] = {}
        for mz, intensity, row in cluster:
            previous = by_row.get(row)
            if previous is None or intensity > previous[1]:
                if previous is not None:
                    duplicate_peaks += 1
                by_row[row] = (mz, intensity)
            else:
                duplicate_peaks += 1
        clusters.append(by_row)
    return clusters, duplicate_peaks


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: list[float]) -> list[float]:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not keep.any():
        return [float("nan")] * len(quantiles)
    values, weights = values[keep], weights[keep]
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return [float(np.interp(q, cumulative, values)) for q in quantiles]


def summarize_array(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return {
            "n": 0, "q10": math.nan, "q25": math.nan, "q50": math.nan,
            "q75": math.nan, "q90": math.nan,
        }
    q10, q25, q50, q75, q90 = np.quantile(array, [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "n": int(array.size), "q10": float(q10), "q25": float(q25),
        "q50": float(q50), "q75": float(q75), "q90": float(q90),
    }


def relation_quantiles(frame: pd.DataFrame, value: str) -> dict[str, dict[str, float | int]]:
    return {
        str(relation): summarize_array(part[value].tolist())
        for relation, part in frame.groupby("relation", observed=True)
    }


def identity_equal_relation_quantiles(
    frame: pd.DataFrame, value: str
) -> dict[str, dict[str, float | int]]:
    """Quantiles in which each identity contributes unit mass per relation."""
    output: dict[str, dict[str, float | int]] = {}
    for relation, part in frame.groupby("relation", observed=True):
        counts = part.groupby("ik14", observed=True)[value].transform("size").to_numpy(dtype=float)
        values = part[value].to_numpy(dtype=float)
        weights = 1.0 / np.maximum(counts, 1.0)
        q10, q25, q50, q75, q90 = weighted_quantile(
            values, weights, [0.10, 0.25, 0.50, 0.75, 0.90]
        )
        output[str(relation)] = {
            "n": int(len(part)),
            "identities": int(part["ik14"].nunique()),
            "formulas": int(part["formula"].nunique()),
            "q10": q10, "q25": q25, "q50": q50, "q75": q75, "q90": q90,
        }
    return output


def e2_dropout_screening_grid(
    summaries: dict[str, dict[str, float | int]], safety_ceiling: float = 0.30
) -> dict[str, dict[str, Any]]:
    """Create conservative E2 doses; these are screens, not estimated optima."""
    output: dict[str, dict[str, Any]] = {}
    for relation, summary in summaries.items():
        eligible = int(summary["n"]) >= 100 and int(summary["identities"]) >= 25
        doses = sorted(
            {
                round(min(float(summary[name]), safety_ceiling), 6)
                for name in ("q10", "q25")
                if np.isfinite(float(summary[name])) and float(summary[name]) > 0
            }
        )
        output[relation] = {
            "eligible_for_e2": bool(eligible and doses),
            "doses": doses if eligible else [],
            "support_pairs": int(summary["n"]),
            "support_identities": int(summary["identities"]),
            "safety_ceiling": safety_ceiling,
        }
    return output


def plot_pairwise_reliability(pairwise: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    sample = pairwise if len(pairwise) <= 50_000 else pairwise.sample(50_000, random_state=20260826)
    axes[0].scatter(
        sample["peak_jaccard"], sample["source_dropout_fraction"],
        c=np.where(sample["reliable_for_dose"], "#3b82f6", "#b8b8b8"),
        s=8, alpha=0.25, linewidths=0,
    )
    axes[0].axvline(0.10, color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("Replicate peak Jaccard")
    axes[0].set_ylabel("Directed missing-peak fraction")
    axes[0].set_title("Reliable pairs retained for dose estimation")
    relation_order = sorted(pairwise["relation"].unique())
    data = [
        pairwise.loc[
            (pairwise["relation"] == relation) & pairwise["reliable_for_dose"],
            "source_dropout_fraction",
        ].to_numpy(dtype=float)
        for relation in relation_order
    ]
    nonempty = [(name, values) for name, values in zip(relation_order, data) if len(values)]
    if nonempty:
        axes[1].boxplot([values for _, values in nonempty], showfliers=False)
        axes[1].set_xticks(
            np.arange(1, len(nonempty) + 1), [name for name, _ in nonempty],
            rotation=28, ha="right",
        )
    axes[1].axhline(0.30, color="#b91c1c", linestyle="--", linewidth=1, label="E2 safety ceiling")
    axes[1].set_ylabel("Directed missing-peak fraction")
    axes[1].set_title("Identity-replicate variation by condition")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_prevalence(consensus: pd.DataFrame, output: Path) -> None:
    table = pd.crosstab(
        pd.cut(consensus["prevalence"], bins=[0, 0.25, 0.5, 0.75, 0.999999, 1.0], include_lowest=True),
        consensus["intensity_band"],
    )
    order = [name for name in ["low<=0.02", "mid<=0.10", "high>0.10"] if name in table.columns]
    table = table.reindex(columns=order)
    values = np.log10(table.to_numpy(dtype=float) + 1.0)
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    image = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(table.columns)), table.columns)
    ax.set_yticks(np.arange(len(table.index)), [str(value) for value in table.index])
    ax.set_xlabel("Consensus peak intensity")
    ax.set_ylabel("Within-identity prevalence")
    ax.set_title("E1 empirical peak presence landscape")
    fig.colorbar(image, ax=ax, label="log10(cluster count + 1)")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_jitter(strata: pd.DataFrame, output: Path) -> None:
    relation_order = [
        "same_instrument_same_ce",
        "same_instrument_cross_ce",
        "cross_instrument",
        "same_instrument_unknown_ce",
        "unknown_instrument",
    ]
    summary = strata.groupby("relation", observed=True).agg(
        intensity_q50=("abs_log_intensity_ratio_q50", "median"),
        intensity_q75=("abs_log_intensity_ratio_q75", "median"),
        mz_q50=("abs_mz_residual_ppm_q50", "median"),
        mz_q75=("abs_mz_residual_ppm_q75", "median"),
    ).reindex(relation_order).dropna(how="all")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    x = np.arange(len(summary))
    axes[0].plot(x, summary["intensity_q50"], marker="o", label="q50")
    axes[0].plot(x, summary["intensity_q75"], marker="s", label="q75")
    axes[0].set_ylabel("|log intensity ratio|")
    axes[0].set_title("Empirical intensity jitter")
    axes[1].plot(x, summary["mz_q50"], marker="o", label="q50")
    axes[1].plot(x, summary["mz_q75"], marker="s", label="q75")
    axes[1].set_ylabel("Absolute m/z residual (ppm)")
    axes[1].set_title("Empirical mass jitter")
    for ax in axes:
        ax.set_xticks(x, summary.index, rotation=28, ha="right")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.data, args.p3_dir, args.e0_dir, args.output_dir = map(
        resolve, (args.data, args.p3_dir, args.e0_dir, args.output_dir)
    )
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite E1 output: {args.output_dir}")
    allow_path = args.p3_dir / "p3_p2_allowed_training_ik14.json"
    e0_manifest_path = args.e0_dir / "e0_manifest.json"
    for path in (args.data, e0_manifest_path):
        if not path.exists():
            raise FileNotFoundError(path)
    smoke_without_allow = args.smoke_use_all_real_train and args.max_groups > 0
    if not allow_path.exists() and not smoke_without_allow:
        raise FileNotFoundError(allow_path)
    e0 = json.loads(e0_manifest_path.read_text(encoding="utf-8"))
    if not e0.get("formal") or not e0.get("historical_sources_complete"):
        if not (args.allow_partial_e0_for_smoke and args.max_groups > 0):
            raise RuntimeError("E1 requires a complete formal E0")
    allow = json.loads(allow_path.read_text(encoding="utf-8")) if allow_path.exists() else None
    allowed_rows = (
        np.sort(np.asarray(allow["real_train_primary"]["rows"], dtype=np.int64)) if allow is not None else None
    )
    allowed_ik14 = set(map(str, allow["real_train_primary"]["ik14"])) if allow is not None else None

    temporary = Path(tempfile.mkdtemp(prefix="noise_e1_", dir=args.output_dir.parent))
    try:
        group_rows: list[dict[str, Any]] = []
        consensus_rows: list[dict[str, Any]] = []
        pairwise_rows: list[dict[str, Any]] = []
        residuals: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
            lambda: {"abs_log_ratio": [], "abs_ppm": [], "signed_log_ratio": [], "signed_ppm": []}
        )
        stratum_identities: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        stratum_formulas: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        addition_values: list[float] = []
        addition_weights: list[float] = []
        duplicate_peak_count = 0
        total_matched_pairs = 0

        with h5py.File(args.data, "r") as handle:
            ik = decode(handle["INCHIKEY"][:])
            ik14 = np.asarray([value[:14] for value in ik], dtype=object)
            adduct = decode(handle["adduct"][:])
            fold = decode(handle["fold"][:])
            simulation = decode(handle["SIMULATION_CHALLENGE"][:])
            instrument = decode(handle["INSTRUMENT_TYPE"][:])
            ce = np.asarray(handle["COLLISION_ENERGY"][:], dtype=float)
            formula = decode(handle["FORMULA"][:])
            if allowed_rows is None:
                allowed_rows = np.flatnonzero((fold == "train") & (simulation == "False")).astype(np.int64)
                allowed_ik14 = set(map(str, ik14[allowed_rows]))
            if np.any(fold[allowed_rows] != "train") or np.any(simulation[allowed_rows] != "False"):
                raise RuntimeError("E1 allow-list contains non-real or non-train spectra")
            if not set(map(str, ik14[allowed_rows])).issubset(allowed_ik14):
                raise RuntimeError("P3 row and identity allow-lists disagree")

            conditions = {
                int(row): f"{clean_instrument(str(instrument[row]))}|ce-{ce_bin(float(ce[row]))}"
                for row in allowed_rows
            }
            groups: dict[tuple[str, str], list[int]] = defaultdict(list)
            for row in allowed_rows:
                groups[(str(ik14[row]), str(adduct[row]))].append(int(row))
            eligible = [(key, rows) for key, rows in groups.items() if len(rows) >= args.minimum_spectra]
            eligible.sort(key=lambda item: item[0])
            if args.max_groups:
                eligible = eligible[: args.max_groups]

            for group_position, ((identity, ion), rows) in enumerate(eligible):
                selected = choose_rows(rows, conditions, args.maximum_spectra, args.seed)
                spectra = {int(row): valid_peaks(np.asarray(handle["spectrum"][int(row)], dtype=float)) for row in selected}
                clusters, duplicates = cluster_group_peaks(spectra, args.fragment_tolerance)
                duplicate_peak_count += duplicates
                group_conditions = {conditions[row] for row in selected}
                group_instruments = {clean_instrument(str(instrument[row])) for row in selected}
                group_formula = str(formula[selected[0]])
                cross_condition = len(group_conditions) >= 2
                group_rows.append(
                    {
                        "ik14": identity,
                        "formula": group_formula,
                        "adduct": ion,
                        "available_spectra": len(rows),
                        "selected_spectra": len(selected),
                        "conditions": len(group_conditions),
                        "instruments": len(group_instruments),
                        "cross_condition": cross_condition,
                        "consensus_clusters": len(clusters),
                    }
                )

                for cluster_index, occurrence in enumerate(clusters):
                    present_rows = sorted(occurrence)
                    mz_values = np.asarray([occurrence[row][0] for row in present_rows], dtype=float)
                    intensities = np.asarray([occurrence[row][1] for row in present_rows], dtype=float)
                    prevalence = len(present_rows) / len(selected)
                    present_conditions = {conditions[row] for row in present_rows}
                    condition_prevalence = len(present_conditions) / max(len(group_conditions), 1)
                    representative_mz = float(np.median(mz_values))
                    representative_intensity = float(np.median(intensities))
                    consensus_rows.append(
                        {
                            "ik14": identity,
                            "formula": group_formula,
                            "adduct": ion,
                            "cluster_index": cluster_index,
                            "mz": representative_mz,
                            "median_intensity": representative_intensity,
                            "intensity_band": intensity_band(representative_intensity),
                            "prevalence": prevalence,
                            "dropout_probability": 1.0 - prevalence,
                            "condition_prevalence": condition_prevalence,
                            "present_spectra": len(present_rows),
                            "group_spectra": len(selected),
                            "cross_condition_group": cross_condition,
                        }
                    )

                    missing_count = len(selected) - len(present_rows)
                    if missing_count > 0 and representative_intensity <= 0.20:
                        addition_values.append(representative_intensity)
                        addition_weights.append(float(missing_count))

                    for row_a, row_b in itertools.combinations(present_rows, 2):
                        mz_a, intensity_a = occurrence[row_a]
                        mz_b, intensity_b = occurrence[row_b]
                        relation = condition_relation(
                            str(instrument[row_a]), float(ce[row_a]), str(instrument[row_b]), float(ce[row_b])
                        )
                        mean_mz = 0.5 * (mz_a + mz_b)
                        mean_intensity = math.sqrt(max(intensity_a * intensity_b, 0.0))
                        key = (relation, mz_band(mean_mz), intensity_band(mean_intensity))
                        log_ratio = math.log((intensity_b + 1e-6) / (intensity_a + 1e-6))
                        ppm_residual = (mz_b - mz_a) / max(mean_mz, 1e-9) * 1e6
                        residuals[key]["abs_log_ratio"].append(abs(log_ratio))
                        residuals[key]["signed_log_ratio"].append(log_ratio)
                        residuals[key]["abs_ppm"].append(abs(ppm_residual))
                        residuals[key]["signed_ppm"].append(ppm_residual)
                        stratum_identities[key].add(identity)
                        stratum_formulas[key].add(group_formula)
                        total_matched_pairs += 1

                # The pair-level missing fraction controls the total dropout dose.
                # Peak prevalence controls only which source peaks are sampled.
                cluster_presence = [set(occurrence) for occurrence in clusters]
                for row_a, row_b in itertools.combinations(selected, 2):
                    count_a = sum(row_a in presence for presence in cluster_presence)
                    count_b = sum(row_b in presence for presence in cluster_presence)
                    matched = sum(
                        row_a in presence and row_b in presence for presence in cluster_presence
                    )
                    union = count_a + count_b - matched
                    relation = condition_relation(
                        str(instrument[row_a]), float(ce[row_a]),
                        str(instrument[row_b]), float(ce[row_b]),
                    )
                    common = {
                        "ik14": identity,
                        "formula": group_formula,
                        "adduct": ion,
                        "relation": relation,
                        "matched_clusters": matched,
                        "union_clusters": union,
                        "peak_jaccard": matched / max(union, 1),
                    }
                    for source_row, target_row, source_count, target_count in (
                        (row_a, row_b, count_a, count_b),
                        (row_b, row_a, count_b, count_a),
                    ):
                        pairwise_rows.append(
                            {
                                **common,
                                "source_row": source_row,
                                "target_row": target_row,
                                "source_peak_clusters": source_count,
                                "target_peak_clusters": target_count,
                                "missing_from_target": source_count - matched,
                                "source_dropout_fraction": (source_count - matched) / max(source_count, 1),
                            }
                        )

                if (group_position + 1) % 250 == 0 or group_position + 1 == len(eligible):
                    print(f"[E1] {group_position + 1:,}/{len(eligible):,} identity-adduct groups", flush=True)

        groups_frame = pd.DataFrame(group_rows)
        consensus_frame = pd.DataFrame(consensus_rows)
        pairwise_frame = pd.DataFrame(pairwise_rows)
        if groups_frame.empty or consensus_frame.empty or pairwise_frame.empty:
            raise RuntimeError("E1 produced no empirical calibration groups")
        groups_frame.to_csv(temporary / "replicate_group_summary.csv", index=False)
        consensus_frame.to_csv(temporary / "consensus_peak_calibration.csv.gz", index=False, compression="gzip")
        pairwise_frame["reliable_for_dose"] = (
            (pairwise_frame["source_peak_clusters"] >= 5)
            & (pairwise_frame["target_peak_clusters"] >= 5)
            & (pairwise_frame["matched_clusters"] >= 3)
            & (pairwise_frame["peak_jaccard"] >= 0.10)
        )
        pairwise_frame.to_csv(
            temporary / "pairwise_spectrum_variation.csv.gz", index=False, compression="gzip"
        )

        stratum_rows = []
        for key in sorted(residuals):
            relation, mz_name, intensity_name = key
            values = residuals[key]
            intensity_summary = summarize_array(values["abs_log_ratio"])
            mz_summary = summarize_array(values["abs_ppm"])
            stratum_rows.append(
                {
                    "relation": relation,
                    "mz_band": mz_name,
                    "intensity_band": intensity_name,
                    "matched_peak_pairs": intensity_summary["n"],
                    "identities": len(stratum_identities[key]),
                    "formulas": len(stratum_formulas[key]),
                    "abs_log_intensity_ratio_q25": intensity_summary["q25"],
                    "abs_log_intensity_ratio_q50": intensity_summary["q50"],
                    "abs_log_intensity_ratio_q75": intensity_summary["q75"],
                    "abs_log_intensity_ratio_q90": intensity_summary["q90"],
                    "abs_mz_residual_ppm_q25": mz_summary["q25"],
                    "abs_mz_residual_ppm_q50": mz_summary["q50"],
                    "abs_mz_residual_ppm_q75": mz_summary["q75"],
                    "abs_mz_residual_ppm_q90": mz_summary["q90"],
                    "signed_log_intensity_ratio_median": float(np.median(values["signed_log_ratio"])),
                    "signed_mz_residual_ppm_median": float(np.median(values["signed_ppm"])),
                }
            )
        strata_frame = pd.DataFrame(stratum_rows)
        strata_frame.to_csv(temporary / "matched_peak_jitter_strata.csv", index=False)

        addition_q = weighted_quantile(
            np.asarray(addition_values), np.asarray(addition_weights), [0.25, 0.50, 0.75]
        )
        global_intensity = summarize_array(
            [value for values in residuals.values() for value in values["abs_log_ratio"]]
        )
        global_mz = summarize_array([value for values in residuals.values() for value in values["abs_ppm"]])

        reliable_pairs = pairwise_frame.loc[pairwise_frame["reliable_for_dose"]].copy()
        descriptive_dropout = relation_quantiles(pairwise_frame, "source_dropout_fraction")
        reliable_dropout = identity_equal_relation_quantiles(
            reliable_pairs, "source_dropout_fraction"
        )
        screening_grid = e2_dropout_screening_grid(reliable_dropout)
        recipe = {
            "status": "noise_final_e1_empirical_recipe_frozen",
            "peak_dropout": {
                "dose_source": (
                    "identity-equal directed missing-peak fraction from reliable same-identity "
                    "same-adduct replicate pairs"
                ),
                "descriptive_unfiltered_by_condition_relation": descriptive_dropout,
                "reliable_identity_equal_by_condition_relation": reliable_dropout,
                "reliability_filter": {
                    "minimum_source_peak_clusters": 5,
                    "minimum_target_peak_clusters": 5,
                    "minimum_matched_clusters": 3,
                    "minimum_peak_jaccard": 0.10,
                },
                "e2_screening_grid": screening_grid,
                "target_weight": "within-identity cluster missing probability (1 - prevalence)",
                "target_weight_cap": 0.95,
                "implementation": (
                    "E2 screens only lower-tail q10/q25 doses under the 0.30 safety ceiling; "
                    "within a dose, weighted-sample source peaks without replacement"
                ),
            },
            "abs_log_intensity_jitter": global_intensity,
            "abs_mz_jitter_ppm": global_mz,
            "low_intensity_addition": {
                "source": "same-identity same-adduct support clusters absent from the target spectrum",
                "q25": addition_q[0],
                "q50": addition_q[1],
                "q75": addition_q[2],
                "cross_fit_required": True,
            },
            "pair_preservation": {
                "policy": "E2 must perturb linked neutral-loss/peak-pair evidence together or leave it unchanged",
                "calibrated_here": False,
            },
            "forbidden": [
                "copying peaks from another molecular identity",
                "using action outcomes to choose an E1 noise magnitude",
                "using P2b or any downstream reranker feature",
                "using global cluster missingness as the whole-spectrum deletion dose",
                "using unfiltered replicate-pair q50/q75 as an E2 training dose",
                "treating the 0.30 safety ceiling as an empirically optimal dose",
            ],
        }
        (temporary / "frozen_empirical_noise_recipe.json").write_text(json.dumps(recipe, indent=2), encoding="utf-8")

        plot_prevalence(consensus_frame, temporary / "e1_peak_prevalence_landscape.png")
        plot_jitter(strata_frame, temporary / "e1_empirical_jitter_by_condition.png")
        plot_pairwise_reliability(pairwise_frame, temporary / "e1_pairwise_dropout_reliability.png")

        cross_condition_groups = int(groups_frame["cross_condition"].sum())
        instruments = {}
        for value in groups_frame["instruments"]:
            instruments[str(int(value))] = instruments.get(str(int(value)), 0) + 1
        dropout_relations = recipe["peak_dropout"]["reliable_identity_equal_by_condition_relation"]
        eligible_dropout_relations = sum(
            int(value["eligible_for_e2"])
            for value in recipe["peak_dropout"]["e2_screening_grid"].values()
        )
        gates = {
            "groups_ge_1000": len(groups_frame) >= 1000,
            "identities_ge_1000": groups_frame["ik14"].nunique() >= 1000,
            "formulas_ge_500": groups_frame["formula"].nunique() >= 500,
            "cross_condition_groups_ge_500": cross_condition_groups >= 500,
            "consensus_clusters_ge_50000": len(consensus_frame) >= 50_000,
            "matched_peak_pairs_ge_250000": total_matched_pairs >= 250_000,
            "reliable_pairwise_variants_ge_50000": len(reliable_pairs) >= 50_000,
            "reliable_pairwise_identities_ge_750": reliable_pairs["ik14"].nunique() >= 750,
            "eligible_dropout_relations_ge_2": eligible_dropout_relations >= 2,
            "all_frozen_quantiles_monotone": (
                all(
                    values["q25"] <= values["q50"] <= values["q75"]
                    for values in dropout_relations.values()
                )
                and global_intensity["q25"] <= global_intensity["q50"] <= global_intensity["q75"]
                and global_mz["q25"] <= global_mz["q50"] <= global_mz["q75"]
            ),
        }
        report = {
            "status": "noise_final_e1_empirical_calibration_complete",
            "formal": args.max_groups == 0,
            "allowed_real_train_rows": len(allowed_rows),
            "eligible_identity_adduct_groups": len(groups_frame),
            "identities": int(groups_frame["ik14"].nunique()),
            "formulas": int(groups_frame["formula"].nunique()),
            "cross_condition_groups": cross_condition_groups,
            "consensus_peak_clusters": len(consensus_frame),
            "matched_peak_pairs": total_matched_pairs,
            "pairwise_variants": len(pairwise_frame),
            "reliable_pairwise_variants": len(reliable_pairs),
            "reliable_pairwise_identities": int(reliable_pairs["ik14"].nunique()),
            "reliable_pairwise_formulas": int(reliable_pairs["formula"].nunique()),
            "eligible_dropout_relations": eligible_dropout_relations,
            "duplicate_peaks_within_cluster": duplicate_peak_count,
            "instrument_count_distribution_per_group": instruments,
            "gates": gates,
            "pass_to_e2": all(gates.values()),
            "parameters": {
                "minimum_spectra": args.minimum_spectra,
                "maximum_spectra": args.maximum_spectra,
                "fragment_tolerance": args.fragment_tolerance,
                "seed": args.seed,
                "max_groups": args.max_groups,
            },
            "provenance": {
                "hdf5_sha256": sha256(args.data),
                "p3_allow_sha256": sha256(allow_path) if allow_path.exists() else None,
                "e0_manifest_sha256": sha256(e0_manifest_path),
                "script_sha256": sha256(Path(__file__)),
                "recipe_sha256": sha256(temporary / "frozen_empirical_noise_recipe.json"),
                "strata_sha256": sha256(temporary / "matched_peak_jitter_strata.csv"),
                "consensus_sha256": sha256(temporary / "consensus_peak_calibration.csv.gz"),
                "pairwise_variation_sha256": sha256(temporary / "pairwise_spectrum_variation.csv.gz"),
            },
            "contracts": {
                "identity_use": "replicate grouping only",
                "P3_identity_overlap": 0,
                "P2b": "forbidden",
                "stage_output": "frozen empirical noise distribution, not embedding performance",
            },
            "claim_limit": (
                "E1 estimates acquisition variation among repeated spectra. It does not show that applying this "
                "noise improves clean-spectrum retrieval; that is tested by E2-E4."
            ),
        }
        if not report["pass_to_e2"] and args.max_groups == 0:
            raise RuntimeError(f"E1 calibration gates failed: {gates}")
        (temporary / "e1_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(args.output_dir)
        print(json.dumps(report, indent=2), flush=True)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
