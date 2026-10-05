"""Strict full-metric decision for routed, shuffled-action and clean v3 arms."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from train_noise_corrected_routed_direct import (
    REGISTERED_BEST_ACTION_V6_EXACT_CONFIGURATION,
    REGISTERED_BEST_ACTION_V6_FLOAT_CONFIGURATION,
    REGISTERED_BEST_ACTION_V6_RESTORED_EXACT_CONFIGURATION,
    REGISTERED_BEST_ACTION_V6_RESTORED_FLOAT_CONFIGURATION,
    REGISTERED_BEST_ACTION_V7_CORRECTIVE_RESTORED_EXACT_CONFIGURATION,
    REGISTERED_BEST_ACTION_V7_CORRECTIVE_RESTORED_FLOAT_CONFIGURATION,
    REGISTERED_BEST_ACTION_V6_INPUT_SHA256,
    REGISTERED_BEST_ACTION_V6_SCHEDULE_GEOMETRY,
    _validate_registered_best_action_v6_schedule_geometry,
)


V6_REGISTERED_SEED = 20260908
V6_RESTORED_REGISTERED_SEED = 20260911
V6_REGISTERED_MINIMUM_DELTA_RECALL1_PP = 4.0
V6_REGISTERED_BOOTSTRAP_RESAMPLES = 10_000
V6_REGISTERED_FORMULA_CI_FAMILYWISE_HYPOTHESES = 4
V6_REGISTERED_HELD_QUERIES = 18_333
V6_HELD_METRIC_EVIDENCE_SCHEMA = "noise_v6_held_metric_evidence_v1"
RETRIEVAL_CUTOFFS = (1, 2, 3, 5, 10, 20)
BEST_ACTION_CONTRACTS = {
    "best_action_v6", "best_action_v6_restored",
    "best_action_v7_corrective_restored",
}
RESTORED_OPTIMIZER_CONTRACTS = {
    "best_action_v6_restored", "best_action_v7_corrective_restored",
}


def _json_native(value: object) -> object:
    """Recursively convert NumPy containers/scalars to strict JSON values.

    Scientific comparisons such as ``np.isfinite`` return ``np.bool_`` even
    when every input is a Python scalar.  Those values are valid booleans for
    gating, but the standard-library JSON encoder deliberately rejects them.
    Keep the scientific report unchanged while normalising only its transport
    representation at the summary boundary.
    """
    if isinstance(value, np.generic):
        return _json_native(value.item())
    if isinstance(value, np.ndarray):
        return _json_native(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    return value


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routed-dir", type=Path, required=True)
    parser.add_argument("--shuffled-dir", type=Path, required=True)
    parser.add_argument("--clean-dir", type=Path, required=True)
    parser.add_argument("--minimum-delta-recall1-pp", type=float, default=4.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--formula-ci-familywise-hypotheses", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash model tensors without depending on archive metadata."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _normalise_near(table: pd.DataFrame) -> np.ndarray:
    values = table["near"]
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.to_numpy(dtype=bool)
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all() and numeric.isin([0, 1]).all():
        return numeric.to_numpy(dtype=np.int8).astype(bool)
    lowered = values.astype(str).str.strip().str.lower()
    if lowered.isin(["true", "false"]).all():
        return lowered.eq("true").to_numpy(dtype=bool)
    raise RuntimeError("v3 held table contains a non-boolean near flag")


def _table_metric_summary(
    table: pd.DataFrame,
    prefix: str,
    mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    selected = table if mask is None else table.loc[np.asarray(mask, dtype=bool)]
    if len(selected) == 0:
        raise RuntimeError(f"v3 held {prefix} metric panel is empty")
    rank = selected[f"{prefix}_rank"].to_numpy(dtype=np.int64)
    output: dict[str, float | int] = {
        "queries": int(len(selected)),
        "mrr": float(np.mean(1.0 / rank)),
        "mean_rank": float(np.mean(rank)),
        "median_rank": float(np.median(rank)),
        "macro_query_auroc": float(selected[f"{prefix}_macro_query_auc"].mean()),
        "macro_query_auprc": float(selected[f"{prefix}_macro_query_auprc"].mean()),
        "mean_positive_vs_best_negative_margin": float(
            selected[f"{prefix}_positive_vs_best_negative_margin"].mean()
        ),
        "mean_top1_top2_gap": float(
            selected[f"{prefix}_top1_top2_gap"].mean()
        ),
        "mean_signed_top1_top2_gap": float(
            selected[f"{prefix}_signed_top1_top2_gap"].mean()
        ),
    }
    for cutoff in RETRIEVAL_CUTOFFS:
        output[f"recall@{cutoff}"] = float(np.mean(rank <= cutoff))
    return output


def _require_numeric_match(
    observed: float | int,
    reported: object,
    context: str,
) -> None:
    try:
        reported_float = float(reported)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"v3 held metric is missing or non-numeric: {context}") from error
    if not np.isfinite(reported_float) or not np.isclose(
        float(observed), reported_float, rtol=1e-10, atol=1e-12,
    ):
        raise RuntimeError(
            f"v3 held CSV/decision metric differs for {context}: "
            f"csv={observed} decision={reported}"
        )


def _validate_metric_summary(
    observed: dict[str, float | int],
    reported: object,
    context: str,
) -> None:
    if not isinstance(reported, dict):
        raise RuntimeError(f"v3 held decision misses metric panel: {context}")
    if int(reported.get("queries", -1)) != int(observed["queries"]):
        raise RuntimeError(
            f"v3 held query count differs for {context}: "
            f"csv={observed['queries']} decision={reported.get('queries')}"
        )
    for key, value in observed.items():
        if key == "queries":
            continue
        _require_numeric_match(value, reported.get(key), f"{context}.{key}")


def _validate_per_query_metrics(table: pd.DataFrame, prefix: str) -> None:
    rank = table[f"{prefix}_rank"].to_numpy(dtype=np.int64)
    reciprocal = table[f"{prefix}_reciprocal_rank"].to_numpy(dtype=np.float64)
    macro_auc = table[f"{prefix}_macro_query_auc"].to_numpy(dtype=np.float64)
    macro_auprc = table[f"{prefix}_macro_query_auprc"].to_numpy(dtype=np.float64)
    margin = table[
        f"{prefix}_positive_vs_best_negative_margin"
    ].to_numpy(dtype=np.float64)
    gap = table[f"{prefix}_top1_top2_gap"].to_numpy(dtype=np.float64)
    signed_gap = table[f"{prefix}_signed_top1_top2_gap"].to_numpy(dtype=np.float64)
    expected_reciprocal = 1.0 / rank
    if not np.allclose(reciprocal, expected_reciprocal, rtol=1e-10, atol=1e-12):
        raise RuntimeError(f"v3 held per-query reciprocal rank differs for {prefix}")
    if not np.allclose(macro_auprc, expected_reciprocal, rtol=1e-10, atol=1e-12):
        raise RuntimeError(f"v3 held per-query macro AUPRC differs for {prefix}")
    if np.any((macro_auc < -1e-12) | (macro_auc > 1.0 + 1e-12)):
        raise RuntimeError(f"v3 held per-query macro AUROC is invalid for {prefix}")
    correct = rank == 1
    if np.any(correct & ~np.isclose(macro_auc, 1.0, rtol=0, atol=1e-12)):
        raise RuntimeError(f"v3 held Top-1 query does not have AUROC=1 for {prefix}")
    if np.any(gap < -1e-12):
        raise RuntimeError(f"v3 held Top1-Top2 gap is negative for {prefix}")
    expected_signed = np.where(correct, gap, -gap)
    if not np.allclose(signed_gap, expected_signed, rtol=1e-10, atol=1e-12):
        raise RuntimeError(f"v3 held signed Top1-Top2 gap differs for {prefix}")
    if np.any(correct & (margin <= 0.0)) or np.any(~correct & (margin > 1e-12)):
        raise RuntimeError(f"v3 held rank/margin direction differs for {prefix}")


def _validate_panel_counts(
    held: dict[str, object],
    table_rows: int,
    near_rows: int,
    *,
    require_official: bool,
) -> None:
    labels = ["initial_E8", "candidate"]
    if "official" in held:
        labels.insert(0, "official")
    elif require_official:
        raise RuntimeError("best-action v6 decision misses the official held panel")
    reference_counts: dict[tuple[str, str], int] = {}
    for label in labels:
        panel = held.get(label)
        if not isinstance(panel, dict):
            raise RuntimeError(f"v3 held decision misses panel: {label}")
        for section, expected in (("retrieval", table_rows), ("near_subset", near_rows)):
            report = panel.get(section)
            if not isinstance(report, dict) or int(report.get("queries", -1)) != expected:
                raise RuntimeError(
                    f"v3 held {label}.{section} query count does not close: "
                    f"expected={expected} observed="
                    f"{report.get('queries') if isinstance(report, dict) else None}"
                )
        micro = panel.get("micro_candidate")
        if not isinstance(micro, dict):
            raise RuntimeError(f"v3 held decision misses {label}.micro_candidate")
        molecules = int(micro.get("molecules", -1))
        if molecules < 2 * table_rows:
            raise RuntimeError(
                f"v3 held {label}.micro_candidate molecule count does not close"
            )
        current_counts = {("micro_candidate", "molecules"): molecules}
        for section in (
            "massspecgym_10ppm_pooled_pairwise",
            "massspecgym_mh_10ppm_pooled_pairwise",
        ):
            report = panel.get(section)
            if not isinstance(report, dict):
                raise RuntimeError(f"v3 held decision misses {label}.{section}")
            total = int(report.get("spectrum_pairs", -1))
            positive = int(report.get("positive_pairs", -1))
            negative = int(report.get("negative_pairs", -1))
            if total <= 0 or positive <= 0 or negative <= 0 or total != positive + negative:
                raise RuntimeError(f"v3 held {label}.{section} pair counts do not close")
            current_counts[(section, "spectrum_pairs")] = total
            current_counts[(section, "positive_pairs")] = positive
            current_counts[(section, "negative_pairs")] = negative
        if not reference_counts:
            reference_counts = current_counts
        elif current_counts != reference_counts:
            raise RuntimeError(
                f"v3 held candidate-ledger panel counts differ for {label}"
            )


def _binary_score_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    context: str,
) -> dict[str, float | int]:
    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise RuntimeError(f"v6 metric evidence shape differs for {context}")
    if labels.dtype != np.uint8 or not np.isin(labels, [0, 1]).all():
        raise RuntimeError(f"v6 metric evidence labels are not binary uint8: {context}")
    if scores.dtype != np.float32 or not np.isfinite(scores).all():
        raise RuntimeError(f"v6 metric evidence scores are not finite float32: {context}")
    if set(map(int, np.unique(labels))) != {0, 1}:
        raise RuntimeError(f"v6 metric evidence misses one class: {context}")
    return {
        "rows": int(len(labels)),
        "positive": int(np.sum(labels == 1)),
        "negative": int(np.sum(labels == 0)),
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
    }


def _validate_evidence_metric_panel(
    observed: dict[str, float | int],
    reported: object,
    context: str,
    *,
    row_key: str,
    include_class_counts: bool,
) -> None:
    if not isinstance(reported, dict):
        raise RuntimeError(f"v6 decision misses metric evidence panel: {context}")
    _require_numeric_match(observed["rows"], reported.get(row_key), f"{context}.{row_key}")
    if include_class_counts:
        _require_numeric_match(
            observed["positive"], reported.get("positive_pairs"),
            f"{context}.positive_pairs",
        )
        _require_numeric_match(
            observed["negative"], reported.get("negative_pairs"),
            f"{context}.negative_pairs",
        )
    for metric in ("auroc", "auprc"):
        _require_numeric_match(observed[metric], reported.get(metric), f"{context}.{metric}")


def _validate_held_metric_evidence(
    path: Path,
    decision: dict[str, object],
    table: pd.DataFrame,
) -> str:
    evidence_path = path / "held_metric_evidence.npz"
    provenance = decision.get("provenance", {})
    reported_hash = provenance.get("held_metric_evidence_sha256")
    reported_rows = provenance.get("held_metric_evidence_rows")
    if not evidence_path.is_file() or not reported_hash or not isinstance(reported_rows, dict):
        raise RuntimeError(f"best-action v6 arm misses held metric evidence: {path}")
    if reported_hash != sha256_file(evidence_path):
        raise RuntimeError(f"v6 held metric evidence hash differs from report: {path}")

    required = {
        "schema_version", "query_index", "molecule_label", "pair_label",
        "pair_is_mh", "official_molecule_score", "official_pair_score",
        "initial_E8_molecule_score", "initial_E8_pair_score",
        "candidate_molecule_score", "candidate_pair_score",
    }
    try:
        with np.load(evidence_path, allow_pickle=False) as package:
            if set(package.files) != required:
                raise RuntimeError(
                    "v6 held metric evidence schema keys differ: "
                    f"missing={sorted(required - set(package.files))} "
                    f"extra={sorted(set(package.files) - required)}"
                )
            schema = np.asarray(package["schema_version"])
            if schema.shape != () or str(schema.item()) != V6_HELD_METRIC_EVIDENCE_SCHEMA:
                raise RuntimeError("v6 held metric evidence schema version differs")
            query_index = np.asarray(package["query_index"])
            molecule_label = np.asarray(package["molecule_label"])
            pair_label = np.asarray(package["pair_label"])
            pair_is_mh = np.asarray(package["pair_is_mh"])
            scores = {
                panel: {
                    "molecule": np.asarray(package[f"{panel}_molecule_score"]),
                    "pair": np.asarray(package[f"{panel}_pair_score"]),
                }
                for panel in ("official", "initial_E8", "candidate")
            }
    except (OSError, ValueError) as error:
        raise RuntimeError(f"cannot read v6 held metric evidence: {path}") from error

    table_query = table["query_index"].to_numpy(dtype=np.int64)
    if (
        query_index.dtype != np.int64
        or query_index.ndim != 1
        or not np.array_equal(query_index, table_query)
    ):
        raise RuntimeError("v6 metric evidence query ledger differs from held CSV")
    if molecule_label.ndim != 1 or pair_label.ndim != 1:
        raise RuntimeError("v6 metric evidence labels must be one-dimensional")
    if pair_is_mh.dtype != np.bool_ or pair_is_mh.shape != pair_label.shape:
        raise RuntimeError("v6 metric evidence [M+H]+ mask differs from pair ledger")

    expected_rows = {
        "held_queries": int(len(query_index)),
        "molecules": int(len(molecule_label)),
        "spectrum_pairs": int(len(pair_label)),
        "mh_spectrum_pairs": int(np.sum(pair_is_mh)),
    }
    if set(reported_rows) != set(expected_rows) or any(
        type(reported_rows[key]) is not int or reported_rows[key] != value
        for key, value in expected_rows.items()
    ):
        raise RuntimeError(
            "v6 held metric evidence row provenance differs: "
            f"npz={expected_rows} decision={reported_rows}"
        )

    held = decision["evaluation"]["formal_held_graph"]
    for panel, panel_scores in scores.items():
        molecule = _binary_score_metrics(
            molecule_label, panel_scores["molecule"], f"{panel}.micro_candidate",
        )
        pair = _binary_score_metrics(
            pair_label, panel_scores["pair"],
            f"{panel}.massspecgym_10ppm_pooled_pairwise",
        )
        mh_pair = _binary_score_metrics(
            pair_label[pair_is_mh], panel_scores["pair"][pair_is_mh],
            f"{panel}.massspecgym_mh_10ppm_pooled_pairwise",
        )
        _validate_evidence_metric_panel(
            molecule, held[panel].get("micro_candidate"),
            f"{panel}.micro_candidate", row_key="molecules",
            include_class_counts=False,
        )
        _validate_evidence_metric_panel(
            pair, held[panel].get("massspecgym_10ppm_pooled_pairwise"),
            f"{panel}.massspecgym_10ppm_pooled_pairwise", row_key="spectrum_pairs",
            include_class_counts=True,
        )
        _validate_evidence_metric_panel(
            mh_pair, held[panel].get("massspecgym_mh_10ppm_pooled_pairwise"),
            f"{panel}.massspecgym_mh_10ppm_pooled_pairwise", row_key="spectrum_pairs",
            include_class_counts=True,
        )

    digest = hashlib.sha256()
    for name, value in (
        ("query_index", query_index), ("molecule_label", molecule_label),
        ("pair_label", pair_label), ("pair_is_mh", pair_is_mh),
    ):
        digest.update(name.encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _validate_registered_v6_configuration(
    decision: dict[str, object],
    path: Path,
) -> None:
    configuration = decision.get("configuration")
    if not isinstance(configuration, dict):
        raise RuntimeError(f"best-action v6 arm misses training configuration: {path}")
    decision_contract = str(decision.get(
        "direct_contract", configuration.get("direct_contract", ""),
    ))
    if decision_contract == "best_action_v7_corrective_restored":
        exact_configuration = (
            REGISTERED_BEST_ACTION_V7_CORRECTIVE_RESTORED_EXACT_CONFIGURATION
        )
        float_configuration = (
            REGISTERED_BEST_ACTION_V7_CORRECTIVE_RESTORED_FLOAT_CONFIGURATION
        )
    elif decision_contract == "best_action_v6_restored":
        exact_configuration = REGISTERED_BEST_ACTION_V6_RESTORED_EXACT_CONFIGURATION
        float_configuration = REGISTERED_BEST_ACTION_V6_RESTORED_FLOAT_CONFIGURATION
    else:
        exact_configuration = REGISTERED_BEST_ACTION_V6_EXACT_CONFIGURATION
        float_configuration = REGISTERED_BEST_ACTION_V6_FLOAT_CONFIGURATION
    mismatches: dict[str, dict[str, object]] = {}
    for name, expected in exact_configuration.items():
        observed = configuration.get(name)
        if observed != expected or type(observed) is not type(expected):
            mismatches[name] = {"observed": observed, "expected": expected}
    for name, expected in float_configuration.items():
        observed = configuration.get(name)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not np.isfinite(float(observed))
            or not np.isclose(float(observed), expected, rtol=1e-12, atol=1e-12)
        ):
            mismatches[name] = {"observed": observed, "expected": expected}
    if mismatches:
        raise RuntimeError(
            f"best-action v6 reported training configuration drifted at {path}: "
            + json.dumps(mismatches, sort_keys=True)
        )


def _validate_registered_v6_schedule(
    decision: dict[str, object],
    path: Path,
) -> None:
    schedule = decision.get("schedule")
    if not isinstance(schedule, dict):
        raise RuntimeError(f"best-action v6 arm misses its schedule report: {path}")
    geometry = schedule.get("schedule_geometry")
    if not isinstance(geometry, dict):
        raise RuntimeError(f"best-action v6 arm misses schedule geometry: {path}")
    _validate_registered_best_action_v6_schedule_geometry(
        geometry, context=f"summary arm {path}",
    )
    epoch_geometry = schedule.get("schedule_geometry_by_epoch")
    required_flags = {
        "schedule_geometry_matches_pre_model_preflight": True,
        "cap_safe_corrective_repartition_used": True,
        "cap_safe_corrective_repartition_preserved_query_order_and_coverage": True,
        "all_unique_action_panels_covered_before_recycling": True,
        "partial_batch_query_mass_scaled_to_registered_size": True,
    }
    mismatches = {
        name: {"observed": schedule.get(name), "expected": expected}
        for name, expected in required_flags.items()
        if schedule.get(name) is not expected
    }
    scalar_expected = {
        "original_corrective_batches": 871,
        "cap_safe_corrective_batches": 877,
        "corrective_batches_added_by_cap_safe_repartition": 6,
        "required_optimizer_steps": 3508,
        "effective_corrective_recycle_factor": 4.0,
        "configured_maximum_corrective_recycle_factor": 4.0,
        "minimum_corrective_action_exposure_per_epoch": 4.0,
        "maximum_corrective_action_exposure_per_epoch": 4.0,
    }
    for name, expected in scalar_expected.items():
        observed = schedule.get(name)
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not np.isfinite(float(observed))
            or not np.isclose(float(observed), expected, rtol=1e-12, atol=1e-12)
        ):
            mismatches[name] = {"observed": observed, "expected": expected}
    if (
        not isinstance(epoch_geometry, list)
        or len(epoch_geometry) != 4
        or any(value != REGISTERED_BEST_ACTION_V6_SCHEDULE_GEOMETRY
               for value in epoch_geometry)
    ):
        mismatches["schedule_geometry_by_epoch"] = {
            "observed": epoch_geometry,
            "expected": [REGISTERED_BEST_ACTION_V6_SCHEDULE_GEOMETRY] * 4,
        }
    contracts = decision.get("contracts", {})
    for name in (
        "schedule_geometry_validated_before_model_load",
        "cap_safe_corrective_repartition_preserves_query_action_panel",
        "best_action_v6_exact_877_batch_schedule_verified",
    ):
        if not isinstance(contracts, dict) or contracts.get(name) is not True:
            mismatches[f"contracts.{name}"] = {
                "observed": contracts.get(name) if isinstance(contracts, dict) else None,
                "expected": True,
            }
    if mismatches:
        raise RuntimeError(
            f"best-action v6 schedule repair is missing or drifted at {path}: "
            + json.dumps(mismatches, sort_keys=True)
        )


def formula_cluster_ci(
    formulas: np.ndarray,
    values: np.ndarray,
    repeats: int,
    seed: int,
    familywise_hypotheses: int = 1,
) -> dict[str, float | int]:
    if repeats < 1 or not len(values) or familywise_hypotheses < 1:
        raise ValueError("formula-cluster CI requires observations and resamples")
    grouped = pd.DataFrame({
        "formula": np.asarray(formulas, dtype=str),
        "value": np.asarray(values, dtype=float),
    }).groupby("formula", sort=True).value.agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(int(seed))
    boot = np.empty(repeats, dtype=float)
    for index in range(repeats):
        take = rng.integers(0, len(grouped), len(grouped))
        boot[index] = 100.0 * sums[take].sum() / counts[take].sum()
    tail = 0.05 / (2 * familywise_hypotheses)
    return {
        "delta_pp": float(100.0 * np.mean(values)),
        "ci_low_pp": float(np.quantile(boot, tail)),
        "ci_high_pp": float(np.quantile(boot, 1.0 - tail)),
        "formula_clusters": int(len(grouped)),
        "familywise_hypotheses": int(familywise_hypotheses),
        "bonferroni_per_interval_alpha": float(0.05 / familywise_hypotheses),
    }


def _load_arm(
    path: Path,
    expected_arm: str,
) -> tuple[dict[str, object], pd.DataFrame, str | None]:
    decision_path = path / "decision.json"
    table_path = path / "held_per_query.csv.gz"
    checkpoint_path = path / "final_shared_encoder.pt"
    if not all(value.is_file() for value in (
        decision_path, table_path, checkpoint_path,
    )):
        raise FileNotFoundError(f"incomplete v3 arm: {path}")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if (
        decision.get("status") != "noise_corrected_routed_direct_complete"
        or decision.get("formal") is not True
        or decision.get("arm") != expected_arm
        or decision.get("corrective_objective_mode") != "v3_direct"
        or "formal_held_graph" not in decision.get("evaluation", {})
    ):
        raise RuntimeError(f"v3 arm contract failed: {path}")
    decision_contract = str(
        decision.get("direct_contract", decision.get("configuration", {}).get(
            "direct_contract", "v3",
        ))
    )
    if decision_contract in BEST_ACTION_CONTRACTS:
        _validate_registered_v6_configuration(decision, path)
        _validate_registered_v6_schedule(decision, path)
    table_sha256 = decision.get("provenance", {}).get("held_per_query_sha256")
    table_rows = decision.get("provenance", {}).get("held_per_query_rows")
    if decision_contract in BEST_ACTION_CONTRACTS and (
        not table_sha256 or table_rows is None
    ):
        raise RuntimeError(
            f"best-action v6 arm misses held-per-query hash/row provenance: {path}"
        )
    if table_sha256 is not None and table_sha256 != sha256_file(table_path):
        raise RuntimeError(f"v3 held-per-query file hash differs from report: {path}")
    if decision.get("provenance", {}).get(
        "final_shared_encoder_sha256"
    ) != sha256_file(checkpoint_path):
        raise RuntimeError(f"v3 final checkpoint file hash differs from report: {path}")
    package = torch.load(checkpoint_path, map_location="cpu")
    package_contract = str(package.get("direct_contract", "v3"))
    if (
        not isinstance(package, dict)
        or not isinstance(package.get("model_state"), dict)
        or package.get("status") != "noise_corrected_routed_direct_shared_encoder"
        or package.get("arm") != expected_arm
        or package.get("corrective_objective_mode") != "v3_direct"
        or int(package.get("outer_fold", -1))
        != int(decision.get("outer_formula_fold", -2))
        or package.get("inference_clean_only") is not True
        or package.get("P2b_used") is not False
        or package_contract != decision_contract
    ):
        raise RuntimeError(f"v3 final checkpoint package is malformed: {path}")
    if decision.get("provenance", {}).get(
        "final_model_state_sha256"
    ) != state_sha256(package["model_state"]):
        raise RuntimeError(f"v3 final model-state hash differs from report: {path}")
    table = pd.read_csv(table_path, low_memory=False)
    required = {
        "query_index", "query_formula", "near", "initial_E8_rank", "candidate_rank",
        "initial_E8_reciprocal_rank", "candidate_reciprocal_rank",
        "initial_E8_macro_query_auc", "candidate_macro_query_auc",
        "initial_E8_macro_query_auprc", "candidate_macro_query_auprc",
        "initial_E8_positive_vs_best_negative_margin",
        "candidate_positive_vs_best_negative_margin",
        "initial_E8_top1_top2_gap", "candidate_top1_top2_gap",
        "initial_E8_signed_top1_top2_gap", "candidate_signed_top1_top2_gap",
        "corrected", "introduced", "risk_net",
    }
    if missing := required - set(table.columns):
        raise RuntimeError(f"v3 held table misses {sorted(missing)}")
    if table.query_index.duplicated().any():
        raise RuntimeError("v3 held table contains duplicate queries")
    if table.query_formula.isna().any() or table.query_formula.astype(str).str.len().eq(0).any():
        raise RuntimeError("v3 held table contains a missing query formula")
    table["near"] = _normalise_near(table)
    numeric = table[list(required - {"query_formula", "near"})].apply(
        pd.to_numeric, errors="coerce",
    )
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError("v3 held table contains a non-finite required value")
    for column in ("initial_E8_rank", "candidate_rank"):
        values = table[column].to_numpy(dtype=float)
        if np.any(values < 1) or not np.equal(values, np.floor(values)).all():
            raise RuntimeError(f"v3 held table contains an invalid rank: {column}")
    if table_rows is not None and int(table_rows) != len(table):
        raise RuntimeError(
            f"v3 held-per-query row provenance differs: "
            f"csv={len(table)} decision={table_rows}"
        )
    if decision_contract in BEST_ACTION_CONTRACTS and len(table) != V6_REGISTERED_HELD_QUERIES:
        raise RuntimeError(
            "best-action v6 requires exactly "
            f"{V6_REGISTERED_HELD_QUERIES} actual held CSV rows; observed={len(table)}"
        )
    graph_scope = decision.get("graph_scope", {})
    try:
        all_queries = int(graph_scope["all_queries"])
        train_queries = int(graph_scope["outer_train_queries"])
        outer_held_queries = int(graph_scope["outer_held_queries"])
        evaluated_held_queries = int(graph_scope["evaluated_outer_held_queries"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("v3 graph_scope is missing or non-integral") from error
    if (
        all_queries != train_queries + outer_held_queries
        or outer_held_queries != len(table)
        or evaluated_held_queries != len(table)
    ):
        raise RuntimeError(
            "v3 graph_scope does not close against the actual held CSV: "
            f"scope={graph_scope} csv_rows={len(table)}"
        )
    held = decision["evaluation"]["formal_held_graph"]
    near = table["near"].to_numpy(dtype=bool)
    evidence_structure_sha256 = None
    if decision_contract in BEST_ACTION_CONTRACTS:
        evidence_structure_sha256 = _validate_held_metric_evidence(
            path, decision, table,
        )
    _validate_panel_counts(
        held, len(table), int(near.sum()),
        require_official=decision_contract in BEST_ACTION_CONTRACTS,
    )
    for label, prefix in (("initial_E8", "initial_E8"), ("candidate", "candidate")):
        _validate_per_query_metrics(table, prefix)
        _validate_metric_summary(
            _table_metric_summary(table, prefix),
            held[label].get("retrieval"),
            f"{label}.retrieval",
        )
        _validate_metric_summary(
            _table_metric_summary(table, prefix, near),
            held[label].get("near_subset"),
            f"{label}.near_subset",
        )
    initial_rank = table["initial_E8_rank"].to_numpy(dtype=np.int64)
    candidate_rank = table["candidate_rank"].to_numpy(dtype=np.int64)
    corrected_rows = (initial_rank > 1) & (candidate_rank == 1)
    introduced_rows = (initial_rank == 1) & (candidate_rank > 1)
    if not np.array_equal(
        pd.to_numeric(table["corrected"], errors="coerce").to_numpy(dtype=float),
        corrected_rows.astype(float),
    ):
        raise RuntimeError("v3 held per-query corrected flags differ from ranks")
    if not np.array_equal(
        pd.to_numeric(table["introduced"], errors="coerce").to_numpy(dtype=float),
        introduced_rows.astype(float),
    ):
        raise RuntimeError("v3 held per-query introduced flags differ from ranks")
    expected_risk_rows = corrected_rows.astype(np.int8) - introduced_rows.astype(np.int8)
    if not np.array_equal(
        pd.to_numeric(table["risk_net"], errors="coerce").to_numpy(dtype=float),
        expected_risk_rows.astype(float),
    ):
        raise RuntimeError("v3 held per-query risk_net differs from ranks")
    corrected = int(corrected_rows.sum())
    introduced = int(introduced_rows.sum())
    near_corrected = int(np.sum(near & (initial_rank > 1) & (candidate_rank == 1)))
    near_introduced = int(np.sum(near & (initial_rank == 1) & (candidate_rank > 1)))
    outcomes = held.get("candidate_vs_initial_E8")
    if not isinstance(outcomes, dict):
        raise RuntimeError("v3 held decision misses candidate-vs-initial outcomes")
    expected_risk = {
        "corrected": corrected,
        "introduced": introduced,
        "risk_net_lambda2": corrected - 2 * introduced,
    }
    for key, expected in expected_risk.items():
        if int(outcomes.get(key, -10**18)) != expected:
            raise RuntimeError(
                f"v3 held CSV/decision risk differs for {key}: "
                f"csv={expected} decision={outcomes.get(key)}"
            )
    near_outcomes = outcomes.get("near")
    if not isinstance(near_outcomes, dict):
        raise RuntimeError("v3 held decision misses near candidate-vs-initial outcomes")
    expected_near_risk = {
        "corrected": near_corrected,
        "introduced": near_introduced,
        "risk_net_lambda2": near_corrected - 2 * near_introduced,
    }
    for key, expected in expected_near_risk.items():
        if int(near_outcomes.get(key, -10**18)) != expected:
            raise RuntimeError(
                f"v3 held CSV/decision near-risk differs for {key}: "
                f"csv={expected} decision={near_outcomes.get(key)}"
            )
    return (
        decision,
        table.sort_values("query_index", kind="stable").reset_index(drop=True),
        evidence_structure_sha256,
    )


def _formula_rank_ci(
    left_rank: np.ndarray,
    right_rank: np.ndarray,
    formulas: np.ndarray,
    cutoff: int,
    repeats: int,
    seed: int,
    familywise_hypotheses: int = 1,
) -> dict[str, float | int]:
    delta = (right_rank <= cutoff).astype(float) - (left_rank <= cutoff).astype(float)
    return formula_cluster_ci(
        formulas, delta, repeats, seed, familywise_hypotheses,
    )


def _metric_checks(
    initial: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, bool]:
    left = initial["retrieval"]
    right = candidate["retrieval"]
    near_left = initial["near_subset"]
    near_right = candidate["near_subset"]
    checks: dict[str, bool] = {}
    for cutoff in (1, 2, 3, 5, 10, 20):
        baseline = float(left[f"recall@{cutoff}"])
        candidate_value = float(right[f"recall@{cutoff}"])
        checks[f"recall@{cutoff}"] = (
            candidate_value >= baseline - 1e-12
            if baseline >= 1.0 - 1e-12 else
            candidate_value > baseline
        )
        near_baseline = float(near_left[f"recall@{cutoff}"])
        near_candidate = float(near_right[f"recall@{cutoff}"])
        checks[f"near_recall@{cutoff}"] = (
            near_candidate >= near_baseline - 1e-12
            if near_baseline >= 1.0 - 1e-12 else
            near_candidate > near_baseline
        )
    for key in (
        "mrr", "macro_query_auroc", "macro_query_auprc",
        "mean_positive_vs_best_negative_margin", "mean_signed_top1_top2_gap",
    ):
        checks[key] = float(right[key]) > float(left[key])
        checks[f"near_{key}"] = float(near_right[key]) > float(near_left[key])
    # Keep the conventional unsigned gap visible, but never use it as a
    # monotone quality gate: confidently wrong predictions also increase it.
    checks["raw_top1_top2_gap_reported_not_direction_gated"] = all(
        "mean_top1_top2_gap" in panel
        for panel in (left, right, near_left, near_right)
    )
    checks["mean_rank"] = float(right["mean_rank"]) < float(left["mean_rank"])
    checks["median_rank_nonworse"] = float(right["median_rank"]) <= float(left["median_rank"])
    checks["near_mean_rank"] = float(near_right["mean_rank"]) < float(near_left["mean_rank"])
    checks["near_median_rank_nonworse"] = (
        float(near_right["median_rank"]) <= float(near_left["median_rank"])
    )
    for panel in ("micro_candidate", "massspecgym_10ppm_pooled_pairwise",
                  "massspecgym_mh_10ppm_pooled_pairwise"):
        for metric in ("auroc", "auprc"):
            checks[f"{panel}.{metric}"] = (
                float(candidate[panel][metric]) > float(initial[panel][metric])
            )
    return checks


def _effective_calibration_vector(decision: dict[str, object]) -> np.ndarray:
    calibration = decision.get("gradient_calibration", {})
    branch = calibration.get("effective_branch_scale", {})
    required = ("transfer", "payload", "consistency", "robust", "harmful")
    if any(name not in branch for name in required):
        raise RuntimeError("v6 calibration report misses an effective branch scale")
    values = [float(branch[name]) for name in required]
    values.extend([
        float(calibration["effective_action_to_risk_scale"]),
        float(calibration["effective_global_gradient_scale"]),
    ])
    vector = np.asarray(values, dtype=np.float64)
    if not np.isfinite(vector).all():
        raise RuntimeError("v6 calibration scale vector is non-finite")
    return vector


def summarize(args: argparse.Namespace) -> tuple[dict[str, object], pd.DataFrame]:
    routed, routed_table, routed_evidence_structure = _load_arm(
        args.routed_dir, "routed_direct",
    )
    shuffled, shuffled_table, shuffled_evidence_structure = _load_arm(
        args.shuffled_dir, "shuffled_action_control",
    )
    clean, clean_table, clean_evidence_structure = _load_arm(
        args.clean_dir, "clean_control",
    )
    arms = {"routed": routed, "shuffled": shuffled, "clean": clean}
    tables = {"routed": routed_table, "shuffled": shuffled_table, "clean": clean_table}
    provenance_keys = (
        "spectrum_data_sha256", "architecture_checkpoint_sha256",
        "candidate_graph_sha256", "graph_report_sha256",
        "source_manifest_sha256",
        "routed_ledger_report_sha256", "training_actions_sha256",
        "action_spectra_sha256", "initial_student_checkpoint_sha256",
        "official_checkpoint_sha256", "v3_core_sha256",
        "v3_transfer_objective_sha256", "v3_action_expansion_sha256",
        "optimizer_update_arbitration_sha256",
        "noise_v3_core_sha256",
        "v3_action_router_sha256", "v3_shuffled_control_sha256",
        "v3_action_panel_sha256",
        "fullgraph_evaluator_sha256",
        "script_sha256",
    )
    for key in provenance_keys:
        values = {str(arm["provenance"].get(key)) for arm in arms.values()}
        if len(values) != 1:
            raise RuntimeError(f"v3 arms differ in provenance: {key}")
    configurations = [json.dumps(arm["configuration"], sort_keys=True) for arm in arms.values()]
    if len(set(configurations)) != 1:
        raise RuntimeError("v3 arms differ in training configuration")
    direct_contracts = {
        str(arm.get("direct_contract", arm["configuration"].get(
            "direct_contract", "v3",
        )))
        for arm in arms.values()
    }
    if len(direct_contracts) != 1:
        raise RuntimeError("v3 arms differ in direct-training contract")
    direct_contract = next(iter(direct_contracts))
    if direct_contract not in ({"v3"} | BEST_ACTION_CONTRACTS):
        raise RuntimeError(f"unregistered direct-training contract: {direct_contract}")
    if (
        direct_contract in RESTORED_OPTIMIZER_CONTRACTS
        and any(
            not arm["provenance"].get("optimizer_update_arbitration_sha256")
            for arm in arms.values()
        )
    ):
        raise RuntimeError("restored v6 arms miss optimizer arbitration provenance")
    if direct_contract in BEST_ACTION_CONTRACTS:
        registered_summary_parameters = {
            "seed": (
                int(args.seed),
                V6_RESTORED_REGISTERED_SEED
                if direct_contract in RESTORED_OPTIMIZER_CONTRACTS
                else V6_REGISTERED_SEED,
            ),
            "minimum_delta_recall1_pp": (
                float(args.minimum_delta_recall1_pp),
                V6_REGISTERED_MINIMUM_DELTA_RECALL1_PP,
            ),
            "bootstrap_resamples": (
                int(args.bootstrap_resamples),
                V6_REGISTERED_BOOTSTRAP_RESAMPLES,
            ),
            "formula_ci_familywise_hypotheses": (
                int(args.formula_ci_familywise_hypotheses),
                V6_REGISTERED_FORMULA_CI_FAMILYWISE_HYPOTHESES,
            ),
        }
        for name, (observed, expected) in registered_summary_parameters.items():
            if observed != expected:
                raise RuntimeError(
                    f"best-action v6 summary requires registered {name}={expected}; "
                    f"observed={observed}"
                )
    if direct_contract in BEST_ACTION_CONTRACTS:
        registered_input_hashes = REGISTERED_BEST_ACTION_V6_INPUT_SHA256
        evidence_structures = {
            routed_evidence_structure, shuffled_evidence_structure,
            clean_evidence_structure,
        }
        if None in evidence_structures or len(evidence_structures) != 1:
            raise RuntimeError(
                "best-action v6 arms do not share one metric evidence ledger"
            )
        selector_hashes = {
            str(arm["provenance"].get("best_action_selector_sha256"))
            for arm in arms.values()
        }
        if len(selector_hashes) != 1 or selector_hashes == {"None"}:
            raise RuntimeError("best-action v6 arms differ in selector provenance")
        for name, arm in arms.items():
            calibration_bank = arm.get("calibration_action_bank", {})
            targeted_hash = calibration_bank.get("targeted_action_bank_sha256")
            control_hash = calibration_bank.get("control_action_bank_sha256")
            training_hash = calibration_bank.get("training_action_bank_sha256")
            calibration_hash = calibration_bank.get("calibration_action_bank_sha256")
            admitted = arm.get("admitted_action_bank", {})
            graph_scope = arm.get("graph_scope", {})
            expected_admitted_counts = {
                "corrective": 32114,
                "harmful": 85959,
                "robust": 179289,
            }
            if (
                any(
                    arm.get("provenance", {}).get(key) != expected
                    for key, expected in registered_input_hashes.items()
                )
                or calibration_bank.get("arm_invariant") is not (
                    arm.get("arm") != "shuffled_action_control"
                )
                or calibration_bank.get("policy")
                != "arm_specific_training_action_bank"
                or not targeted_hash
                or not control_hash
                or calibration_hash != training_hash
                or (training_hash != targeted_hash)
                != (arm.get("arm") == "shuffled_action_control")
                or calibration_bank.get(
                    "calibration_occurs_after_admission_and_shuffle"
                ) is not True
                or arm.get("contracts", {}).get(
                    "causal_arm_training_action_bank_hash_contract"
                ) is not True
                or arm.get("contracts", {}).get(
                    "each_arm_calibrated_on_actual_post_shuffle_training_bank"
                ) is not True
                or arm.get("contracts", {}).get(
                    "action_admission_precedes_tensor_subset_reindex_and_shuffle"
                ) is not True
                or arm.get("corrective_admission", {}).get("mode") != "strict_top1"
                or arm.get("corrective_admission", {}).get(
                    "one_best_query_compression_used"
                ) is not False
                or arm.get("corrective_query_materialization", {}).get(
                    "exact_action_id_set_preserved"
                ) is not True
                or arm.get("corrective_query_materialization", {}).get(
                    "one_boundary_example_per_query"
                ) is not True
                or arm.get("contracts", {}).get(
                    "selected_corrective_action_ids_exactly_preserved_in_query_blocks"
                ) is not True
                or admitted.get("strategy")
                != "admission_then_exact_tensor_subset_then_reindex_before_shuffle"
                or admitted.get("admitted_rows") != 297362
                or admitted.get("admitted_rows_by_supervision")
                != expected_admitted_counts
                or not admitted.get("admitted_action_ids_sha256")
                or not admitted.get("admitted_semantic_boundary_sha256")
                or admitted.get(
                    "shuffle_donor_pool_equals_optimizer_admitted_panel"
                ) is not True
                or admitted.get(
                    "rejected_corrective_rows_can_be_shuffle_donors"
                ) is not False
                or arm.get("action_forward_memory_contract", {}).get(
                    "gate_passed"
                ) is not True
                or graph_scope != {
                    "all_queries": 83619,
                    "outer_train_queries": 65286,
                    "outer_held_queries": 18333,
                    "evaluated_outer_held_queries": 18333,
                }
                or (
                    arm.get("arm") == "shuffled_action_control"
                    and (
                        arm.get("action_control", {}).get(
                            "donors_restricted_to_optimizer_admitted_panel"
                        ) is not True
                        or arm.get("action_control", {}).get(
                            "all_nonfallback_donors_inside_input_panel"
                        ) is not True
                        or not arm.get("action_control", {}).get(
                            "donor_tensor_index_sha256"
                        )
                    )
                )
            ):
                raise RuntimeError(f"best-action v6 arm contract failed: {name}")
        for key, source in (
            ("targeted_action_bank_sha256", "calibration_action_bank"),
            ("control_action_bank_sha256", "calibration_action_bank"),
            ("admitted_action_ids_sha256", "admitted_action_bank"),
            ("admitted_semantic_boundary_sha256", "admitted_action_bank"),
        ):
            if len({arm[source][key] for arm in arms.values()}) != 1:
                raise RuntimeError(f"best-action v6 arms differ in admitted asset: {key}")
        admission = [
            json.dumps(arm["corrective_admission"], sort_keys=True)
            for arm in arms.values()
        ]
        if len(set(admission)) != 1:
            raise RuntimeError("best-action v6 arms differ in corrective admission")
        for name, arm in (("routed", routed), ("shuffled", shuffled)):
            calibration = arm.get("gradient_calibration", {})
            if (
                calibration.get("dense_corrective_to_risk_target_exactly_reached")
                is not True
                or calibration.get("action_to_risk_scale_hit_cap") is not False
                or any(calibration.get("branch_scale_hit_cap", {}).values())
                or not np.isclose(
                    float(calibration.get(
                        "effective_dense_corrective_to_risk_gradient_ratio", np.nan,
                    )),
                    float(arm["configuration"][
                        "target_dense_corrective_to_risk_ratio"
                    ]),
                    rtol=1e-10,
                    atol=1e-12,
                )
            ):
                raise RuntimeError(
                    f"best-action v6 {name} arm did not independently hit its "
                    "registered realized gradient budget"
                )
    training_seeds = {
        int(arm["configuration"].get("seed", -1)) for arm in arms.values()
    }
    if training_seeds != {int(args.seed)}:
        raise RuntimeError(
            "v3 arm training seed does not match the registered summary seed: "
            f"observed={sorted(training_seeds)} expected={args.seed}"
        )
    initial_ledger_columns = [
        "query_index", "query_formula", "near", "initial_E8_rank",
        "initial_E8_reciprocal_rank", "initial_E8_macro_query_auc",
        "initial_E8_macro_query_auprc",
        "initial_E8_positive_vs_best_negative_margin",
        "initial_E8_top1_top2_gap", "initial_E8_signed_top1_top2_gap",
    ]
    query_ledgers = [
        table[initial_ledger_columns].to_dict("list") for table in tables.values()
    ]
    if not all(value == query_ledgers[0] for value in query_ledgers[1:]):
        raise RuntimeError("v3 arms do not share one initial held-query ledger")

    held = routed["evaluation"]["formal_held_graph"]
    initial = held["initial_E8"]
    candidate = held["candidate"]
    metric_checks = _metric_checks(initial, candidate)
    initial_rank = routed_table.initial_E8_rank.to_numpy(np.int64)
    routed_rank = routed_table.candidate_rank.to_numpy(np.int64)
    shuffled_rank = shuffled_table.candidate_rank.to_numpy(np.int64)
    clean_rank = clean_table.candidate_rank.to_numpy(np.int64)
    formulas = routed_table.query_formula.to_numpy(str)
    near = routed_table.near.to_numpy(bool)
    candidate_ci = _formula_rank_ci(
        initial_rank, routed_rank, formulas, 1,
        args.bootstrap_resamples, args.seed, args.formula_ci_familywise_hypotheses,
    )
    candidate_near_ci = _formula_rank_ci(
        initial_rank[near], routed_rank[near], formulas[near], 1,
        args.bootstrap_resamples, args.seed + 1,
        args.formula_ci_familywise_hypotheses,
    )
    routed_vs_shuffled_ci = _formula_rank_ci(
        shuffled_rank, routed_rank, formulas, 1,
        args.bootstrap_resamples, args.seed + 2,
        args.formula_ci_familywise_hypotheses,
    )
    routed_vs_clean_ci = _formula_rank_ci(
        clean_rank, routed_rank, formulas, 1,
        args.bootstrap_resamples, args.seed + 3,
        args.formula_ci_familywise_hypotheses,
    )
    outcomes = held["candidate_vs_initial_E8"]
    near_corrected = int(np.sum(near & (initial_rank > 1) & (routed_rank == 1)))
    near_introduced = int(np.sum(near & (initial_rank == 1) & (routed_rank > 1)))
    risk_checks = {
        "corrected_exceeds_introduced": int(outcomes["corrected"]) > int(outcomes["introduced"]),
        "risk_net_lambda2_positive": int(outcomes["risk_net_lambda2"]) > 0,
        "near_corrected_exceeds_introduced": near_corrected > near_introduced,
        "near_risk_net_lambda2_positive": near_corrected - 2 * near_introduced > 0,
    }
    routed_norm = float(
        routed["gradient_calibration"][
            "effective_dense_corrective_to_risk_gradient_ratio"
        ]
    )
    shuffled_norm = float(
        shuffled["gradient_calibration"][
            "effective_dense_corrective_to_risk_gradient_ratio"
        ]
    )
    target_norm = float(
        routed["configuration"]["target_dense_corrective_to_risk_ratio"]
    )
    control_checks = {
        "routed_norm_matches_target": bool(np.isclose(routed_norm, target_norm, rtol=0.05)),
        "shuffled_norm_matches_target": bool(np.isclose(shuffled_norm, target_norm, rtol=0.05)),
        "shuffled_cross_query_fraction_ge_95pct": float(
            shuffled["action_control"].get("cross_query_fraction", 0.0)
        ) >= 0.95,
        "shuffled_matches_supervision_source_family_exact_recipe": (
            shuffled["contracts"].get(
                "shuffled_control_matches_supervision_source_family_and_exact_recipe"
            ) is True
            and shuffled["action_control"].get("strategy")
            == "supervision_source_family_exact_recipe_matched_cross_query_cyclic_shuffle"
            and shuffled["action_control"].get(
                "dose_and_action_semantics_preserved"
            ) is True
            and shuffled["action_control"].get(
                "family_semantics_preserved"
            ) is True
        ),
        "routed_beats_shuffled_recall1": float(routed_vs_shuffled_ci["delta_pp"]) > 0,
        "routed_beats_clean_recall1": float(routed_vs_clean_ci["delta_pp"]) > 0,
        "routed_vs_shuffled_formula_ci_positive": float(
            routed_vs_shuffled_ci["ci_low_pp"]
        ) > 0,
        "routed_vs_clean_formula_ci_positive": float(routed_vs_clean_ci["ci_low_pp"]) > 0,
    }
    restored_optimizer_contract = direct_contract in RESTORED_OPTIMIZER_CONTRACTS
    corrective_restored_contract = (
        direct_contract == "best_action_v7_corrective_restored"
    )
    optimizer_sampling = (
        "every_action_active_optimizer_step"
        if restored_optimizer_contract
        else "uniform_epoch_step_positions_including_endpoints"
    )
    minimum_optimizer_observations = 4 * 3508 if restored_optimizer_contract else 64
    schedule_checks = {
        "registered_full_dose_recycling": all(
            arm["schedule"].get("corrective_recycle_full_dose") is True
            for arm in arms.values()
        ),
        "actual_scheduled_corrective_mechanism_epoch_mass_equalized": all(
            arm["schedule"].get(
                "actual_scheduled_corrective_mechanism_epoch_mass_equalized"
            ) is True
            for arm in arms.values()
        ),
        "registered_formal_direct_configuration_verified": all(
            arm["contracts"].get(
                (
                    "registered_formal_best_action_v6_configuration_verified"
                    if direct_contract in BEST_ACTION_CONTRACTS
                    else "registered_formal_v3_configuration_verified"
                )
            ) is True
            and arm["contracts"].get(
                "formal_query_truncation_disabled"
            ) is True
            and arm["contracts"].get(
                "near_corrected_introduced_risk_reported"
            ) is True
            for arm in arms.values()
        ),
        "post_shuffle_actual_bank_calibration": (
            all(
                arm.get("calibration_action_bank", {}).get("policy")
                == "arm_specific_training_action_bank"
                and arm.get("calibration_action_bank", {}).get(
                    "calibration_action_bank_sha256"
                ) == arm.get("calibration_action_bank", {}).get(
                    "training_action_bank_sha256"
                )
                and arm["contracts"].get(
                    "each_arm_calibrated_on_actual_post_shuffle_training_bank"
                ) is True
                for arm in arms.values()
            )
            if direct_contract in BEST_ACTION_CONTRACTS else True
        ),
        "strict_top1_multi_action_admission": (
            all(
                arm.get("corrective_admission", {}).get("mode") == "strict_top1"
                and arm.get("corrective_admission", {}).get(
                    "one_best_query_compression_used"
                ) is False
                and arm["contracts"].get("strict_top1_corrective_admission") is True
                for arm in arms.values()
            )
            if direct_contract in BEST_ACTION_CONTRACTS else True
        ),
        "best_action_v6_cap_safe_871_to_877_schedule_verified": (
            all(
                arm.get("schedule", {}).get("original_corrective_batches") == 871
                and arm.get("schedule", {}).get("cap_safe_corrective_batches") == 877
                and arm.get("schedule", {}).get("required_optimizer_steps") == 3508
                and np.isclose(
                    float(arm.get("schedule", {}).get(
                        "effective_corrective_recycle_factor", np.nan,
                    )),
                    4.0,
                    rtol=1e-12,
                    atol=1e-12,
                )
                and arm.get("schedule", {}).get(
                    "schedule_geometry_matches_pre_model_preflight"
                ) is True
                and arm.get("contracts", {}).get(
                    "best_action_v6_exact_877_batch_schedule_verified"
                ) is True
                for arm in arms.values()
            )
            if direct_contract in BEST_ACTION_CONTRACTS else True
        ),
        "source_manifest_exact_graph_alignment": all(
            arm["contracts"].get("source_manifest_exact_graph_alignment") is True
            and arm["contracts"].get(
                "route_and_training_formula_fold_seed_exactly_aligned"
            ) is True
            for arm in arms.values()
        ),
        "protective_accumulation_at_most_four": all(
            int(arm["schedule"]["maximum_protective_microbatches_per_step"]) <= 4
            for arm in arms.values()
        ),
        "auxiliary_accumulation_at_most_registered_four": all(
            int(arm["schedule"].get(
                "maximum_auxiliary_microbatches_per_step", 5,
            )) <= 4
            and int(arm["schedule"].get(
                "configured_maximum_auxiliary_microbatches_per_step", -1,
            )) == 4
            and arm["schedule"].get(
                "dense_auxiliary_epoch_mass_capped_at_one_batch_per_step"
            ) is True
            for arm in arms.values()
        ),
        "corrective_recycle_at_most_four": all(
            float(arm["schedule"]["maximum_corrective_action_exposure_per_epoch"]) <= 4
            for arm in arms.values()
        ),
        "active_arm_signal_gates_pass": all(
            arm["signal_transmission"]["passed"] is True
            and arm["signal_transmission"]["clip_gate_passed"] is True
            and arm["signal_transmission"]["optimizer_action_alignment_gate_passed"] is True
            for arm in (routed, shuffled)
        ),
        "active_arm_internal_combination_gates_pass": all(
            arm["gradient_calibration"]["internal_action_combination_gate_passed"] is True
            for arm in (routed, shuffled)
        ),
        "inner_semantic_projection_preserves_corrective": all(
            arm["gradient_calibration"].get(
                "corrective_direction_preserved_through_inner_semantic_projection"
            ) is True
            and arm["schedule"].get(
                "corrective_direction_preserved_through_inner_semantic_projection"
            ) is True
            and arm["contracts"].get(
                "auxiliary_projected_against_corrective_before_fullgraph_risk"
            ) is True
            for arm in arms.values()
        ),
        "active_arm_calibration_caps_not_silently_truncated": all(
            arm["contracts"].get(
                "action_calibration_caps_not_silently_truncated"
            ) is True
            and arm["gradient_calibration"].get(
                "action_to_risk_scale_hit_cap"
            ) is False
            and not any(
                arm["gradient_calibration"].get(
                    "branch_scale_hit_cap", {}
                ).values()
            )
            and arm["gradient_calibration"].get(
                "dense_corrective_to_risk_target_exactly_reached"
            ) is True
            for arm in (routed, shuffled)
        ),
        "routed_semantic_transfer_gate_pass": (
            routed["gradient_calibration"].get(
                "active_transfer_fraction_gate_applicable"
            ) is True
            and routed["gradient_calibration"].get(
                "active_transfer_fraction_gate_passed"
            ) is True
        ),
        "routed_each_mechanism_transfer_gate_pass": (
            routed["gradient_calibration"].get(
                "mechanism_active_transfer_gate_applicable"
            ) is True
            and routed["gradient_calibration"].get(
                "mechanism_active_transfer_gate_passed"
            ) is True
            and bool(routed["gradient_calibration"].get(
                "corrective_mechanism_active_fraction"
            ))
        ),
        "shuffled_semantic_transfer_is_measured_not_forced": (
            shuffled["gradient_calibration"].get(
                "active_transfer_fraction_gate_applicable"
            ) is False
            and shuffled["gradient_calibration"].get(
                "active_transfer_fraction_gate_passed"
            ) is None
            and bool(np.isfinite(float(
                shuffled["gradient_calibration"].get(
                    "corrective_semantic_edge_fraction_median", {}
                ).get("active_transfer_fraction", np.nan)
            )))
        ),
        "shuffled_each_mechanism_transfer_is_measured_not_forced": (
            shuffled["gradient_calibration"].get(
                "mechanism_active_transfer_gate_applicable"
            ) is False
            and shuffled["gradient_calibration"].get(
                "mechanism_active_transfer_gate_passed"
            ) is None
            and bool(shuffled["gradient_calibration"].get(
                "corrective_mechanism_active_fraction"
            ))
        ),
        "control_semantics_and_global_mechanism_mass_are_explicit": all(
            arm["contracts"].get("control_semantics_explicit_and_separate") is True
            and arm["contracts"].get(
                "corrective_N_P_A4_global_epoch_mass_equalized_across_queries"
            ) is True
            and arm["contracts"].get(
                "P_repeated_generation_source_does_not_multiply_family_dose"
            ) is True
            and arm["contracts"].get(
                "selector_balances_source_then_family_before_recipe_multiplicity"
            ) is True
            and arm["contracts"].get(
                "robust_harmful_use_query_local_not_sparse_global_equalization"
            ) is True
            and arm["contracts"].get(
                "sparse_auxiliary_batches_evenly_interleaved"
            ) is True
            and arm["contracts"].get(
                "dense_auxiliary_panels_cannot_multiply_calibrated_step_mass"
            ) is True
            and arm["contracts"].get(
                "auxiliary_projected_against_corrective_before_fullgraph_risk"
            ) is True
            and arm["contracts"].get(
                "corrective_direction_preserved_through_inner_semantic_projection"
            ) is True
            and arm["contracts"].get(
                "semantic_action_branches_have_separate_gradient_ledgers"
            ) is True
            and arm.get("gradient_calibration", {}).get(
                "corrective_robust_harmful_protective_gradients_separately_calibrated"
            ) is True
            and arm.get("schedule", {}).get(
                "corrective_robust_harmful_protective_gradients_separately_recorded"
            ) is True
            and arm.get("schedule", {}).get(
                "each_action_branch_vs_protective_cosine_recorded"
            ) is True
            and arm.get("gradient_calibration", {}).get(
                "corrective_direction_preserved_through_inner_semantic_projection"
            ) is True
            and arm.get("schedule", {}).get(
                "corrective_direction_preserved_through_inner_semantic_projection"
            ) is True
            and arm["contracts"].get(
                "global_mechanism_coefficient_cap_passed"
            ) is True
            and arm["contracts"].get(
                "corrective_scale_uses_dense_corrective_branches_only"
            ) is True
            and arm["contracts"].get(
                "sparse_auxiliary_branches_excluded_from_corrective_scale_denominator"
            ) is True
            and arm["contracts"].get(
                "training_epoch_transfer_edge_strata_are_pooled_not_batch_averaged"
            ) is True
            and arm.get("schedule", {}).get(
                "robust_harmful_single_exposure_bounded_scale"
            ) is True
            and arm.get("schedule", {}).get(
                "dense_auxiliary_epoch_mass_capped_at_one_batch_per_step"
            ) is True
            and arm.get("schedule", {}).get(
                "maximum_auxiliary_microbatches_per_step", float("inf")
            ) <= arm.get("schedule", {}).get(
                "configured_maximum_auxiliary_microbatches_per_step", -1
            )
            and arm.get("schedule", {}).get(
                "sparse_auxiliary_global_duty_compensation_applied"
            ) is False
            and arm.get("schedule", {}).get(
                "robust_harmful_evenly_interleaved_across_epoch"
            ) is True
            and arm.get("schedule", {}).get(
                "robust_harmful_avoidable_step_overlap"
            ) is False
            and arm.get("schedule", {}).get(
                "partial_batch_query_mass_scaled_to_registered_size"
            ) is True
            and arm.get("schedule", {}).get(
                "protective_microbatches_have_equal_epoch_weight"
            ) is True
            and arm.get("schedule", {}).get(
                "clean_control_skips_no_gradient_action_forwards"
            ) is True
            and arm.get("schedule", {}).get(
                "active_arm_epoch_transfer_edge_strata_recorded"
            ) is True
            and arm.get("gradient_calibration", {}).get(
                "corrective_scale_uses_dense_corrective_branches_only"
            ) is True
            and arm.get("gradient_calibration", {}).get(
                "sparse_auxiliary_branches_excluded_from_corrective_scale_denominator"
            ) is True
            and arm.get("mechanism_balance", {}).get(
                "query_or_action_dropped_for_balance"
            ) is False
            and arm.get("mechanism_balance", {}).get(
                "coefficient_cap_passed"
            ) is True
            and arm.get("mechanism_balance", {}).get(
                "corrective", {}
            ).get("balanced") is True
            and arm.get("mechanism_balance", {}).get("strategy", {}) == {
                "corrective": "global_identity_weighted_inverse_mechanism_incidence",
                "robust": "query_local_mechanism_mean",
                "harmful": "query_local_mechanism_mean",
            }
            for arm in arms.values()
        ),
        "all_available_mechanisms_enter_calibration_prefix": all(
            arm["gradient_calibration"].get(
                "mechanism_prioritized_formula_diverse_prefix"
            ) is True
            and all(
                values.get("all_available_present") is True
                for values in arm["gradient_calibration"].get(
                    "calibration_mechanism_coverage", {}
                ).values()
            )
            and bool(arm["gradient_calibration"].get(
                "calibration_mechanism_coverage"
            ))
            for arm in arms.values()
        ),
        "action_conditioned_candidate_boundaries_consumed": all(
            arm["contracts"].get(
                "action_and_control_candidate_switch_molecules_retained"
            ) is True
            and arm["contracts"].get(
                "action_control_hard_rows_survive_epoch_refresh"
            ) is True
            and arm["contracts"].get("clean_action_control_topk_edge_union") is True
            and arm["contracts"].get(
                "E8_symmetric_live_action_consistency_restored"
            ) is True
            and arm["contracts"].get(
                "mature_E8_gradient_clip_restored"
            ) is True
            and arm["action_conditioned_boundary"].get("enabled") is True
            and all(
                arm["action_conditioned_boundary"][panel].get(
                    "all_selected_action_control_boundaries_retained"
                ) is True
                and arm["action_conditioned_boundary"][panel].get(
                    "routed_hard_rows_never_truncated_within_molecule"
                ) is True
                and arm["action_conditioned_boundary"][panel].get(
                    "candidate_boundary_cap_truncation"
                ) is False
                for panel in ("corrective", "robust", "harmful")
            )
            for arm in arms.values()
        ),
        "formula_stratified_calibration_consumed": all(
            arm["contracts"].get("formula_stratified_gradient_calibration") is True
            and arm["gradient_calibration"].get(
                "formula_stratified_without_replacement"
            ) is True
            for arm in arms.values()
        ),
        "bounded_memory_action_backprop_consumed": all(
            arm["contracts"].get(
                "bounded_memory_sequential_action_branch_backprop"
            ) is True
            and arm["schedule"].get(
                "simultaneous_corrective_robust_harmful_graph_retention"
            ) is False
            and arm["gradient_calibration"].get(
                "simultaneous_corrective_robust_harmful_risk_graph_retention"
            ) is False
            for arm in arms.values()
        ),
        "active_arm_optimizer_counterfactual_attribution_valid": all(
            arm["contracts"].get(
                "optimizer_counterfactual_action_attribution_measured"
            ) is True
            and (
                arm["contracts"].get(
                    "optimizer_update_restoration_materialized_every_active_step"
                ) is True
                if restored_optimizer_contract
                else arm["contracts"].get(
                    "optimizer_parameter_clones_limited_to_registered_audit_steps"
                ) is True
            )
            and arm["contracts"].get(
                "head_and_backbone_action_signal_measured_separately"
            ) is True
            and arm["contracts"].get(
                "corrective_direction_traced_through_risk_clip_and_adamw"
            ) is True
            and arm["signal_transmission"].get(
                "optimizer_counterfactual_virtual_step_gate_passed"
            ) is True
            and float(arm["signal_transmission"].get(
                "optimizer_counterfactual_virtual_step_max_relative_error",
                np.inf,
            )) <= 1e-3
            and int(arm["signal_transmission"].get(
                "optimizer_counterfactual_attribution_steps", 0,
            )) >= minimum_optimizer_observations
            and arm["signal_transmission"].get(
                "optimizer_counterfactual_attribution_sampling"
            ) == optimizer_sampling
            and int(arm["signal_transmission"].get(
                "optimizer_action_alignment_steps", 0,
            )) >= minimum_optimizer_observations
            and arm["signal_transmission"].get(
                "optimizer_action_alignment_sampling"
            ) == optimizer_sampling
            and arm["signal_transmission"].get(
                "parameter_group_action_signal_gate_passed"
            ) is True
            and arm["signal_transmission"].get(
                "parameter_group_action_signal_sampling"
            ) == optimizer_sampling
            and set(arm["signal_transmission"].get(
                "parameter_group_action_signal", {}
            )) == {"head", "backbone"}
            and all(
                int(values.get("observations", 0)) >= minimum_optimizer_observations
                and values.get(
                    "action_gradient_reaches_group_on_every_audit_step"
                ) is True
                and float(values.get(
                    "corrective_direction_retention_after_risk_and_clip_p10",
                    -np.inf,
                )) >= float(arm["configuration"].get(
                    "minimum_corrective_direction_retention_p10", np.inf,
                ))
                and float(values.get(
                    "optimizer_action_attributable_corrective_alignment_p10",
                    -np.inf,
                )) >= float(arm["configuration"].get(
                    "minimum_optimizer_action_alignment_p10", np.inf,
                ))
                and values.get("gate_passed") is True
                for values in arm["signal_transmission"].get(
                    "parameter_group_action_signal", {}
                ).values()
            )
            and np.isfinite(float(arm["signal_transmission"].get(
                "optimizer_action_attributable_update_fraction_p10", np.nan,
            )))
            and arm["signal_transmission"].get(
                "optimizer_action_attributable_update_fraction_gate_passed"
            ) is True
            and float(arm["signal_transmission"].get(
                "optimizer_action_attributable_update_fraction_p10", -np.inf,
            )) >= float(arm["configuration"].get(
                "minimum_optimizer_action_attributable_fraction_p10", np.inf,
            ))
            and np.isfinite(float(arm["signal_transmission"].get(
                "optimizer_action_attributable_alignment_p10", np.nan,
            )))
            and arm["signal_transmission"].get(
                "optimizer_action_attributable_alignment_gate_passed"
            ) is True
            and float(arm["signal_transmission"].get(
                "optimizer_action_attributable_alignment_p10", -np.inf,
            )) >= float(arm["configuration"].get(
                "minimum_optimizer_action_alignment_p10", np.inf,
            ))
            and arm["signal_transmission"].get(
                "corrective_direction_retention_gate_passed"
            ) is True
            and float(arm["signal_transmission"].get(
                "corrective_direction_retention_after_risk_and_clip_p10", -np.inf,
            )) >= float(arm["configuration"].get(
                "minimum_corrective_direction_retention_p10", np.inf,
            ))
            and arm["signal_transmission"].get(
                "optimizer_action_attributable_corrective_alignment_gate_passed"
            ) is True
            and float(arm["signal_transmission"].get(
                "optimizer_action_attributable_corrective_alignment_p10", -np.inf,
            )) >= float(arm["configuration"].get(
                "minimum_optimizer_action_alignment_p10", np.inf,
            ))
            and arm["signal_transmission"].get(
                "legacy_90pct_end_to_end_loss_reproduced"
            ) is False
            for arm in (routed, shuffled)
        ),
        "optimizer_update_restoration_contract_valid": (
            all(
                arm["configuration"].get(
                    "materialize_optimizer_update_restoration"
                ) is True
                and arm["configuration"].get(
                    "continue_after_signal_gate_failure"
                ) is True
                and arm["contracts"].get(
                    (
                        "registered_formal_best_action_v7_corrective_restored_"
                        "configuration_verified"
                        if corrective_restored_contract
                        else "registered_formal_best_action_v6_restored_"
                        "configuration_verified"
                    )
                ) is True
                and arm["signal_transmission"].get(
                    "optimizer_update_restoration_materialized_every_active_step"
                ) is True
                and arm["signal_transmission"].get(
                    "signal_gate_failure"
                ) is False
                and float(arm["signal_transmission"].get(
                    "optimizer_update_restoration_minimum_observed_group_risk_retention",
                    -np.inf,
                )) >= float(arm["configuration"].get(
                    "optimizer_restoration_minimum_risk_component_retention",
                    np.inf,
                )) - 1e-10
                and (
                    not corrective_restored_contract
                    or (
                        arm["signal_transmission"].get(
                            "optimizer_update_restoration_target_coverage_gate_passed"
                        ) is True
                        and float(arm["signal_transmission"].get(
                            "optimizer_update_restoration_all_group_targets_reached_fraction",
                            -np.inf,
                        )) >= float(arm["configuration"].get(
                            "minimum_optimizer_restoration_target_reached_fraction",
                            np.inf,
                        )) - 1e-10
                    )
                )
                and (
                    arm["signal_transmission"].get(
                        "optimizer_update_restoration_scope"
                    ) == "corrective_only"
                    and arm["signal_transmission"].get(
                        "optimizer_update_restoration_corrective_only"
                    ) is True
                    and arm["signal_transmission"].get(
                        "optimizer_update_restoration_noncorrective_baseline"
                    ) == "protective_plus_projected_robust_harmful"
                    and arm["signal_transmission"].get(
                        "protective_gradient_reaches_every_parameter_on_every_active_step"
                    ) is True
                    and arm["signal_transmission"].get(
                        "restored_adamw_first_moment_reconciled_every_active_step"
                    ) is True
                    and float(arm["signal_transmission"].get(
                        "restored_adamw_first_moment_max_reconstruction_relative_error",
                        np.inf,
                    )) <= 1e-6
                    and np.isfinite(float(arm["signal_transmission"].get(
                        "restored_adamw_max_materialized_update_relative_error",
                        np.nan,
                    )))
                    and np.isfinite(float(arm["signal_transmission"].get(
                        "restored_adamw_max_fp32_parameter_replay_relative_error",
                        np.nan,
                    )))
                    and arm["signal_transmission"].get(
                        "restored_adamw_reconstruction_target"
                    ) == "realized_materialized_parameter_displacement"
                    and arm["signal_transmission"].get(
                        "restored_adamw_verification_arithmetic"
                    ) == "stable_decay_plus_adaptive_displacement"
                    and arm["signal_transmission"].get(
                        "restored_adamw_second_moment_source"
                    ) == "actual_combined_gradient"
                    and arm["contracts"].get(
                        "corrective_only_optimizer_residual_restored"
                    ) is True
                    and arm["contracts"].get(
                        "restored_adamw_first_moment_reconciled"
                    ) is True
                    if corrective_restored_contract else True
                )
                for arm in (routed, shuffled)
            )
            and clean["contracts"].get(
                "optimizer_update_restoration_materialized_every_active_step"
            ) is True
            and clean["signal_transmission"].get(
                "optimizer_update_restoration_clean_control_noop"
            ) is True
            if restored_optimizer_contract else True
        ),
    }
    target_checks = {
        "recall1_gain_at_least_threshold": float(candidate_ci["delta_pp"]) >= args.minimum_delta_recall1_pp,
        "formula_cluster_ci_strict_positive": float(candidate_ci["ci_low_pp"]) > 0,
        "near_formula_cluster_ci_strict_positive": float(candidate_near_ci["ci_low_pp"]) > 0,
        "all_required_metrics_improve_or_allowed_nonworse": all(metric_checks.values()),
        "risk_checks_pass": all(risk_checks.values()),
        "matched_controls_pass": all(control_checks.values()),
        "registered_schedule_and_signal_gates_pass": all(schedule_checks.values()),
    }
    paired = pd.DataFrame({
        "query_index": routed_table.query_index,
        "query_formula": formulas,
        "near": near,
        "initial_rank": initial_rank,
        "routed_rank": routed_rank,
        "shuffled_rank": shuffled_rank,
        "clean_rank": clean_rank,
    })
    single_seed_gate_pass = bool(all(target_checks.values()))
    report: dict[str, object] = {
        "status": (
            "noise_corrected_best_action_v7_arm_summary_complete"
            if direct_contract == "best_action_v7_corrective_restored"
            else (
                "noise_corrected_best_action_v6_arm_summary_complete"
                if direct_contract in BEST_ACTION_CONTRACTS
                else "noise_corrected_direct_v3_arm_summary_complete"
            )
        ),
        "direct_contract": direct_contract,
        "training_seed": int(args.seed),
        "minimum_delta_recall1_pp": args.minimum_delta_recall1_pp,
        "bootstrap_resamples": int(args.bootstrap_resamples),
        "formula_ci_familywise_hypotheses": args.formula_ci_familywise_hypotheses,
        "candidate_vs_initial": {
            "formula_cluster_recall1": candidate_ci,
            "near_formula_cluster_recall1": candidate_near_ci,
            "metric_checks": metric_checks,
            "risk_checks": risk_checks,
            "near_risk_counts": {
                "corrected": near_corrected,
                "introduced": near_introduced,
                "risk_net_lambda2": near_corrected - 2 * near_introduced,
            },
            "full_metrics": candidate,
            "initial_metrics": initial,
        },
        "causal_controls": {
            "routed_vs_shuffled_formula_cluster_recall1": routed_vs_shuffled_ci,
            "routed_vs_clean_formula_cluster_recall1": routed_vs_clean_ci,
            "routed_effective_dense_corrective_to_risk_gradient_ratio": routed_norm,
            "shuffled_effective_dense_corrective_to_risk_gradient_ratio": shuffled_norm,
            "target_dense_corrective_to_risk_gradient_ratio": target_norm,
            "checks": control_checks,
        },
        "schedule_and_signal_checks": schedule_checks,
        "promotion_checks": target_checks,
        "single_seed_gate_pass": single_seed_gate_pass,
        # Compatibility field retained for older readers.  This script only
        # sees one seed, so it is never authorised to make a final promotion.
        "promote": False,
        "promotion_status": (
            "pending_multiseed" if single_seed_gate_pass else "single_seed_gate_failed"
        ),
        "promote_compatibility": {
            "field_semantics": "final_registered_multiseed_promotion_only",
            "single_seed_summary_can_promote": False,
        },
        "claim_limit": (
            "One outer-formula fold and one seed; promotion still requires the registered multi-seed gate."
        ),
        "provenance": {
            "shared_training_actions_sha256": routed["provenance"]["training_actions_sha256"],
            "shared_candidate_graph_sha256": routed["provenance"]["candidate_graph_sha256"],
            "shared_source_manifest_sha256": routed["provenance"]["source_manifest_sha256"],
            "shared_initial_student_checkpoint_sha256": routed["provenance"][
                "initial_student_checkpoint_sha256"
            ],
            "shared_official_checkpoint_sha256": routed["provenance"][
                "official_checkpoint_sha256"
            ],
            "shared_initial_student_decision_sha256": routed["provenance"].get(
                "initial_student_decision_sha256"
            ),
            "routed_held_per_query_sha256": sha256_file(
                args.routed_dir / "held_per_query.csv.gz"
            ),
            "shuffled_held_per_query_sha256": sha256_file(
                args.shuffled_dir / "held_per_query.csv.gz"
            ),
            "clean_held_per_query_sha256": sha256_file(
                args.clean_dir / "held_per_query.csv.gz"
            ),
            "held_per_query_rows_per_arm": {
                name: int(len(table)) for name, table in tables.items()
            },
            "routed_decision_sha256": sha256_file(args.routed_dir / "decision.json"),
            "shuffled_decision_sha256": sha256_file(args.shuffled_dir / "decision.json"),
            "clean_decision_sha256": sha256_file(args.clean_dir / "decision.json"),
            "routed_final_shared_encoder_sha256": routed["provenance"][
                "final_shared_encoder_sha256"
            ],
            "routed_final_model_state_sha256": routed["provenance"][
                "final_model_state_sha256"
            ],
            "shuffled_final_model_state_sha256": shuffled["provenance"][
                "final_model_state_sha256"
            ],
            "clean_final_model_state_sha256": clean["provenance"][
                "final_model_state_sha256"
            ],
            "script_sha256": sha256_file(Path(__file__)),
        },
    }
    report = _json_native(report)
    # Fail here (and in the lightweight contract tests), before any atomic
    # output publication, if a future report field introduces an unsupported
    # transport type.
    json.dumps(report)
    return report, paired


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    report, paired = summarize(args)
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{args.output_dir.name}.", dir=args.output_dir.parent,
    ))
    try:
        paired.to_csv(staging / "paired_held_queries.csv.gz", index=False, compression="gzip")
        report["provenance"]["paired_held_queries_sha256"] = sha256_file(
            staging / "paired_held_queries.csv.gz"
        )
        (staging / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
