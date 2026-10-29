#!/usr/bin/env python
"""Evaluate Full23 multisource OOF with non-duplicated scientific endpoints.

The multisource trainer emits panel-level candidate scores.  This companion
evaluator adds three views without changing or refitting the model:

1. a source-specific primary endpoint in which a query occurring in both the
   identity and formula panels has total mass one;
2. a GNPS identity-panel sensitivity analysis collapsed by query IK14; and
3. panel-level paired Full23-v1 comparisons including formula-cluster CIs,
   MRR, near-query performance, and macro/micro AUC/AUPRC.

Enveda query keys come from the released OOF ledger.  GNPS query keys come
from the frozen candidate-score files and IK14 values come from the aligned
GNPS axis ledger.  No identifier is inferred from array position.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from gnps_pair_score_cache import sha256_file
from train_enveda_unified_candidate_integrator import (
    PANELS,
    PanelData,
    _metric_summary,
    _strict_ranks,
)
from train_full23_centered_residual_from_frozen_ledger import PARENT
from train_full23_multisource_additive import METHOD as DEFAULT_METHOD


SOURCE_PREFIX = {
    "enveda": {name: name + "__" for name in PANELS},
    "gnps": {
        "identity_disjoint": "identity__",
        "formula_disjoint": "formula__",
    },
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-output", "--multisource-output", dest="run_output", type=Path,
        required=True,
        help="Run directory containing report.json and candidate_predictions_oof.npz.",
    )
    parser.add_argument(
        "--prediction-file", type=Path,
        help="Optional prediction NPZ override; defaults to RUN_OUTPUT/candidate_predictions_oof.npz.",
    )
    parser.add_argument(
        "--run-report", type=Path,
        help="Optional run-report override; defaults to RUN_OUTPUT/report.json.",
    )
    parser.add_argument(
        "--method-name",
        help="Prediction key to evaluate; defaults to the run report's method field.",
    )
    parser.add_argument(
        "--expected-status",
        help="Optional exact run status contract. Without it, any *_COMPLETE status is accepted.",
    )
    parser.add_argument("--enveda-ledger", type=Path, required=True)
    parser.add_argument("--gnps-score-dir", type=Path, required=True)
    parser.add_argument("--gnps-axis-ledger", type=Path, required=True)
    parser.add_argument("--gnps-identity-panel", type=Path, required=True)
    parser.add_argument("--gnps-formula-panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20261029)
    return parser.parse_args()


def _formula_cluster_ci(
    delta: np.ndarray, formulas: np.ndarray, resamples: int, seed: int,
) -> dict:
    """Formula-cluster bootstrap equivalent to the canonical evaluator, O(n)."""
    values = np.asarray(delta, dtype=np.float64)
    formulas = np.asarray(formulas).astype("U")
    unique, inverse = np.unique(formulas, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(resamples, dtype=np.float64)
    # Keep the temporary index matrix below roughly one million int64 values.
    # This preserves the canonical resampling semantics while avoiding 5,000
    # Python-level loops for the large cross-panel endpoint.
    batch_size = max(1, min(64, 1_000_000 // len(unique)))
    for start in range(0, resamples, batch_size):
        stop = min(resamples, start + batch_size)
        chosen = rng.integers(0, len(unique), (stop - start, len(unique)))
        bootstrap[start:stop] = (
            np.sum(sums[chosen], axis=1) / np.sum(counts[chosen], axis=1)
        )
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    return {
        "delta": float(np.mean(values)),
        "delta_pp": float(100 * np.mean(values)),
        "ci_low": float(low),
        "ci_high": float(high),
        "ci_low_pp": float(100 * low),
        "ci_high_pp": float(100 * high),
        "formula_clusters": int(len(unique)),
        "bootstrap_resamples": int(resamples),
    }


def _query_auc(scores: np.ndarray, panel: PanelData) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    output = []
    for left, right in zip(panel.query_ptr[:-1], panel.query_ptr[1:]):
        left, right = int(left), int(right)
        local_label = panel.labels[left:right]
        positive_index = np.flatnonzero(local_label > 0)
        if len(positive_index) != 1:
            raise RuntimeError("query does not have exactly one positive")
        positive = float(values[left + int(positive_index[0])])
        negative = values[left:right][local_label == 0]
        if len(negative) == 0:
            output.append(1.0)
        else:
            output.append(float(
                np.mean(positive > negative) + 0.5 * np.mean(positive == negative)
            ))
    return np.asarray(output, dtype=np.float64)


def _metadata(
    source: str,
    panel_name: str,
    enveda: np.lib.npyio.NpzFile,
    gnps_scores: dict[str, np.lib.npyio.NpzFile],
    gnps_axis: np.lib.npyio.NpzFile,
    gnps_panels: dict[str, np.lib.npyio.NpzFile],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    if source == "enveda":
        prefix = SOURCE_PREFIX[source][panel_name]
        return (
            np.asarray(enveda[prefix + "query_keys"]).astype("U"),
            np.asarray(enveda[prefix + "near"], dtype=bool),
            np.asarray(enveda[prefix + "formula"]).astype("U"),
            None,
        )
    body = gnps_scores[panel_name]
    axis_prefix = SOURCE_PREFIX[source][panel_name]
    query_keys = np.asarray(body["query_key"]).astype("U")
    near = np.asarray(gnps_axis[axis_prefix + "near"], dtype=bool)
    formulas = np.asarray(gnps_axis[axis_prefix + "formula"]).astype("U")
    ik14 = np.asarray(gnps_axis[axis_prefix + "query_ids"]).astype("U")
    frozen_panel = gnps_panels[panel_name]
    expected_keys = np.asarray([
        f"row={int(value)}" for value in frozen_panel["query_row"]
    ], dtype="U")
    checks = {
        "query_key_to_row": np.array_equal(query_keys, expected_keys),
        "ik14": np.array_equal(
            ik14, np.asarray(frozen_panel["query_ik14"]).astype("U")
        ),
        "formula": np.array_equal(
            formulas, np.asarray(frozen_panel["query_formula"]).astype("U")
        ),
        "near": np.array_equal(
            near, np.asarray(frozen_panel["near_query"], dtype=bool)
        ),
        "query_ptr": np.array_equal(
            np.asarray(body["query_ptr"], dtype=np.int64),
            np.asarray(frozen_panel["query_ptr"], dtype=np.int64),
        ),
        "labels": np.array_equal(
            np.asarray(body["molecule_label"], dtype=np.int8),
            np.asarray(frozen_panel["molecule_label"], dtype=np.int8),
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"GNPS stable query/IK14 mapping drifted: {panel_name}: {checks}")
    if np.any(np.char.str_len(ik14) != 14):
        raise RuntimeError(f"GNPS query identity is not stable IK14: {panel_name}")
    return query_keys, near, formulas, ik14


def _load_panel(
    predictions: np.lib.npyio.NpzFile,
    source: str,
    panel_name: str,
    query_keys: np.ndarray,
    near: np.ndarray,
    metadata_formula: np.ndarray,
    enveda: np.lib.npyio.NpzFile,
    gnps_scores: dict[str, np.lib.npyio.NpzFile],
    method_name: str,
    parent_name: str,
) -> tuple[PanelData, dict[str, np.ndarray]]:
    prefix = f"{source}__{panel_name}__"
    ptr = np.asarray(predictions[prefix + "query_ptr"], dtype=np.int64)
    labels = np.asarray(predictions[prefix + "labels"], dtype=np.int8)
    formulas = np.asarray(predictions[prefix + "formula"]).astype("U")
    methods = {
        method_name: np.asarray(predictions[prefix + method_name], dtype=np.float32),
        parent_name: np.asarray(predictions[prefix + parent_name], dtype=np.float32),
    }
    if source == "enveda":
        raw_prefix = SOURCE_PREFIX[source][panel_name]
        raw_ptr = np.asarray(enveda[raw_prefix + "query_ptr"], dtype=np.int64)
        raw_labels = np.asarray(enveda[raw_prefix + "labels"], dtype=np.int8)
    else:
        body = gnps_scores[panel_name]
        raw_ptr = np.asarray(body["query_ptr"], dtype=np.int64)
        raw_labels = np.asarray(body["molecule_label"], dtype=np.int8)
    checks = {
        "query_ptr": np.array_equal(ptr, raw_ptr),
        "labels": np.array_equal(labels, raw_labels),
        "formula": np.array_equal(formulas, metadata_formula),
        "query_keys": (
            prefix + "query_keys" not in predictions.files
            or np.array_equal(
                np.asarray(predictions[prefix + "query_keys"]).astype("U"), query_keys,
            )
        ),
        "near": (
            prefix + "near" not in predictions.files
            or np.array_equal(
                np.asarray(predictions[prefix + "near"], dtype=bool), near,
            )
        ),
        "query_count": len(query_keys) == len(ptr) - 1 == len(near),
        "score_length": all(len(value) == len(labels) for value in methods.values()),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"multisource predictions drifted from {source}/{panel_name}: {checks}"
        )
    panel = PanelData(
        name=f"{source}__{panel_name}", graph=None,
        scalar=np.empty((len(labels), 0), dtype=np.float32), feature_names=(),
        labels=labels, query_ptr=ptr, formulas=formulas, query_keys=query_keys,
        near=near, query_weights=np.ones(len(near), dtype=np.float64), scores=methods,
    )
    return panel, methods


def _paired_panel_report(
    panel: PanelData, methods: dict[str, np.ndarray], method_name: str,
    parent_name: str, resamples: int, seed: int,
) -> dict:
    candidate_metrics, candidate_rank = _metric_summary(methods[method_name], panel)
    parent_metrics, parent_rank = _metric_summary(methods[parent_name], panel)
    candidate_auc = _query_auc(methods[method_name], panel)
    parent_auc = _query_auc(methods[parent_name], panel)
    candidate_ok = candidate_rank == 1
    parent_ok = parent_rank == 1
    comparison = {
        "corrected": int(np.sum(candidate_ok & ~parent_ok)),
        "introduced": int(np.sum(~candidate_ok & parent_ok)),
        "risk_net_lambda2": int(
            np.sum(candidate_ok & ~parent_ok) - 2 * np.sum(~candidate_ok & parent_ok)
        ),
        "recall1_formula_cluster_ci": _formula_cluster_ci(
            candidate_ok.astype(float) - parent_ok.astype(float),
            panel.formulas, resamples, seed,
        ),
        "mrr_delta": float(np.mean(1.0 / candidate_rank - 1.0 / parent_rank)),
        "mrr_formula_cluster_ci": _formula_cluster_ci(
            1.0 / candidate_rank - 1.0 / parent_rank,
            panel.formulas, resamples, seed + 1,
        ),
        "macro_query_auroc_delta": float(np.mean(candidate_auc - parent_auc)),
        "macro_query_auroc_formula_cluster_ci": _formula_cluster_ci(
            candidate_auc - parent_auc, panel.formulas, resamples, seed + 2,
        ),
        "micro_candidate_auroc_delta": float(
            candidate_metrics["micro_candidate_auroc"]
            - parent_metrics["micro_candidate_auroc"]
        ),
        "micro_candidate_auprc_delta": float(
            candidate_metrics["micro_candidate_auprc"]
            - parent_metrics["micro_candidate_auprc"]
        ),
    }
    if np.any(panel.near):
        selected = panel.near
        comparison["near"] = {
            "queries": int(np.sum(selected)),
            "recall1_delta": float(np.mean(
                candidate_ok[selected].astype(float) - parent_ok[selected].astype(float)
            )),
            "recall1_formula_cluster_ci": _formula_cluster_ci(
                candidate_ok[selected].astype(float) - parent_ok[selected].astype(float),
                panel.formulas[selected], resamples, seed + 3,
            ),
            "mrr_delta": float(np.mean(
                1.0 / candidate_rank[selected] - 1.0 / parent_rank[selected]
            )),
            "mrr_formula_cluster_ci": _formula_cluster_ci(
                1.0 / candidate_rank[selected] - 1.0 / parent_rank[selected],
                panel.formulas[selected], resamples, seed + 4,
            ),
        }
    else:
        comparison["near"] = {"queries": 0}
    return {
        "queries": int(len(panel.query_ptr) - 1),
        "candidate_rows": int(len(panel.labels)),
        "methods": {method_name: candidate_metrics, parent_name: parent_metrics},
        f"{method_name}_vs_{parent_name}": comparison,
    }


def _collapse_records(
    panels: dict[str, PanelData], methods: dict[str, dict[str, np.ndarray]],
    identifiers: dict[str, np.ndarray], method_name: str, parent_name: str,
) -> tuple[dict[str, dict[str, tuple[float, float, float]]], dict[str, str]]:
    records: dict[str, dict[str, list[tuple[float, float, float]]]] = {
        method: defaultdict(list) for method in (method_name, parent_name)
    }
    formula_by_id: dict[str, str] = {}
    for panel_name in panels:
        panel = panels[panel_name]
        ids = identifiers[panel_name]
        if len(ids) != len(panel.query_ptr) - 1:
            raise RuntimeError(f"identifier axis drifted: {panel.name}")
        for method in (method_name, parent_name):
            rank = _strict_ranks(methods[panel_name][method], panel)
            auc = _query_auc(methods[panel_name][method], panel)
            for identifier, formula, value, local_auc in zip(
                ids, panel.formulas, rank, auc, strict=True,
            ):
                key = str(identifier)
                previous = formula_by_id.setdefault(key, str(formula))
                if previous != str(formula):
                    raise RuntimeError(f"identifier formula drifted: {key}")
                records[method][key].append((
                    float(value == 1), float(1.0 / value), float(local_auc),
                ))
    if set(records[method_name]) != set(records[parent_name]):
        raise RuntimeError("candidate and Full23-v1 identifier coverage differs")
    collapsed = {
        method: {
            key: tuple(float(np.mean([row[index] for row in values])) for index in range(3))
            for key, values in records[method].items()
        }
        for method in (method_name, parent_name)
    }
    return collapsed, formula_by_id


def _collapsed_report(
    collapsed: dict[str, dict[str, tuple[float, float, float]]],
    formula_by_id: dict[str, str], method_name: str, parent_name: str,
    resamples: int, seed: int, semantics: str,
) -> dict:
    keys = sorted(collapsed[method_name])
    formulas = np.asarray([formula_by_id[key] for key in keys], dtype="U")
    arrays = {
        method: np.asarray([collapsed[method][key] for key in keys], dtype=np.float64)
        for method in (method_name, parent_name)
    }
    delta = arrays[method_name] - arrays[parent_name]
    return {
        "unique_units": int(len(keys)),
        "semantics": semantics,
        "methods": {
            method: {
                "top1_utility": float(np.mean(values[:, 0])),
                "mrr_utility": float(np.mean(values[:, 1])),
                "macro_query_auroc_utility": float(np.mean(values[:, 2])),
            }
            for method, values in arrays.items()
        },
        f"{method_name}_vs_{parent_name}": {
            "top1_delta": float(np.mean(delta[:, 0])),
            "top1_delta_pp": float(100 * np.mean(delta[:, 0])),
            "top1_formula_cluster_ci": _formula_cluster_ci(
                delta[:, 0], formulas, resamples, seed,
            ),
            "mrr_delta": float(np.mean(delta[:, 1])),
            "mrr_formula_cluster_ci": _formula_cluster_ci(
                delta[:, 1], formulas, resamples, seed + 1,
            ),
            "macro_query_auroc_delta": float(np.mean(delta[:, 2])),
            "macro_query_auroc_formula_cluster_ci": _formula_cluster_ci(
                delta[:, 2], formulas, resamples, seed + 2,
            ),
            "units_improved_top1_utility": int(np.sum(delta[:, 0] > 0)),
            "units_worsened_top1_utility": int(np.sum(delta[:, 0] < 0)),
            "units_unchanged_top1_utility": int(np.sum(delta[:, 0] == 0)),
        },
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.bootstrap_resamples < 100:
        raise ValueError("bootstrap-resamples must be at least 100")
    prediction_path = args.prediction_file or (
        args.run_output / "candidate_predictions_oof.npz"
    )
    report_path = args.run_report or (args.run_output / "report.json")
    if not prediction_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("multisource output lacks predictions or report")
    trainer_report = json.loads(report_path.read_text(encoding="utf-8"))
    run_status = str(trainer_report.get("status", ""))
    if args.expected_status is not None:
        if run_status != args.expected_status:
            raise RuntimeError(
                f"run status mismatch: expected {args.expected_status}, got {run_status}"
            )
    elif not run_status.endswith("_COMPLETE"):
        raise RuntimeError(f"input is not a completed Full23 run: {run_status!r}")
    method_name = args.method_name or trainer_report.get("method") or DEFAULT_METHOD
    method_name = str(method_name)
    reported_method = trainer_report.get("method")
    if reported_method is not None and str(reported_method) != method_name:
        raise RuntimeError(
            f"method-name disagrees with run report: {method_name} != {reported_method}"
        )
    parent_name = PARENT

    gnps_paths = {
        name: args.gnps_score_dir / f"candidate_scores_{name}.npz" for name in PANELS
    }
    gnps_panel_paths = {
        "identity_disjoint": args.gnps_identity_panel,
        "formula_disjoint": args.gnps_formula_panel,
    }
    for path in (
        args.enveda_ledger, args.gnps_axis_ledger, *gnps_paths.values(),
        *gnps_panel_paths.values(),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    panel_reports: dict[str, dict] = {"enveda": {}, "gnps": {}}
    source_panels: dict[str, dict[str, PanelData]] = {"enveda": {}, "gnps": {}}
    source_methods: dict[str, dict[str, dict[str, np.ndarray]]] = {
        "enveda": {}, "gnps": {},
    }
    source_keys: dict[str, dict[str, np.ndarray]] = {"enveda": {}, "gnps": {}}
    gnps_ik14: dict[str, np.ndarray] = {}

    with (
        np.load(prediction_path, allow_pickle=False) as predictions,
        np.load(args.enveda_ledger, allow_pickle=False) as enveda,
        np.load(args.gnps_axis_ledger, allow_pickle=False) as axis,
        np.load(gnps_paths["identity_disjoint"], allow_pickle=False) as gnps_identity,
        np.load(gnps_paths["formula_disjoint"], allow_pickle=False) as gnps_formula,
        np.load(gnps_panel_paths["identity_disjoint"], allow_pickle=False) as identity_panel,
        np.load(gnps_panel_paths["formula_disjoint"], allow_pickle=False) as formula_panel,
    ):
        gnps_scores = {
            "identity_disjoint": gnps_identity,
            "formula_disjoint": gnps_formula,
        }
        gnps_panels = {
            "identity_disjoint": identity_panel,
            "formula_disjoint": formula_panel,
        }
        for source_index, source in enumerate(("enveda", "gnps")):
            for panel_index, panel_name in enumerate(PANELS):
                keys, near, formulas, ik14 = _metadata(
                    source, panel_name, enveda, gnps_scores, axis, gnps_panels,
                )
                panel, methods = _load_panel(
                    predictions, source, panel_name, keys, near, formulas,
                    enveda, gnps_scores, method_name, parent_name,
                )
                source_panels[source][panel_name] = panel
                source_methods[source][panel_name] = methods
                source_keys[source][panel_name] = keys
                if ik14 is not None:
                    gnps_ik14[panel_name] = ik14
                panel_reports[source][panel_name] = _paired_panel_report(
                    panel, methods, method_name, parent_name, args.bootstrap_resamples,
                    args.seed + source_index * 10_000 + panel_index * 100,
                )

    joint_reports = {}
    for source_index, source in enumerate(("enveda", "gnps")):
        collapsed, formulas = _collapse_records(
            source_panels[source], source_methods[source], source_keys[source],
            method_name, parent_name,
        )
        overlap = len(
            set(map(str, source_keys[source]["identity_disjoint"]))
            & set(map(str, source_keys[source]["formula_disjoint"]))
        )
        joint_reports[source] = {
            "panel_occurrences": int(sum(
                len(values) for values in source_keys[source].values()
            )),
            "cross_panel_repeated_queries": int(overlap),
            **_collapsed_report(
                collapsed, formulas, method_name, parent_name, args.bootstrap_resamples,
                args.seed + 30_000 + source_index * 100,
                (
                    "Each stable source query key has total mass one; a key present in both "
                    "panels contributes the mean of its panel-specific outcomes."
                ),
            ),
        }

    identity_panel = {
        "identity_disjoint": source_panels["gnps"]["identity_disjoint"]
    }
    identity_methods = {
        "identity_disjoint": source_methods["gnps"]["identity_disjoint"]
    }
    identity_ids = {"identity_disjoint": gnps_ik14["identity_disjoint"]}
    ik14_collapsed, ik14_formulas = _collapse_records(
        identity_panel, identity_methods, identity_ids, method_name, parent_name,
    )
    ik14_report = {
        "spectrum_queries": int(len(identity_ids["identity_disjoint"])),
        **_collapsed_report(
            ik14_collapsed, ik14_formulas, method_name, parent_name,
            args.bootstrap_resamples,
            args.seed + 40_000,
            (
                "GNPS identity-panel spectra are grouped by aligned 14-character query "
                "InChIKey block; each IK14 has total mass one and its repeated-spectrum "
                "outcomes are averaged."
            ),
        ),
    }

    report = {
        "status": "FULL23_GENERIC_ENDPOINT_EVALUATION_COMPLETE",
        "evaluated_run_status": run_status,
        "method": method_name,
        "baseline": parent_name,
        "dataset_role": "consumed multisource development OOF only",
        "allow_final_external_claim": False,
        "panels": panel_reports,
        "query_balanced_primary_endpoints": joint_reports,
        "gnps_identity_ik14_collapse_sensitivity": ik14_report,
        "inputs": {
            "multisource_report": str(report_path),
            "multisource_report_sha256": sha256_file(report_path),
            "multisource_predictions": str(prediction_path),
            "multisource_predictions_sha256": sha256_file(prediction_path),
            "enveda_ledger": str(args.enveda_ledger),
            "enveda_ledger_sha256": sha256_file(args.enveda_ledger),
            "gnps_axis_ledger": str(args.gnps_axis_ledger),
            "gnps_axis_ledger_sha256": sha256_file(args.gnps_axis_ledger),
            **{
                f"gnps_{name}_scores_sha256": sha256_file(path)
                for name, path in gnps_paths.items()
            },
            **{
                f"gnps_{name}_panel_sha256": sha256_file(path)
                for name, path in gnps_panel_paths.items()
            },
        },
        "claim_boundary": (
            "Both Enveda and GNPS are consumed development resources. Query-balanced and "
            "IK14-collapsed views correct multiplicity in the reported endpoint but do not "
            "convert these OOF estimates into independent external evidence."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        (staging / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": report["status"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
