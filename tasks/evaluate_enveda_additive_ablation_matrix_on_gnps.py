#!/usr/bin/env python
"""Apply the fixed Enveda Additive contribution matrix once to consumed GNPS."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from enveda_unified_additive_ablation_core import (
    ABLATION_ORDER,
    CHECKPOINT_SCHEMA,
    FULL,
    VARIANT_ORDER,
    VARIANTS,
    WSE,
    build_variant_features,
    predict_variant,
)
from enveda_unified_candidate_integrator_core import CONTROL_ARM, predict_from_payload
from gnps_pair_score_cache import sha256_file
from evaluate_enveda_integrator_on_gnps import load_checkpoint as load_full_checkpoint
from train_enveda_unified_candidate_integrator import (
    PANELS,
    _apply_feature_policy,
    _comparison,
    _formula_ci,
    _joint_query_balanced,
    _make_panels,
    _metric_summary,
    named_path,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--full-checkpoint", type=Path, required=True,
        help="Exact frozen 23-feature Additive checkpoint from run 2413878.",
    )
    parser.add_argument("--score-bundle", type=Path, required=True)
    parser.add_argument("--msg-cache", type=Path, required=True)
    parser.add_argument("--stage1-embeddings", type=Path, required=True)
    parser.add_argument("--phasea-embeddings", type=Path, required=True)
    parser.add_argument("--fullmsg-embeddings", type=Path, required=True)
    parser.add_argument("--chem-dir", type=Path, required=True)
    parser.add_argument("--comparator-embedding", action="append", type=named_path, default=[])
    parser.add_argument("--comparator-score", action="append", type=named_path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bootstrap-resamples", type=int, default=5_000)
    return parser.parse_args()


def _load(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("GNPS ablation evaluation received an unsupported checkpoint")
    if payload.get("reference_variant") != FULL:
        raise RuntimeError("GNPS ablation reference variant drifted")
    if payload.get("selection_from_ablation_outcomes") is not False:
        raise RuntimeError("ablation outcomes were used to select a deployment model")
    if tuple(payload.get("variants", {})) != ABLATION_ORDER:
        raise RuntimeError("checkpoint ablation registry drifted")
    return payload


def _joint_comparison(full: dict, baseline: dict, resamples: int, seed: int) -> dict:
    keys = sorted(full)
    if keys != sorted(baseline):
        raise RuntimeError("GNPS joint comparison query axes differ")
    delta = np.asarray([full[key][0] - baseline[key][0] for key in keys], dtype=np.float64)
    formula = np.asarray([full[key][2] for key in keys], dtype="U")
    result = _formula_ci(delta, formula, resamples, seed)
    result["semantics"] = "positive means full Additive has higher query-balanced Top-1 utility"
    return result


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.bootstrap_resamples < 100:
        raise ValueError("bootstrap-resamples must be at least 100")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("fixed GNPS matrix requested CUDA without an allocated GPU")
    payload = _load(args.checkpoint)
    full_payload = load_full_checkpoint(args.full_checkpoint)
    expected_full_sha = payload.get("frozen_full_reference", {}).get("checkpoint_sha256")
    observed_full_sha = sha256_file(args.full_checkpoint)
    if expected_full_sha != observed_full_sha:
        raise RuntimeError("fixed GNPS full Additive checkpoint differs from the frozen reference")
    base, panel_paths = _make_panels(args)
    policy, excluded = _apply_feature_policy(base, True)
    if policy != payload.get("feature_policy") or tuple(excluded) != tuple(
        payload.get("excluded_query_constant_features", ())
    ):
        raise RuntimeError("GNPS feature policy drifted from the Enveda matrix")

    predictions: dict[str, dict[str, np.ndarray]] = {panel: {} for panel in PANELS}
    for panel_name in PANELS:
        panel = base[panel_name]
        full_names = tuple(panel.feature_names)
        predictions[panel_name][FULL] = predict_from_payload(
            full_payload, full_names, panel.scalar,
            arm=CONTROL_ARM, device=args.device,
        )
        unavailable = np.asarray(panel.scalar, dtype=np.float32).copy()
        v2_columns = [
            index for index, name in enumerate(full_names)
            if name.startswith("chemaware_v2_")
        ]
        if len(v2_columns) != 6:
            raise RuntimeError("fixed GNPS V2-unavailable replay expected six V2 fields")
        unavailable[:, v2_columns] = 0.0
        predictions[panel_name]["full_additive_v2_unavailable_replay"] = predict_from_payload(
            full_payload, full_names, unavailable,
            arm=CONTROL_ARM, device=args.device,
        )
        for variant in ABLATION_ORDER:
            scalar, names = build_variant_features(
                panel.scores, panel.query_ptr, panel.scalar, panel.feature_names, variant,
            )
            predictions[panel_name][variant] = predict_variant(
                payload["variants"][variant], names, scalar, args.device,
            )

    method_scores: dict[str, dict[str, np.ndarray]] = {}
    panel_reports: dict[str, dict] = {}
    for panel_index, panel_name in enumerate(PANELS):
        panel = base[panel_name]
        comparators = {
            name: values for name, values in panel.scores.items()
            if name in {"official_dreams", WSE, "B_gnps_nested_oof"}
        }
        method_scores[panel_name] = {**comparators, **predictions[panel_name]}
        metrics = {
            method: _metric_summary(scores, panel)[0]
            for method, scores in method_scores[panel_name].items()
        }
        baselines = [name for name in ("official_dreams", WSE, "B_gnps_nested_oof") if name in comparators]
        baselines.extend(VARIANT_ORDER[1:])
        full_vs = {
            baseline: _comparison(
                predictions[panel_name][FULL], method_scores[panel_name][baseline], panel,
                args.bootstrap_resamples, 20261026 + 10_000 * panel_index + index,
            )
            for index, baseline in enumerate(baselines)
        }
        panel_reports[panel_name] = {
            "queries": int(len(panel.query_ptr) - 1),
            "candidate_molecules": int(len(panel.labels)),
            "metrics": metrics,
            "fixed_full_additive_vs": full_vs,
            "deployment_missingness_sensitivity": {
                "comparison": "full_23_additive_vs_full_additive_v2_unavailable_replay",
                "result": _comparison(
                    predictions[panel_name][FULL],
                    predictions[panel_name]["full_additive_v2_unavailable_replay"],
                    panel, args.bootstrap_resamples,
                    20261026 + 30_000 + panel_index,
                ),
                "interpretation": (
                    "Frozen-checkpoint replay only; this is not the strict no-V2 refit "
                    "and does not estimate the V2 group's training contribution."
                ),
            },
        }

    methods = tuple(method_scores[PANELS[0]])
    if any(tuple(method_scores[panel]) != methods for panel in PANELS):
        raise RuntimeError("GNPS method registry differs across panels")
    joint, collapsed = {}, {}
    for method in methods:
        joint[method], collapsed[method] = _joint_query_balanced(base, method_scores, method)
    joint_full_vs = {
        baseline: _joint_comparison(
            collapsed[FULL], collapsed[baseline], args.bootstrap_resamples,
            20261026 + 90_000 + index,
        )
        for index, baseline in enumerate(method for method in methods if method != FULL)
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        for panel_name in PANELS:
            panel = base[panel_name]
            np.savez_compressed(
                staging / f"candidate_scores_{panel_name}.npz",
                query_ptr=panel.query_ptr, molecule_label=panel.labels,
                query_formula=panel.formulas, query_key=panel.query_keys,
                **method_scores[panel_name],
            )
        report = {
            "status": "ENVEDA_ADDITIVE_ABLATION_FIXED_GNPS_DEVELOPMENT_EVALUATION_COMPLETE",
            "parameter_fitting_on_gnps": False,
            "model_selection_in_this_evaluation": False,
            "gnps_previously_consumed_for_project_decisions": True,
            "dataset": "GNPS Gold/Silver 10 ppm primary panels",
            "evaluation_role": "development_consumed_fixed_contribution_comparison",
            "allow_final_external_claim": False,
            "checkpoint": {
                "path": str(args.checkpoint), "sha256": sha256_file(args.checkpoint),
                "reference_variant": FULL,
            },
            "frozen_full_checkpoint": {
                "path": str(args.full_checkpoint), "sha256": observed_full_sha,
            },
            "variants": {
                name: {
                    "experts": list(VARIANTS[name].experts),
                    "include_chemistry": VARIANTS[name].include_chemistry,
                }
                for name in VARIANT_ORDER
            },
            "panels": panel_reports,
            "query_balanced_joint": joint,
            "query_balanced_joint_full_vs": joint_full_vs,
            "claim_boundary": (
                "GNPS is consumed development evidence and influenced the project before this "
                "fixed comparison. No model is refit or selected here; a genuinely new external "
                "collection is required for a final generalization or SOTA claim."
            ),
            "scientific_contract": {
                "dependent_reliability_summaries_recomputed_after_removal": True,
                "fixed_enveda_models_only": True,
                "gnps_parameter_fitting": False,
                "gnps_model_selection_in_this_evaluation": False,
                "gnps_previously_consumed_for_project_decisions": True,
                "v2_unavailable_replay_is_not_a_refit": True,
            },
            "input_sha256": {
                "score_bundle": sha256_file(args.score_bundle),
                "msg_report": sha256_file(args.msg_cache / "report.json"),
                "stage1_embeddings": sha256_file(args.stage1_embeddings),
                "phasea_embeddings": sha256_file(args.phasea_embeddings),
                "fullmsg_embeddings": sha256_file(args.fullmsg_embeddings),
                "chem_report": sha256_file(args.chem_dir / "report.json"),
                **{f"panel_{panel}": sha256_file(panel_paths[panel]) for panel in PANELS},
            },
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({"status": report["status"], "output": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
