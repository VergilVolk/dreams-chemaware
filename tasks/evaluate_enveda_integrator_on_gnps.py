#!/usr/bin/env python
"""Apply the Enveda-fitted candidate integrator once to consumed GNPS.

No parameter, normalization statistic, feature choice, threshold or model arm
is fitted here.  The nonlinear primary arm was declared before Enveda OOF and
is always the deployed arm; the lower-capacity additive model is reported only
as a fixed comparison.  GNPS therefore remains a consumed, read-only
development benchmark rather than a selection surface.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from enveda_unified_candidate_integrator_core import (
    CHECKPOINT_SCHEMA,
    CONTROL_ARM,
    PRIMARY_ARM,
    predict_from_payload,
)
from gnps_pair_score_cache import sha256_file
from train_enveda_unified_candidate_integrator import (
    PANELS,
    WSE,
    _comparison,
    _joint_query_balanced,
    _make_panels,
    _metric_summary,
    named_path,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--score-bundle", type=Path, required=True)
    parser.add_argument("--msg-cache", type=Path, required=True)
    parser.add_argument("--stage1-embeddings", type=Path, required=True)
    parser.add_argument("--phasea-embeddings", type=Path, required=True)
    parser.add_argument("--fullmsg-embeddings", type=Path, required=True)
    parser.add_argument("--chem-dir", type=Path, required=True)
    parser.add_argument(
        "--comparator-embedding", action="append", type=named_path, default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--comparator-score", action="append", type=named_path, default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bootstrap-resamples", type=int, default=5_000)
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("fixed GNPS evaluation received an unsupported checkpoint")
    if payload.get("selection_class") != "candidate_evidence_integration":
        raise RuntimeError("checkpoint lacks the candidate-evidence integration contract")
    if payload.get("primary_arm") != PRIMARY_ARM or payload.get("control_arm") != CONTROL_ARM:
        raise RuntimeError("checkpoint arm registry drifted")
    if payload.get("arm_selection_from_oof") is not False:
        raise RuntimeError("checkpoint selected an arm from Enveda outcomes")
    return payload


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.bootstrap_resamples < 100:
        raise ValueError("bootstrap-resamples must be at least 100")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("fixed GNPS evaluation requested CUDA without an allocated GPU")
    payload = load_checkpoint(args.checkpoint)
    panels, panel_paths = _make_panels(args)
    feature_names = tuple(panels[PANELS[0]].feature_names)
    if feature_names != tuple(payload.get("scalar_feature_names", ())):
        raise RuntimeError("GNPS scalar feature contract differs from Enveda refit")

    method_scores: dict[str, dict[str, np.ndarray]] = {}
    for panel in PANELS:
        method_scores[panel] = dict(panels[panel].scores)
        for arm in (PRIMARY_ARM, CONTROL_ARM):
            method_scores[panel][arm] = predict_from_payload(
                payload, feature_names, panels[panel].scalar,
                arm=arm, device=args.device,
            )
    common = set.intersection(*(set(method_scores[panel]) for panel in PANELS))
    preferred = (
        "official_dreams", WSE, "noise_v1",
        "p2b_noise_v1_frozen", "noise_msg_fusion_stack_v1",
        "chemaware_stage1", "chemaware_phasea", "noise_chemaware_fullmsg",
        "B_gnps_nested_oof", CONTROL_ARM, PRIMARY_ARM,
    )
    ordered = [name for name in preferred if name in common]
    ordered.extend(sorted(common - set(ordered)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        panel_reports: dict[str, dict] = {}
        for panel_index, panel_name in enumerate(PANELS):
            panel = panels[panel_name]
            metrics = {
                method: _metric_summary(method_scores[panel_name][method], panel)[0]
                for method in ordered
            }
            comparisons = {
                f"{PRIMARY_ARM}_vs_{baseline}": _comparison(
                    method_scores[panel_name][PRIMARY_ARM],
                    method_scores[panel_name][baseline], panel,
                    args.bootstrap_resamples,
                    20261026 + panel_index * 1000 + index,
                )
                for index, baseline in enumerate(ordered)
                if baseline != PRIMARY_ARM
            }
            panel_reports[panel_name] = {
                "queries": int(len(panel.query_ptr) - 1),
                "candidate_molecules": int(len(panel.labels)),
                "metrics": metrics,
                "fixed_primary_comparisons": comparisons,
            }
            np.savez_compressed(
                staging / f"candidate_scores_{panel_name}.npz",
                query_ptr=panel.query_ptr,
                molecule_label=panel.labels,
                query_formula=panel.formulas,
                query_key=panel.query_keys,
                **{method: method_scores[panel_name][method] for method in ordered},
            )

        joint = {
            method: _joint_query_balanced(panels, method_scores, method)[0]
            for method in ordered
        }
        report = {
            "status": "ENVEDA_INTEGRATOR_FIXED_GNPS_DEVELOPMENT_EVALUATION_COMPLETE",
            "training_or_selection_on_gnps": False,
            "dataset": "GNPS Gold/Silver 10 ppm primary panels",
            "evaluation_role": "development_consumed_fixed_comparison",
            "allow_final_external_claim": False,
            "checkpoint": {
                "path": str(args.checkpoint),
                "sha256": sha256_file(args.checkpoint),
                "training_role": payload.get("training", {}).get("role"),
                "deployed_arm": PRIMARY_ARM,
                "additive_arm_role": "capacity-lower fixed comparator; never selected here",
            },
            "panels": panel_reports,
            "query_balanced_joint": joint,
            "claim_boundary": (
                "Both Enveda and GNPS are consumed development resources.  This fixed run "
                "does not refit or select the Enveda-fitted model after reading this GNPS run. "
                "GNPS nevertheless influenced historical architecture and module development; "
                "a genuinely new external dataset is required for a final SOTA or "
                "generalization claim."
            ),
            "input_sha256": {
                "score_bundle": sha256_file(args.score_bundle),
                "msg_report": sha256_file(args.msg_cache / "report.json"),
                "stage1_embeddings": sha256_file(args.stage1_embeddings),
                "phasea_embeddings": sha256_file(args.phasea_embeddings),
                "fullmsg_embeddings": sha256_file(args.fullmsg_embeddings),
                "chem_report": sha256_file(args.chem_dir / "report.json"),
                **{
                    f"panel_{panel}": sha256_file(panel_paths[panel])
                    for panel in PANELS
                },
            },
        }
        (staging / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"],
        "deployed_arm": PRIMARY_ARM,
        "output": str(args.output),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
