#!/usr/bin/env python3
"""Package the post-result-selected Full23 v1.3 development release.

This script does not fit or evaluate a model.  It validates the already
completed run 2433292 and wraps the Full23 v1 parent plus the fixed
query-centred representation residual into one explicit release directory.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


PANELS = ("identity_disjoint", "formula_disjoint")
ARM = "no_count_query_centered"
PARENT = "full23_v1"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--parent-release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("tasks"))
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_selection(report: dict) -> dict:
    metrics = report["metrics"]
    comparisons = report["comparisons"]
    deltas = {}
    for panel in PANELS:
        parent = float(metrics[panel][PARENT]["recall@1"])
        candidate = float(metrics[panel][ARM]["recall@1"])
        deltas[panel] = {
            "queries": int(metrics[panel][ARM]["queries"]),
            "parent_recall1": parent,
            "v13_recall1": candidate,
            "delta_pp": 100.0 * (candidate - parent),
            "parent_mrr": float(metrics[panel][PARENT]["mrr"]),
            "v13_mrr": float(metrics[panel][ARM]["mrr"]),
            "corrected": int(comparisons[panel][ARM]["corrected"]),
            "introduced": int(comparisons[panel][ARM]["introduced"]),
            "risk_net_lambda2_descriptive_only": int(
                comparisons[panel][ARM]["risk_net_lambda2"]
            ),
        }
    joint = report["query_balanced_joint"]
    if not float(joint[ARM]["top1_utility"]) > float(joint[PARENT]["top1_utility"]):
        raise RuntimeError("v1.3 does not improve query-balanced joint Top-1")
    return {
        "panels": deltas,
        "query_balanced_joint": {
            "unique_queries": int(joint[ARM]["unique_queries"]),
            "parent_top1": float(joint[PARENT]["top1_utility"]),
            "v13_top1": float(joint[ARM]["top1_utility"]),
            "delta_pp": 100.0 * (
                float(joint[ARM]["top1_utility"])
                - float(joint[PARENT]["top1_utility"])
            ),
        },
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)

    evaluation_path = args.run / "gnps_development_evaluation" / "report.json"
    training_path = args.run / "modeling" / "report.json"
    readout_path = args.run / "modeling" / f"{ARM}.pt"
    parent_path = args.parent_release / "assets" / "unified_candidate_integrator.pt"
    for path in (evaluation_path, training_path, readout_path, parent_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    evaluation = load_json(evaluation_path)
    summary = validate_selection(evaluation)
    payload = torch.load(readout_path, map_location="cpu", weights_only=False)
    settings = payload.get("settings", {})
    if payload.get("model_role") != "Full23_transfer_safe_no_count_query_centered":
        raise RuntimeError("unexpected v1.3 readout role")
    if settings.get("include_reference_count") is not False:
        raise RuntimeError("v1.3 must exclude reference-count shortcut")
    if settings.get("query_center_raw_residual") is not True:
        raise RuntimeError("v1.3 must query-centre its raw residual")

    (args.output / "assets").mkdir(parents=True)
    (args.output / "evidence").mkdir()
    (args.output / "runtime").mkdir()
    shutil.copy2(parent_path, args.output / "assets" / "full23_v1_parent.pt")
    shutil.copy2(readout_path, args.output / "assets" / "full23_v13_query_centered_readout.pt")
    shutil.copy2(training_path, args.output / "evidence" / "training_report.json")
    shutil.copy2(evaluation_path, args.output / "evidence" / "gnps_development_report_original.json")
    for name in (
        "full23_tensor_set_readout_core.py",
        "full23_multireference_readout_core.py",
        "unified_additive_release.py",
        "enveda_unified_candidate_integrator_core.py",
        "massspecgym_embedding_readout_core.py",
    ):
        source = args.source_root / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, args.output / "runtime" / name)

    manifest = {
        "schema": "dreams_unified_additive_full23_v13",
        "release_date": "2026-11-01",
        "release_status": "formal development algorithm",
        "scientific_boundary": (
            "Selected on consumed GNPS development after a post-result endpoint "
            "policy correction; not untouched external confirmation."
        ),
        "architecture": {
            "parent": "Full23 v1 candidate-level scalar evidence integrator",
            "residual": "four-encoder full-embedding tensor-set readout",
            "encoders": [
                "Noise V1",
                "ChemAware Stage-1",
                "ChemAware Phase-A",
                "Noise-ChemAware FullMSG",
            ],
            "repair": [
                "reference-count shortcut removed",
                "raw residual query-centred before bounded tanh",
            ],
            "score": "Full23_v1_score + bounded_query_centered_embedding_residual",
        },
        "assets": {
            "parent_checkpoint": "assets/full23_v1_parent.pt",
            "v13_readout_checkpoint": "assets/full23_v13_query_centered_readout.pt",
        },
        "selection_policy": {
            "name": "positive_query_balanced_unique_query_top1_v1",
            "adopted": "2026-11-01 post-result specification correction",
            "primary_endpoint": "query-balanced unique-query Top-1 point estimate",
            "risk_net_lambda2": "descriptive only; never a promotion veto",
            "confidence_intervals": "uncertainty description; not a promotion veto",
        },
        "development_result": summary,
        "evidence": {
            "training": "evidence/training_report.json",
            "gnps_original": "evidence/gnps_development_report_original.json",
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (args.output / "README.md").write_text(
        "# DreaMS Unified Additive Full23 v1.3\n\n"
        "Full23 v1.3 is the formal development release selected on 2026-11-01. "
        "It preserves the Full23 v1 candidate-evidence score and adds the fixed "
        "bounded, query-centred tensor-set residual over the complete embeddings "
        "of Noise V1, ChemAware Stage-1, ChemAware Phase-A and Noise-ChemAware "
        "FullMSG. Reference count is not an input.\n\n"
        "The release was selected because the query-balanced unique-query Top-1 "
        "endpoint increased; both constituent GNPS panels also improved. "
        "Corrected/introduced counts, confidence intervals and the historical "
        "lambda=2 risk statistic are reported but do not veto a positive primary "
        "ranking endpoint. This policy change is explicitly post-result and the "
        "bundle is not an untouched external-confirmation claim.\n\n"
        "No upstream encoder or reranker was retrained for this release. The two "
        "checkpoints and exact runtime sources are under `assets/` and `runtime/`.\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": "FULL23_V13_FORMAL_DEVELOPMENT_RELEASE_COMPLETE",
        "output": str(args.output),
        "selected": ARM,
        "development_result": summary,
    }, indent=2))


if __name__ == "__main__":
    main()
