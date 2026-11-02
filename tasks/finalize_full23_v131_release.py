#!/usr/bin/env python3
"""Package the selected Full23 v1.3.1 consumed-development release.

This is a packaging operation only. It validates the completed cumulative
repair run, copies the already selected ``v13_01`` checkpoint, and reuses the
Full23 v1.3 parent asset and runtime without fitting or scoring any model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch


SELECTED = "v13_01"
EXPECTED_SHA256 = (
    "6603013a04fdc26946445a6ba1ed566d768bac49423046ce3922daac79fe637e"
)
PANELS = ("identity_disjoint", "formula_disjoint")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--v13-release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path("tasks"))
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)

    modeling = args.run / "modeling"
    evaluation = args.run / "gnps_development_evaluation"
    checkpoint = modeling / f"{SELECTED}.pt"
    selected_copy = evaluation / "selected_model.pt"
    evaluation_report_path = evaluation / "report.json"
    candidate_ledgers = {
        panel: evaluation / f"candidate_predictions_{panel}.npz"
        for panel in PANELS
    }
    training_report_path = modeling / "report.json"
    contract_path = modeling / "contract.json"
    parent_checkpoint = args.v13_release / "assets" / "full23_v1_parent.pt"
    parent_manifest = args.v13_release / "manifest.json"
    required = (
        checkpoint,
        selected_copy,
        evaluation_report_path,
        training_report_path,
        contract_path,
        *candidate_ledgers.values(),
        parent_checkpoint,
        parent_manifest,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing release inputs: {missing}")

    evaluation_report = load_json(evaluation_report_path)
    if evaluation_report.get("selected") != SELECTED:
        raise RuntimeError("cumulative run did not select v13_01")
    if sha256_file(checkpoint) != EXPECTED_SHA256:
        raise RuntimeError("selected checkpoint hash drifted")
    if sha256_file(selected_copy) != EXPECTED_SHA256:
        raise RuntimeError("selected_model.pt differs from v13_01.pt")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    settings = payload.get("settings", {})
    expected_settings = {
        "pair_dimension": 64,
        "include_reference_count": False,
        "query_center_raw_residual": True,
        "include_anchor_context": False,
        "include_signed_difference": False,
    }
    for field, expected in expected_settings.items():
        if settings.get(field) != expected:
            raise RuntimeError(
                f"selected checkpoint setting drifted: {field}={settings.get(field)!r}"
            )
    if payload.get("training", {}).get("anchor_correct_multiplier") != 1.0:
        raise RuntimeError("selected checkpoint did not use multiplier 1")
    if "penalize_query_centered_raw_residual" not in payload.get(
        "interventions", ()
    ):
        raise RuntimeError("selected checkpoint did not use the raw-space penalty")

    joint = evaluation_report["query_balanced_joint"]
    baseline = joint["full23_v13"]
    selected = joint[SELECTED]
    if not float(selected["top1_utility"]) > float(baseline["top1_utility"]):
        raise RuntimeError("selected checkpoint does not improve the primary endpoint")

    (args.output / "assets").mkdir(parents=True)
    (args.output / "evidence").mkdir()
    (args.output / "runtime").mkdir()
    runtime_names = (
        "enveda_unified_candidate_integrator_core.py",
        "massspecgym_embedding_readout_core.py",
        "unified_additive_release.py",
        "full23_tensor_set_readout_core.py",
        "full23_v131_release.py",
    )
    for name in runtime_names:
        source = args.source_root / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, args.output / "runtime" / name)
    shutil.copy2(parent_checkpoint, args.output / "assets" / "full23_v1_parent.pt")
    readout_name = "full23_v131_query_centered_raw_penalty_readout.pt"
    shutil.copy2(checkpoint, args.output / "assets" / readout_name)
    shutil.copy2(training_report_path, args.output / "evidence" / "training_report.json")
    shutil.copy2(contract_path, args.output / "evidence" / "training_contract.json")
    shutil.copy2(
        evaluation_report_path,
        args.output / "evidence" / "gnps_development_selection_report.json",
    )
    for panel, path in candidate_ledgers.items():
        shutil.copy2(
            path,
            args.output / "evidence" / f"candidate_predictions_{panel}.npz",
        )
    shutil.copy2(parent_manifest, args.output / "evidence" / "parent_v13_manifest.json")

    panel_results = {}
    for panel in PANELS:
        formal = evaluation_report["metrics"][panel]["full23_v13"]
        winner = evaluation_report["metrics"][panel][SELECTED]
        comparison = evaluation_report["comparisons"][panel][
            f"{SELECTED}_vs_formal_v13"
        ]
        panel_results[panel] = {
            "queries": int(winner["queries"]),
            "formal_v13_recall1": float(formal["recall@1"]),
            "v131_recall1": float(winner["recall@1"]),
            "delta_pp": 100.0
            * (float(winner["recall@1"]) - float(formal["recall@1"])),
            "formal_v13_mrr": float(formal["mrr"]),
            "v131_mrr": float(winner["mrr"]),
            "corrected": int(comparison["corrected"]),
            "introduced": int(comparison["introduced"]),
            "risk_net_lambda2_descriptive_only": int(
                comparison["risk_net_lambda2"]
            ),
        }

    manifest = {
        "schema": "dreams_unified_additive_full23_v131",
        "release_date": "2026-11-02",
        "release_status": "formal consumed-development incremental release",
        "scientific_boundary": (
            "Fit on consumed Enveda-180 meta-development and selected on "
            "consumed GNPS development; not external confirmation."
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
            "score": "Full23_v1_score + bounded_query_centered_embedding_residual",
            "v131_training_repairs": [
                "anchor-correct multiplier reduced from 2 to 1",
                "regularize query-centred raw residual before bounded tanh",
            ],
        },
        "assets": {
            "parent_checkpoint": "assets/full23_v1_parent.pt",
            "parent_checkpoint_sha256": sha256_file(parent_checkpoint),
            "readout_checkpoint": f"assets/{readout_name}",
            "readout_sha256": EXPECTED_SHA256,
        },
        "runtime": {
            "entrypoint": "runtime/full23_v131_release.py",
            "class": "FrozenFull23V131",
            "missing_representation_semantics": "keep_parent_score",
        },
        "selection_policy": evaluation_report["selection_policy"],
        "source_run": str(args.run).replace("\\", "/"),
        "readout_contract": {
            "schema": payload.get("schema"),
            "model_role": payload.get("model_role"),
            "settings": settings,
            "interventions": list(payload.get("interventions", ())),
            "training": payload.get("training"),
        },
        "data_roles": {
            "enveda180": {
                "role": "consumed meta-development fitting",
                "identity_queries": 25000,
                "formula_queries": 9435,
            },
            "gnps_gold_silver": {
                "role": "consumed development and model selection",
                "identity_queries": 10995,
                "formula_queries": 5261,
                "query_balanced_unique_queries": 11019,
            },
        },
        "development_result": {
            "panels": panel_results,
            "query_balanced_joint": {
                "unique_queries": int(selected["unique_queries"]),
                "formal_v13_top1": float(baseline["top1_utility"]),
                "v131_top1": float(selected["top1_utility"]),
                "delta_pp": 100.0
                * (
                    float(selected["top1_utility"])
                    - float(baseline["top1_utility"])
                ),
                "formal_v13_mrr": float(baseline["mrr_utility"]),
                "v131_mrr": float(selected["mrr_utility"]),
            },
        },
        "evidence": {
            "training": "evidence/training_report.json",
            "training_contract": "evidence/training_contract.json",
            "gnps_selection": "evidence/gnps_development_selection_report.json",
            "candidate_predictions_identity": "evidence/candidate_predictions_identity_disjoint.npz",
            "candidate_predictions_formula": "evidence/candidate_predictions_formula_disjoint.npz",
            "parent_manifest": "evidence/parent_v13_manifest.json",
            "checksums": "checksums.sha256",
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (args.output / "README.md").write_text(
        "# DreaMS Unified Additive Full23 v1.3.1\n\n"
        "This bundle is the incremental Full23 v1.3.1 consumed-development "
        "release. It reuses the Full23 v1 parent and v1.3 runtime, and replaces "
        "only the tensor-set readout with the selected `v13_01` checkpoint. "
        "The executable composition entrypoint is "
        "`runtime/full23_v131_release.py::FrozenFull23V131`.\n\n"
        "The change fixes residual saturation by regularizing the query-centred "
        "raw residual and removes the historical twofold training multiplier. "
        "It was fit on consumed Enveda-180 meta-development and selected on "
        "11,019 unique GNPS development queries. It is not untouched external "
        "confirmation.\n\n"
        "The exact result and limitations are recorded in "
        "`docs/FULL23_V13_CUMULATIVE_REPAIRS_RESULT_20261102.md`.\n",
        encoding="utf-8",
    )
    checksum_lines = []
    for path in sorted(p for p in args.output.rglob("*") if p.is_file()):
        relative = path.relative_to(args.output).as_posix()
        checksum_lines.append(f"{sha256_file(path)}  {relative}")
    (args.output / "checksums.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "FULL23_V131_INCREMENTAL_RELEASE_COMPLETE",
                "output": str(args.output),
                "selected": SELECTED,
                "checkpoint_sha256": EXPECTED_SHA256,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
