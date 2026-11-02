#!/usr/bin/env python
"""Evaluate a Full23 successor from frozen GNPS candidate ledgers on CPU.

The frozen Full23 GNPS score files contain all seven candidate-level experts.
Together with the aligned continuous V2 evidence they reconstruct the exact
23-feature decision input, so no spectrum, encoder or reranker is recomputed.
The released parent checkpoint is replayed first and must reproduce its stored
candidate scores before the successor is evaluated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import tempfile

import numpy as np

from enveda_unified_candidate_integrator_core import (
    CONTROL_ARM,
    predict_from_payload,
    query_constant_context_names,
)
from gnps_pair_score_cache import sha256_file
from massspecgym_embedding_readout_core import build_independent_scalar_features
from train_enveda_unified_candidate_integrator import (
    MSG,
    PANELS,
    WSE,
    PanelData,
    _comparison,
    _metric_summary,
)
from train_full23_centered_residual_from_frozen_ledger import (
    EXPERTS,
    PARENT,
    SUCCESSOR,
    load_checkpoint,
)


AXIS_PREFIX = {
    "identity_disjoint": "identity__",
    "formula_disjoint": "formula__",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--axis-ledger", type=Path, required=True)
    parser.add_argument("--chem-dir", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--successor-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=5_000)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _strict_top(scores: np.ndarray, ptr: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    return np.asarray([
        int(left) + int(np.argmax(values[int(left):int(right)]))
        for left, right in zip(ptr[:-1], ptr[1:])
    ], dtype=np.int64)


def _panel(
    name: str, score_dir: Path, axis: np.lib.npyio.NpzFile,
    chem_dir: Path, expected_names: tuple[str, ...],
) -> tuple[PanelData, np.ndarray]:
    score_path = score_dir / f"candidate_scores_{name}.npz"
    if not score_path.is_file():
        raise FileNotFoundError(score_path)
    with np.load(score_path, allow_pickle=False) as body:
        ptr = np.asarray(body["query_ptr"], dtype=np.int64)
        labels = np.asarray(body["molecule_label"], dtype=np.int8)
        formulas = np.asarray(body["query_formula"]).astype("U")
        query_keys = np.asarray(body["query_key"]).astype("U")
        scores = {
            expert: np.asarray(body[expert], dtype=np.float32)
            for expert in EXPERTS
        }
        stored_parent = np.asarray(body[CONTROL_ARM], dtype=np.float32)
    prefix = AXIS_PREFIX[name]
    axis_ptr = np.asarray(axis[prefix + "query_ptr"], dtype=np.int64)
    axis_labels = np.asarray(axis[prefix + "labels"], dtype=np.int8)
    axis_formulas = np.asarray(axis[prefix + "formula"]).astype("U")
    near = np.asarray(axis[prefix + "near"], dtype=bool)
    query_ids = np.asarray(axis[prefix + "query_ids"]).astype("U")
    candidate_ids = np.asarray(axis[prefix + "candidate_ids"]).astype("U")
    if (
        not np.array_equal(ptr, axis_ptr)
        or not np.array_equal(labels, axis_labels)
        or not np.array_equal(formulas, axis_formulas)
        or len(query_keys) != len(query_ids)
    ):
        raise RuntimeError(f"frozen Full23 and GNPS axis ledgers drifted: {name}")

    chem_path = chem_dir / f"continuous_chem_{name}.npz"
    with np.load(chem_path, allow_pickle=False) as chemistry:
        checks = (
            np.array_equal(chemistry["query_ptr"], ptr),
            np.array_equal(chemistry["query_ids"].astype("U"), query_ids),
            np.array_equal(chemistry["candidate_ids"].astype("U"), candidate_ids),
            np.array_equal(chemistry["query_formula"].astype("U"), formulas),
            np.array_equal(chemistry["near_query"].astype(bool), near),
        )
        if not all(checks):
            raise RuntimeError(f"continuous V2 evidence drifted from GNPS axis: {name}")
        chemical = {
            "chem_valid": np.asarray(chemistry["availability"], dtype=bool),
            "candidate_utility_correct": np.asarray(
                chemistry["utility_correct"], dtype=np.float32,
            ),
            "candidate_utility_zero": np.asarray(
                chemistry["utility_zero"], dtype=np.float32,
            ),
            "candidate_utility_reversed": np.asarray(
                chemistry["utility_reversed"], dtype=np.float32,
            ),
            "candidate_utility_rotated": np.asarray(
                chemistry["utility_candidate_mismatch"], dtype=np.float32,
            ),
        }
    graph = SimpleNamespace(
        query_ptr=ptr,
        molecule_ptr=np.arange(len(labels) + 1, dtype=np.int64),
        molecule_label=labels,
    )
    scalar, feature_names = build_independent_scalar_features(
        {
            "v1_cosine": scores["noise_v1"],
            "weighted_entropy": scores[WSE],
            "p2b_fused": scores["p2b_noise_v1_frozen"],
        },
        graph,
        {
            "noise_v1": scores["noise_v1"],
            "chemaware_stage1": scores["chemaware_stage1"],
            "chemaware_phasea": scores["chemaware_phasea"],
            "noise_chemaware_fullmsg": scores["noise_chemaware_fullmsg"],
        },
        "noise_v1",
        chemical,
        {MSG: scores[MSG]},
        include_reliability_context=True,
    )
    excluded = set(query_constant_context_names(feature_names))
    keep = np.asarray([feature not in excluded for feature in feature_names], dtype=bool)
    retained_names = tuple(feature for feature in feature_names if feature not in excluded)
    if len(excluded) != 31 or retained_names != expected_names:
        raise RuntimeError(f"reconstructed GNPS Full23 feature contract drifted: {name}")
    panel = PanelData(
        name=name, graph=graph, scalar=scalar[:, keep], feature_names=retained_names,
        labels=labels, query_ptr=ptr, formulas=formulas, query_keys=query_keys,
        near=near, query_weights=np.ones(len(near), dtype=np.float32), scores=scores,
    )
    return panel, stored_parent


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.bootstrap_resamples < 100:
        raise ValueError("bootstrap-resamples must be at least 100")
    parent = load_checkpoint(args.parent_checkpoint)
    successor = load_checkpoint(args.successor_checkpoint)
    expected_names = tuple(map(str, parent["scalar_feature_names"]))
    if tuple(map(str, successor["scalar_feature_names"])) != expected_names:
        raise RuntimeError("parent and successor feature contracts differ")

    panels = {}
    stored_parent = {}
    with np.load(args.axis_ledger, allow_pickle=False) as axis:
        for name in PANELS:
            panels[name], stored_parent[name] = _panel(
                name, args.score_dir, axis, args.chem_dir, expected_names,
            )

    reports = {}
    predictions = {}
    for index, name in enumerate(PANELS):
        panel = panels[name]
        replay = predict_from_payload(
            parent, panel.feature_names, panel.scalar, arm=CONTROL_ARM, device=args.device,
        )
        centered = predict_from_payload(
            successor, panel.feature_names, panel.scalar,
            arm=CONTROL_ARM, device=args.device,
        )
        max_abs = float(np.max(np.abs(replay - stored_parent[name])))
        top_agreement = float(np.mean(
            _strict_top(replay, panel.query_ptr)
            == _strict_top(stored_parent[name], panel.query_ptr)
        ))
        if max_abs > 2e-5 or top_agreement != 1.0:
            raise RuntimeError(
                f"released Full23 GNPS replay drifted: {name}, max_abs={max_abs}, "
                f"top_agreement={top_agreement}"
            )
        scores = {
            WSE: panel.scores[WSE], PARENT: stored_parent[name], SUCCESSOR: centered,
        }
        reports[name] = {
            "queries": len(panel.query_ptr) - 1,
            "candidate_molecules": len(panel.labels),
            "parent_replay_max_abs_error": max_abs,
            "parent_replay_top1_agreement": top_agreement,
            "methods": {
                method: _metric_summary(value, panel)[0]
                for method, value in scores.items()
            },
            "centered_vs_full23_v1": _comparison(
                centered, stored_parent[name], panel,
                args.bootstrap_resamples, 20271029 + index,
            ),
            "centered_vs_wse": _comparison(
                centered, panel.scores[WSE], panel,
                args.bootstrap_resamples, 20272029 + index,
            ),
        }
        predictions[name] = scores

    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        for name in PANELS:
            panel = panels[name]
            np.savez_compressed(
                staging / f"candidate_scores_{name}.npz",
                query_ptr=panel.query_ptr, molecule_label=panel.labels,
                query_formula=panel.formulas, query_key=panel.query_keys,
                **predictions[name],
            )
        report = {
            "status": "FULL23_CENTERED_RESIDUAL_FROZEN_GNPS_EVALUATION_COMPLETE",
            "dataset": "GNPS Gold/Silver 10 ppm primary panels",
            "evaluation_role": "development_consumed_fixed_comparison",
            "training_or_selection_on_gnps": False,
            "allow_final_external_claim": False,
            "panels": reports,
            "checkpoint": {
                "parent": str(args.parent_checkpoint),
                "parent_sha256": sha256_file(args.parent_checkpoint),
                "successor": str(args.successor_checkpoint),
                "successor_sha256": sha256_file(args.successor_checkpoint),
                "arm": CONTROL_ARM,
            },
            "inputs": {
                "axis_ledger": str(args.axis_ledger),
                "axis_ledger_sha256": sha256_file(args.axis_ledger),
                "chem_report_sha256": sha256_file(args.chem_dir / "report.json"),
                **{
                    f"score_{name}_sha256": sha256_file(
                        args.score_dir / f"candidate_scores_{name}.npz"
                    )
                    for name in PANELS
                },
            },
            "claim_boundary": (
                "GNPS is a consumed development resource and influenced historical method "
                "development. This fixed replay tests transfer direction only, not external SOTA."
            ),
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"], "output": str(args.output),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
