#!/usr/bin/env python3
"""Consumed-GNPS development adjudication of the frozen representation residual."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import torch

from full23_multireference_readout_core import (
    ARMS,
    ENCODER_NAMES,
    SCHEMA,
    TOP1_ARM,
    TOPK_ARM,
    TopKReferenceIndex,
    model_from_payload,
)
from train_enveda_unified_candidate_integrator import (
    PANELS,
    _comparison,
    _joint_query_balanced,
    _metric_summary,
    named_path,
)
from train_full23_multireference_readout import (
    ANCHOR_STRENGTH,
    BATCH_QUERIES,
    EPOCHS,
    FULL23_OOF_SHA256,
    FUSION_HIDDEN,
    LEARNING_RATE,
    PROJECTION_DIMENSION,
    TRAIN_SEED,
    WEIGHT_DECAY,
    RepresentationPanel,
    _embedding_paths,
    _load_banks,
    _make_representation_panels,
    _pack,
    _reference_multiplicity,
)
from unified_additive_release import CHECKPOINT_SHA256, sha256_file, validate_checkpoint


BOOTSTRAP_RESAMPLES = 5000
PERMUTED = "top3_candidate_reference_permuted"
RESERVED_COMPARATOR_NAMES = {
    "full23_v1", TOP1_ARM, TOPK_ARM, PERMUTED,
    "official_dreams", "weighted_spectral_entropy", "noise_v1",
    "p2b_noise_v1_frozen", "noise_msg_fusion_stack_v1",
    "chemaware_stage1", "chemaware_phasea", "noise_chemaware_fullmsg",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--score-bundle", type=Path, required=True)
    parser.add_argument("--msg-cache", type=Path, required=True)
    parser.add_argument("--noise-embeddings", type=Path, required=True)
    parser.add_argument("--stage1-embeddings", type=Path, required=True)
    parser.add_argument("--phasea-embeddings", type=Path, required=True)
    parser.add_argument("--fullmsg-embeddings", type=Path, required=True)
    parser.add_argument("--chem-dir", type=Path, required=True)
    parser.add_argument("--full23-checkpoint", type=Path, required=True)
    parser.add_argument("--readout-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--comparator-score", action="append", type=named_path, default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.set_defaults(comparator_embedding=[])
    parser.set_defaults(full23_oof=None)
    return parser.parse_args()


def _load_torch(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _permuted_panels(
    panels: dict[str, RepresentationPanel],
) -> tuple[dict[str, RepresentationPanel], dict[str, dict[str, dict[str, float]]]]:
    """Candidate-reference mismatch retaining K and mask within each query."""
    output: dict[str, RepresentationPanel] = {}
    audit: dict[str, dict[str, dict[str, float]]] = {}
    for panel_name, panel in panels.items():
        altered: dict[str, TopKReferenceIndex] = {}
        audit[panel_name] = {}
        ptr = panel.base.query_ptr
        for encoder in ENCODER_NAMES:
            source = panel.topk[encoder]
            positions = source.reference_positions.copy()
            rows = source.reference_rows.copy()
            mask = source.reference_mask.copy()
            for left, right in zip(ptr[:-1], ptr[1:]):
                left, right = int(left), int(right)
                counts = mask[left:right].sum(axis=1)
                for count in np.unique(counts):
                    group = np.flatnonzero(counts == count) + left
                    if len(group) > 1:
                        positions[group] = np.roll(positions[group], 1, axis=0)
                        rows[group] = np.roll(rows[group], 1, axis=0)
            changed_slots = np.asarray((rows != source.reference_rows) & mask, dtype=bool)
            changed_candidates = np.any(changed_slots, axis=1)
            audit[panel_name][encoder] = {
                "candidates": int(len(changed_candidates)),
                "changed_candidates": int(np.sum(changed_candidates)),
                "changed_candidate_fraction": float(np.mean(changed_candidates)),
                "valid_reference_slots": int(np.sum(mask)),
                "changed_valid_reference_slots": int(np.sum(changed_slots)),
            }
            altered[encoder] = replace(
                source, reference_positions=positions, reference_rows=rows, reference_mask=mask,
            )
        output[panel_name] = RepresentationPanel(
            base=panel.base, anchor=panel.anchor, topk=altered,
        )
        if not any(row["changed_candidates"] > 0 for row in audit[panel_name].values()):
            raise RuntimeError(f"candidate-reference permutation is a no-op for {panel_name}")
    return output, audit


@torch.inference_mode()
def _predict(
    model: torch.nn.Module,
    arm: str,
    panels: dict[str, RepresentationPanel],
    banks: dict[str, torch.Tensor],
    device: str,
) -> dict[str, np.ndarray]:
    output = {
        name: np.full(len(panel.base.labels), np.nan, dtype=np.float32)
        for name, panel in panels.items()
    }
    model.eval()
    for panel_name in PANELS:
        panel = panels[panel_name]
        queries = np.arange(len(panel.base.query_ptr) - 1, dtype=np.int64)
        for start in range(0, len(queries), BATCH_QUERIES):
            chosen = [(panel_name, int(query)) for query in queries[start:start + BATCH_QUERIES]]
            anchor, _, query, refs, mask, ptr, _ = _pack(panels, chosen, arm)
            values = model(
                torch.as_tensor(anchor, device=device),
                torch.as_tensor(query, device=device),
                torch.as_tensor(refs, device=device),
                torch.as_tensor(mask, device=device),
                banks,
            ).float().cpu().numpy()
            cursor = 0
            for _, query_index in chosen:
                left, right = map(int, panel.base.query_ptr[query_index:query_index + 2])
                width = right - left
                output[panel_name][left:right] = values[cursor:cursor + width]
                cursor += width
            if cursor != int(ptr[-1]):
                raise RuntimeError("prediction packing cursor drifted")
    if any(not np.all(np.isfinite(value)) for value in output.values()):
        raise RuntimeError("multi-reference prediction coverage is incomplete")
    return output


def _validate_readout_checkpoint(readout: dict) -> None:
    """Fail closed if the one registered training recipe or parent changed."""
    if readout.get("schema") != SCHEMA:
        raise RuntimeError("multi-reference checkpoint schema mismatch")
    if readout.get("full23_checkpoint_sha256") != CHECKPOINT_SHA256:
        raise RuntimeError("multi-reference anchor differs from Full23 v1")
    if readout.get("official_embedding_direct_input") is not False:
        raise RuntimeError("official DreaMS entered the new representation head")
    if readout.get("full23_anchor_has_indirect_official_dependency") is not True:
        raise RuntimeError("Full23 indirect official dependency is not declared")
    expected = {
        "source": "Enveda-180 consumed meta-development",
        "parent_prediction_role": "five-fold formula-OOF Full23 v1",
        "parent_oof_sha256": FULL23_OOF_SHA256,
        "evaluation_on_training_source": False,
        "epochs": EPOCHS,
        "batch_queries": BATCH_QUERIES,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "anchor_strength": ANCHOR_STRENGTH,
        "seed": TRAIN_SEED,
        "unique_query_occurrence_weighting": True,
    }
    training = readout.get("training", {})
    drifted = {
        key: {"expected": value, "observed": training.get(key)}
        for key, value in expected.items() if training.get(key) != value
    }
    settings = readout.get("settings", {})
    expected_settings = {
        "embedding_dimension": 1024,
        "projection_dimension": PROJECTION_DIMENSION,
        "fusion_hidden": FUSION_HIDDEN,
        "top_k": 3,
    }
    drifted.update({
        f"settings.{key}": {"expected": value, "observed": settings.get(key)}
        for key, value in expected_settings.items() if settings.get(key) != value
    })
    if drifted:
        raise RuntimeError(f"multi-reference registered recipe drifted: {drifted}")


def _near_recall1(metric: dict) -> float | None:
    near = metric.get("near", {})
    if int(near.get("queries", 0)) == 0:
        return None
    return float(near["recall@1"])


def _increment_gate(
    candidate: str,
    baseline: str,
    metrics: dict[str, dict[str, dict]],
    comparisons: dict[str, dict[str, dict]],
) -> dict:
    """Two-panel consumed-development gate for one paired comparison."""
    panels: dict[str, dict] = {}
    for panel_name in PANELS:
        comparison = comparisons[panel_name][f"{candidate}_vs_{baseline}"]
        candidate_near = _near_recall1(metrics[panel_name][candidate])
        baseline_near = _near_recall1(metrics[panel_name][baseline])
        checks = {
            "formula_cluster_ci_low_gt_zero": float(
                comparison["recall1_formula_cluster_ci"]["ci_low"]
            ) > 0.0,
            "corrected_gt_introduced": int(comparison["corrected"]) > int(
                comparison["introduced"]
            ),
            "near_recall1_not_lower": (
                candidate_near is None or baseline_near is None
                or candidate_near >= baseline_near
            ),
        }
        panels[panel_name] = {
            "pass": bool(all(checks.values())),
            "checks": checks,
            "candidate_near_recall1": candidate_near,
            "baseline_near_recall1": baseline_near,
        }
    return {
        "pass": bool(all(row["pass"] for row in panels.values())),
        "panels": panels,
    }


def _input_hashes(args: argparse.Namespace) -> dict[str, str]:
    paths: dict[str, Path] = {
        "manifest": args.benchmark / "manifest.csv.gz",
        "score_bundle": args.score_bundle,
        "msg_report": args.msg_cache / "report.json",
        "chem_report": args.chem_dir / "report.json",
        "full23_checkpoint": args.full23_checkpoint,
        "readout_checkpoint": args.readout_checkpoint,
        **{f"embedding_{name}": path for name, path in _embedding_paths(args).items()},
    }
    for panel_name in PANELS:
        paths[f"panel_{panel_name}"] = args.benchmark / f"panel_{panel_name}.npz"
        paths[f"msg_pairs_{panel_name}"] = (
            args.msg_cache / f"pair_scores_{panel_name}.npy"
        )
        paths[f"chem_{panel_name}"] = (
            args.chem_dir / f"continuous_chem_{panel_name}.npz"
        )
    for name, path in args.comparator_score:
        if path.is_file():
            paths[f"comparator_{name}"] = path
        elif path.is_dir() and (path / "candidate_predictions.npz").is_file():
            paths[f"comparator_{name}"] = path / "candidate_predictions.npz"
        elif path.is_dir() and (path / "report.json").is_file():
            paths[f"comparator_{name}_report"] = path / "report.json"
            for panel_name in PANELS:
                pair_path = path / f"pair_scores_{panel_name}.npy"
                if pair_path.is_file():
                    paths[f"comparator_{name}_{panel_name}"] = pair_path
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"hash registry misses inputs: {missing}")
    return {name: sha256_file(path) for name, path in paths.items()}


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("one allocated GPU is required")
    comparator_names = [name for name, _ in args.comparator_score]
    collided = sorted(set(comparator_names) & RESERVED_COMPARATOR_NAMES)
    if collided:
        raise ValueError(f"reserved comparator names: {collided}")
    full23 = validate_checkpoint(args.full23_checkpoint)
    if sha256_file(args.full23_checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("Full23 v1 checkpoint identity drifted")
    readout = _load_torch(args.readout_checkpoint)
    _validate_readout_checkpoint(readout)
    manifest = pd.read_csv(args.benchmark / "manifest.csv.gz", usecols=["row"])
    cpu_banks = _load_banks(args, len(manifest))
    panels = _make_representation_panels(args, full23, cpu_banks)
    bank_dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    gpu_banks = {
        name: torch.as_tensor(embedding, dtype=bank_dtype, device=args.device)
        for name, (_, embedding) in cpu_banks.items()
    }
    models = {arm: model_from_payload(readout, arm, args.device) for arm in ARMS}
    method_scores: dict[str, dict[str, np.ndarray]] = {
        name: {
            "full23_v1": panel.anchor,
            "official_dreams": panel.base.scores["official_dreams"],
            "weighted_spectral_entropy": panel.base.scores["weighted_spectral_entropy"],
            "noise_v1": panel.base.scores["noise_v1"],
            "p2b_noise_v1_frozen": panel.base.scores["p2b_noise_v1_frozen"],
            "noise_msg_fusion_stack_v1": panel.base.scores["noise_msg_fusion_stack_v1"],
            "chemaware_stage1": panel.base.scores["chemaware_stage1"],
            "chemaware_phasea": panel.base.scores["chemaware_phasea"],
            "noise_chemaware_fullmsg": panel.base.scores["noise_chemaware_fullmsg"],
            **{
                comparator: values
                for comparator, values in panel.base.scores.items()
                if comparator not in {
                    "official_dreams", "weighted_spectral_entropy", "noise_v1",
                    "p2b_noise_v1_frozen", "noise_msg_fusion_stack_v1",
                    "chemaware_stage1", "chemaware_phasea", "noise_chemaware_fullmsg",
                }
            },
        }
        for name, panel in panels.items()
    }
    for arm in ARMS:
        predicted = _predict(
            models[arm], arm, panels, gpu_banks, args.device,
        )
        for panel_name in PANELS:
            method_scores[panel_name][arm] = predicted[panel_name]
    permuted_panels, permutation_audit = _permuted_panels(panels)
    permuted = _predict(
        models[TOPK_ARM], TOPK_ARM, permuted_panels, gpu_banks, args.device,
    )
    for panel_name in PANELS:
        method_scores[panel_name][PERMUTED] = permuted[panel_name]

    metrics: dict[str, dict[str, dict]] = {name: {} for name in PANELS}
    comparisons: dict[str, dict[str, dict]] = {name: {} for name in PANELS}
    for panel_index, panel_name in enumerate(PANELS):
        panel = panels[panel_name].base
        for method, scores in method_scores[panel_name].items():
            metrics[panel_name][method], _ = _metric_summary(scores, panel)
        registered = (
            (TOPK_ARM, "full23_v1"),
            (TOPK_ARM, TOP1_ARM),
            (TOP1_ARM, "full23_v1"),
            (TOPK_ARM, PERMUTED),
        )
        for comparison_index, (candidate, baseline) in enumerate(registered):
            comparisons[panel_name][f"{candidate}_vs_{baseline}"] = _comparison(
                method_scores[panel_name][candidate], method_scores[panel_name][baseline],
                panel, BOOTSTRAP_RESAMPLES,
                20261027 + panel_index * 100 + comparison_index,
            )

    joint_metrics = {}
    for method in ("full23_v1", TOP1_ARM, TOPK_ARM):
        joint_metrics[method], _ = _joint_query_balanced(
            {name: panel.base for name, panel in panels.items()}, method_scores, method,
        )
    eligibility = {
        arm: _increment_gate(arm, "full23_v1", metrics, comparisons)
        for arm in ARMS
    }
    eligible = [arm for arm in ARMS if eligibility[arm]["pass"]]
    complexity = {TOP1_ARM: 1, TOPK_ARM: 2}
    selected = (
        min(
            eligible,
            key=lambda method: (
                -joint_metrics[method]["top1_utility"],
                -joint_metrics[method]["mrr_utility"],
                complexity[method],
            ),
        )
        if eligible else "full23_v1"
    )
    multireference_vs_top1 = _increment_gate(
        TOPK_ARM, TOP1_ARM, metrics, comparisons,
    )
    content_vs_permuted = _increment_gate(
        TOPK_ARM, PERMUTED, metrics, comparisons,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        for panel_name in PANELS:
            panel = panels[panel_name].base
            np.savez_compressed(
                staging / f"candidate_predictions_{panel_name}.npz",
                query_ptr=panel.query_ptr,
                query_ids=panel.query_keys,
                candidate_ids=panel.graph.molecule_ik14,
                labels=panel.labels,
                **method_scores[panel_name],
            )
        report = {
            "status": "FULL23_MULTIREFERENCE_GNPS_DEVELOPMENT_ADJUDICATION_COMPLETE",
            "dataset": {
                "name": "GNPS Gold/Silver",
                "role": "consumed development adjudication",
                "allow_final_external_claim": False,
                "queries": {
                    name: len(panel.base.query_ptr) - 1 for name, panel in panels.items()
                },
            },
            "anchor": "frozen Full23 Additive v1",
            "full23_checkpoint_sha256": CHECKPOINT_SHA256,
            "readout_checkpoint_sha256": sha256_file(args.readout_checkpoint),
            "encoder_names": list(ENCODER_NAMES),
            "official_embedding_direct_input": False,
            "full23_anchor_has_indirect_official_dependency": True,
            "predeclared_representation_arm": TOPK_ARM,
            "mechanism_control": TOP1_ARM,
            "development_selection": {
                "selected": selected,
                "rule": (
                    "An embedding arm must beat Full23 in both panels with formula-"
                    "cluster CI lower bound >0, corrected>introduced and no near-"
                    "Recall@1 loss. Among eligible arms choose unique-query-balanced "
                    "Top-1 then MRR, preferring Top1 on an exact tie; otherwise retain v1."
                ),
                "promotion_from_full23_v1": selected != "full23_v1",
                "selection_data_role": "consumed GNPS development",
                "eligibility": eligibility,
            },
            "mechanism_adjudication": {
                "top3_beats_capacity_matched_top1": multireference_vs_top1,
                "top3_beats_candidate_reference_permutation": content_vs_permuted,
                "multi_reference_content_supported": bool(
                    multireference_vs_top1["pass"] and content_vs_permuted["pass"]
                ),
                "boundary": (
                    "Selection of an embedding arm and support for multi-reference "
                    "content are separate claims. Top1 may improve the readout without "
                    "establishing a multi-reference mechanism."
                ),
            },
            "top_k": int(readout["top_k"]),
            "parameter_count": models[TOPK_ARM].trainable_parameters,
            "reference_multiplicity": _reference_multiplicity(panels),
            "metrics": metrics,
            "comparisons": comparisons,
            "inferential_comparisons": [
                f"{TOPK_ARM}_vs_full23_v1",
                f"{TOPK_ARM}_vs_{TOP1_ARM}",
                f"{TOP1_ARM}_vs_full23_v1",
                f"{TOPK_ARM}_vs_{PERMUTED}",
            ],
            "query_balanced_joint": joint_metrics,
            "falsification": {
                "name": "candidate_reference_permutation_within_query_and_reference_count",
                "uses_truth": False,
                "permutation_audit": permutation_audit,
                "interpretation": (
                    "If the primary gain survives candidate-reference permutation, "
                    "coverage/multiplicity rather than embedding content is a plausible shortcut."
                ),
            },
            "metric_semantics": {
                "tie_policy": (
                    "strict rank: every candidate tied with or above the positive counts "
                    "ahead; Recall@1 therefore requires a unique positive top score"
                ),
                "macro_query_auprc": (
                    "equals MRR because every closed-set query has exactly one positive"
                ),
                "micro_candidate_metrics": (
                    "candidate-row weighted; queries with more candidates contribute "
                    "more rows"
                ),
                "top1_top2_margin": (
                    "descriptive within a method only; raw score scales differ across methods"
                ),
                "multiplicity": (
                    "query recurrence across panels is balanced only in query_balanced_joint"
                ),
            },
            "input_sha256": _input_hashes(args),
            "claim_boundary": (
                "Top3 vs Full23 measures the complete representation increment; Top3 vs "
                "capacity-matched Top1 isolates multi-reference use. GNPS has already been "
                "consumed and is used here for development adjudication, not untouched "
                "external confirmation or a SOTA claim. All other method metrics are "
                "descriptive rather than additional inferential tests."
            ),
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        os.replace(staging, args.output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps({
        "status": "FULL23_MULTIREFERENCE_GNPS_DEVELOPMENT_ADJUDICATION_COMPLETE",
        "selected": selected,
        "output": str(args.output),
    }), flush=True)


if __name__ == "__main__":
    main()
