#!/usr/bin/env python
"""Apply the frozen GNPS-trained B ranker to the fixed Enveda candidate axes.

This is a replay-only evaluator.  It never fits a model and deliberately reads
only the frozen B checkpoint, the three required upstream score channels from
the Enveda prediction ledger, and the already-exported continuous ChemAware
evidence.  Other learned Enveda arms in the prediction ledger are not loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from conditional_null_energy_core import AnchoredEvidenceEnergy
from train_wse_conditional_residual import (
    CHEMICAL_FEATURE_NAMES,
    SPECTRAL_FEATURE_NAMES,
    build_chemical_features,
    query_score_features,
    transform_features,
)


PANELS = ("identity_disjoint", "formula_disjoint")
WSE = "weighted_spectral_entropy"
NOISE = "noise_v1"
P2B = "p2b_noise_v1_frozen"
SCORE_CHANNELS = (WSE, NOISE, P2B)
CHEM_SCHEMA = "gnps_chemaware_continuous_null_evidence_v1"
OUTPUT_SCHEMA = "frozen_B_on_enveda_candidate_scores_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prediction-ledger", type=Path, required=True)
    parser.add_argument("--chem-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--batch-candidates", type=int, default=131_072)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.bootstrap_resamples < 1:
        parser.error("--bootstrap-resamples must be positive")
    if args.batch_candidates < 1:
        parser.error("--batch-candidates must be positive")
    return args


def load_checkpoint(path: Path, device: str) -> tuple[dict, AnchoredEvidenceEnergy]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before weights_only was added.
        payload = torch.load(path, map_location="cpu")
    expected_features = (*SPECTRAL_FEATURE_NAMES, *CHEMICAL_FEATURE_NAMES)
    checks = {
        "model_class": payload.get("model_class") == "AnchoredEvidenceEnergy",
        "selected_arm": payload.get("selected_arm") == "B_joint_residual",
        "mode": payload.get("mode") == "joint",
        "spectral_feature_names": tuple(payload.get("spectral_feature_names", ()))
        == tuple(SPECTRAL_FEATURE_NAMES),
        "chemical_feature_names": tuple(payload.get("chemical_feature_names", ()))
        == tuple(CHEMICAL_FEATURE_NAMES),
        "spectral_feature_count": int(payload.get("spectral_feature_count", -1))
        == len(SPECTRAL_FEATURE_NAMES),
        "enveda_unopened_when_frozen": payload.get("enveda_accessed") is False,
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise RuntimeError(f"checkpoint is not the frozen B contract: {failed}")
    mean = np.asarray(payload.get("scaler_mean"), dtype=np.float32)
    scale = np.asarray(payload.get("scaler_scale"), dtype=np.float32)
    if (
        mean.shape != (len(expected_features),)
        or scale.shape != mean.shape
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(scale))
        or np.any(scale <= 0)
        or np.any(mean[len(SPECTRAL_FEATURE_NAMES) :] != 0)
    ):
        raise RuntimeError("frozen B scaler violates its feature contract")
    bound = float(payload.get("residual_bound_raw_wse_units", float("nan")))
    hidden = int(payload.get("hidden", 0))
    if not np.isfinite(bound) or bound <= 0 or hidden < 1:
        raise RuntimeError("frozen B architecture metadata is invalid")
    model = AnchoredEvidenceEnergy(
        len(expected_features),
        mode="joint",
        spectral_feature_count=len(SPECTRAL_FEATURE_NAMES),
        hidden=hidden,
        residual_bound=bound,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    return payload, model


def _require_keys(available: set[str], required: set[str], label: str) -> None:
    missing = required - available
    if missing:
        raise RuntimeError(f"{label} misses required fields: {sorted(missing)}")


def _validate_query_ptr(query_ptr: np.ndarray, candidate_count: int, panel: str) -> None:
    ptr = np.asarray(query_ptr, dtype=np.int64)
    if (
        ptr.ndim != 1
        or len(ptr) < 2
        or int(ptr[0]) != 0
        or int(ptr[-1]) != candidate_count
        or np.any(np.diff(ptr) < 2)
    ):
        raise RuntimeError(f"{panel} query_ptr does not define >=2-candidate blocks")


def _candidate_ids_unique_within_query(
    candidate_ids: np.ndarray, query_ptr: np.ndarray, panel: str,
) -> None:
    ids = np.asarray(candidate_ids).astype("U")
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        block = ids[int(left) : int(right)]
        if len(np.unique(block)) != len(block):
            raise RuntimeError(f"{panel} query {query} repeats a candidate identifier")


def load_panel(
    ledger: np.lib.npyio.NpzFile, chem_dir: Path, panel: str,
) -> dict[str, np.ndarray]:
    prefix = f"{panel}__"
    ledger_fields = {
        "query_ptr", "labels", "query_keys", "formula", "near", *SCORE_CHANNELS,
    }
    _require_keys(
        set(ledger.files), {prefix + field for field in ledger_fields},
        f"prediction ledger/{panel}",
    )
    raw = {
        field: np.asarray(ledger[prefix + field])
        for field in ledger_fields
    }
    candidate_count = len(raw[WSE])
    _validate_query_ptr(raw["query_ptr"], candidate_count, panel)
    query_count = len(raw["query_ptr"]) - 1
    if (
        raw["labels"].shape != (candidate_count,)
        or raw["query_keys"].shape != (query_count,)
        or raw["formula"].shape != (query_count,)
        or raw["near"].shape != (query_count,)
        or len(np.unique(raw["query_keys"].astype("U"))) != query_count
    ):
        raise RuntimeError(f"prediction ledger/{panel} query or candidate axis is malformed")
    if np.any(np.char.strip(raw["formula"].astype("U")) == ""):
        raise RuntimeError(f"prediction ledger/{panel} contains an empty formula")
    for channel in SCORE_CHANNELS:
        values = np.asarray(raw[channel], dtype=np.float32)
        if values.shape != (candidate_count,) or not np.all(np.isfinite(values)):
            raise RuntimeError(f"prediction ledger/{panel}/{channel} is invalid")
        raw[channel] = values
    labels = np.asarray(raw["labels"], dtype=np.int8)
    if np.any((labels != 0) & (labels != 1)):
        raise RuntimeError(f"prediction ledger/{panel} labels are not binary")
    if any(
        np.sum(labels[int(left) : int(right)] > 0) != 1
        for left, right in zip(raw["query_ptr"][:-1], raw["query_ptr"][1:])
    ):
        raise RuntimeError(f"prediction ledger/{panel} does not have one positive per query")
    raw["labels"] = labels
    raw["near"] = np.asarray(raw["near"], dtype=bool)
    raw["query_ptr"] = np.asarray(raw["query_ptr"], dtype=np.int64)
    raw["query_keys"] = raw["query_keys"].astype("U")
    raw["formula"] = np.char.strip(raw["formula"].astype("U"))

    chem_path = chem_dir / f"continuous_chem_{panel}.npz"
    if not chem_path.is_file():
        raise FileNotFoundError(chem_path)
    required_chem = {
        "schema", "panel", "query_ptr", "query_ids", "candidate_ids",
        "query_formula", "near_query", "availability", "utility_correct",
        "utility_zero", "utility_reversed", "utility_candidate_mismatch",
        "policy_sha256", "panel_sha256",
    }
    with np.load(chem_path, allow_pickle=False) as body:
        _require_keys(set(body.files), required_chem, f"continuous chemistry/{panel}")
        chem = {field: np.asarray(body[field]) for field in required_chem}
    if str(chem["schema"].item()) != CHEM_SCHEMA or str(chem["panel"].item()) != panel:
        raise RuntimeError(f"continuous chemistry/{panel} schema or panel is invalid")
    axis_checks = {
        "query_ptr": np.array_equal(chem["query_ptr"], raw["query_ptr"]),
        "query_formula": np.array_equal(
            np.char.strip(chem["query_formula"].astype("U")), raw["formula"]
        ),
        "near_query": np.array_equal(chem["near_query"].astype(bool), raw["near"]),
        "candidate_count": chem["candidate_ids"].shape == (candidate_count,),
        "query_count": chem["query_ids"].shape == (query_count,),
    }
    failed = [name for name, valid in axis_checks.items() if not valid]
    if failed:
        raise RuntimeError(f"Enveda candidate axis drifted for {panel}: {failed}")
    _candidate_ids_unique_within_query(chem["candidate_ids"], raw["query_ptr"], panel)
    policy_sha = str(chem["policy_sha256"].item())
    panel_sha = str(chem["panel_sha256"].item())
    if len(policy_sha) != 64 or len(panel_sha) != 64:
        raise RuntimeError(f"continuous chemistry/{panel} provenance hash is malformed")
    raw["chem"] = chem
    raw["chem_path"] = np.asarray(str(chem_path))
    raw["chem_sha256"] = np.asarray(sha256(chem_path))
    return raw


def build_features(panel: dict[str, np.ndarray]) -> np.ndarray:
    spectral = np.column_stack(tuple(
        query_score_features(panel[channel], panel["query_ptr"])
        for channel in SCORE_CHANNELS
    )).astype(np.float32)
    chemical, _nulls, _available = build_chemical_features(panel["chem"])
    features = np.column_stack((spectral, chemical)).astype(np.float32)
    expected = len(SPECTRAL_FEATURE_NAMES) + len(CHEMICAL_FEATURE_NAMES)
    if features.shape != (len(panel[WSE]), expected) or not np.all(np.isfinite(features)):
        raise RuntimeError("frozen B feature construction drifted")
    return features


@torch.inference_mode()
def score_b(
    model: AnchoredEvidenceEnergy,
    checkpoint: dict,
    panel: dict[str, np.ndarray],
    batch_candidates: int,
    device: str,
) -> np.ndarray:
    features = transform_features(
        build_features(panel),
        np.asarray(checkpoint["scaler_mean"], dtype=np.float32),
        np.asarray(checkpoint["scaler_scale"], dtype=np.float32),
    )
    base = np.asarray(panel[WSE], dtype=np.float32)
    output = np.empty_like(base)
    for left in range(0, len(base), batch_candidates):
        right = min(left + batch_candidates, len(base))
        output[left:right] = model(
            torch.as_tensor(base[left:right], dtype=torch.float32, device=device),
            torch.as_tensor(features[left:right], dtype=torch.float32, device=device),
        ).cpu().numpy().astype(np.float32)
    if not np.all(np.isfinite(output)):
        raise RuntimeError("frozen B produced non-finite candidate scores")
    return output


def strict_ranks(scores: np.ndarray, panel: dict[str, np.ndarray]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    ranks = np.empty(len(panel["query_ptr"]) - 1, dtype=np.int32)
    for query, (left, right) in enumerate(
        zip(panel["query_ptr"][:-1], panel["query_ptr"][1:])
    ):
        left, right = int(left), int(right)
        positive = int(np.flatnonzero(panel["labels"][left:right] > 0)[0])
        target = values[left + positive]
        # Strict policy: every tie, including the positive itself, is ahead.
        ranks[query] = int(np.sum(values[left:right] >= target))
    return ranks


def metric_summary(ranks: np.ndarray, near: np.ndarray) -> dict:
    result = {
        "queries": int(len(ranks)),
        "recall@1": float(np.mean(ranks == 1)),
        "mrr": float(np.mean(1.0 / ranks)),
    }
    near_ranks = ranks[np.asarray(near, dtype=bool)]
    result["near"] = {
        "queries": int(len(near_ranks)),
        "recall@1": float(np.mean(near_ranks == 1)) if len(near_ranks) else None,
        "mrr": float(np.mean(1.0 / near_ranks)) if len(near_ranks) else None,
    }
    return result


def formula_cluster_ci(
    delta: np.ndarray, formulas: np.ndarray, resamples: int, seed: int,
) -> dict:
    values = np.asarray(delta, dtype=np.float64)
    formula = np.asarray(formulas).astype("U")
    unique, inverse = np.unique(formula, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique))
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(resamples, dtype=np.float64)
    # Chunking bounds memory while retaining the exact formula-cluster draw.
    for start in range(0, resamples, 512):
        stop = min(start + 512, resamples)
        chosen = rng.integers(0, len(unique), size=(stop - start, len(unique)))
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
        "seed": int(seed),
    }


def compare_b_to_wse(
    b_ranks: np.ndarray,
    wse_ranks: np.ndarray,
    panel: dict[str, np.ndarray],
    resamples: int,
    seed: int,
) -> dict:
    b_top1 = b_ranks == 1
    wse_top1 = wse_ranks == 1
    delta = b_top1.astype(np.float64) - wse_top1.astype(np.float64)
    near = np.asarray(panel["near"], dtype=bool)
    comparison = {
        "corrected": int(np.sum(b_top1 & ~wse_top1)),
        "introduced": int(np.sum(~b_top1 & wse_top1)),
        "formula_cluster_bootstrap_95ci": formula_cluster_ci(
            delta, panel["formula"], resamples, seed,
        ),
    }
    near_delta = delta[near]
    comparison["near"] = {
        "queries": int(np.sum(near)),
        "corrected": int(np.sum(b_top1[near] & ~wse_top1[near])),
        "introduced": int(np.sum(~b_top1[near] & wse_top1[near])),
        "delta_recall@1": float(np.mean(near_delta)) if len(near_delta) else None,
        "delta_recall@1_pp": float(100 * np.mean(near_delta)) if len(near_delta) else None,
    }
    return comparison


def main() -> None:
    args = arguments()
    for path in (args.checkpoint, args.prediction_ledger):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.chem_dir.is_dir():
        raise FileNotFoundError(args.chem_dir)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    checkpoint, model = load_checkpoint(args.checkpoint, args.device)
    panels: dict[str, dict[str, np.ndarray]] = {}
    with np.load(args.prediction_ledger, allow_pickle=False) as ledger:
        for panel in PANELS:
            panels[panel] = load_panel(ledger, args.chem_dir, panel)

    staging = args.output.with_name(f".{args.output.name}.staging.{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    try:
        score_arrays: dict[str, np.ndarray] = {
            "schema": np.asarray(OUTPUT_SCHEMA),
            "checkpoint_sha256": np.asarray(sha256(args.checkpoint)),
        }
        panel_reports: dict[str, dict] = {}
        for panel_index, panel_name in enumerate(PANELS):
            panel = panels[panel_name]
            b_score = score_b(
                model, checkpoint, panel, args.batch_candidates, args.device,
            )
            wse_score = np.asarray(panel[WSE], dtype=np.float32)
            b_ranks = strict_ranks(b_score, panel)
            wse_ranks = strict_ranks(wse_score, panel)
            prefix = f"{panel_name}__"
            score_arrays.update({
                prefix + "query_ptr": panel["query_ptr"],
                prefix + "query_keys": panel["query_keys"],
                prefix + "query_ids": panel["chem"]["query_ids"].astype("U"),
                prefix + "candidate_ids": panel["chem"]["candidate_ids"].astype("U"),
                prefix + "weighted_spectral_entropy": wse_score,
                prefix + "B_frozen_GNPS_fullfit": b_score,
            })
            panel_reports[panel_name] = {
                "queries": int(len(panel["query_ptr"]) - 1),
                "candidate_molecules": int(len(wse_score)),
                "chem_available_candidates": int(np.sum(panel["chem"]["availability"])),
                "methods": {
                    "weighted_spectral_entropy": metric_summary(wse_ranks, panel["near"]),
                    "B_frozen_GNPS_fullfit": metric_summary(b_ranks, panel["near"]),
                },
                "B_vs_weighted_spectral_entropy": compare_b_to_wse(
                    b_ranks, wse_ranks, panel, args.bootstrap_resamples,
                    args.seed + 1000 * panel_index,
                ),
                "axis": {
                    "minimum_candidates_per_query": int(np.min(np.diff(panel["query_ptr"]))),
                    "maximum_candidates_per_query": int(np.max(np.diff(panel["query_ptr"]))),
                    "policy_sha256": str(panel["chem"]["policy_sha256"].item()),
                    "panel_sha256": str(panel["chem"]["panel_sha256"].item()),
                    "continuous_chem_sha256": str(panel["chem_sha256"].item()),
                },
            }
            print(
                f"[frozen B] {panel_name}: "
                f"B R@1={panel_reports[panel_name]['methods']['B_frozen_GNPS_fullfit']['recall@1']:.6f}; "
                f"WSE R@1={panel_reports[panel_name]['methods']['weighted_spectral_entropy']['recall@1']:.6f}",
                flush=True,
            )

        score_path = staging / "candidate_scores.npz"
        np.savez_compressed(score_path, **score_arrays)
        report = {
            "status": "FROZEN_B_ON_ENVEDA_POSTHOC_TRANSFER_EVALUATION_COMPLETE",
            "dataset": "Enveda-180 cross-condition benchmark primary panels",
            "evaluation_role": "consumed_development_posthoc_reverse_transfer_diagnostic",
            "training_or_selection_on_enveda": False,
            "model_selection_results_read": False,
            "checkpoint": {
                "path": str(args.checkpoint),
                "sha256": sha256(args.checkpoint),
                "model_class": "AnchoredEvidenceEnergy",
                "selected_arm": "B_joint_residual",
                "fit_scope": "all GNPS identity development queries",
                "training_queries": int(checkpoint["training_queries"]),
                "training_seed": int(checkpoint["training_seed"]),
                "residual_bound_raw_wse_units": float(
                    checkpoint["residual_bound_raw_wse_units"]
                ),
                "enveda_accessed_when_frozen": bool(checkpoint["enveda_accessed"]),
            },
            "panels": panel_reports,
            "candidate_scores": {
                "path": "candidate_scores.npz",
                "sha256": sha256(score_path),
                "contains_truth": False,
            },
            "input_sha256": {
                "prediction_ledger": sha256(args.prediction_ledger),
                **{
                    f"continuous_chem_{panel}": str(
                        panels[panel]["chem_sha256"].item()
                    )
                    for panel in PANELS
                },
            },
            "evaluation": {
                "strict_tie_policy": "all tied candidates count ahead of the positive",
                "formula_cluster_bootstrap_resamples": args.bootstrap_resamples,
                "seed": args.seed,
                "no_retraining": True,
                "no_reencoding": True,
            },
            "claim_boundary": (
                "B is a frozen GNPS-trained full-fit model applied unchanged to consumed Enveda. "
                "This estimates GNPS-to-Enveda transfer on fixed candidate axes; it is not an "
                "untouched external or SOTA result and is not an architecture-only comparison "
                "against heads trained with Enveda labels."
            ),
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
        "output": str(args.output),
        "candidate_scores_sha256": report["candidate_scores"]["sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
