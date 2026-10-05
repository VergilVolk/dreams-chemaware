#!/usr/bin/env python
"""A2 falsification pilot for chemical-to-biological-context transfer.

Models are deliberately simple and auditable:
  B0: global observation rate after repository exposure;
  B1: B0 plus coarse analytical detectability groups;
  M0: ClassyFire-superclass context profile;
  M1: Morgan-fingerprint neighbour transfer of context profiles.

No missing match is called a biological absence.  The likelihood concerns the
public-repository *observation process* (project-level spectral observations).
"""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import gzip
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


EXCLUDED_PRIMARY_CONTEXTS = {
    "missing value",
    "reference material",
    "blank_analysis",
    "blank_QC",
    "blank_extraction",
    "blank_culturemedia",
    "pool_QC",
}


def _read(path: Path) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _decode_fp(value: str) -> np.ndarray:
    record = ast.literal_eval(value)
    if not isinstance(record, dict) or not record.get("$base64") or "encoded" not in record:
        raise ValueError("unexpected fingerprint serialization")
    raw = base64.b64decode(record["encoded"])
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    if bits.size == 0:
        raise ValueError("empty fingerprint")
    return bits.astype(np.float32)


def _polarity(value: str) -> str:
    text = str(value).lower()
    if "pos" in text or "+" in text:
        return "positive"
    if "neg" in text or "-" in text:
        return "negative"
    return "unknown"


def _adduct_family(value: str) -> str:
    text = str(value).upper()
    if "M+H" in text:
        return "protonated"
    if "M-H" in text:
        return "deprotonated"
    if "M+NA" in text or "M+K" in text:
        return "alkali"
    return "other"


def _technical_key(anchor: dict[str, str]) -> tuple[str, str, int]:
    try:
        mz = float(anchor.get("precursor_mz") or "nan")
    except ValueError:
        mz = math.nan
    mass_bin = min(5, max(0, int(mz // 200))) if math.isfinite(mz) else -1
    return _polarity(anchor.get("ion_mode", "")), _adduct_family(anchor.get("adduct", "")), mass_bin


def _rate(counts: np.ndarray, opportunity: np.ndarray, prior: np.ndarray, alpha: float) -> np.ndarray:
    return (counts + alpha * prior) / np.maximum(opportunity + alpha, 1e-12)


def _tanimoto(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    intersection = matrix @ query
    union = matrix.sum(axis=1) + query.sum() - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def _evaluate(
    identities: list[str],
    y: np.ndarray,
    opportunity: np.ndarray,
    scores: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    result: dict[str, Any] = {}
    per_identity: dict[str, np.ndarray] = {}
    for name, rate in scores.items():
        rate = np.clip(rate, 1e-9, 1.0 - 1e-9)
        lamb = np.clip(rate * opportunity[None, :], 1e-9, None)
        nll_cells = lamb - y * np.log(lamb)
        positive_weight = y.sum(axis=1)
        identity_mrr = np.full(len(identities), np.nan, dtype=np.float64)
        identity_r3 = np.full(len(identities), np.nan, dtype=np.float64)
        for index in range(len(identities)):
            positive = np.flatnonzero(y[index] > 0)
            if positive.size == 0:
                continue
            ranks = np.array(
                [1 + np.sum(rate[index] >= rate[index, context]) - 1 for context in positive],
                dtype=np.float64,
            )
            weights = y[index, positive]
            identity_mrr[index] = np.average(1.0 / ranks, weights=weights)
            identity_r3[index] = np.average((ranks <= 3).astype(float), weights=weights)
        valid = np.isfinite(identity_mrr)
        result[name] = {
            "identities_with_positive_observations": int(valid.sum()),
            "project_weighted_mrr": float(np.average(identity_mrr[valid], weights=positive_weight[valid])),
            "identity_macro_mrr": float(identity_mrr[valid].mean()),
            "project_weighted_recall_at_3": float(np.average(identity_r3[valid], weights=positive_weight[valid])),
            "observation_poisson_nll_per_cell_without_constant": float(nll_cells.mean()),
        }
        per_identity[name] = identity_mrr
    return result, per_identity


def _bootstrap_delta(
    candidate: np.ndarray,
    baseline: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    valid = np.isfinite(candidate) & np.isfinite(baseline)
    delta = candidate[valid] - baseline[valid]
    if delta.size < 2:
        return {"mean": float(delta.mean()) if delta.size else math.nan, "ci_low": math.nan, "ci_high": math.nan, "n": int(delta.size)}
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, delta.size, size=(repeats, delta.size))
    means = delta[sampled].mean(axis=1)
    return {
        "mean": float(delta.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "n": int(delta.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a1-dir", type=Path, default=Path("data/validation/reverse_context_joint_a1c"))
    parser.add_argument("--output", type=Path, default=Path("data/validation/reverse_context_joint_a2/report.json"))
    parser.add_argument("--minimum-development-project-observations", type=int, default=20)
    parser.add_argument("--neighbours", type=int, default=16)
    parser.add_argument("--shrinkage-projects", type=float, default=5.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise RuntimeError(f"refusing to overwrite A2 result: {args.output}")

    manifest_path = args.a1_dir / "manifest.json"
    anchor_path = args.a1_dir / "anchors.csv.gz"
    observation_path = args.a1_dir / "positive_observations.csv.gz"
    split_path = args.a1_dir / "split_assignments.csv.gz"
    opportunity_path = Path("data/validation/reverse_context_joint_a0c/opportunities.csv.gz")
    for path in (manifest_path, anchor_path, observation_path, split_path, opportunity_path):
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("pass_to_model_baselines"):
        raise RuntimeError("A1 did not pass")
    anchors = {row["ik14"]: row for row in _read(anchor_path)}
    observations = _read(observation_path)
    opportunities = _read(opportunity_path)
    splits = _read(split_path)
    project_split = {row["key"]: row["split"] for row in splits if row["axis"] == "project"}
    identity_split = {row["key"]: row["split"] for row in splits if row["axis"] == "identity"}

    dev_context_counts = Counter()
    seen_triplets: set[tuple[str, str, str]] = set()
    for row in observations:
        context = row["sample_type"]
        if row["observation_type"] != "biological" or context in EXCLUDED_PRIMARY_CONTEXTS:
            continue
        triplet = (row["ik14"], row["dataset"], context)
        if triplet in seen_triplets:
            continue
        seen_triplets.add(triplet)
        if row["project_split"] == "development" and row["identity_split"] == "development":
            dev_context_counts[context] += 1
    contexts = sorted(
        context for context, count in dev_context_counts.items()
        if count >= args.minimum_development_project_observations
    )
    if len(contexts) < 6:
        raise RuntimeError(f"too few development-selected contexts: {contexts}")
    context_index = {context: index for index, context in enumerate(contexts)}

    opportunity_projects: dict[str, dict[str, set[str]]] = {
        "development": defaultdict(set), "test": defaultdict(set)
    }
    for row in opportunities:
        context = row.get("sample_type")
        dataset = row.get("dataset")
        if context not in context_index or dataset not in project_split:
            continue
        opportunity_projects[project_split[dataset]][context].add(dataset)
    exposure = {
        split: np.array([len(opportunity_projects[split][context]) for context in contexts], dtype=np.float64)
        for split in ("development", "test")
    }
    if np.any(exposure["development"] == 0) or np.any(exposure["test"] == 0):
        raise RuntimeError("selected context lacks opportunity projects in a split")

    identities = sorted(anchors)
    identity_index = {identity: index for index, identity in enumerate(identities)}
    counts = {
        "development": np.zeros((len(identities), len(contexts)), dtype=np.float64),
        "test": np.zeros((len(identities), len(contexts)), dtype=np.float64),
    }
    seen_triplets.clear()
    for row in observations:
        identity, dataset, context = row["ik14"], row["dataset"], row["sample_type"]
        if row["observation_type"] != "biological" or context not in context_index:
            continue
        triplet = (identity, dataset, context)
        if triplet in seen_triplets:
            continue
        seen_triplets.add(triplet)
        split = project_split[dataset]
        counts[split][identity_index[identity], context_index[context]] += 1.0

    train_ids = [identity for identity in identities if identity_split[identity] == "development"]
    train_indices = np.array([identity_index[identity] for identity in train_ids], dtype=int)
    dev_counts = counts["development"][train_indices]
    global_count = dev_counts.sum(axis=0)
    global_denom = len(train_ids) * exposure["development"]
    global_prior = (global_count + 0.5) / (global_denom + 1.0)

    technical_groups: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    superclass_groups: dict[str, list[int]] = defaultdict(list)
    for local_index, identity in enumerate(train_ids):
        technical_groups[_technical_key(anchors[identity])].append(local_index)
        superclass_groups[anchors[identity]["classyfire_superclass"]].append(local_index)

    technical_rates: dict[tuple[str, str, int], np.ndarray] = {}
    for key, local_indices in technical_groups.items():
        group_count = dev_counts[local_indices].sum(axis=0)
        group_opp = len(local_indices) * exposure["development"]
        technical_rates[key] = _rate(group_count, group_opp, global_prior, args.shrinkage_projects)
    superclass_rates: dict[str, np.ndarray] = {}
    for key, local_indices in superclass_groups.items():
        group_count = dev_counts[local_indices].sum(axis=0)
        group_opp = len(local_indices) * exposure["development"]
        superclass_rates[key] = _rate(group_count, group_opp, global_prior, args.shrinkage_projects)

    train_fps = np.stack([_decode_fp(anchors[identity]["fp_morgan"]) for identity in train_ids])
    if len({fp.size for fp in train_fps}) != 1:
        raise RuntimeError("inconsistent fingerprint dimensions")
    individual_rates = np.empty_like(dev_counts)
    for local_index, identity in enumerate(train_ids):
        prior = technical_rates.get(_technical_key(anchors[identity]), global_prior)
        individual_rates[local_index] = _rate(
            dev_counts[local_index], exposure["development"], prior, args.shrinkage_projects
        )

    def score_identities(eval_ids: list[str]) -> dict[str, np.ndarray]:
        b0 = np.tile(global_prior, (len(eval_ids), 1))
        b1 = np.stack([
            technical_rates.get(_technical_key(anchors[identity]), global_prior)
            for identity in eval_ids
        ])
        m0 = np.stack([
            superclass_rates.get(anchors[identity]["classyfire_superclass"], b1[index])
            for index, identity in enumerate(eval_ids)
        ])
        m1_rows = []
        for index, identity in enumerate(eval_ids):
            similarity = _tanimoto(_decode_fp(anchors[identity]["fp_morgan"]), train_fps)
            if identity in train_ids:
                similarity[train_ids.index(identity)] = -1.0
            chosen = np.argsort(-similarity)[: args.neighbours]
            positive = chosen[similarity[chosen] > 0]
            if positive.size == 0:
                m1_rows.append(m0[index])
                continue
            weights = np.square(similarity[positive])
            neighbour = np.average(individual_rates[positive], axis=0, weights=weights)
            support = float(weights.sum())
            m1_rows.append((support * neighbour + args.shrinkage_projects * m0[index]) / (support + args.shrinkage_projects))
        return {"B0": b0, "B1": b1, "M0": m0, "M1": np.stack(m1_rows)}

    panels = {
        "project_holdout": [identity for identity in train_ids if counts["test"][identity_index[identity]].sum() > 0],
        "identity_holdout": [identity for identity in identities if identity_split[identity] == "test" and counts["development"][identity_index[identity]].sum() > 0],
        "joint_holdout": [identity for identity in identities if identity_split[identity] == "test" and counts["test"][identity_index[identity]].sum() > 0],
    }
    panel_reports: dict[str, Any] = {}
    bootstraps: dict[str, Any] = {}
    for panel_name, eval_ids in panels.items():
        project_axis = "test" if panel_name in {"project_holdout", "joint_holdout"} else "development"
        eval_indices = np.array([identity_index[identity] for identity in eval_ids], dtype=int)
        y = counts[project_axis][eval_indices]
        result, per_identity = _evaluate(eval_ids, y, exposure[project_axis], score_identities(eval_ids))
        panel_reports[panel_name] = {
            "identities": len(eval_ids),
            "positive_project_context_observations": int(y.sum()),
            "contexts": contexts,
            "models": result,
        }
        bootstraps[panel_name] = {
            "M1_minus_B1_identity_macro_mrr": _bootstrap_delta(
                per_identity["M1"], per_identity["B1"], args.bootstrap_resamples, args.seed + len(bootstraps)
            ),
            "M1_minus_M0_identity_macro_mrr": _bootstrap_delta(
                per_identity["M1"], per_identity["M0"], args.bootstrap_resamples, args.seed + 100 + len(bootstraps)
            ),
        }

    project_ci = bootstraps["project_holdout"]["M1_minus_B1_identity_macro_mrr"]
    identity_ci = bootstraps["identity_holdout"]["M1_minus_B1_identity_macro_mrr"]
    joint_ci = bootstraps["joint_holdout"]["M1_minus_B1_identity_macro_mrr"]
    gates = {
        "project_holdout_M1_beats_B1_ci_positive": project_ci["ci_low"] > 0,
        "identity_holdout_M1_beats_B1_point_positive": identity_ci["mean"] > 0,
        "joint_holdout_M1_not_worse_than_B1_point": joint_ci["mean"] >= 0,
        "project_holdout_M1_poisson_nll_not_worse": (
            panel_reports["project_holdout"]["models"]["M1"]["observation_poisson_nll_per_cell_without_constant"]
            <= panel_reports["project_holdout"]["models"]["B1"]["observation_poisson_nll_per_cell_without_constant"]
        ),
    }
    report = {
        "status": "reverse_context_joint_a2_falsification_complete",
        "formal": True,
        "task": "project-level positive observation intensity across biological sample types",
        "contexts_selected_on_development_only": contexts,
        "panels": panel_reports,
        "bootstrap": bootstraps,
        "gates": gates,
        "pass_to_dreams_embedding_model": all(gates.values()),
        "interpretation": (
            "M1 tests whether structural neighbourhood transfers biological-context observation "
            "profiles beyond global and analytical baselines. This is not metabolite identification."
        ),
        "zero_contract": "zero means no repository spectral observation under recorded opportunity, not biological absence",
        "provenance": {
            "a1_manifest_sha256": _sha256(manifest_path),
            "anchors_sha256": _sha256(anchor_path),
            "observations_sha256": _sha256(observation_path),
            "splits_sha256": _sha256(split_path),
            "opportunities_sha256": _sha256(opportunity_path),
            "script_sha256": _sha256(Path(__file__)),
        },
        "parameters": {
            "minimum_development_project_observations": args.minimum_development_project_observations,
            "neighbours": args.neighbours,
            "shrinkage_projects": args.shrinkage_projects,
            "bootstrap_resamples": args.bootstrap_resamples,
            "seed": args.seed,
        },
        "claim_limit": (
            "A passing pilot only justifies replacing structural fingerprints with frozen DreaMS "
            "embeddings under the same splits. It does not establish biological mechanism, disease "
            "association, or superior metabolite annotation."
        ),
    }
    _atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
