#!/usr/bin/env python
"""Paired formula-cluster comparison of biology-relation vs spectral-only embeddings."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from noise_final_core import CandidateGraph


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(root: Path, seed: int, n: int) -> tuple[np.ndarray, np.ndarray, dict]:
    old = np.full(n, -1, np.int16); new = np.full(n, -1, np.int16)
    hashes, reports = {}, []
    for fold in range(5):
        directory = root / f"fold_{fold}" / f"seed_{seed}"
        path = directory / "heldout_predictions.npz"
        report_path = directory / "report.json"
        with np.load(path) as body:
            query = np.asarray(body["query"], np.int64)
            if np.any(old[query] != -1):
                raise RuntimeError("duplicate OOF query")
            old[query] = body["old_rank"]; new[query] = body["new_rank"]
        reports.append(json.loads(report_path.read_text(encoding="utf-8")))
        hashes[str(fold)] = {"prediction": sha256(path), "report": sha256(report_path)}
    if np.any(old < 1) or np.any(new < 1):
        raise RuntimeError("incomplete OOF coverage")
    return old, new, {"hashes": hashes, "reports": reports}


def aggregate_relation_readout(reports: list[dict]) -> dict:
    """Pool fold diagnostics by their strictly held-out pair counts."""
    totals = {"n": 0, "official_correct": 0.0, "adapted_correct": 0.0,
              "official_reaction_recalled": 0.0, "adapted_reaction_recalled": 0.0,
              "reaction_n": 0}
    for report in reports:
        readout = report.get("heldout_relation_readout", {})
        official = readout.get("official", {})
        adapted = readout.get("adapted", {})
        n = int(official.get("n") or 0)
        if n == 0:
            continue
        if int(adapted.get("n") or 0) != n:
            raise RuntimeError("official/adapted relation readout sizes differ")
        reaction_n = int(official.get("class_counts", {}).get("reaction", 0))
        totals["n"] += n
        totals["official_correct"] += float(official["accuracy"]) * n
        totals["adapted_correct"] += float(adapted["accuracy"]) * n
        totals["official_reaction_recalled"] += float(official["reaction_recall"]) * reaction_n
        totals["adapted_reaction_recalled"] += float(adapted["reaction_recall"]) * reaction_n
        totals["reaction_n"] += reaction_n
    if totals["n"] == 0:
        return {"n": 0, "status": "unavailable"}
    official_accuracy = totals["official_correct"] / totals["n"]
    adapted_accuracy = totals["adapted_correct"] / totals["n"]
    reaction_n = totals["reaction_n"]
    return {
        "n": totals["n"], "reaction_n": reaction_n,
        "official_accuracy": official_accuracy,
        "adapted_accuracy": adapted_accuracy,
        "adapted_minus_official_accuracy": adapted_accuracy - official_accuracy,
        "official_reaction_recall": (
            totals["official_reaction_recalled"] / reaction_n if reaction_n else None
        ),
        "adapted_reaction_recall": (
            totals["adapted_reaction_recalled"] / reaction_n if reaction_n else None
        ),
        "protocol": "pooled strict both-endpoint-formula-heldout pairs; diagnostic only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=Path("data/validation/g8r_error_atlas_listwise_cache.npz"))
    parser.add_argument("--spectral-root", type=Path, default=Path("data/validation/bioaware_embedding_adapter_v2_spectral_only"))
    parser.add_argument("--biology-root", type=Path, default=Path("data/validation/bioaware_embedding_adapter_v2_biology_relation"))
    parser.add_argument("--output", type=Path, default=Path("data/validation/bioaware_embedding_adapter_v2_ablation.json"))
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"fail-closed: output exists: {args.output}")
    graph = CandidateGraph(args.graph)
    old_s, spectral, provenance_s = load(args.spectral_root, args.seed, graph.n_queries)
    old_b, biology, provenance_b = load(args.biology_root, args.seed, graph.n_queries)
    if not np.array_equal(old_s, old_b):
        raise RuntimeError("ablation arms do not share the official baseline")
    frame = pd.DataFrame({"formula": graph.query_formula.astype(str),
                          "spectral": (spectral == 1).astype(int),
                          "biology": (biology == 1).astype(int)})
    frame["delta"] = frame.biology - frame.spectral
    groups = {key: group for key, group in frame.groupby("formula", sort=True)}
    keys = sorted(groups); rng = np.random.default_rng(args.seed); draws = np.empty(args.bootstrap)
    for index in range(args.bootstrap):
        selected = rng.choice(keys, len(keys), replace=True)
        draws[index] = pd.concat([groups[key] for key in selected], ignore_index=True).delta.mean()
    corrected = int(((spectral != 1) & (biology == 1)).sum())
    introduced = int(((spectral == 1) & (biology != 1)).sum())
    near = graph.query_has_near
    relation_readout = aggregate_relation_readout(provenance_b["reports"])
    relation_readout_nonnegative = (
        relation_readout.get("status") != "unavailable"
        and relation_readout["adapted_minus_official_accuracy"] >= 0
        and relation_readout["adapted_reaction_recall"] >= relation_readout["official_reaction_recall"]
    )
    report = {
        "status": "bioaware_embedding_v2_paired_ablation_complete", "formal": True,
        "queries": graph.n_queries,
        "official_recall1": float((old_s == 1).mean()),
        "spectral_only_recall1": float((spectral == 1).mean()),
        "biology_relation_recall1": float((biology == 1).mean()),
        "biology_minus_spectral_delta": float(frame.delta.mean()),
        "biology_minus_spectral_near_delta": float((biology[near] == 1).mean() - (spectral[near] == 1).mean()),
        "biology_vs_spectral_corrected": corrected,
        "biology_vs_spectral_introduced": introduced,
        "biology_arm_relation_readout": relation_readout,
        "formula_cluster_bootstrap": {"mean": float(frame.delta.mean()),
            "ci_low": float(np.quantile(draws, .025)), "ci_high": float(np.quantile(draws, .975)),
            "clusters": len(keys), "resamples": args.bootstrap},
        "gates": {"biology_formula_ci_positive": float(np.quantile(draws, .025)) > 0,
                  "biology_corrected_gt_introduced": corrected > introduced,
                  "biology_near_nonnegative": float((biology[near] == 1).mean() - (spectral[near] == 1).mean()) >= 0,
                  "biology_relation_readout_nonnegative": relation_readout_nonnegative},
        "contracts": {"only_difference": "lambda_relation 0.10 versus 0.0",
                      "reaction_neighbour_is_positive": False, "P2b": "forbidden", "P3": "not opened"},
        "provenance": {"graph_sha256": sha256(args.graph), "spectral": provenance_s["hashes"],
                       "biology": provenance_b["hashes"]},
        "claim_limit": "Formula-OOF ablation; independent transfer is still required for an embedding claim.",
    }
    report["gates"]["pass"] = all(report["gates"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
