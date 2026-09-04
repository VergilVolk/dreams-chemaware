#!/usr/bin/env python
"""Apply the frozen BioAware v2 expert to a KGMN-200STD hidden-seed test.

The standard-mixture identities are split before any network features are
constructed.  For a hidden identity, every exact feature carrying that
identity is removed from the seed set.  Candidate identities and mass-window
feature assignments remain deployment-visible inputs.  The frozen expert is
loaded without fitting or threshold selection.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from annotation.bioaware_negative_expert import (  # noqa: E402
    FrozenNegativeBioAwareExpert,
    apply_frozen_negative_bioaware_expert,
)
from dreams.utils.io import read_msp  # noqa: E402
from metdna3_similarity import metdna3_reverse_dot  # noqa: E402


FEATURES = (
    "spectral_score", "known_mass_candidate_fraction", "known_path_fraction",
    "known_inverse_depth_mean", "known_log_seed_support_mean", "known_log_degree",
    "edge0_complete_fraction", "edge0_bottleneck_mean", "edge1_complete_fraction",
    "edge1_bottleneck_mean", "predicted_edge_increment",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adjacency(edges: pd.DataFrame, maximum_step: int) -> dict[str, set[str]]:
    selected = edges[pd.to_numeric(edges["minimum_step"], errors="raise") <= maximum_step]
    graph: dict[str, set[str]] = defaultdict(set)
    for left, right in selected[["ik14_a", "ik14_b"]].itertuples(index=False):
        left, right = str(left), str(right)
        graph[left].add(right)
        graph[right].add(left)
    return graph


def shortest_paths(
    candidate: str,
    graph: dict[str, set[str]],
    seeds: set[str],
    observed: set[str],
    maximum_depth: int = 2,
    maximum_paths: int = 500,
) -> list[tuple[str, ...]]:
    if candidate not in observed:
        return []
    frontier = [(candidate,)]
    for _depth in range(1, maximum_depth + 1):
        reached: list[tuple[str, ...]] = []
        following: list[tuple[str, ...]] = []
        for path in frontier:
            for target in sorted(graph.get(path[-1], set())):
                if target in path:
                    continue
                extended = path + (target,)
                if target in seeds:
                    reached.append(extended)
                elif target in observed:
                    following.append(extended)
        if reached:
            return sorted(set(reached))[:maximum_paths]
        frontier = following
        if not frontier:
            break
    return []


def msp_tensors(path: Path, required_names: set[str], maximum_peaks: int = 100) -> dict[str, np.ndarray]:
    frame = read_msp(path)
    tensors: dict[str, np.ndarray] = {}
    for row in frame.itertuples(index=False):
        name = str(row.name)
        if name not in required_names:
            continue
        spectrum = np.asarray(row.spectrum, dtype=float)
        if spectrum.ndim != 2:
            raise RuntimeError(f"invalid MSP spectrum for {name}")
        fragments = spectrum.T if spectrum.shape[0] == 2 else spectrum
        if fragments.shape[1] != 2:
            raise RuntimeError(f"invalid MSP spectrum shape for {name}: {spectrum.shape}")
        valid = (
            np.isfinite(fragments).all(axis=1)
            & (fragments[:, 0] > 0) & (fragments[:, 1] > 0)
        )
        fragments = fragments[valid]
        if len(fragments) > maximum_peaks:
            keep = np.argsort(fragments[:, 1], kind="stable")[-maximum_peaks:]
            fragments = fragments[keep]
        fragments = fragments[np.argsort(fragments[:, 0], kind="stable")]
        if fragments.size == 0:
            raise RuntimeError(f"empty required MSP spectrum: {name}")
        fragments[:, 1] /= fragments[:, 1].max()
        tensors[name] = np.vstack([[float(row.precursor_mz), 1.0], fragments])
    missing = sorted(required_names - set(tensors))
    if missing:
        raise RuntimeError(f"required MSP feature spectra are missing: {missing[:10]}")
    return tensors


def best_path_bottleneck(
    paths: list[tuple[str, ...]],
    query_name: str,
    node_options: dict[str, list[str]],
    tensors: dict[str, np.ndarray],
    cache: dict[tuple[str, str], float],
) -> tuple[int, float]:
    completed = 0
    best = 0.0
    for identity_path in paths:
        states: dict[str, float] = {query_name: 1.0}
        for identity in identity_path[1:]:
            updated: dict[str, float] = {}
            for source, previous in states.items():
                for target in node_options.get(identity, []):
                    if source == target or target == query_name:
                        continue
                    key = tuple(sorted((source, target)))
                    if key not in cache:
                        cache[key] = metdna3_reverse_dot(tensors[source], tensors[target])
                    score = min(previous, cache[key])
                    updated[target] = max(updated.get(target, 0.0), score)
            states = updated
            if not states:
                break
        if states:
            completed += 1
            best = max(best, max(states.values()))
    return completed, float(best)


def identity_cluster_bootstrap(
    frame: pd.DataFrame, repeats: int, seed: int
) -> dict[str, float | int]:
    identities = frame["truth_candidate_id"].astype(str).unique()
    groups = {
        identity: frame.loc[
            frame["truth_candidate_id"].astype(str).eq(identity), "delta"
        ].to_numpy(float)
        for identity in identities
    }
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=float)
    for index in range(repeats):
        sampled = rng.choice(identities, len(identities), replace=True)
        values[index] = np.mean(np.concatenate([groups[item] for item in sampled]))
    return {
        "mean": float(frame["delta"].mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "clusters": int(len(identities)),
        "resamples": int(repeats),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-dir", type=Path,
        default=Path("data/validation/bioaware_kgmn200std_confirmation_manifest_v2"),
    )
    parser.add_argument(
        "--msp", type=Path,
        default=Path("third_party/MetDNA2/inst/extdata/spectra_200STD_neg_200805.msp"),
    )
    parser.add_argument(
        "--network", type=Path,
        default=Path("data/reference/metdna2_emrn_network_20260828/metdna2_emrn_edges.csv.gz"),
    )
    parser.add_argument(
        "--artifact", type=Path,
        default=Path(
            "data/validation/bioaware_metdna3_negative_network_expert_v2_chemically_filtered/"
            "artifact.json"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_kgmn200std_hidden_seed_v1"),
    )
    parser.add_argument("--maximum-depth", type=int, default=2)
    parser.add_argument("--maximum-paths", type=int, default=500)
    parser.add_argument("--maximum-peaks", type=int, default=100)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    files = {
        "manifest_report": args.manifest_dir / "report.json",
        "queries": args.manifest_dir / "queries.csv.gz",
        "candidates": args.manifest_dir / "candidate_scores.csv.gz",
        "seed_features": args.manifest_dir / "seed_features.csv.gz",
        "splits": args.manifest_dir / "hidden_seed_splits.csv.gz",
        "msp": args.msp, "network": args.network, "artifact": args.artifact,
    }
    for path in files.values():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_dir}")
    manifest_report = json.loads(files["manifest_report"].read_text(encoding="utf-8"))
    if not manifest_report.get("pass_to_frozen_hidden_seed_confirmation"):
        raise RuntimeError("confirmation manifest did not pass")
    if sha256(args.artifact) != manifest_report["frozen_expert"]["sha256"]:
        raise RuntimeError("frozen expert hash changed after manifest construction")

    queries = pd.read_csv(files["queries"])
    candidate_scores = pd.read_csv(files["candidates"])
    seed_features = pd.read_csv(files["seed_features"])
    splits = pd.read_csv(files["splits"])
    edge_table = pd.read_csv(files["network"])
    graph0 = adjacency(edge_table, 0)
    graph1 = adjacency(edge_table, 1)
    required_names = set(seed_features["feature_name"].astype(str)) | set(queries["query_id"].astype(str))
    tensors = msp_tensors(args.msp, required_names, args.maximum_peaks)
    expert = FrozenNegativeBioAwareExpert.load(args.artifact)
    if tuple(expert.feature_names) != FEATURES:
        raise RuntimeError("frozen expert feature contract changed")

    mass_options: dict[str, set[str]] = defaultdict(set)
    for row in candidate_scores.itertuples(index=False):
        mass_options[str(row.candidate_id)].add(str(row.query_id))
    exact_options: dict[str, set[str]] = defaultdict(set)
    for row in seed_features.itertuples(index=False):
        exact_options[str(row.ik14)].add(str(row.feature_name))
    all_seed_identities = set(seed_features["ik14"].astype(str))
    result_rows: list[dict] = []
    feature_rows: list[dict] = []
    similarity_cache: dict[tuple[str, str], float] = {}

    for repeat in sorted(splits["repeat"].unique()):
        split = splits[splits["repeat"].eq(repeat)]
        visible_seeds = set(split.loc[split["role"].eq("seed"), "ik14"].astype(str))
        hidden = all_seed_identities - visible_seeds
        heldout_queries = queries[queries["truth_candidate_id"].astype(str).isin(hidden)].copy()
        if heldout_queries.empty:
            raise RuntimeError(f"repeat {repeat} has no evaluable hidden queries")
        query_ids = set(heldout_queries["query_id"].astype(str))
        local_candidates = candidate_scores[
            candidate_scores["query_id"].astype(str).isin(query_ids)
        ].copy()
        observed = set(candidate_scores["candidate_id"].astype(str)) | visible_seeds
        node_options = {
            identity: sorted(
                mass_options.get(identity, set())
                | (exact_options.get(identity, set()) if identity in visible_seeds else set())
            )
            for identity in observed
        }
        local_feature_rows: list[dict] = []
        for row in local_candidates.itertuples(index=False):
            candidate = str(row.candidate_id)
            query_id = str(row.query_id)
            known_paths = shortest_paths(
                candidate, graph0, visible_seeds, observed,
                args.maximum_depth, args.maximum_paths,
            )
            predicted_paths = shortest_paths(
                candidate, graph1, visible_seeds, observed,
                args.maximum_depth, args.maximum_paths,
            )
            complete0, bottleneck0 = best_path_bottleneck(
                known_paths, query_id, node_options, tensors, similarity_cache
            )
            complete1, bottleneck1 = best_path_bottleneck(
                predicted_paths, query_id, node_options, tensors, similarity_cache
            )
            minimum_depth = len(known_paths[0]) - 1 if known_paths else 0
            shortest_seed_count = len({path[-1] for path in known_paths})
            local_feature_rows.append({
                "query_id": query_id, "candidate_id": candidate,
                "spectral_score": float(row.spectral_score),
                "known_mass_candidate_fraction": 1.0,
                "known_path_fraction": float(bool(known_paths)),
                "known_inverse_depth_mean": 1.0 / minimum_depth if minimum_depth else 0.0,
                "known_log_seed_support_mean": float(np.log1p(shortest_seed_count)),
                "known_log_degree": float(np.log1p(len(graph0.get(candidate, set())))),
                "edge0_complete_fraction": float(complete0 > 0),
                "edge0_bottleneck_mean": bottleneck0,
                "edge1_complete_fraction": float(complete1 > 0),
                "edge1_bottleneck_mean": bottleneck1,
                "predicted_edge_increment": bottleneck1 - bottleneck0,
                "repeat": int(repeat),
                "known_shortest_paths": len(known_paths),
                "predicted_shortest_paths": len(predicted_paths),
                "raw_complete_known_paths": int(complete0),
                "raw_complete_predicted_paths": int(complete1),
            })
        local_features = pd.DataFrame(local_feature_rows)
        inference = local_features[["query_id", "candidate_id", *FEATURES]].copy()
        scored, decisions = apply_frozen_negative_bioaware_expert(inference, expert)
        if len(decisions) != len(heldout_queries):
            raise RuntimeError("expert decision coverage changed")
        truth_meta = heldout_queries[[
            "query_id", "truth_candidate_id", "truth_formula",
            "baseline_candidate_id", "baseline_correct",
        ]]
        evaluated = decisions.merge(truth_meta, on="query_id", validate="one_to_one")
        evaluated["repeat"] = int(repeat)
        evaluated["final_correct"] = evaluated["final_candidate_id"].astype(str).eq(
            evaluated["truth_candidate_id"].astype(str)
        )
        evaluated["corrected"] = ~evaluated["baseline_correct"].astype(bool) & evaluated["final_correct"]
        evaluated["introduced"] = evaluated["baseline_correct"].astype(bool) & ~evaluated["final_correct"]
        evaluated["delta"] = evaluated["final_correct"].astype(int) - evaluated["baseline_correct"].astype(int)
        result_rows.extend(evaluated.to_dict("records"))
        feature_rows.extend(local_feature_rows)

    results = pd.DataFrame(result_rows)
    features = pd.DataFrame(feature_rows)
    if results.empty or results.groupby(["repeat", "query_id"]).size().ne(1).any():
        raise RuntimeError("invalid repeated confirmation denominator")
    per_repeat = []
    for repeat, group in results.groupby("repeat", sort=True):
        per_repeat.append({
            "repeat": int(repeat), "queries": int(len(group)),
            "baseline_recall1": float(group["baseline_correct"].mean()),
            "bioaware_recall1": float(group["final_correct"].mean()),
            "delta_recall1": float(group["delta"].mean()),
            "corrected": int(group["corrected"].sum()),
            "introduced": int(group["introduced"].sum()),
            "interventions": int(group["intervene"].sum()),
        })
    ci = identity_cluster_bootstrap(results, args.bootstrap_resamples, args.seed)
    corrected = int(results["corrected"].sum())
    introduced = int(results["introduced"].sum())
    gates = {
        "identity_cluster_ci_low_positive": ci["ci_low"] > 0,
        "corrected_gt_introduced": corrected > introduced,
        "corrected_minus_2x_introduced_positive": corrected - 2 * introduced > 0,
        "all_repeats_nonnegative": all(row["delta_recall1"] >= 0 for row in per_repeat),
        "interventions_nonzero": bool(results["intervene"].sum()),
    }
    args.output_dir.mkdir(parents=True)
    result_path = args.output_dir / "transitions.csv.gz"
    feature_path = args.output_dir / "candidate_features.csv.gz"
    results.to_csv(result_path, index=False, compression="gzip")
    features.to_csv(feature_path, index=False, compression="gzip")
    report = {
        "status": "bioaware_kgmn200std_hidden_seed_confirmation_complete",
        "formal": True,
        "protocol": (
            "10-repeat identity-hidden artificial-standard confirmation; exact hidden identity removed from all "
            "seed features; frozen artifact and thresholds; no fitting"
        ),
        "query_rotations": int(len(results)),
        "unique_queries": int(results["query_id"].nunique()),
        "truth_identities": int(results["truth_candidate_id"].nunique()),
        "baseline_recall1": float(results["baseline_correct"].mean()),
        "bioaware_recall1": float(results["final_correct"].mean()),
        "delta_recall1": float(results["delta"].mean()),
        "corrected": corrected,
        "introduced": introduced,
        "risk_weighted_net": corrected - 2 * introduced,
        "interventions": int(results["intervene"].sum()),
        "identity_cluster_bootstrap": ci,
        "per_repeat": per_repeat,
        "network_coverage": {
            "candidate_rows": int(len(features)),
            "known_path_rows": int((features["known_path_fraction"] > 0).sum()),
            "raw_step0_validated_rows": int(
                ((features["edge0_complete_fraction"] > 0) & (features["edge0_bottleneck_mean"] > 0)).sum()
            ),
            "predicted_edge_increment_nonzero_rows": int(
                (np.abs(features["predicted_edge_increment"]) > 1e-12).sum()
            ),
        },
        "gates": gates,
        "pass": all(gates.values()),
        "provenance": {name: sha256(path) for name, path in files.items()} | {
            "candidate_features": sha256(feature_path),
            "transitions": sha256(result_path),
        },
        "contracts": {
            "artifact_frozen": True,
            "truth_or_outcome_columns_passed_to_expert": False,
            "hidden_truth_identity_used_as_seed": False,
            "phenotype_used": False,
            "P2b_used": False,
        },
        "claim_limit": (
            "This test concerns network-assisted recovery in an artificial standard mixture. Passing would not "
            "establish biological-cohort generalization, SOTA, or shared-embedding improvement."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
