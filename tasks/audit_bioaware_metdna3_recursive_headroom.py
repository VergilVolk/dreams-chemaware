#!/usr/bin/env python
"""Audit recursive MetDNA-style headroom on stable observed MS1 features.

This stage measures reachability only.  It does not fit a score or inspect the
sealed RP benchmark.  Ties in graph distance count against a rescue claim.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from audit_bioaware_metdna3_ms1_premapping import MassCandidateIndex
except ModuleNotFoundError:  # package import used by tests
    from tasks.audit_bioaware_metdna3_ms1_premapping import MassCandidateIndex


FILE_RE = re.compile(r"^urine_(pos|neg)_(70_300|70_1200|290_600|590_1200)_([12])\.mzML$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def greedy_pairs(a: pd.DataFrame, b: pd.DataFrame, ppm: float, rt_sec: float) -> list[tuple[int, int]]:
    """Match technical-replicate features using the pilot's frozen ordering."""
    if a.empty or b.empty:
        return []
    b_ordered = b.sort_values("mz").copy()
    b_mz = b_ordered["mz"].to_numpy(float)
    b_rt = b_ordered["rt_sec"].to_numpy(float)
    b_index = b_ordered.index.to_numpy(int)
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for left in a.sort_values("intensity", ascending=False).itertuples():
        tolerance = float(left.mz) * ppm * 1e-6
        low = bisect.bisect_left(b_mz, float(left.mz) - tolerance)
        high = bisect.bisect_right(b_mz, float(left.mz) + tolerance)
        best_position = None
        best_cost = float("inf")
        for position in range(low, high):
            right_index = int(b_index[position])
            if right_index in used:
                continue
            delta_rt = abs(float(left.rt_sec) - b_rt[position])
            if delta_rt > rt_sec:
                continue
            delta_ppm = abs(b_mz[position] - float(left.mz)) / float(left.mz) * 1e6
            cost = (delta_ppm / ppm) ** 2 + (delta_rt / rt_sec) ** 2
            if cost < best_cost:
                best_position, best_cost = position, cost
        if best_position is not None:
            right_index = int(b_index[best_position])
            used.add(right_index)
            pairs.append((int(left.Index), right_index))
    return pairs


class UnionFind:
    def __init__(self, size: int, labels: list[str] | None = None) -> None:
        self.parent = list(range(size))
        self.labels = [set() for _ in range(size)] if labels is None else [{value} for value in labels]

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> bool:
        a, b = self.find(left), self.find(right)
        if a == b:
            return True
        # A consensus node may contain at most one feature from each acquisition
        # window.  This prevents transitive chains from collapsing two distinct
        # within-window chromatographic peaks.
        if self.labels[a] & self.labels[b]:
            return False
        self.parent[b] = a
        self.labels[a].update(self.labels[b])
        self.labels[b].clear()
        return True


def merge_acquisition_windows(frame: pd.DataFrame, ppm: float, rt_sec: float) -> pd.DataFrame:
    """Merge only cross-window duplicates; never collapse within-window peaks."""
    records: list[dict] = []
    for polarity, group in frame.groupby("polarity", sort=True):
        ordered = group.sort_values("mz").reset_index(drop=True)
        mz = ordered["mz"].to_numpy(float)
        rt = ordered["rt_sec"].to_numpy(float)
        windows = ordered["mass_window"].astype(str).to_numpy()
        union = UnionFind(len(ordered), windows.tolist())
        for i in range(len(ordered)):
            upper_mz = mz[i] / (1.0 - ppm * 1e-6)
            stop = int(np.searchsorted(mz, upper_mz, side="right"))
            for j in range(i + 1, stop):
                if windows[i] == windows[j] or abs(rt[i] - rt[j]) > rt_sec:
                    continue
                union.union(i, j)
        components: dict[int, list[int]] = defaultdict(list)
        for i in range(len(ordered)):
            components[union.find(i)].append(i)
        for members in components.values():
            subset = ordered.iloc[members]
            if subset["mass_window"].duplicated().any():
                raise RuntimeError("cross-window clustering collapsed within-window peaks")
            records.append({
                "polarity": polarity,
                "mz": float(subset["mz"].mean()),
                "rt_sec": float(subset["rt_sec"].mean()),
                "intensity": float(subset["intensity"].median()),
                "window_support": int(subset["mass_window"].nunique()),
                "pair_feature_count": int(len(subset)),
                "mass_windows": ";".join(sorted(set(subset["mass_window"].astype(str)))),
            })
    result = pd.DataFrame(records).sort_values(["polarity", "mz", "rt_sec"]).reset_index(drop=True)
    result.insert(0, "feature_node", np.arange(len(result), dtype=int))
    return result


def nearest_feature(
    nodes: pd.DataFrame, polarity: str, mz: float, rt: float, ppm: float, rt_sec: float
) -> int | None:
    subset = nodes[nodes["polarity"].eq(polarity)]
    values = subset["mz"].to_numpy(float)
    tolerance = mz * ppm * 1e-6
    low = int(np.searchsorted(values, mz - tolerance, side="left"))
    high = int(np.searchsorted(values, mz + tolerance, side="right"))
    if high <= low:
        return None
    candidates = subset.iloc[low:high].copy()
    candidates = candidates[abs(candidates["rt_sec"] - rt) <= rt_sec]
    if candidates.empty:
        return None
    candidates["cost"] = (
        ((candidates["mz"] - mz).abs() / mz * 1e6 / ppm) ** 2
        + ((candidates["rt_sec"] - rt).abs() / rt_sec) ** 2
    )
    return int(candidates.sort_values(["cost", "feature_node"]).iloc[0]["feature_node"])


def bounded_distances(
    adjacency: dict[str, set[str]], seeds: set[str], observed: set[str], max_depth: int
) -> dict[str, int]:
    distance = {seed: 0 for seed in seeds}
    queue = deque(sorted(seeds))
    while queue:
        source = queue.popleft()
        depth = distance[source]
        if depth >= max_depth:
            continue
        for target in adjacency.get(source, set()):
            if target not in observed or target in distance:
                continue
            distance[target] = depth + 1
            queue.append(target)
    return distance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_ms1_feature_pilot_v1"),
    )
    parser.add_argument(
        "--index-dir", type=Path,
        default=Path("data/reference/metdna2_emrn_mass_adduct_20260828"),
    )
    parser.add_argument(
        "--network-dir", type=Path,
        default=Path("data/reference/metdna2_emrn_network_20260828"),
    )
    parser.add_argument(
        "--development-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_development_v1"),
    )
    parser.add_argument(
        "--query-cache", type=Path,
        default=Path("data/validation/bioaware_metdna3_dreams_cache_v2/queries.csv.gz"),
    )
    parser.add_argument(
        "--baseline-transitions", type=Path,
        default=Path("data/validation/bioaware_metdna3_development_eval_v1/raw_transitions.csv.gz"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/validation/bioaware_metdna3_recursive_headroom_v1"),
    )
    parser.add_argument("--ppm", type=float, default=15.0)
    parser.add_argument("--rt-sec", type=float, default=25.0)
    parser.add_argument("--maximum-depth", type=int, default=3)
    parser.add_argument(
        "--noise-threshold", type=float, default=None,
        help=("Override the outcome-blind threshold selected by the MS1 pilot. "
              "Formal runs normally leave this unset."),
    )
    parser.add_argument("--truth-name", default="development_level1.csv.gz")
    parser.add_argument("--scope", choices=("development", "internal_rplc", "external"), default="development")
    args = parser.parse_args()
    paths = {
        "pilot": args.pilot_dir / "report.json",
        "mass": args.index_dir / "emrn_compound_mass.csv.gz",
        "adduct": args.index_dir / "metdna2_adducts.csv",
        "edges": args.network_dir / "metdna2_emrn_edges.csv.gz",
        "truth": args.development_dir / args.truth_name,
        "splits": args.development_dir / "identity_splits.csv.gz",
        "queries": args.query_cache,
        "transitions": args.baseline_transitions,
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {output}")

    pilot = json.loads(paths["pilot"].read_text(encoding="utf-8"))
    if pilot.get("status") != "bioaware_metdna3_ms1_feature_pilot_complete":
        raise RuntimeError("MS1 pilot is not frozen")
    if pilot.get("contracts", {}).get("selection_is_outcome_blind") is not True:
        raise RuntimeError("MS1 threshold is not outcome-blind")
    selected = pilot.get("selected_noise_threshold")
    if selected is None:
        raise RuntimeError("MS1 pilot does not contain an outcome-blind selected threshold")
    threshold = float(selected if args.noise_threshold is None else args.noise_threshold)
    if threshold not in {float(value) for value in pilot["thresholds"]}:
        raise RuntimeError("fixed primary noise threshold was not evaluated in the pilot")
    if args.scope in {"internal_rplc", "external"} and args.noise_threshold is not None:
        raise RuntimeError("formal validation run forbids overriding the pilot-selected threshold")
    pair_consensus: list[dict] = []
    for polarity in ("pos", "neg"):
        for window in ("70_300", "70_1200", "290_600", "590_1200"):
            tables = []
            for replicate in (1, 2):
                pattern = f"*_{polarity}_{window}_{replicate}__noise_{threshold:g}.csv.gz"
                matches = sorted((args.pilot_dir / "features").glob(pattern))
                if not matches:
                    # One targeted file is absent in the published Mouse-brain
                    # RPLC release.  A technical consensus requires both
                    # replicates, so that acquisition role is omitted.
                    tables = []
                    break
                if len(matches) != 1:
                    raise RuntimeError(f"ambiguous acquisition role {pattern}: {matches}")
                path = matches[0]
                tables.append(pd.read_csv(path))
            if len(tables) != 2:
                continue
            for left, right in greedy_pairs(tables[0], tables[1], args.ppm, args.rt_sec):
                a, b = tables[0].loc[left], tables[1].loc[right]
                pair_consensus.append({
                    "polarity": "positive" if polarity == "pos" else "negative",
                    "mass_window": window,
                    "mz": float((a.mz + b.mz) / 2),
                    "rt_sec": float((a.rt_sec + b.rt_sec) / 2),
                    "intensity": float(np.sqrt(max(float(a.intensity), 0) * max(float(b.intensity), 0))),
                })
    nodes = merge_acquisition_windows(pd.DataFrame(pair_consensus), args.ppm, args.rt_sec)
    node_path = output / "stable_ms1_feature_nodes.csv.gz"
    nodes.to_csv(node_path, index=False, compression="gzip")

    truth = pd.read_csv(paths["truth"])
    truth["feature_node"] = [
        nearest_feature(nodes, str(row.polarity), float(row.mz), float(row.rt), args.ppm, args.rt_sec)
        for row in truth.itertuples(index=False)
    ]
    truth["feature_recovered"] = truth["feature_node"].notna()
    queries = pd.read_csv(paths["queries"])
    queries["feature_node"] = [
        nearest_feature(
            nodes, str(row.polarity), float(row.feature_mz), float(row.feature_rt_sec),
            args.ppm, args.rt_sec,
        )
        for row in queries.itertuples(index=False)
    ]
    transitions = pd.read_csv(paths["transitions"])
    baseline = transitions.groupby("query_id", as_index=False).agg({
        "truth_candidate_id": "first", "baseline_top_candidate": "first",
        "baseline_correct": "first",
    })
    queries = queries.merge(baseline, on="query_id", validate="one_to_one")
    expected_queries = int(pd.read_csv(paths["queries"], usecols=["query_id"])["query_id"].nunique())
    if len(queries) != expected_queries or queries.query_id.nunique() != expected_queries:
        raise RuntimeError("frozen DreaMS query coverage changed during recursive audit")

    masses = pd.read_csv(paths["mass"])
    adducts = pd.read_csv(paths["adduct"])
    if adducts["default_annotation"].dtype != bool:
        adducts["default_annotation"] = adducts["default_annotation"].astype(str).str.lower().eq("true")
    mass_index = MassCandidateIndex(masses, adducts)
    assignment_records: list[dict] = []
    feature_candidates: dict[int, dict[int, set[str]]] = {0: {}, 1: {}}
    for step in (0, 1):
        for row in nodes.itertuples(index=False):
            candidates = mass_index.query(
                float(row.mz), str(row.polarity), step, args.ppm, None
            )
            feature_candidates[step][int(row.feature_node)] = candidates
            assignment_records.extend({
                "maximum_step": step, "feature_node": int(row.feature_node), "candidate_ik14": value
            } for value in sorted(candidates))
    assignment_path = output / "feature_candidate_assignments.csv.gz"
    pd.DataFrame(assignment_records).to_csv(assignment_path, index=False, compression="gzip")

    edges = pd.read_csv(paths["edges"])
    splits = pd.read_csv(paths["splits"])
    records: list[dict] = []
    summaries: dict[str, dict] = {}
    for graph_step in (0, 1):
        observed = set().union(*feature_candidates[graph_step].values())
        graph_edges = edges[edges["minimum_step"].le(graph_step)]
        adjacency: dict[str, set[str]] = defaultdict(set)
        for edge in graph_edges.itertuples(index=False):
            left, right = str(edge.ik14_a), str(edge.ik14_b)
            if left in observed or right in observed:
                adjacency[left].add(right)
                adjacency[right].add(left)
        for depth in range(1, args.maximum_depth + 1):
            for fold in range(10):
                seed_ids = set(splits[(splits["fold"].eq(fold)) & splits["role"].eq("seed")]["ik14"])
                recovered_seed_ids = set(truth.loc[truth["feature_recovered"], "ik14"]) & seed_ids
                distances = bounded_distances(adjacency, recovered_seed_ids, observed, depth)
                heldout = set(splits[(splits["fold"].eq(fold)) & splits["role"].eq("heldout")]["ik14"])
                for row in queries.itertuples(index=False):
                    if str(row.truth_ik14) not in heldout:
                        continue
                    node = None if pd.isna(row.feature_node) else int(row.feature_node)
                    candidates = set() if node is None else feature_candidates[graph_step].get(node, set())
                    truth_id = str(row.truth_candidate_id)
                    wrong_id = str(row.baseline_top_candidate)
                    truth_distance = distances.get(truth_id)
                    wrong_distance = distances.get(wrong_id)
                    truth_supported = truth_id in candidates and truth_distance is not None
                    wrong_supported = wrong_id in candidates and wrong_distance is not None
                    strict_rescue = truth_supported and (
                        not wrong_supported or int(truth_distance) < int(wrong_distance)
                    )
                    records.append({
                        "graph_step": graph_step, "maximum_depth": depth, "fold": fold,
                        "query_id": str(row.query_id), "truth_candidate_id": truth_id,
                        "baseline_top_candidate": wrong_id,
                        "baseline_correct": bool(row.baseline_correct),
                        "feature_recovered": node is not None,
                        "truth_mass_candidate": truth_id in candidates,
                        "wrong_mass_candidate": wrong_id in candidates,
                        "truth_distance": truth_distance,
                        "wrong_distance": wrong_distance,
                        "truth_supported": truth_supported,
                        "wrong_supported": wrong_supported,
                        "strict_rescue_headroom": strict_rescue,
                    })
    result = pd.DataFrame(records)
    result_path = output / "per_rotation.csv.gz"
    result.to_csv(result_path, index=False, compression="gzip")
    for (step, depth), group in result.groupby(["graph_step", "maximum_depth"]):
        errors = group[~group["baseline_correct"]]
        summaries[f"step{int(step)}|depth{int(depth)}"] = {
            "rotations": int(len(group)),
            "feature_recovered": int(group["feature_recovered"].sum()),
            "official_error_rotations": int(len(errors)),
            "error_truth_mass_candidate": int(errors["truth_mass_candidate"].sum()),
            "error_truth_supported": int(errors["truth_supported"].sum()),
            "error_wrong_supported": int(errors["wrong_supported"].sum()),
            "error_strict_rescue_headroom": int(errors["strict_rescue_headroom"].sum()),
            "unique_error_queries_with_strict_rescue": int(
                errors.loc[errors["strict_rescue_headroom"], "query_id"].nunique()
            ),
        }
    best = max(
        summaries.values(), key=lambda item: item["unique_error_queries_with_strict_rescue"]
    )
    report = {
        "status": "bioaware_metdna3_recursive_headroom_complete",
        "formal": True,
        "scope": args.scope,
        "selected_noise_threshold": threshold,
        "technical_pair_consensus_features": int(len(pair_consensus)),
        "stable_ms1_feature_nodes": int(len(nodes)),
        "level1_feature_recovery": int(truth["feature_recovered"].sum()),
        "level1_rows": int(len(truth)),
        "dreams_feature_recovery": int(queries["feature_node"].notna().sum()),
        "dreams_queries": int(len(queries)),
        "results": summaries,
        "primary_gate": {
            "at_least_two_unique_error_queries_have_strict_recursive_headroom": (
                best["unique_error_queries_with_strict_rescue"] >= 2
            ),
            "maximum_unique_error_queries_with_strict_rescue": int(
                best["unique_error_queries_with_strict_rescue"]
            ),
        },
        "provenance": {key: sha256(path) for key, path in paths.items()} | {
            "feature_nodes_sha256": sha256(node_path),
            "feature_assignments_sha256": sha256(assignment_path),
            "per_rotation_sha256": sha256(result_path),
        },
        "contracts": {
            "feature_selection_outcome_blind": True,
            "primary_noise_threshold": (
                "frozen before this panel and technically audited without outcome access"
                if pilot.get("contracts", {}).get("within_panel_threshold_selection") is False else
                "selected from technical-replicate stability in the outcome-blind MS1 pilot"
            ),
            "seed_query_identity_disjoint": True,
            "ties_count_against_rescue": True,
            "external_test_opened": False,
            "P2b_used": False,
        },
        "claim_limit": "Reachability headroom only; no ranking model or annotation gain.",
    }
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
