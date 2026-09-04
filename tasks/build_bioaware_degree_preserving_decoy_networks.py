#!/usr/bin/env python
"""Build frozen degree-preserving decoys for the MetDNA2/eMRN graph.

The graph used by BioAware is undirected at the IK14 level.  For every
``minimum_step`` stratum needed by the frozen V3 expert (steps 0 and 1), this
script performs double-edge swaps.  A swap ``(a,b),(c,d)->(a,d),(c,b)`` keeps
the degree of every compound and the number of edges in the stratum exactly
unchanged while destroying the biological pairing.

These files are negative controls.  They must be passed through the same path,
raw-MS2 edge and router pipeline as the real graph; shuffling final scores is
not an acceptable substitute.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


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


def canon(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def degree(edges: list[tuple[str, str]]) -> Counter[str]:
    result: Counter[str] = Counter()
    for left, right in edges:
        result[left] += 1
        result[right] += 1
    return result


def rewire(
    edges: list[tuple[str, str]], *, seed: int, swaps_per_edge: int
) -> tuple[list[tuple[str, str]], int, int]:
    """Return a simple undirected degree-preserving rewiring."""
    if len(edges) < 2:
        raise RuntimeError("degree-preserving rewiring requires at least two edges")
    original_degree = degree(edges)
    current = [canon(*edge) for edge in edges]
    occupied = set(current)
    if len(occupied) != len(current):
        raise RuntimeError("input graph contains duplicate undirected edges in a stratum")
    rng = np.random.default_rng(seed)
    target = int(max(1, swaps_per_edge * len(current)))
    accepted = 0
    attempts = 0
    maximum_attempts = max(target * 100, 10000)
    while accepted < target and attempts < maximum_attempts:
        attempts += 1
        i, j = rng.integers(0, len(current), size=2)
        if i == j:
            continue
        a, b = current[int(i)]
        c, d = current[int(j)]
        if len({a, b, c, d}) < 4:
            continue
        # Randomly choose one of the two valid undirected switch orientations.
        if bool(rng.integers(0, 2)):
            candidate_i, candidate_j = canon(a, d), canon(c, b)
        else:
            candidate_i, candidate_j = canon(a, c), canon(b, d)
        if candidate_i[0] == candidate_i[1] or candidate_j[0] == candidate_j[1]:
            continue
        old_i, old_j = current[int(i)], current[int(j)]
        if candidate_i in occupied or candidate_j in occupied or candidate_i == candidate_j:
            continue
        occupied.remove(old_i)
        occupied.remove(old_j)
        current[int(i)] = candidate_i
        current[int(j)] = candidate_j
        occupied.add(candidate_i)
        occupied.add(candidate_j)
        accepted += 1
    if accepted < target:
        raise RuntimeError(
            f"rewiring accepted only {accepted}/{target} swaps after {attempts} attempts"
        )
    if degree(current) != original_degree:
        raise RuntimeError("degree sequence changed during decoy construction")
    if len(set(current)) != len(current):
        raise RuntimeError("decoy graph is not simple")
    return current, accepted, attempts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edges", type=Path,
        default=Path("data/reference/metdna2_emrn_network_20260828/metdna2_emrn_edges.csv.gz"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("data/reference/bioaware_degree_preserving_decoys_v1"),
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--swaps-per-edge", type=int, default=10)
    parser.add_argument("--maximum-step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()
    if args.repeats < 10:
        raise RuntimeError("formal BioAware decoy audit requires at least 10 repeats")
    if args.swaps_per_edge < 5:
        raise RuntimeError("formal decoys require at least five swaps per edge")
    if not args.edges.exists():
        raise FileNotFoundError(args.edges)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {output}")

    source = pd.read_csv(args.edges)
    required = {"minimum_step", "ik14_a", "ik14_b"}
    if not required.issubset(source.columns):
        raise RuntimeError(f"edge table lacks columns: {sorted(required - set(source.columns))}")
    source = source[source.minimum_step.le(args.maximum_step)].copy()
    source["ik14_a"] = source.ik14_a.astype(str)
    source["ik14_b"] = source.ik14_b.astype(str)
    reports: list[dict] = []
    for repeat in range(args.repeats):
        parts: list[pd.DataFrame] = []
        strata: dict[str, dict] = {}
        for step, group in source.groupby("minimum_step", sort=True):
            original_edges = [canon(str(a), str(b)) for a, b in zip(group.ik14_a, group.ik14_b)]
            rewired, accepted, attempts = rewire(
                original_edges,
                seed=args.seed + repeat * 1009 + int(step) * 9176,
                swaps_per_edge=args.swaps_per_edge,
            )
            frame = group.reset_index(drop=True).copy()
            frame["ik14_a"] = [edge[0] for edge in rewired]
            frame["ik14_b"] = [edge[1] for edge in rewired]
            frame["source_ik14"] = frame.ik14_a
            frame["target_ik14"] = frame.ik14_b
            frame["source_id"] = "DECOY:" + frame.ik14_a
            frame["target_id"] = "DECOY:" + frame.ik14_b
            frame["edge_source"] = "degree_preserving_decoy"
            frame["edge_label"] = "degree_preserving_decoy"
            frame["reaction_id"] = pd.NA
            frame["reaction"] = pd.NA
            parts.append(frame)
            original_set, decoy_set = set(original_edges), set(rewired)
            strata[str(int(step))] = {
                "edges": int(len(group)),
                "nodes": int(len(degree(original_edges))),
                "accepted_swaps": int(accepted),
                "attempts": int(attempts),
                "edge_overlap_fraction": float(len(original_set & decoy_set) / len(original_set)),
                "degree_sequence_exact": degree(original_edges) == degree(rewired),
            }
        decoy = pd.concat(parts, ignore_index=True)
        repeat_dir = output / f"repeat_{repeat:02d}"
        repeat_dir.mkdir()
        path = repeat_dir / "metdna2_emrn_edges.csv.gz"
        decoy.to_csv(path, index=False, compression="gzip")
        report = {
            "repeat": repeat,
            "seed": args.seed + repeat * 1009,
            "strata": strata,
            "edge_file_sha256": sha256(path),
        }
        atomic_json(repeat_dir / "report.json", report)
        reports.append(report)
        print(f"[decoy {repeat + 1}/{args.repeats}] {path}", flush=True)

    report_path = output / "report.json"
    atomic_json(report_path, {
        "status": "bioaware_degree_preserving_decoy_networks_complete",
        "formal": True,
        "repeats": args.repeats,
        "swaps_per_edge": args.swaps_per_edge,
        "maximum_step": args.maximum_step,
        "source_edges_sha256": sha256(args.edges),
        "decoys": reports,
        "contract": (
            "Each minimum-step stratum preserves every compound degree and edge count; "
            "biological pairings are destroyed before path and raw-MS2 evidence are recomputed."
        ),
        "claim_limit": "Negative-control graph construction only; no BioAware performance.",
    })


if __name__ == "__main__":
    main()
