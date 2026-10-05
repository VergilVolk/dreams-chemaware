#!/usr/bin/env python
"""Build an outcome-blind BioAware B40 soft graph-completion cache.

The cache projects non-currency Rhea hyperedges to a degree-normalised
metabolite graph and propagates each visible seed set for a fixed four-hop
radius.  Three deterministic component/degree-matched seed permutations are
computed as structural nulls.  No identity truth, DreaMS score, rank or
phenotype is read.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components


FORBIDDEN = {
    "truth_candidate_id", "truth_formula", "is_positive", "spectral_score",
    "baseline_candidate_id", "baseline_correct", "baseline_gap", "corrected",
    "introduced", "final_correct", "delta", "phenotype", "group",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_csv_gzip(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, suffix=".csv.gz", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(
            temporary, index=False,
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = set(required) - set(frame.columns)
    if missing:
        raise RuntimeError(f"{label} misses columns: {sorted(missing)}")


def build_rhea_projection(participants: pd.DataFrame) -> tuple[list[str], sparse.csr_matrix, dict[str, Any]]:
    participants = participants.loc[~parse_bool(participants["is_currency"])].copy()
    participants["compound_id"] = participants["compound_id"].astype(str)
    nodes = sorted(participants["compound_id"].unique())
    position = {node: index for index, node in enumerate(nodes)}
    edge_weights: dict[tuple[int, int], float] = defaultdict(float)
    usable_reactions = 0
    skipped_one_side = 0
    for _reaction, local in participants.groupby("reaction_id", sort=False):
        left = sorted(set(local.loc[local["side"].astype(str).eq("left"), "compound_id"]))
        right = sorted(set(local.loc[local["side"].astype(str).eq("right"), "compound_id"]))
        if not left or not right:
            skipped_one_side += 1
            continue
        pairs = {(source, target) for source in left for target in right if source != target}
        if not pairs:
            continue
        usable_reactions += 1
        contribution = 1.0 / len(pairs)
        for source, target in pairs:
            i, j = position[source], position[target]
            edge_weights[(i, j)] += contribution
            edge_weights[(j, i)] += contribution
    rows = np.fromiter((edge[0] for edge in edge_weights), dtype=np.int32)
    columns = np.fromiter((edge[1] for edge in edge_weights), dtype=np.int32)
    data = np.fromiter(edge_weights.values(), dtype=np.float64)
    adjacency = sparse.csr_matrix((data, (rows, columns)), shape=(len(nodes), len(nodes)))
    adjacency.sum_duplicates()
    row_sum = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.zeros_like(row_sum)
    inverse[row_sum > 0] = 1.0 / row_sum[row_sum > 0]
    transition = sparse.diags(inverse).dot(adjacency).tocsr()
    components, labels = connected_components(adjacency, directed=False, return_labels=True)
    report = {
        "nodes": int(len(nodes)),
        "directed_projection_edges": int(adjacency.nnz),
        "undirected_projection_edges": int(adjacency.nnz // 2),
        "usable_reactions": int(usable_reactions),
        "skipped_one_side_reactions": int(skipped_one_side),
        "connected_components": int(components),
        "largest_component": int(np.bincount(labels).max()),
    }
    return nodes, transition, {**report, "component_labels": labels, "weighted_degree": row_sum}


def diffuse(
    starts: list[list[int]],
    transition: sparse.csr_matrix,
    *,
    hops: int,
    decay: float,
) -> tuple[np.ndarray, list[np.ndarray]]:
    matrix = np.zeros((len(starts), transition.shape[0]), dtype=np.float32)
    for row, indices in enumerate(starts):
        if indices:
            matrix[row, indices] = 1.0 / len(indices)
    current = matrix
    total = np.zeros_like(matrix)
    hop_values: list[np.ndarray] = []
    for hop in range(1, hops + 1):
        current = transition.transpose().dot(current.transpose()).transpose().astype(np.float32, copy=False)
        hop_values.append(current.copy())
        total += (decay ** (hop - 1)) * current
    return total, hop_values


def stable_index(*items: str, modulus: int) -> int:
    digest = hashlib.sha256("\x1f".join(items).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % modulus


def degree_bins(degrees: np.ndarray) -> np.ndarray:
    logged = np.log1p(degrees)
    positive = logged[degrees > 0]
    if not len(positive):
        return np.zeros(len(degrees), dtype=np.int8)
    cuts = np.unique(np.quantile(positive, np.linspace(0, 1, 11)))
    return np.searchsorted(cuts[1:-1], logged, side="right").astype(np.int8)


def permute_seed_indices(
    real: list[int],
    *,
    forbidden: set[int],
    context_key: str,
    repeat: int,
    components: np.ndarray,
    bins: np.ndarray,
    exact_pools: dict[tuple[int, int], list[int]],
    component_pools: dict[int, list[int]],
) -> tuple[list[int], int]:
    def choose(pool: list[int], key: tuple[str, ...], blocked: set[int]) -> int | None:
        if not pool:
            return None
        start = stable_index(*key, modulus=len(pool))
        for offset in range(len(pool)):
            value = pool[(start + offset) % len(pool)]
            if value not in blocked:
                return value
        return None

    output: list[int] = []
    fallbacks = 0
    blocked = set(forbidden) | set(real)
    for seed in real:
        exact = exact_pools.get((int(components[seed]), int(bins[seed])), [])
        chosen = choose(exact, (context_key, str(repeat), str(seed), "exact"), blocked)
        if chosen is None:
            fallbacks += 1
            chosen = choose(
                component_pools[int(components[seed])],
                (context_key, str(repeat), str(seed), "component"),
                blocked,
            )
        if chosen is None:
            continue
        output.append(chosen)
        blocked.add(chosen)
    return sorted(set(output)), fallbacks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m0-dir", type=Path, required=True)
    parser.add_argument("--participants", type=Path, default=Path("data/reference/bioaware_rhea_offline_20260827/rhea_participants.csv.gz"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hops", type=int, default=4)
    parser.add_argument("--decay", type=float, default=0.65)
    parser.add_argument("--null-repeats", type=int, default=3)
    parser.add_argument("--batch-contexts", type=int, default=128)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {args.output_dir}")
    if args.hops != 4 or abs(args.decay - 0.65) > 1e-12 or args.null_repeats != 3:
        raise RuntimeError("formal B40-M0 requires fixed hops=4 decay=0.65 null_repeats=3")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    m0_report_path = args.m0_dir / "report.json"
    contexts_path = args.m0_dir / "candidate_context_semantics.csv.gz"
    seed_contexts_path = args.m0_dir / "visible_seed_contexts.csv.gz"
    for path in (m0_report_path, contexts_path, seed_contexts_path, args.participants):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    m0_report = json.loads(m0_report_path.read_text(encoding="utf-8"))
    if m0_report.get("provenance", {}).get("candidate_contexts") != sha256(contexts_path):
        raise RuntimeError("B39-M0 candidate-context provenance mismatch")
    if m0_report.get("provenance", {}).get("visible_seed_contexts") != sha256(seed_contexts_path):
        raise RuntimeError("B39-M0 visible-seed provenance mismatch")
    contexts = pd.read_csv(contexts_path, low_memory=False)
    seed_contexts = pd.read_csv(seed_contexts_path, low_memory=False)
    participants = pd.read_csv(args.participants, low_memory=False)
    require_columns(contexts, ["source", "context_class", "query_id", "candidate_id", "seed_stratum"], "contexts")
    require_columns(seed_contexts, ["source", "context_class", "query_id", "seed_stratum", "visible_seed_identity_list", "visible_seed_count"], "seed contexts")
    require_columns(participants, ["compound_id", "reaction_id", "side", "is_currency"], "participants")
    leaked = FORBIDDEN & (set(contexts.columns) | set(seed_contexts.columns))
    if leaked:
        raise RuntimeError(f"outcome leakage in B40 input: {sorted(leaked)}")
    nodes, transition, graph_report = build_rhea_projection(participants)
    position = {node: index for index, node in enumerate(nodes)}
    components = np.asarray(graph_report.pop("component_labels"), dtype=np.int32)
    weighted_degree = np.asarray(graph_report.pop("weighted_degree"), dtype=float)
    bins = degree_bins(weighted_degree)
    exact_pools: dict[tuple[int, int], list[int]] = defaultdict(list)
    component_pools: dict[int, list[int]] = defaultdict(list)
    for index in range(len(nodes)):
        exact_pools[(int(components[index]), int(bins[index]))].append(index)
        component_pools[int(components[index])].append(index)

    context_table = seed_contexts[["source", "context_class", "query_id", "seed_stratum", "visible_seed_identity_list", "visible_seed_count"]].drop_duplicates()
    if context_table.duplicated(["query_id", "seed_stratum"]).any():
        raise RuntimeError("one query/seed stratum maps to multiple visible seed sets")
    candidates_by_context = {
        (str(query), str(stratum)): sorted(set(local["candidate_id"].astype(str)))
        for (query, stratum), local in contexts.groupby(["query_id", "seed_stratum"], sort=False)
    }
    records: list[dict[str, Any]] = []
    unmapped_seed_contexts = 0
    null_fallbacks = 0
    total_visible_seed_identities = 0
    mapped_visible_seed_identities = 0
    rows = list(context_table.itertuples(index=False))
    for batch_start in range(0, len(rows), args.batch_contexts):
        batch = rows[batch_start: batch_start + args.batch_contexts]
        starts: list[list[int]] = []
        metadata: list[tuple[Any, list[str], list[int], list[str]]] = []
        null_starts: list[list[int]] = []
        for row in batch:
            visible = [] if pd.isna(row.visible_seed_identity_list) else [value for value in str(row.visible_seed_identity_list).split(";") if value]
            if len(set(visible)) != int(row.visible_seed_count):
                raise RuntimeError(f"visible seed list/count mismatch: {(row.query_id, row.seed_stratum)}")
            mapped = sorted({position[value] for value in visible if value in position})
            total_visible_seed_identities += len(set(visible))
            mapped_visible_seed_identities += len(mapped)
            if not mapped:
                unmapped_seed_contexts += 1
            candidates = candidates_by_context[(str(row.query_id), str(row.seed_stratum))]
            forbidden = {position[value] for value in candidates if value in position}
            starts.append(mapped)
            local_nulls = []
            for repeat in range(args.null_repeats):
                permuted, fallbacks = permute_seed_indices(
                    mapped,
                    forbidden=forbidden,
                    context_key=f"{row.query_id}|{row.seed_stratum}",
                    repeat=repeat,
                    components=components,
                    bins=bins,
                    exact_pools=exact_pools,
                    component_pools=component_pools,
                )
                null_fallbacks += fallbacks
                null_starts.append(permuted)
                local_nulls.append(permuted)
            metadata.append((row, visible, mapped, candidates))
        real_total, real_hops = diffuse(starts, transition, hops=args.hops, decay=args.decay)
        null_total, _null_hops = diffuse(null_starts, transition, hops=args.hops, decay=args.decay)
        for batch_index, (row, visible, mapped, candidates) in enumerate(metadata):
            for candidate in candidates:
                candidate_index = position.get(candidate)
                if candidate_index is None:
                    real_score = 0.0
                    hop_scores = [0.0] * args.hops
                    null_scores = [0.0] * args.null_repeats
                else:
                    real_score = float(real_total[batch_index, candidate_index])
                    hop_scores = [float(values[batch_index, candidate_index]) for values in real_hops]
                    null_scores = [
                        float(null_total[batch_index * args.null_repeats + repeat, candidate_index])
                        for repeat in range(args.null_repeats)
                    ]
                record = {
                    "source": str(row.source),
                    "context_class": str(row.context_class),
                    "query_id": str(row.query_id),
                    "seed_stratum": str(row.seed_stratum),
                    "candidate_id": candidate,
                    "visible_seed_count": int(len(set(visible))),
                    "mapped_seed_count": int(len(mapped)),
                    "candidate_rhea_mapped": bool(candidate_index is not None),
                    "candidate_rhea_degree": float(weighted_degree[candidate_index]) if candidate_index is not None else 0.0,
                    "real_diffusion_score": real_score,
                    "null_diffusion_mean": float(np.mean(null_scores)),
                    "null_diffusion_max": float(np.max(null_scores)),
                    "real_minus_null_mean": float(real_score - np.mean(null_scores)),
                }
                for hop, value in enumerate(hop_scores, 1):
                    record[f"real_hop_{hop}"] = value
                for repeat, value in enumerate(null_scores):
                    record[f"null_{repeat}_diffusion_score"] = value
                records.append(record)
        completed = min(batch_start + len(batch), len(rows))
        if completed % 1024 < len(batch) or completed == len(rows):
            print(f"[B40-M0] {completed:,}/{len(rows):,} contexts", flush=True)

    scores = pd.DataFrame(records)
    if FORBIDDEN & set(scores.columns):
        raise RuntimeError("B40 output acquired a forbidden outcome column")
    score_path = args.output_dir / "candidate_context_diffusion.csv.gz"
    atomic_csv_gzip(score_path, scores)
    group = scores.groupby(["query_id", "seed_stratum"], sort=False)
    discriminative = group["real_diffusion_score"].agg(lambda values: float(values.max() - values.min()) > 1e-12)
    mapped_two = group["candidate_rhea_mapped"].sum().ge(2)
    positive_any = group["real_diffusion_score"].max().gt(0)
    null_excess = group["real_minus_null_mean"].max()
    candidate_degree = scores.loc[scores["candidate_rhea_mapped"] & scores["real_diffusion_score"].gt(0), ["candidate_rhea_degree", "real_diffusion_score"]]
    degree_correlation = float(candidate_degree.corr(method="spearman").iloc[0, 1]) if len(candidate_degree) >= 3 else 1.0
    source_report = {}
    query_mapping = (
        scores.groupby(["source", "query_id", "candidate_id"], sort=False)["candidate_rhea_mapped"]
        .max().reset_index()
        .groupby(["source", "query_id"], sort=False)["candidate_rhea_mapped"].sum().reset_index()
    )
    for source, local in scores.groupby("source", sort=False):
        local_group = local.groupby(["query_id", "seed_stratum"], sort=False)
        mapped_queries = query_mapping.loc[query_mapping["source"].eq(source)]
        source_report[str(source)] = {
            "contexts": int(local_group.ngroups),
            "contexts_with_positive_score": int(local_group["real_diffusion_score"].max().gt(0).sum()),
            "contexts_with_candidate_discrimination": int(local_group["real_diffusion_score"].agg(lambda values: float(values.max() - values.min()) > 1e-12).sum()),
            "candidate_mapping_fraction": float(local["candidate_rhea_mapped"].mean()),
            "queries_with_at_least_two_mapped_candidates": int(mapped_queries["candidate_rhea_mapped"].ge(2).sum()),
        }
    queries_with_two_mapped = int(query_mapping["candidate_rhea_mapped"].ge(2).sum())
    gates = {
        "contexts_ge_5000": bool(len(context_table) >= 5000),
        "mapped_seed_fraction_ge_0_50": bool(mapped_visible_seed_identities / max(total_visible_seed_identities, 1) >= 0.50),
        "queries_with_two_mapped_candidates_ge_500": bool(queries_with_two_mapped >= 500),
        "every_source_has_100_two_mapped_candidate_queries": bool(all(item["queries_with_at_least_two_mapped_candidates"] >= 100 for item in source_report.values())),
        "positive_score_contexts_ge_1000": bool(positive_any.sum() >= 1000),
        "discriminative_contexts_ge_500": bool(discriminative.sum() >= 500),
        "median_best_real_minus_null_positive": bool(float(null_excess.median()) > 0),
        "absolute_degree_correlation_lt_0_80": bool(abs(degree_correlation) < 0.80),
    }
    report = {
        "status": "bioaware_b40_soft_graph_completion_cache_complete",
        "formal": True,
        "outcomes_read": False,
        "models_fitted": False,
        "graph": graph_report,
        "contexts": int(len(context_table)),
        "queries": int(context_table["query_id"].nunique()),
        "candidate_context_rows": int(len(scores)),
        "candidate_mapping_fraction": float(scores["candidate_rhea_mapped"].mean()),
        "visible_seed_identities": int(total_visible_seed_identities),
        "mapped_visible_seed_identities": int(mapped_visible_seed_identities),
        "mapped_seed_fraction": float(mapped_visible_seed_identities / max(total_visible_seed_identities, 1)),
        "contexts_without_mapped_seeds": int(unmapped_seed_contexts),
        "contexts_with_at_least_two_mapped_candidates": int(mapped_two.sum()),
        "queries_with_at_least_two_mapped_candidates": queries_with_two_mapped,
        "contexts_with_positive_score": int(positive_any.sum()),
        "contexts_with_candidate_discrimination": int(discriminative.sum()),
        "median_best_real_minus_null": float(null_excess.median()),
        "positive_best_real_minus_null_fraction": float(null_excess.gt(0).mean()),
        "spearman_score_vs_candidate_degree": degree_correlation,
        "degree_bin_fallbacks_in_null_sampling": int(null_fallbacks),
        "source_report": source_report,
        "parameters": {"hops": args.hops, "decay": args.decay, "null_repeats": args.null_repeats},
        "gates": gates,
        "pass_to_fixed_action_evaluation": bool(all(gates.values())),
        "provenance": {
            "m0_report": sha256(m0_report_path),
            "candidate_contexts": sha256(contexts_path),
            "visible_seed_contexts": sha256(seed_contexts_path),
            "rhea_participants": sha256(args.participants),
            "scores": sha256(score_path),
            "script": sha256(Path(__file__)),
        },
        "contract": {
            "knowledge_layer": "non-currency Rhea opposite-side hyperedge projection",
            "data_layer": "visible seed identities only",
            "null": "three deterministic component and weighted-degree-decile matched seed permutations",
            "truth_or_rank_used": False,
            "P2b_used": False,
        },
        "claim_limit": "Outcome-blind coverage and structural-null preflight only. It is not a ranking result and does not establish reaction-specificity, external generalisation or embedding gain.",
    }
    atomic_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
