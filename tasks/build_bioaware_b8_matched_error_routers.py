#!/usr/bin/env python
"""Build paired graph-selected and generic-hard-error routers for BioAware B8.

The unit of matching is a truth identity, matching the downstream identity-
uniform sampler.  Exactly one representative query is selected from each
matched identity.  The representative graph and generic queries must come from
the same biological acquisition unit and the same frozen difficulty strata.
This avoids the positivity failure caused by requiring one generic identity to
reproduce *every* query and acquisition unit of a graph-selected identity.

Both output arms therefore contain exactly one query per identity, the same
number of identities, and the same row-wise unit distribution.  Formal runs
also use one reference spectrum per candidate, so raw identity/query/reference
multiplicity cannot alter training dose.

This is a causal *selection* control.  It does not create a new BioAware score
and it does not use an embedding outcome, P2b, phenotype, or held-fold result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


GRAPH_ARM = "matched_graph_direct"
GENERIC_ARM = "matched_generic_direct"
STATUS = "bioaware_b8_matched_error_routers_frozen"
NUMERIC_FEATURES = (
    "training_truth_margin",
    "training_baseline_gap",
)
INFEASIBLE = 1.0e9


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--b7-router-dir", type=Path,
        default=ROOT / "data/validation/bioaware_b7_direct_graph_injection_2331964.partial/action_router",
    )
    parser.add_argument(
        "--manifest-dir", type=Path,
        default=ROOT / "data/validation/bioaware_b4_direct_shared_embedding_v1_20260905/manifest",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--minimum-matched-identities", type=int, default=12)
    parser.add_argument("--maximum-absolute-smd", type=float, default=0.25)
    return parser.parse_args()


def stable_bucket(value: int, edges: tuple[int, ...]) -> int:
    return int(np.searchsorted(np.asarray(edges), int(value), side="right"))


def standardised_mean_difference(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) != len(right) or not len(left):
        raise ValueError("SMD inputs must be non-empty paired vectors")
    if len(left) < 2:
        return 0.0 if np.isclose(np.mean(left), np.mean(right)) else INFEASIBLE
    pooled = np.sqrt((np.var(left, ddof=1) + np.var(right, ddof=1)) / 2.0)
    if not np.isfinite(pooled) or pooled < 1e-12:
        return 0.0 if np.isclose(np.mean(left), np.mean(right)) else INFEASIBLE
    return float((np.mean(left) - np.mean(right)) / pooled)


def robust_scales(frame: pd.DataFrame) -> dict[str, float]:
    output: dict[str, float] = {}
    for column in NUMERIC_FEATURES:
        values = frame[column].to_numpy(float)
        scale = float(np.nanstd(values, ddof=1))
        if not np.isfinite(scale) or scale < 1e-8:
            scale = 1.0
        output[column] = scale
    return output


def query_cost(left: pd.Series, right: pd.Series, scales: dict[str, float]) -> float:
    """Difficulty distance with large, but non-oracular, stratum penalties."""
    # These are baseline/pre-training covariates.  Hard calipers prevent the
    # optimizer from buying identity coverage with a scientifically invalid
    # easy-versus-hard comparison.
    for column in NUMERIC_FEATURES:
        if abs(float(left[column]) - float(right[column])) / scales[column] > 1.0:
            return INFEASIBLE
    # These quantities change the listwise task or the amount/quality of
    # identity evidence.  With one query per identity there is enough freedom
    # to match the raw counts exactly instead of relying on broad buckets.
    if (int(left["training_baseline_rank_numeric"]) != int(right["training_baseline_rank_numeric"])
            or int(left["training_candidate_count"]) != int(right["training_candidate_count"])
            or int(left["positive_reference_count"]) != int(right["positive_reference_count"])
            or int(left["identity_query_count"]) != int(right["identity_query_count"])):
        return INFEASIBLE
    value = 0.0
    for column in NUMERIC_FEATURES:
        value += ((float(left[column]) - float(right[column])) / scales[column]) ** 2
    value += 2.0 * float(left["identity_count_bucket"] != right["identity_count_bucket"])
    return float(value)


@dataclass(frozen=True)
class IdentityEdge:
    cost: float
    query_pairs: tuple[tuple[str, str], ...]


@dataclass
class FlowEdge:
    to: int
    reverse: int
    capacity: int
    cost: float


def minimum_cost_bipartite(cost: np.ndarray) -> list[tuple[int, int]]:
    """Maximum-cardinality, minimum-cost matching without a SciPy dependency.

    A successive shortest augmenting path solver is sufficient here because
    every B8 stratum contains at most hundreds of identities and usually only
    one or two queries per identity.  Bellman-Ford is used deliberately: reverse
    residual edges have negative costs, so a naive greedy assignment is not an
    acceptable scientific matching implementation.
    """
    matrix = np.asarray(cost, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("matching cost must be a matrix")
    n_left, n_right = matrix.shape
    source = 0
    left_offset = 1
    right_offset = left_offset + n_left
    sink = right_offset + n_right
    graph: list[list[FlowEdge]] = [[] for _ in range(sink + 1)]

    def add_edge(start: int, end: int, capacity: int, value: float) -> FlowEdge:
        forward = FlowEdge(end, len(graph[end]), capacity, value)
        reverse = FlowEdge(start, len(graph[start]), 0, -value)
        graph[start].append(forward)
        graph[end].append(reverse)
        return forward

    for left in range(n_left):
        add_edge(source, left_offset + left, 1, 0.0)
    for right in range(n_right):
        add_edge(right_offset + right, sink, 1, 0.0)
    original: dict[tuple[int, int], FlowEdge] = {}
    for left in range(n_left):
        for right in range(n_right):
            value = float(matrix[left, right])
            if np.isfinite(value) and value < INFEASIBLE:
                original[(left, right)] = add_edge(
                    left_offset + left, right_offset + right, 1, value
                )

    nodes = len(graph)
    while True:
        distance = [float("inf")] * nodes
        predecessor: list[tuple[int, int] | None] = [None] * nodes
        distance[source] = 0.0
        # A residual augmenting path is simple, hence at most V-1 relaxations.
        for _ in range(nodes - 1):
            changed = False
            for node, edges in enumerate(graph):
                if not np.isfinite(distance[node]):
                    continue
                for edge_index, edge in enumerate(edges):
                    if edge.capacity <= 0:
                        continue
                    candidate = distance[node] + edge.cost
                    if candidate + 1e-12 < distance[edge.to]:
                        distance[edge.to] = candidate
                        predecessor[edge.to] = (node, edge_index)
                        changed = True
            if not changed:
                break
        if predecessor[sink] is None:
            break
        node = sink
        while node != source:
            prior, edge_index = predecessor[node]  # type: ignore[misc]
            edge = graph[prior][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = prior
    return sorted(
        (left, right)
        for (left, right), edge in original.items()
        if edge.capacity == 0
    )


def identity_edge(
    graph: pd.DataFrame,
    generic: pd.DataFrame,
    scales: dict[str, float],
) -> IdentityEdge | None:
    """Return the best admissible same-unit representative-query pair.

    Identity-uniform training gives every selected identity the same expected
    dose.  Requiring all graph queries to be covered by one generic identity
    instead selected for data-set multiplicity and destroyed common support.
    One query per identity is the estimand used by the corrected experiment.
    """
    if str(graph["truth_candidate_id"].iloc[0]) == str(generic["truth_candidate_id"].iloc[0]):
        return None
    if str(graph["truth_formula"].iloc[0]) == str(generic["truth_formula"].iloc[0]):
        return None
    best: tuple[float, str, str] | None = None
    for unit, left in graph.groupby("unit_id", sort=True):
        right = generic[generic["unit_id"].astype(str).eq(str(unit))]
        if right.empty:
            continue
        for _, left_item in left.iterrows():
            for _, right_item in right.iterrows():
                cost = query_cost(left_item, right_item, scales)
                if not np.isfinite(cost) or cost >= INFEASIBLE:
                    continue
                candidate = (
                    float(cost), str(left_item["query_id"]),
                    str(right_item["query_id"]),
                )
                if best is None or candidate < best:
                    best = candidate
    if best is None:
        return None
    return IdentityEdge(cost=best[0], query_pairs=((best[1], best[2]),))


def match_identity_clusters(
    graph: pd.DataFrame,
    pool: pd.DataFrame,
    scales: dict[str, float],
) -> tuple[pd.DataFrame, list[str]]:
    """Maximum-cardinality, minimum-cost identity-cluster matching."""
    graph_groups = [group.copy() for _, group in graph.groupby("truth_candidate_id", sort=True)]
    pool_groups = [group.copy() for _, group in pool.groupby("truth_candidate_id", sort=True)]
    if not graph_groups or not pool_groups:
        return pd.DataFrame(), [str(group["truth_candidate_id"].iloc[0]) for group in graph_groups]
    edges: dict[tuple[int, int], IdentityEdge] = {}
    cost = np.full((len(graph_groups), len(pool_groups)), INFEASIBLE)
    for i, left in enumerate(graph_groups):
        for j, right in enumerate(pool_groups):
            edge = identity_edge(left, right, scales)
            if edge is not None:
                edges[(i, j)] = edge
                cost[i, j] = edge.cost
    assignment = minimum_cost_bipartite(cost)
    matched_left = {left for left, _ in assignment}
    records: list[dict] = []
    unmatched = [
        str(group["truth_candidate_id"].iloc[0])
        for index, group in enumerate(graph_groups) if index not in matched_left
    ]
    for i, j in assignment:
        graph_identity = str(graph_groups[int(i)]["truth_candidate_id"].iloc[0])
        edge = edges[(int(i), int(j))]
        generic_identity = str(pool_groups[int(j)]["truth_candidate_id"].iloc[0])
        for graph_query, generic_query in edge.query_pairs:
            records.append({
                "graph_query_id": graph_query,
                "generic_query_id": generic_query,
                "graph_identity": graph_identity,
                "generic_identity": generic_identity,
                "identity_pair_cost": edge.cost,
            })
    result = pd.DataFrame.from_records(records)
    if len(result):
        if result["graph_query_id"].duplicated().any() or result["generic_query_id"].duplicated().any():
            raise RuntimeError("query reused within an outer-fold matching")
        mapping = result[["graph_identity", "generic_identity"]].drop_duplicates()
        if mapping["graph_identity"].duplicated().any() or mapping["generic_identity"].duplicated().any():
            raise RuntimeError("identity matching is not one-to-one")
    return result, unmatched


def enrich(actions: pd.DataFrame, body: dict[str, np.ndarray]) -> pd.DataFrame:
    query_ids = list(map(str, body["query_id"]))
    position = {value: index for index, value in enumerate(query_ids)}
    if set(actions["query_id"].astype(str)) - set(position):
        raise RuntimeError("B7 routes contain queries absent from B4 manifest")
    identity_counts = pd.Series(body["query_ik14"].astype(str)).value_counts().to_dict()
    records: list[dict] = []
    for row in actions.itertuples(index=False):
        query_id = str(row.query_id)
        query = position[query_id]
        left, right = map(int, body["query_ptr"][query:query + 2])
        scores = body["molecule_official_score"][left:right].astype(float)
        if len(scores) < 2:
            raise RuntimeError(f"{query_id}: fewer than two candidates")
        rank = 1 + int(np.sum(scores[1:] >= scores[0]))
        order = np.sort(scores)[::-1]
        positive_left, positive_right = map(int, body["reference_ptr"][left:left + 2])
        truth_identity = str(body["query_ik14"][query])
        truth_formula = str(body["query_formula"][query])
        if (str(row.truth_candidate_id) != truth_identity
                or str(row.truth_formula) != truth_formula
                or str(row.unit_id) != str(body["unit_id"][query])):
            raise RuntimeError(f"{query_id}: B7/manifest identity, formula, or unit mismatch")
        if rank != int(body["baseline_rank"][query]):
            raise RuntimeError(f"{query_id}: recomputed/manifest rank mismatch")
        if bool(row.baseline_correct) != (rank == 1):
            raise RuntimeError(f"{query_id}: B7/manifest baseline mismatch")
        record = row._asdict()
        record.update({
            "query_index": query,
            "truth_candidate_id": truth_identity,
            "truth_formula": truth_formula,
            "unit_id": str(body["unit_id"][query]),
            "baseline_rank_numeric": float(rank),
            "candidate_count": int(right - left),
            "positive_reference_count": int(positive_right - positive_left),
            "identity_query_count": int(identity_counts[truth_identity]),
            "truth_margin": float(scores[0] - np.max(scores[1:])),
            "baseline_gap": float(order[0] - order[1]),
            "log_candidate_count": float(np.log1p(right - left)),
            "log_positive_reference_count": float(np.log1p(positive_right - positive_left)),
            "log_identity_query_count": float(np.log1p(identity_counts[truth_identity])),
            "baseline_rank_bucket": stable_bucket(rank, (2, 3, 5)),
            "candidate_count_bucket": stable_bucket(right - left, (2, 3, 5, 8)),
            "identity_count_bucket": stable_bucket(identity_counts[truth_identity], (1, 2, 4)),
            "positive_reference_count_bucket": stable_bucket(
                positive_right - positive_left, (1, 2, 4, 8)
            ),
        })
        records.append(record)
    result = pd.DataFrame.from_records(records)
    if result["query_id"].astype(str).duplicated().any():
        raise RuntimeError("enriched outer routes contain duplicate queries")
    if not np.isfinite(result[list(NUMERIC_FEATURES)].to_numpy(float)).all():
        raise RuntimeError("non-finite matching feature")
    return result


def attach_training_graph(
    enriched: pd.DataFrame,
    body: dict[str, np.ndarray],
    held_formulas: set[str],
    outer: int,
) -> pd.DataFrame:
    """Recompute the exact candidate graph seen by the training loss."""
    output = enriched.copy()
    training_metrics: dict[str, dict[str, float | bool]] = {}
    for row in output.itertuples(index=False):
        query = int(row.query_index)
        left, right = map(int, body["query_ptr"][query:query + 2])
        allowed = [
            molecule for molecule in range(left, right)
            if str(body["molecule_formula"][molecule]) not in held_formulas
        ]
        if left not in allowed:
            raise RuntimeError(
                f"outer={outer} query={row.query_id}: truth candidate was filtered"
            )
        scores = body["molecule_official_score"][allowed].astype(float)
        rank = 1 + int(np.sum(scores[1:] >= scores[0])) if len(scores) >= 2 else 1
        order = np.sort(scores)[::-1]
        training_metrics[str(row.query_id)] = {
            "training_eligible": bool(len(allowed) >= 2 and rank > 1),
            "training_candidate_count": int(len(allowed)),
            "training_baseline_rank_numeric": int(rank),
            "training_truth_margin": (
                float(scores[0] - np.max(scores[1:])) if len(scores) >= 2 else 0.0
            ),
            "training_baseline_gap": (
                float(order[0] - order[1]) if len(scores) >= 2 else 0.0
            ),
        }
    for column in (
        "training_eligible", "training_candidate_count",
        "training_baseline_rank_numeric", "training_truth_margin",
        "training_baseline_gap",
    ):
        output[column] = output["query_id"].astype(str).map(
            lambda query_id, name=column: training_metrics[query_id][name]
        )
    return output


def matching_balance(matches: pd.DataFrame, enriched: pd.DataFrame) -> dict:
    if matches.empty:
        return {
            "absolute_standardised_mean_differences": {
                column: INFEASIBLE for column in NUMERIC_FEATURES
            },
            "paired_exact_fraction": {
                column: 0.0 for column in (
                    "unit_id", "baseline_rank_bucket", "candidate_count_bucket",
                    "identity_count_bucket", "positive_reference_count_bucket",
                    "baseline_rank_numeric", "candidate_count",
                    "positive_reference_count", "identity_query_count",
                )
            },
            "balance_weighting": "one mean vector per matched truth-identity pair",
            "identity_pairs": 0,
            "identity_or_formula_collisions": 0,
        }
    lookup = enriched.set_index("query_id")
    graph = lookup.loc[matches["graph_query_id"]]
    generic = lookup.loc[matches["generic_query_id"]]
    # The trainer samples truth identities uniformly and then a query within
    # identity.  Balance must therefore be assessed on one mean vector per
    # matched identity pair, not by giving identities with more queries more
    # weight.
    identity_rows: list[dict[str, float]] = []
    for (graph_identity, generic_identity), pair_rows in matches.groupby(
        ["graph_identity", "generic_identity"], sort=False
    ):
        graph_ids = pair_rows["graph_query_id"].astype(str).tolist()
        generic_ids = pair_rows["generic_query_id"].astype(str).tolist()
        item: dict[str, float] = {}
        for column in NUMERIC_FEATURES:
            item[f"graph_{column}"] = float(lookup.loc[graph_ids, column].mean())
            item[f"generic_{column}"] = float(lookup.loc[generic_ids, column].mean())
        identity_rows.append(item)
    identity = pd.DataFrame(identity_rows)
    smd = {
        column: standardised_mean_difference(
            identity[f"graph_{column}"].to_numpy(float),
            identity[f"generic_{column}"].to_numpy(float),
        )
        for column in NUMERIC_FEATURES
    }
    return {
        "absolute_standardised_mean_differences": {
            key: abs(value) for key, value in smd.items()
        },
        "paired_exact_fraction": {
            "unit_id": float(np.mean(
                graph["unit_id"].astype(str).to_numpy()
                == generic["unit_id"].astype(str).to_numpy()
            )),
            "baseline_rank_bucket": float(np.mean(
                graph["baseline_rank_bucket"].to_numpy()
                == generic["baseline_rank_bucket"].to_numpy()
            )),
            "candidate_count_bucket": float(np.mean(
                graph["candidate_count_bucket"].to_numpy()
                == generic["candidate_count_bucket"].to_numpy()
            )),
            "identity_count_bucket": float(np.mean(
                graph["identity_count_bucket"].to_numpy()
                == generic["identity_count_bucket"].to_numpy()
            )),
            "positive_reference_count_bucket": float(np.mean(
                graph["positive_reference_count_bucket"].to_numpy()
                == generic["positive_reference_count_bucket"].to_numpy()
            )),
            "baseline_rank_numeric": float(np.mean(
                graph["training_baseline_rank_numeric"].to_numpy()
                == generic["training_baseline_rank_numeric"].to_numpy()
            )),
            "candidate_count": float(np.mean(
                graph["training_candidate_count"].to_numpy()
                == generic["training_candidate_count"].to_numpy()
            )),
            "positive_reference_count": float(np.mean(
                graph["positive_reference_count"].to_numpy()
                == generic["positive_reference_count"].to_numpy()
            )),
            "identity_query_count": float(np.mean(
                graph["identity_query_count"].to_numpy()
                == generic["identity_query_count"].to_numpy()
            )),
        },
        "balance_weighting": "one mean vector per matched truth-identity pair",
        "identity_pairs": int(len(identity)),
        "identity_or_formula_collisions": int(sum(
            (str(row.graph_identity) == str(row.generic_identity))
            or (
                str(lookup.loc[str(row.graph_query_id), "truth_formula"])
                == str(lookup.loc[str(row.generic_query_id), "truth_formula"])
            )
            for row in matches.itertuples(index=False)
        )),
    }


def trim_to_common_support(
    matches: pd.DataFrame,
    enriched: pd.DataFrame,
    maximum_absolute_smd: float,
    minimum_identities: int,
) -> tuple[pd.DataFrame, list[dict]]:
    """Deterministically trim whole identity pairs until balance is achieved.

    This is outcome-blind cardinality trimming.  A graph and its paired generic
    identity are always removed together, so training mass remains symmetric.
    """
    current = matches.copy()
    history: list[dict] = []
    while current[["graph_identity", "generic_identity"]].drop_duplicates().shape[0] > minimum_identities:
        report = matching_balance(current, enriched)
        maximum = max(report["absolute_standardised_mean_differences"].values())
        history.append({
            "identity_pairs": int(report["identity_pairs"]),
            "maximum_absolute_smd": float(maximum),
        })
        if maximum <= maximum_absolute_smd:
            break
        pairs = current[["graph_identity", "generic_identity"]].drop_duplicates()
        best_key: tuple[str, str] | None = None
        best_maximum = float("inf")
        for row in pairs.itertuples(index=False):
            candidate = current[
                ~(
                    current["graph_identity"].astype(str).eq(str(row.graph_identity))
                    & current["generic_identity"].astype(str).eq(str(row.generic_identity))
                )
            ]
            candidate_report = matching_balance(candidate, enriched)
            candidate_maximum = max(
                candidate_report["absolute_standardised_mean_differences"].values()
            )
            key = (str(row.graph_identity), str(row.generic_identity))
            if (candidate_maximum < best_maximum - 1e-12
                    or (np.isclose(candidate_maximum, best_maximum) and (best_key is None or key < best_key))):
                best_maximum = float(candidate_maximum)
                best_key = key
        if best_key is None:
            break
        current = current[
            ~(
                current["graph_identity"].astype(str).eq(best_key[0])
                & current["generic_identity"].astype(str).eq(best_key[1])
            )
        ].copy()
    final = matching_balance(current, enriched)
    history.append({
        "identity_pairs": int(final["identity_pairs"]),
        "maximum_absolute_smd": float(max(
            final["absolute_standardised_mean_differences"].values()
        )),
    })
    return current.reset_index(drop=True), history


def route_frame(
    base: pd.DataFrame,
    arm: str,
    selected_queries: list[str],
) -> pd.DataFrame:
    output = base.copy()
    if len(selected_queries) != len(set(selected_queries)):
        raise RuntimeError("selected corrective route contains duplicate queries")
    selected_order = {value: index for index, value in enumerate(selected_queries)}
    output["arm"] = arm
    output["corrective_selected"] = output["query_id"].astype(str).isin(selected_order)
    output["source_graph_corrected"] = output["corrected"].astype(bool)
    # The same graph-introduced queries are safety cases in both arms.  The
    # generic error set is never mislabeled as having been corrected by graph.
    output["introduced"] = output["introduced"].astype(bool)
    mapped_order = output["query_id"].astype(str).map(selected_order)
    output["paired_schedule_order"] = np.where(
        mapped_order.notna(), mapped_order.to_numpy(float),
        len(selected_order) + np.arange(len(output)),
    ).astype(int)
    return output.sort_values("paired_schedule_order", kind="stable").reset_index(drop=True)


def main() -> None:
    args = arguments()
    if args.folds != 5:
        raise ValueError("B8 reuses exactly five B7 formula folds")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_dir}")
    action_path = args.b7_router_dir / "action_routes.csv.gz"
    router_report_path = args.b7_router_dir / "report.json"
    manifest_path = args.manifest_dir / "manifest.npz"
    manifest_report_path = args.manifest_dir / "report.json"
    for path in (action_path, router_report_path, manifest_path, manifest_report_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    router_report = json.loads(router_report_path.read_text(encoding="utf-8"))
    manifest_report = json.loads(manifest_report_path.read_text(encoding="utf-8"))
    if (router_report.get("status") != "bioaware_b7_graph_action_router_frozen"
            or not router_report.get("pass_to_direct_gradient_gate")):
        raise RuntimeError("B7 router is not the frozen passing action ledger")
    if manifest_report.get("status") != "bioaware_b4_direct_manifest_frozen":
        raise RuntimeError("wrong B4 manifest status")
    if manifest_report["provenance"].get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("B4 manifest hash mismatch")
    with np.load(manifest_path, allow_pickle=False) as loaded:
        body = {key: loaded[key] for key in loaded.files}
    required = {
        "query_id", "unit_id", "query_ik14", "query_formula", "formula_fold",
        "query_ptr", "molecule_formula", "molecule_official_score",
        "reference_ptr", "baseline_rank",
    }
    if missing := required - set(body):
        raise RuntimeError(f"manifest lacks {sorted(missing)}")
    if len(body["query_id"]) != 548:
        raise RuntimeError("B8 requires the exact 548-query B7 manifest")

    actions = pd.read_csv(action_path)
    if set(actions["arm"].astype(str)) != {"graph_prior_direct"}:
        raise RuntimeError("B7 action file contains an unexpected arm")
    route_parts: list[pd.DataFrame] = []
    match_parts: list[pd.DataFrame] = []
    fold_reports: dict[str, dict] = {}
    all_query_ids = set(map(str, body["query_id"]))

    for outer in range(args.folds):
        base = actions[actions["outer_fold"].eq(outer)].copy()
        expected = {
            str(body["query_id"][query]) for query in range(len(body["query_id"]))
            if int(body["formula_fold"][query]) != outer
        }
        if base["query_id"].astype(str).duplicated().any() or set(base["query_id"].astype(str)) != expected:
            raise RuntimeError(f"outer={outer}: B7 action coverage changed")
        enriched = enrich(base, body)
        held_formulas = set(map(
            str, body["query_formula"][body["formula_fold"].astype(int) == outer]
        ))
        enriched = attach_training_graph(enriched, body, held_formulas, outer)
        eligible = enriched[enriched["training_eligible"].astype(bool)].copy()
        graph = eligible[eligible["corrected"].astype(bool)].copy()
        generic_pool = eligible[
            ~eligible["baseline_correct"].astype(bool)
            & ~eligible["corrected"].astype(bool)
        ].copy()
        scales = robust_scales(eligible[~eligible["baseline_correct"].astype(bool)])
        matches, unmatched = match_identity_clusters(graph, generic_pool, scales)
        pretrim_identities = int(matches["graph_identity"].nunique()) if not matches.empty else 0
        if matches.empty:
            matches = pd.DataFrame(columns=(
                "graph_identity", "generic_identity", "graph_query_id",
                "generic_query_id", "identity_pair_cost",
            ))
            trimming_history = []
        else:
            matches, trimming_history = trim_to_common_support(
                matches, enriched, args.maximum_absolute_smd,
                args.minimum_matched_identities,
            )
        matches["outer_fold"] = outer
        balance = matching_balance(matches, enriched)
        graph_query_order = matches["graph_query_id"].astype(str).tolist()
        generic_query_order = matches["generic_query_id"].astype(str).tolist()
        graph_queries = set(graph_query_order)
        generic_queries = set(generic_query_order)
        graph_identities = int(matches["graph_identity"].nunique())
        generic_identities = int(matches["generic_identity"].nunique())
        if (len(graph_queries) != len(generic_queries)
                or graph_identities != generic_identities
                or len(graph_queries) != graph_identities
                or graph_queries & generic_queries
                or not graph_queries | generic_queries <= all_query_ids):
            raise RuntimeError(f"outer={outer}: matched-arm symmetry failed")
        graph_total_identities = int(graph["truth_candidate_id"].nunique())
        retention = graph_identities / graph_total_identities
        maximum_smd = max(balance["absolute_standardised_mean_differences"].values())
        fold_reports[str(outer)] = {
            "available_graph_corrective_queries": int(len(graph)),
            "available_graph_corrective_identities": graph_total_identities,
            "post_held_formula_filter_queries": int(len(eligible)),
            "matched_queries_per_arm": int(len(matches)),
            "matched_identities_per_arm": graph_identities,
            "pretrim_matched_identities": pretrim_identities,
            "graph_identity_retention": float(retention),
            "unmatched_graph_identities": len(unmatched),
            "identity_pairs_removed_for_balance": int(
                pretrim_identities - graph_identities
            ),
            "common_support_trimming_history": trimming_history,
            "maximum_absolute_smd": float(maximum_smd),
            "balance": balance,
            "same_harm_queries_per_arm": int(enriched["introduced"].astype(bool).sum()),
        }
        fold_reports[str(outer)]["pass_to_training"] = bool(
            graph_identities >= args.minimum_matched_identities
            and maximum_smd <= args.maximum_absolute_smd
            and balance["paired_exact_fraction"]["unit_id"] == 1.0
            and balance["identity_or_formula_collisions"] == 0
        )
        route_parts.extend((
            route_frame(base, GRAPH_ARM, graph_query_order),
            route_frame(base, GENERIC_ARM, generic_query_order),
        ))
        match_parts.append(matches)

    routes = pd.concat(route_parts, ignore_index=True)
    matches = pd.concat(match_parts, ignore_index=True)
    for (outer, arm), group in routes.groupby(["outer_fold", "arm"]):
        expected = {
            str(body["query_id"][query]) for query in range(len(body["query_id"]))
            if int(body["formula_fold"][query]) != int(outer)
        }
        if group["query_id"].astype(str).duplicated().any() or set(group["query_id"].astype(str)) != expected:
            raise RuntimeError(f"outer={outer} arm={arm}: route coverage mismatch")
    gates = {
        "every_fold_identities_ge_minimum": all(
            item["matched_identities_per_arm"] >= args.minimum_matched_identities
            for item in fold_reports.values()
        ),
        "every_fold_maximum_absolute_smd_le_threshold": all(
            item["maximum_absolute_smd"] <= args.maximum_absolute_smd
            for item in fold_reports.values()
        ),
        "every_fold_unit_matching_exact": all(
            item["balance"]["paired_exact_fraction"]["unit_id"] == 1.0
            for item in fold_reports.values()
        ),
        "every_fold_no_identity_or_formula_collision": all(
            item["balance"]["identity_or_formula_collisions"] == 0
            for item in fold_reports.values()
        ),
        "every_fold_arm_identity_and_query_mass_equal": True,
        "every_fold_individual_gate_passed": all(
            item["pass_to_training"] for item in fold_reports.values()
        ),
    }
    args.output_dir.mkdir(parents=True)
    routes_path = args.output_dir / "action_routes.csv.gz"
    matches_path = args.output_dir / "matched_pairs.csv.gz"
    routes.to_csv(routes_path, index=False, compression="gzip")
    matches.to_csv(matches_path, index=False, compression="gzip")
    fold0_pass = bool(fold_reports["0"]["pass_to_training"])
    all_folds_pass = bool(all(
        item["pass_to_training"] for item in fold_reports.values()
    ))
    report = {
        "status": STATUS if fold0_pass else "bioaware_b8_matched_error_control_not_identifiable",
        "formal": True,
        "protocol": (
            "one representative query per identity; identity-cluster matched "
            "graph-selected versus generic baseline errors; same-unit query "
            "pairing; equal identity/query/reference training mass"
        ),
        "arms": [GRAPH_ARM, GENERIC_ARM],
        "outer_folds": fold_reports,
        "gates": {key: bool(value) for key, value in gates.items()},
        "pass_to_fold0_training": fold0_pass,
        "all_folds_ready": all_folds_pass,
        "contracts": {
            "exact_b7_548_query_protocol": True,
            "identity_cluster_is_matching_unit": True,
            "one_representative_query_per_identity": True,
            "generic_identity_used_without_replacement_within_outer_fold": True,
            "biological_unit_exact_within_query_pairs": True,
            "graph_and_generic_training_mass_equal": True,
            "paired_corrective_identity_sampling_order": True,
            "formal_training_references_per_candidate": 1,
            "calipers": (
                "same unit and exact post-held-formula baseline-rank/candidate-count plus "
                "raw positive-reference-count/identity-query-count; maximum one training-pool "
                "SD on post-held-formula truth margin and baseline Top1-Top2 gap"
            ),
            "matching_graph_equals_backprop_graph": True,
            "balance_weighting_matches_identity_uniform_trainer": True,
            "common_support_trimming_is_pairwise_and_outcome_blind": True,
            "same_graph_introduced_harm_queries_in_both_arms": True,
            "held_embedding_outcome_used_for_matching": False,
            "P2b_used": False,
            "phenotype_used": False,
        },
        "provenance": {
            "b7_router_report_sha256": sha256_file(router_report_path),
            "b7_action_routes_sha256": sha256_file(action_path),
            "b4_manifest_sha256": sha256_file(manifest_path),
            "b4_manifest_report_sha256": sha256_file(manifest_report_path),
            "routes_sha256": sha256_file(routes_path),
            "matches_sha256": sha256_file(matches_path),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "A matched training-router control only. It does not show graph-specific "
            "embedding gain until both shared-encoder arms are trained and compared."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    if not report["pass_to_fold0_training"]:
        print(
            "[BioAware B8] NOT IDENTIFIABLE on fold 0; no GPU training is authorised. "
            f"Diagnostics: {fold_reports['0']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
