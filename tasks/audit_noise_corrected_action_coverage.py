"""Independently validate the corrected graph and remap mature N actions by row."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from noise_final_core import (
    CandidateGraph, load_embedding_cache, sha256_file, stable_fold, strict_rank,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--r0-dir", type=Path, required=True)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--score-probes", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    graph_path = args.graph_dir / "candidate_graph.npz"
    cache_path = args.graph_dir / "official_embeddings.npz"
    graph_report_path = args.graph_dir / "report.json"
    r0_path = args.r0_dir / "training_actions.csv.gz"
    r0_report_path = args.r0_dir / "report.json"
    for path in (graph_path, cache_path, graph_report_path, r0_path, r0_report_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    graph_report = json.loads(graph_report_path.read_text(encoding="utf-8"))
    if (
        graph_report.get("status") != "noise_corrected_candidate_graph_complete"
        or graph_report.get("formal_training_authorized") is not True
        or graph_report.get("data_contract") != "train_primary_all_p3_disjoint_v1"
        or graph_report.get("provenance", {}).get("candidate_graph_sha256")
        != sha256_file(graph_path)
        or graph_report.get("provenance", {}).get("embedding_cache_sha256")
        != sha256_file(cache_path)
    ):
        raise RuntimeError("corrected graph report or hashes are invalid")

    # Load through the production consumers, not through builder-local arrays.
    graph = CandidateGraph(graph_path)
    rows, embeddings, row_index = load_embedding_cache(cache_path)
    if any(int(row) not in row_index for row in graph.query_row):
        raise RuntimeError("production embedding loader misses a query row")
    rng = np.random.default_rng(args.seed)
    probe_count = min(args.score_probes, len(graph.pair_candidate_row))
    probes = rng.choice(len(graph.pair_candidate_row), size=probe_count, replace=False)
    molecule_query = np.repeat(
        np.arange(graph.n_queries, dtype=np.int64), np.diff(graph.query_ptr),
    )
    pair_query = np.repeat(molecule_query, np.diff(graph.molecule_ptr))
    direct = np.einsum(
        "ij,ij->i",
        embeddings[[row_index[int(graph.query_row[pair_query[index]])] for index in probes]],
        embeddings[[row_index[int(graph.pair_candidate_row[index])] for index in probes]],
    )
    stored = graph.features[probes, graph.dreams_column]
    if not np.allclose(direct, stored, rtol=0, atol=2e-6):
        raise RuntimeError("independent official-score probes disagree with graph")
    ranks = np.asarray([
        strict_rank(graph.official_molecule_scores(query))
        for query in range(graph.n_queries)
    ], dtype=np.int16)
    observed_baseline = {
        "queries": int(len(ranks)),
        "recall1": float(np.mean(ranks == 1)),
        "mrr": float(np.mean(1.0 / ranks)),
        "errors": int(np.sum(ranks != 1)),
        "near_queries": int(graph.query_has_near.sum()),
        "near_recall1": float(np.mean(ranks[graph.query_has_near] == 1)),
    }
    expected_baseline = graph_report["official_baseline"]
    for key in observed_baseline:
        if not np.isclose(observed_baseline[key], expected_baseline[key], rtol=0, atol=1e-12):
            raise RuntimeError(f"production graph baseline drifted: {key}")

    actions = pd.read_csv(r0_path, low_memory=False)
    required = {
        "query_index", "query_row", "query_ik14", "query_formula", "formula_fold",
        "selector", "attenuation", "step", "target_path", "matched_control_paths",
        "hard_negative_row",
    }
    if missing := required - set(actions.columns):
        raise RuntimeError(f"R0 lacks remap columns: {sorted(missing)}")
    new_query_by_row = {int(row): index for index, row in enumerate(graph.query_row)}
    mapped = actions["query_row"].astype(int).map(new_query_by_row)
    retained = actions.loc[mapped.notna()].copy()
    retained.insert(0, "legacy_query_index", retained["query_index"].astype(np.int64))
    retained["query_index"] = mapped[mapped.notna()].astype(np.int64).to_numpy()
    query = retained["query_index"].to_numpy(np.int64)
    if not np.array_equal(retained["query_row"].to_numpy(np.int64), graph.query_row[query]):
        raise RuntimeError("action row remap is not exact")
    if not np.array_equal(retained["query_ik14"].astype(str).to_numpy(), graph.query_ik14[query]):
        raise RuntimeError("action identity changed during corrected graph remap")
    if not np.array_equal(retained["query_formula"].astype(str).to_numpy(), graph.query_formula[query]):
        raise RuntimeError("action formula changed during corrected graph remap")
    expected_fold = np.asarray([
        stable_fold(str(formula), 5, args.formula_fold_seed)
        for formula in retained["query_formula"].astype(str)
    ], dtype=np.int8)
    if not np.array_equal(expected_fold, retained["formula_fold"].to_numpy(np.int8)):
        raise RuntimeError("formula-fold assignment changed during corrected graph remap")

    hard_negative_present = np.zeros(len(retained), dtype=bool)
    for position, row in enumerate(retained[["query_index", "hard_negative_row"]].itertuples(index=False)):
        _, candidate_rows, local_ptr, _ = graph.query_block(int(row.query_index))
        positive_end = int(local_ptr[1])
        hard_negative_present[position] = int(row.hard_negative_row) in set(
            map(int, candidate_rows[positive_end:])
        )
    retained["corrected_graph_hard_negative_present"] = hard_negative_present
    report = {
        "status": "noise_corrected_action_coverage_complete",
        "formal_training_authorized": False,
        "production_loader_baseline": observed_baseline,
        "official_score_probes": probe_count,
        "actions": {
            "source_rows": int(len(actions)),
            "retained_rows": int(len(retained)),
            "retained_fraction": float(len(retained) / len(actions)),
            "source_queries": int(actions["query_row"].nunique()),
            "retained_queries": int(retained["query_row"].nunique()),
            "retained_identities": int(retained["query_ik14"].astype(str).nunique()),
            "retained_formulas": int(retained["query_formula"].astype(str).nunique()),
            "hard_negative_present_rows": int(hard_negative_present.sum()),
            "hard_negative_present_fraction": float(hard_negative_present.mean()),
        },
        "contracts": {
            "production_candidate_graph_loader_passed": True,
            "production_embedding_cache_loader_passed": True,
            "independent_official_score_probes_exact": True,
            "query_row_remap_only": True,
            "identity_formula_unchanged": True,
            "formula_folds_unchanged": True,
            "action_outcomes_consumed": False,
            "P2b": "forbidden",
            "P3_consumed": False,
        },
        "next_gate": (
            "Recompute target/control outcomes on the corrected candidate graph at the "
            "frozen initialization; old-graph route labels are forbidden."
        ),
        "claim_limit": "Coverage and baseline audit only; no action has been qualified for training.",
        "provenance": {
            "graph_report_sha256": sha256_file(graph_report_path),
            "r0_report_sha256": sha256_file(r0_report_path),
            "r0_actions_sha256": sha256_file(r0_path),
            "script_sha256": sha256_file(Path(__file__)),
        },
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".noise_corrected_action_", dir=args.output_dir.parent))
    try:
        retained.to_csv(
            staging / "remapped_training_actions.csv.gz", index=False, compression="gzip",
        )
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
