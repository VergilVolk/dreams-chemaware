#!/usr/bin/env python
"""Score B47 candidates and freeze cross-sample-consensus seeds without truth."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from bioaware_b47_seed_core import (  # noqa: E402
    PRIMARY_POLICY, STRICT_POLICY, attach_absolute_gates, feature_consensus,
    query_summaries, select_sample_seeds,
)
from bioaware_b47_truthblind_io import sha256_file  # noqa: E402


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


def parse_bool(value: object) -> bool:
    return str(value).strip().casefold() in {"1", "true", "t", "yes"}


def quantiles(values: pd.Series) -> dict[str, float | int]:
    if values.empty:
        return {"minimum": 0, "median": 0.0, "p90": 0.0, "maximum": 0}
    return {
        "minimum": int(values.min()), "median": float(values.median()),
        "p90": float(values.quantile(0.9)), "maximum": int(values.max()),
    }


def load_and_verify(graph: Path, embeddings: Path) -> tuple[dict, dict]:
    graph_report_path = graph / "report.json"
    embedding_report_path = embeddings / "report.json"
    graph_report = json.loads(graph_report_path.read_text(encoding="utf-8"))
    embedding_report = json.loads(embedding_report_path.read_text(encoding="utf-8"))
    if not graph_report.get("pass_to_query_embedding_and_seed_construction"):
        raise RuntimeError("B47 candidate graph gate failed")
    if not embedding_report.get("pass_to_truthblind_seed_construction"):
        raise RuntimeError("B47 embedding gate failed")
    expected_graph_contract = {
        "P2b_used": False, "algorithm_outputs_opened": False,
        "phenotype_used": False, "truth_opened": False,
    }
    if graph_report.get("contracts") != expected_graph_contract:
        raise RuntimeError("B47 candidate graph truth-blind contract changed")
    expected_embedding_false = (
        "truth_opened", "phenotype_used", "algorithm_outputs_opened",
        "candidate_scores_computed", "seed_selection_performed", "model_fitted", "P2b_used",
    )
    embedding_contract = embedding_report.get("contracts", {})
    if embedding_contract.get("one_shared_official_encoder") is not True or any(
        embedding_contract.get(key) is not False for key in expected_embedding_false
    ):
        raise RuntimeError("B47 embedding truth-blind contract changed")
    if embedding_report.get("provenance", {}).get("graph_report_sha256") != sha256_file(graph_report_path):
        raise RuntimeError("B47 graph/embedding provenance mismatch")
    for key, filename in (
        ("queries_sha256", "queries.csv.gz"),
        ("candidate_references_sha256", "candidate_references.csv.gz"),
    ):
        observed = sha256_file(graph / filename)
        if graph_report["provenance"].get(key) != observed:
            raise RuntimeError(f"B47 graph file hash mismatch: {filename}")
        if embedding_report["provenance"].get(key) != observed:
            raise RuntimeError(f"B47 embedding input hash mismatch: {filename}")
    for key, filename in (
        ("query_embeddings_sha256", "query_embeddings.npy"),
        ("reference_embeddings_sha256", "reference_embeddings.npy"),
        ("reference_rows_sha256", "reference_rows.npy"),
        ("query_index_sha256", "query_index.csv"),
    ):
        if embedding_report["provenance"].get(key) != sha256_file(embeddings / filename):
            raise RuntimeError(f"B47 embedding output hash mismatch: {filename}")
    return graph_report, embedding_report


def score_reference_pairs(
    query_embedding: np.ndarray,
    reference_embedding: np.ndarray,
    query_positions: np.ndarray,
    reference_positions: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    # The embedding arrays are normally read-only ``mmap`` objects.  PyTorch
    # warns that wrapping such arrays could be unsafe if a tensor were ever
    # mutated.  Make the ownership boundary explicit before moving to the GPU;
    # this scorer is read-only, but it must not rely on that implicit promise.
    q_host = np.array(query_embedding, dtype=np.float32, copy=True, order="C")
    r_host = np.array(reference_embedding, dtype=np.float32, copy=True, order="C")
    if not q_host.flags.writeable or not r_host.flags.writeable:
        raise RuntimeError("B47 host embedding copies are unexpectedly read-only")
    q = torch.from_numpy(q_host).to(device=device)
    r = torch.from_numpy(r_host).to(device=device)
    del q_host, r_host
    output = np.empty(len(query_positions), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(output), batch_size):
            end = min(len(output), start + batch_size)
            qpos = torch.from_numpy(query_positions[start:end]).to(device=device)
            rpos = torch.from_numpy(reference_positions[start:end]).to(device=device)
            value = (q.index_select(0, qpos) * r.index_select(0, rpos)).sum(dim=1)
            output[start:end] = value.float().cpu().numpy()
            if end % (batch_size * 10) == 0 or end == len(output):
                print(f"[B47 score] {end:,}/{len(output):,}", flush=True)
    if not np.isfinite(output).all() or output.min() < -1.0001 or output.max() > 1.0001:
        raise RuntimeError("invalid B47 cosine scores")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--participants", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-batch-size", type=int, default=65536)
    parser.add_argument("--maximum-seed-degree", type=int, default=250)
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if args.score_batch_size <= 0 or args.maximum_seed_degree <= 0:
        raise ValueError("batch size and maximum degree must be positive")
    graph, embeddings = args.graph_dir.resolve(), args.embedding_dir.resolve()
    for path in (graph, embeddings):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if not args.participants.is_file():
        raise FileNotFoundError(args.participants)
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite output: {output}")
    output.mkdir(parents=True, exist_ok=False)
    graph_report, embedding_report = load_and_verify(graph, embeddings)

    queries = pd.read_csv(graph / "queries.csv.gz")
    query_index = pd.read_csv(embeddings / "query_index.csv")
    if queries["query_id"].astype(str).tolist() != query_index["query_id"].astype(str).tolist():
        raise RuntimeError("B47 query/index order mismatch")
    if not np.array_equal(query_index["embedding_position"].to_numpy(int), np.arange(len(queries))):
        raise RuntimeError("B47 query positions are not canonical")
    query_position = pd.Series(
        query_index["embedding_position"].to_numpy(np.int64),
        index=query_index["query_id"].astype(str),
    )
    reference_rows = np.load(embeddings / "reference_rows.npy", mmap_mode="r", allow_pickle=False)
    query_embedding = np.load(embeddings / "query_embeddings.npy", mmap_mode="r", allow_pickle=False)
    reference_embedding = np.load(
        embeddings / "reference_embeddings.npy", mmap_mode="r", allow_pickle=False
    )
    if query_embedding.shape[0] != len(queries) or reference_embedding.shape[0] != len(reference_rows):
        raise RuntimeError("B47 embedding/manifest shape mismatch")

    candidate = pd.read_csv(
        graph / "candidate_references.csv.gz",
        usecols=["query_id", "candidate_id", "reference_row", "reference_adduct", "candidate_formula"],
        dtype={"query_id": str, "candidate_id": str, "reference_row": np.int64,
               "reference_adduct": str, "candidate_formula": str},
    )
    if len(candidate) != int(graph_report["candidate_reference_rows"]):
        raise RuntimeError("B47 candidate-reference row count mismatch")
    qpos = candidate["query_id"].map(query_position)
    if qpos.isna().any():
        raise RuntimeError("B47 candidate has unknown query")
    rpos = np.searchsorted(reference_rows, candidate["reference_row"].to_numpy(np.int64))
    if np.any(rpos >= len(reference_rows)) or not np.array_equal(
        np.asarray(reference_rows[rpos]), candidate["reference_row"].to_numpy(np.int64)
    ):
        raise RuntimeError("B47 candidate reference row is absent from embedding index")
    candidate["spectral_score"] = score_reference_pairs(
        query_embedding, reference_embedding, qpos.to_numpy(np.int64),
        rpos.astype(np.int64), torch.device(args.device), args.score_batch_size,
    )

    group_columns = ["query_id", "candidate_id"]
    reference_counts = candidate.groupby(group_columns, sort=False).size().rename("reference_spectra")
    best_index = candidate.groupby(group_columns, sort=False)["spectral_score"].idxmax()
    scores = candidate.loc[best_index].copy()
    scores = scores.merge(reference_counts, on=group_columns, how="left", validate="one_to_one")
    scores = scores.rename(columns={"reference_row": "best_reference_row",
                                    "reference_adduct": "best_reference_adduct"})
    scores = scores.sort_values(
        ["query_id", "spectral_score", "candidate_id"],
        ascending=[True, False, True], kind="stable",
    ).reset_index(drop=True)
    scores["spectral_rank_ties_adverse"] = (
        scores.groupby("query_id", sort=False)["spectral_score"]
        .rank(method="max", ascending=False).astype(int)
    )
    summary = query_summaries(scores)
    summary = queries.merge(summary, on="query_id", how="left", validate="one_to_one")
    if summary["top_candidate_id"].isna().any():
        raise RuntimeError("B47 query lost its spectral summary")
    policies = (PRIMARY_POLICY, STRICT_POLICY)
    summary = attach_absolute_gates(summary, policies)
    consensus = feature_consensus(summary, policies)

    participants = pd.read_csv(args.participants)
    required_participant = {"compound_id", "reaction_id", "is_currency"}
    if not required_participant.issubset(participants.columns):
        raise RuntimeError("Rhea participant cache lacks required columns")
    participants["compound_id"] = participants["compound_id"].astype(str).str[:14].str.upper()
    degree = participants.groupby("compound_id")["reaction_id"].nunique().astype(int).to_dict()
    currency = set(
        participants.loc[participants["is_currency"].map(parse_bool), "compound_id"].astype(str)
    )
    primary, primary_audit = select_sample_seeds(
        summary, consensus, PRIMARY_POLICY, degree, currency, args.maximum_seed_degree
    )
    strict, strict_audit = select_sample_seeds(
        summary, consensus, STRICT_POLICY, degree, currency, args.maximum_seed_degree
    )

    paths = {
        "candidate_scores": output / "candidate_scores.csv.gz",
        "query_summaries": output / "query_summaries.csv.gz",
        "feature_consensus": output / "feature_consensus.csv.gz",
        "seeds_primary": output / "seeds_primary.csv.gz",
        "seeds_strict": output / "seeds_strict.csv.gz",
    }
    write_csv(scores, paths["candidate_scores"])
    write_csv(summary, paths["query_summaries"])
    write_csv(consensus, paths["feature_consensus"])
    write_csv(primary, paths["seeds_primary"])
    write_csv(strict, paths["seeds_strict"])

    primary_per_sample = primary.groupby(["study", "sample"]).size()
    primary_source = primary.groupby("study").agg(
        seed_rows=("seed_query_id", "size"), seed_identities=("seed_compound_id", "nunique"),
        seed_features=("seed_feature_id", "nunique"), samples=("sample", "nunique"),
    ).to_dict(orient="index") if len(primary) else {}
    gates = {
        "all_queries_scored": len(summary) == int(graph_report["queries"]),
        "candidate_sets_preserved": scores.groupby("query_id")["candidate_id"].nunique().ge(2).all(),
        "primary_seed_rows_ge_500": len(primary) >= 500,
        "primary_seed_identities_ge_200": primary["seed_compound_id"].nunique() >= 200,
        "both_sources_ge_100_primary_seeds": all(
            int(primary_source.get(source, {}).get("seed_rows", 0)) >= 100
            for source in ("ST001122", "ST003356")
        ),
        "truth_columns_absent": not any(
            "truth" in column.casefold() for column in (*scores.columns, *queries.columns)
        ),
        "phenotype_columns_absent": not any(
            token in column.casefold() for column in summary.columns
            for token in ("disease", "case_control", "phenotype")
        ),
        "P2b_absent": True,
    }
    report = {
        "status": "bioaware_b47_truthblind_seed_construction_complete",
        "formal": True,
        "queries": int(len(summary)), "candidate_identities": int(len(scores)),
        "spectral_score_range": [float(scores["spectral_score"].min()),
                                 float(scores["spectral_score"].max())],
        "unique_top1_queries": int(summary["unique_top1"].sum()),
        "primary_absolute_gate_queries": int(summary["primary_absolute_gate"].sum()),
        "strict_absolute_gate_queries": int(summary["strict_absolute_gate"].sum()),
        "features": int(len(consensus)),
        "primary_consensus_features": int(consensus["primary_feature_gate"].sum()),
        "strict_consensus_features": int(consensus["strict_feature_gate"].sum()),
        "primary_seeds": {
            "rows": int(len(primary)),
            "identities": int(primary["seed_compound_id"].nunique()),
            "features": int(primary["seed_feature_id"].nunique()),
            "samples": int(primary[["study", "sample"]].drop_duplicates().shape[0]),
            "per_sample": quantiles(primary_per_sample), "per_source": primary_source,
            "audit": primary_audit,
        },
        "strict_sensitivity_seeds": {
            "rows": int(len(strict)),
            "identities": int(strict["seed_compound_id"].nunique()),
            "features": int(strict["seed_feature_id"].nunique()),
            "audit": strict_audit,
        },
        "policies": {
            policy.name: {
                "minimum_score": policy.minimum_score,
                "minimum_margin": policy.minimum_margin,
                "minimum_modal_samples": policy.minimum_modal_samples,
                "minimum_modal_fraction": policy.minimum_modal_fraction,
            } for policy in policies
        },
        "parameters": {
            "score_batch_size": int(args.score_batch_size),
            "maximum_seed_degree": int(args.maximum_seed_degree),
            "candidate_aggregation": "maximum cosine across reference spectra per IK14",
            "candidate_ties": "adverse; a tied Top-1 cannot become a seed",
        },
        "gates": {key: bool(value) for key, value in gates.items()},
        "pass_to_truthblind_event_graph": all(bool(value) for value in gates.values()),
        "contracts": {
            "seed_source": "frozen DreaMS spectral score plus cross-sample feature consensus",
            "truth_opened": False, "phenotype_used": False, "performance_computed": False,
            "P2b_used": False,
            "original_seed_set_frozen_before_truth": True,
            "evaluation_requirement": (
                "sealed evaluator must retain only queries whose truth identity was absent "
                "from the original same-sample seed set"
            ),
        },
        "provenance": {
            "graph_report_sha256": sha256_file(graph / "report.json"),
            "embedding_report_sha256": sha256_file(embeddings / "report.json"),
            "participants_sha256": sha256_file(args.participants),
            **{f"{key}_sha256": sha256_file(path) for key, path in paths.items()},
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": (
            "Truth-blind spectral seed construction only. Seed precision, truth coverage, "
            "DreaMS errors, BioAware gain, and SOTA remain unevaluated."
        ),
    }
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
