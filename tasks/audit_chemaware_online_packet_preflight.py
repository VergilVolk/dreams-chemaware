"""Build the online query-packet pool under a supplied frozen embedding cache.

This is a zero-update engineering preflight.  Formal online packets are rebuilt
inside the live training process from the current Phase-A/online weights.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from chemaware_online_packet_core import EmbeddingStore, build_online_packet_pool
from build_chemaware_dreams_native_triplets import audit_identity_edges
from chemaware_v2_triplet_eval_core import numerical_rank_replay_audit


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-bank", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dreams-replay-pool", type=Path, required=True)
    parser.add_argument("--embedding-rows", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=0.1)
    parser.add_argument("--packet-width", type=int, default=3)
    parser.add_argument("--dreams-replay-events", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    base_report = json.loads(
        (args.base_bank / "report.json").read_text(encoding="utf-8")
    )
    if (
        base_report.get("status") != "CHEMAWARE_ACTION_HARD_NATIVE_TRIPLETS_COMPLETE"
        or base_report.get("roles", {}).get("optimization") != "formula roles 0-1"
        or base_report.get("roles", {}).get("model_validation") != "formula role 3"
        or base_report.get("roles", {}).get("outer") != "formula role 4 untouched"
    ):
        raise RuntimeError("base bank does not prove the frozen formula-role contract")
    manifest = load_npz(args.manifest)
    evidence = load_npz(args.evidence)
    embedding_rows = np.load(args.embedding_rows, mmap_mode="r")
    embeddings = np.load(args.embeddings, mmap_mode="r")
    pool, report = build_online_packet_pool(
        manifest=manifest, evidence=evidence,
        base=load_npz(args.base_bank / "train_pool.npz"),
        replay=load_npz(args.dreams_replay_pool),
        embedding_rows=embedding_rows, embeddings=embeddings,
        margin=args.margin, packet_width=args.packet_width,
        replay_events=args.dreams_replay_events, seed=args.seed,
    )
    report = dict(report)
    report["weights_updated"] = False
    report["engineering_preflight_only"] = True
    report["formal_packets_built_inside_live_training"] = True
    report["identity_audit"] = audit_identity_edges(pool, args.data)
    rank_by_query = {
        int(query): int(rank)
        for query, rank in zip(pool["source_query"], pool["current_rank"], strict=True)
        if int(query) >= 0
    }
    queries = np.asarray(evidence["query"], dtype=np.int64)
    expected_rank = np.asarray(evidence["baseline_rank"], dtype=np.int32)
    observed_rank = np.asarray(
        [rank_by_query[int(query)] for query in queries], dtype=np.int32,
    )
    normalized = EmbeddingStore(embedding_rows, embeddings).embeddings
    stable, replay_audit = numerical_rank_replay_audit(
        normalized, np.asarray(embedding_rows, dtype=np.int64), manifest,
        queries, expected_rank, observed_rank,
    )
    report["frozen_baseline_rank_mismatches"] = int(np.sum(
        observed_rank != expected_rank
    ))
    report["numerical_boundary_exclusions"] = int(np.sum(~stable))
    report["numerical_boundary_audit"] = replay_audit
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chem_online_preflight_", dir=args.output.parent))
    try:
        np.savez_compressed(temporary / "packet_pool.npz", **pool)
        (temporary / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8",
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
