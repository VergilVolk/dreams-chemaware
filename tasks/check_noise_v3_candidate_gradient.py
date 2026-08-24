"""Fast fail-closed preflight for the formal noise-v3 candidate-gradient audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
for item in (ROOT, ROOT / "tasks"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from audit_noise_v3_candidate_gradient import (  # noqa: E402
    DEFAULT_ARCHITECTURE, DEFAULT_CACHE, DEFAULT_DATA, DEFAULT_EMBEDDINGS,
    DEFAULT_OFFICIAL, DEFAULT_P3, load_embeddings, query_candidate_block,
)
from build_g8r_real_error_atlas import Cache, load_p3_identities, sha256_file  # noqa: E402
from noise_v3_core import candidate_representatives  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--embedding-cache", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--official-checkpoint", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--architecture-checkpoint", type=Path, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--p3-dir", type=Path, default=DEFAULT_P3)
    parser.add_argument("--minimum-queries", type=int, default=20000)
    parser.add_argument("--minimum-errors", type=int, default=1000)
    args = parser.parse_args()
    for path in (
        args.cache, args.embedding_cache, args.data, args.official_checkpoint,
        args.architecture_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    cache = Cache(args.cache)
    if cache.n_queries < args.minimum_queries:
        raise RuntimeError(f"candidate graph is unexpectedly small: {cache.n_queries}")
    p3 = load_p3_identities(args.p3_dir)
    overlap = set(map(str, cache.query_ik14)) & p3
    if overlap:
        raise RuntimeError(f"P3 leakage: {len(overlap)} identities")
    score_column = cache.feature_names.index("dreams_similarity")
    needed = set(map(int, cache.query_row))
    errors = near_queries = pair_edges = 0
    candidate_counts = []
    for query in range(cache.n_queries):
        scores, rows, ptr = query_candidate_block(cache, query, score_column)
        rep = candidate_representatives(scores, rows, ptr, 5)
        errors += int(rep.positive_score <= rep.hardest_negative_score)
        near_queries += int(cache.query_has_near[query])
        needed.update(map(int, rows))
        pair_edges += len(rows)
        candidate_counts.append(len(ptr) - 1)
    if errors < args.minimum_errors:
        raise RuntimeError(f"too few official errors for formal audit: {errors}")
    _, _, embedding_index = load_embeddings(args.embedding_cache)
    missing = needed - set(embedding_index)
    if missing:
        raise RuntimeError(f"embedding cache misses {len(missing)} required rows")
    report = {
        "status": "noise_v3_candidate_gradient_preflight_passed",
        "queries": cache.n_queries,
        "identities": int(len(set(map(str, cache.query_ik14)))),
        "official_errors": errors,
        "near_queries": near_queries,
        "candidate_spectrum_pairs": pair_edges,
        "reachable_rows": len(needed),
        "candidate_molecules_per_query": {
            "median": float(np.median(candidate_counts)),
            "p90": float(np.quantile(candidate_counts, 0.9)),
            "maximum": int(max(candidate_counts)),
        },
        "p3_identity_overlap": 0,
        "provenance": {
            "cache_sha256": sha256_file(args.cache),
            "embedding_cache_sha256": sha256_file(args.embedding_cache),
            "hdf5_sha256": sha256_file(args.data),
            "official_checkpoint_sha256": sha256_file(args.official_checkpoint),
        },
        "pass": True,
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
