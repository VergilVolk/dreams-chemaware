"""Build and audit candidate-conditioned peak-token evidence on the full graph.

This stage does not train a model.  It measures whether contextual fragment
tokens add deployable within-query ranking information beyond the frozen
DreaMS/RAW candidate graph.  Every feature is label-free; identity labels are
used only after feature construction for evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from audit_e0_observability_residual import greedy_matches
from build_g8r_real_error_atlas import Cache
from g8r_p2_listwise_core import evaluate_query_scores


ROOT = Path(__file__).resolve().parent.parent
TOKEN_FEATURES = [
    "token_cosine_mean",
    "token_cosine_weighted",
    "token_cosine_min",
    "token_cosine_p25",
    "token_cosine_p75",
    "token_low_similarity_fraction",
    "token_high_similarity_fraction",
    "token_match_fraction_min",
    "token_conflict_weighted",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    p.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/g8r_noise_v3_c2b_candidate_tokens")
    p.add_argument("--output", type=Path, default=ROOT / "data/validation/g8r_noise_v3_c2b_token_pair_cache.npz")
    p.add_argument("--tolerance", type=float, default=0.02)
    p.add_argument("--max-queries", type=int, default=0, help="Smoke only")
    return p.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def token_pair_summary(
    left_token: np.ndarray, right_token: np.ndarray,
    left_mz: np.ndarray, right_mz: np.ndarray,
    left_intensity: np.ndarray, right_intensity: np.ndarray,
    left_valid: np.ndarray, right_valid: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    lmz = np.asarray(left_mz[left_valid], dtype=np.float64)
    rmz = np.asarray(right_mz[right_valid], dtype=np.float64)
    lint = np.asarray(left_intensity[left_valid], dtype=np.float64)
    rint = np.asarray(right_intensity[right_valid], dtype=np.float64)
    ltok = np.asarray(left_token[left_valid], dtype=np.float32)
    rtok = np.asarray(right_token[right_valid], dtype=np.float32)
    matches = greedy_matches(lmz, rmz, tolerance)
    if not matches:
        return np.zeros(len(TOKEN_FEATURES), dtype=np.float32)
    li = np.fromiter((value[0] for value in matches), dtype=np.int64)
    ri = np.fromiter((value[1] for value in matches), dtype=np.int64)
    cosine = np.einsum("ij,ij->i", ltok[li], rtok[ri], optimize=True).astype(np.float64)
    cosine = np.clip(cosine, -1.0, 1.0)
    weights = np.sqrt(np.clip(lint[li], 0, None) * np.clip(rint[ri], 0, None))
    weight_sum = float(weights.sum())
    if weight_sum <= 1e-12:
        weights = np.full(len(cosine), 1.0 / len(cosine))
    else:
        weights /= weight_sum
    weighted = float(np.sum(weights * cosine))
    match_fraction = len(matches) / max(min(len(lmz), len(rmz)), 1)
    return np.asarray([
        float(cosine.mean()), weighted, float(cosine.min()),
        float(np.quantile(cosine, 0.25)), float(np.quantile(cosine, 0.75)),
        float(np.mean(cosine < 0.5)), float(np.mean(cosine > 0.8)),
        float(match_fraction), float(np.sum(weights * (1.0 - cosine))),
    ], dtype=np.float32)


def metric_for_feature(graph: Cache, values: np.ndarray, query_limit: int, feature: str) -> dict:
    base_column = graph.feature_names.index("dreams_similarity")
    base, final, near = [], [], []
    for query in range(query_limit):
        ml, mr = map(int, graph.query_ptr[query:query + 2])
        pl, pr = int(graph.molecule_ptr[ml]), int(graph.molecule_ptr[mr])
        ptr = graph.molecule_ptr[ml:mr + 1] - pl
        base_result = evaluate_query_scores(graph.features[pl:pr, base_column], ptr, 0)
        result = evaluate_query_scores(values[pl:pr], ptr, 0)
        base.append(bool(base_result["top1"])); final.append(bool(result["top1"]))
        near.append(bool(graph.query_has_near[query]))
    base = np.asarray(base, bool); final = np.asarray(final, bool); near = np.asarray(near, bool)
    return {
        "feature": feature, "queries": int(query_limit),
        "baseline_recall1": float(base.mean()), "recall1": float(final.mean()),
        "delta_recall1": float(final.mean() - base.mean()),
        "corrected": int(np.sum((~base) & final)),
        "introduced": int(np.sum(base & (~final))),
        "near_queries": int(near.sum()),
        "baseline_near_recall1": float(base[near].mean()) if near.any() else None,
        "near_recall1": float(final[near].mean()) if near.any() else None,
        "near_delta_recall1": float(final[near].mean() - base[near].mean()) if near.any() else None,
    }


def main() -> None:
    args = parse_args()
    if args.tolerance <= 0 or args.max_queries < 0:
        raise ValueError("invalid C2-B0 parameters")
    for path in (args.graph, args.token_dir / "report.json", args.token_dir / "rows.npy"):
        if not path.is_file(): raise FileNotFoundError(path)
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(f"refusing to overwrite C2-B0 output {args.output}")
    report_token = json.loads((args.token_dir / "report.json").read_text(encoding="utf-8"))
    if report_token.get("row_scope") != "reachable" or int(report_token.get("spectra", 0)) != 25275:
        raise RuntimeError("C2-B0 requires the formal 25,275-row reachable token cache")

    graph = Cache(args.graph)
    query_limit = min(graph.n_queries, args.max_queries or graph.n_queries)
    last_molecule = int(graph.query_ptr[query_limit])
    last_pair = int(graph.molecule_ptr[last_molecule])
    rows = np.load(args.token_dir / "rows.npy", mmap_mode="r")
    tokens = np.load(args.token_dir / "tokens_f16.npy", mmap_mode="r")
    mz = np.load(args.token_dir / "mz_f32.npy", mmap_mode="r")
    intensity = np.load(args.token_dir / "intensity_f32.npy", mmap_mode="r")
    valid = np.load(args.token_dir / "valid.npy", mmap_mode="r")

    def positions(values: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(rows, values)
        if np.any(pos >= len(rows)):
            raise RuntimeError("reachable token cache misses a graph spectrum row")
        if np.any(rows[pos] != values):
            raise RuntimeError("reachable token cache row lookup is not exact")
        return pos.astype(np.int64)

    candidate_position = positions(graph.pair_candidate_row[:last_pair])
    token_features = np.empty((last_pair, len(TOKEN_FEATURES)), dtype=np.float32)
    for query in range(query_limit):
        ml, mr = map(int, graph.query_ptr[query:query + 2])
        pl, pr = int(graph.molecule_ptr[ml]), int(graph.molecule_ptr[mr])
        qpos = int(positions(np.asarray([graph.query_row[query]], np.int64))[0])
        for pair in range(pl, pr):
            cpos = int(candidate_position[pair])
            token_features[pair] = token_pair_summary(
                tokens[qpos], tokens[cpos], mz[qpos], mz[cpos],
                intensity[qpos], intensity[cpos], valid[qpos], valid[cpos], args.tolerance,
            )
        if (query + 1) % 500 == 0 or query + 1 == query_limit:
            print(f"[C2-B0] {query + 1:,}/{query_limit:,} queries; {pr:,} pairs", flush=True)

    diagnostics = []
    higher_is_better = {
        "token_cosine_mean", "token_cosine_weighted", "token_cosine_min",
        "token_cosine_p25", "token_cosine_p75", "token_high_similarity_fraction",
        "token_match_fraction_min",
    }
    for index, name in enumerate(TOKEN_FEATURES):
        score = token_features[:, index]
        if name not in higher_is_better:
            score = -score
        diagnostics.append(metric_for_feature(graph, score, query_limit, name))

    with np.load(args.graph, allow_pickle=True) as source:
        payload = {name: source[name] for name in source.files}
    if query_limit != graph.n_queries:
        # Smoke output is deliberately not a reusable graph cache.
        payload = {}
    formal = query_limit == graph.n_queries
    augmented = np.concatenate((graph.features[:last_pair], token_features), axis=1)
    if formal:
        payload["features"] = augmented
        payload["feature_names"] = np.asarray([*graph.feature_names, *TOKEN_FEATURES], dtype=object)
        staging = Path(tempfile.mkdtemp(prefix="c2b_pair_", dir=args.output.parent))
        try:
            temporary = staging / args.output.name
            np.savez_compressed(temporary, **payload)
            temporary.replace(args.output)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    best = max(diagnostics, key=lambda value: (value["delta_recall1"], value["corrected"] - value["introduced"]))
    feature_std = np.std(token_features.astype(np.float64), axis=0)
    decision = {
        "status": "noise_v3_c2b_token_pair_cache_complete", "formal": formal,
        "queries": int(query_limit), "pairs": int(last_pair),
        "token_features": TOKEN_FEATURES, "single_feature_diagnostics": diagnostics,
        "token_feature_standard_deviation": {
            name: float(value) for name, value in zip(TOKEN_FEATURES, feature_std)
        },
        "best_single_token_feature": best,
        "gates": {
            "full_graph_covered": bool(formal and query_limit == 23876),
            "candidate_tokens_reachable": bool(len(rows) == 25275),
            "diagnostic_any_single_feature_has_positive_net": bool(any(
                value["corrected"] > value["introduced"] for value in diagnostics
            )),
            "diagnostic_any_single_feature_near_nonnegative": bool(any(
                value["near_delta_recall1"] is not None and value["near_delta_recall1"] >= 0
                for value in diagnostics
            )),
            "token_evidence_is_not_constant": bool(np.sum(feature_std > 1e-5) >= 5),
        },
        "claim_limit": "Single-feature results are headroom diagnostics, not a trained or deployable model.",
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "token_report_sha256": sha256_file(args.token_dir / "report.json"),
            "token_rows_sha256": sha256_file(args.token_dir / "rows.npy"),
            "output_sha256": sha256_file(args.output) if formal else None,
        },
    }
    decision["gates"]["pass"] = bool(
        decision["gates"]["full_graph_covered"]
        and decision["gates"]["candidate_tokens_reachable"]
        and decision["gates"]["token_evidence_is_not_constant"]
    )
    args.output.with_suffix(".json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == "__main__":
    main()
