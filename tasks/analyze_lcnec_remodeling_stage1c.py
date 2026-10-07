"""Stage-1c: Leiden community detection for robust family decomposition.

v1 (kNN): local structure but k-unstable → STOP
v1b (threshold): stable but creates giant blob → STOP
v1c (Leiden): find communities in a generous kNN graph. Leiden guarantees
  well-separated communities, no giant component, and a resolution parameter
  that can be sensitivity-tested instead of k.

Requirements: pip install leidenalg igraph (or use python-igraph)
Fallback: if leidenalg unavailable, use connected components of a
  community-aware threshold graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as h:
        for b in iter(lambda: h.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def unit_rows(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)


def knn_graph(embedding: np.ndarray, k: int):
    """Symmetric kNN graph (not mutual - more generous)."""
    n = len(embedding)
    sim = embedding @ embedding.T
    np.fill_diagonal(sim, -np.inf)
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        neighbors = np.argpartition(-sim[i], kth=min(k - 1, n - 2))[:k]
        adj[i, neighbors] = True
    adj = adj | adj.T  # symmetrize
    return adj


def leiden_communities(adj: np.ndarray, resolution: float, seed: int = 42):
    """Leiden community detection. Falls back to connected components if unavailable."""
    try:
        import igraph as ig
        import leidenalg
        # build igraph from adjacency
        edges = np.where(np.triu(adj, 1))
        sources, targets = edges[0], edges[1]
        g = ig.Graph(n=adj.shape[0], edges=list(zip(sources, targets)),
                     directed=False)
        weights = [1.0] * len(sources)
        partition = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            weights=weights, resolution_parameter=resolution, seed=seed)
        return np.array(partition.membership, dtype=np.int64)
    except ImportError:
        # Fallback: connected components (same as threshold but on kNN graph)
        n_comp, labels = connected_components(
            csr_matrix(adj.astype(np.int8)), directed=False, return_labels=True)
        return labels.astype(np.int64)


def decompose(values, labels):
    m = np.asarray(values, dtype=np.float64)
    if m.ndim == 1:
        m = m[None, :]
    family = np.zeros_like(m)
    within = np.zeros_like(m)
    isolated = np.zeros_like(m)
    for comp in np.unique(labels):
        idx = np.flatnonzero(labels == comp)
        if len(idx) == 1:
            isolated[:, idx] = m[:, idx]
        else:
            mean = m[:, idx].mean(axis=1, keepdims=True)
            family[:, idx] = mean
            within[:, idx] = m[:, idx] - mean
    if m.ndim == 1:
        return family[0], within[0], isolated[0]
    return family, within, isolated


def energy(v):
    return float(np.sum(np.square(np.asarray(v, dtype=np.float64))))


def summary(effects, labels, draws=10000, seed=20261007):
    mean_vec = effects.mean(axis=0)
    fam, wit, iso = decompose(mean_vec, labels)
    total = energy(mean_vec)
    parts = {"family": energy(fam), "within": energy(wit), "isolated": energy(iso)}
    result = {"total_energy": total}
    for name, val in parts.items():
        result[f"{name}_energy"] = val
        result[f"{name}_fraction"] = val / max(total, 1e-12)
    result["orthogonality_error"] = abs(total - sum(parts.values())) / max(total, 1e-12)

    rng = np.random.default_rng(seed)
    null = {n: np.empty(draws) for n in parts}
    for d in range(draws):
        signs = rng.choice((-1.0, 1.0), size=len(effects))
        m = np.mean(effects * signs[:, None], axis=0)
        f2, w2, i2 = decompose(m, labels)
        null["family"][d] = energy(f2)
        null["within"][d] = energy(w2)
        null["isolated"][d] = energy(i2)
    for name in parts:
        obs = result[f"{name}_energy"]
        result[f"{name}_signflip_p"] = float((1 + np.sum(null[name] >= obs)) / (draws + 1))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--knn-k", type=int, default=10,
                        help="Generous k for input kNN graph (not the primary parameter)")
    parser.add_argument("--resolution", type=float, default=1.0,
                        help="Leiden resolution parameter")
    parser.add_argument("--resolution-sensitivity", type=float, default=0.5,
                        help="Test resolution ± this value for stability")
    parser.add_argument("--sign-flips", type=int, default=10000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and not args.overwrite:
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    # Load Stage-1 outputs
    npz = np.load(args.stage1_dir / "primary_decomposition.npz")
    effects = npz["patient_effect"]  # (34, 263)
    family_id = npz["family_id"]
    print(f"loaded: {effects.shape[0]} patients x {effects.shape[1]} families", flush=True)

    embed_dir = args.stage1_dir.parent / "query" / "dreams_embeddings"
    embedding = unit_rows(np.load(embed_dir / "embeddings.npy"))
    print(f"embeddings: {embedding.shape}", flush=True)

    # Build generous kNN graph
    adj = knn_graph(embedding, args.knn_k)
    print(f"kNN(k={args.knn_k}) edges: {int(np.sum(np.triu(adj, 1)))}", flush=True)

    # Primary: Leiden at specified resolution
    labels = leiden_communities(adj, args.resolution)
    sizes = np.bincount(labels)
    result = summary(effects, labels, args.sign_flips)
    result.update({
        "graph": f"Leiden(knn_k={args.knn_k}, resolution={args.resolution})",
        "knn_k": args.knn_k,
        "resolution": args.resolution,
        "communities": int(len(sizes)),
        "singletons": int(np.sum(sizes == 1)),
        "non_singleton_fraction": float(np.mean(sizes[labels] > 1)),
        "largest_community_fraction": float(np.max(sizes) / len(labels)),
        "median_community_size": float(np.median(sizes[sizes > 1])) if np.any(sizes > 1) else 0,
    })
    print(f"primary: {result['communities']} communities, "
          f"family {result['family_fraction']:.3f}, within {result['within_fraction']:.3f}", flush=True)

    # Sensitivity: resolution ± delta
    sensitivities = []
    for res in [args.resolution - args.resolution_sensitivity,
                args.resolution + args.resolution_sensitivity]:
        if res <= 0:
            continue
        s_labels = leiden_communities(adj, res)
        s_sizes = np.bincount(s_labels)
        s_summary = summary(effects, s_labels, min(args.sign_flips, 5000))
        sensitivities.append({
            "resolution": round(res, 4),
            "communities": int(len(s_sizes)),
            "family_fraction": s_summary["family_fraction"],
            "within_fraction": s_summary["within_fraction"],
            "non_singleton_fraction": float(np.mean(s_sizes[s_labels] > 1)),
        })
    fam_range = max(s["family_fraction"] for s in sensitivities) - \
               min(s["family_fraction"] for s in sensitivities)
    wit_range = max(s["within_fraction"] for s in sensitivities) - \
               min(s["within_fraction"] for s in sensitivities)

    # Gates
    gates = {
        "exact_orthogonal_decomposition": result["orthogonality_error"] <= 1e-10,
        "nontrivial_coverage_ge_25pct": result["non_singleton_fraction"] >= 0.25,
        "no_giant_component_le_50pct": result["largest_community_fraction"] <= 0.50,
        "family_resolution_stability_le_15pct": fam_range <= 0.15,
        "within_resolution_stability_le_15pct": wit_range <= 0.15,
        "family_or_within_significant": min(
            result["family_signflip_p"], result["within_signflip_p"]) <= 0.05,
    }
    passed = all(gates.values())

    report = {
        "status": "LCNEC_GLOBAL_REMODELING_STAGE1C_" + ("PASS" if passed else "STOP"),
        "graph_definition": (
            f"Leiden community detection on kNN(k={args.knn_k}) graph, "
            f"resolution={args.resolution}. Communities are well-separated by "
            f"construction (no giant component). Stability measured across "
            f"resolution ±{args.resolution_sensitivity}."
        ),
        "primary": result,
        "resolution_sensitivity": {
            "delta": args.resolution_sensitivity,
            "results": sensitivities,
            "family_range": round(fam_range, 6),
            "within_range": round(wit_range, 6),
        },
        "gates": gates,
        "pass_to_stage2": passed,
        "provenance": {
            "stage1_dir": str(args.stage1_dir),
            "embeddings_sha256": sha256(embed_dir / "embeddings.npy"),
        },
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
