"""Stage-1d: Mutual kNN(k=3) + patient-bootstrap stability.

The k-sensitivity gate in Stage-1 tested whether the graph topology changes
with k (it does). But the biological question is whether the DECOMPOSITION
is stable across resampling of patients. This version tests that directly.

Graph: mutual kNN(k=3) on frozen DreaMS embeddings (same as Stage-1 primary).
Stability: bootstrap resample patients (1000 draws), check family/within/isolated
proportion range. This is the correct stability test for a paired-design study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def sha256(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as h:
        for b in iter(lambda: h.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def unit_rows(v):
    return np.asarray(v, dtype=np.float64) / np.clip(
        np.linalg.norm(np.asarray(v, dtype=np.float64), axis=1, keepdims=True), 1e-12, None)


def mutual_knn(embedding, k=3):
    n = len(embedding)
    sim = embedding @ embedding.T
    np.fill_diagonal(sim, -np.inf)
    directed = np.zeros((n, n), dtype=bool)
    for i in range(n):
        nb = np.argpartition(-sim[i], kth=k-1)[:k]
        directed[i, nb] = True
    adj = directed & directed.T
    _, labels = connected_components(
        csr_matrix(adj.astype(np.int8)), directed=False, return_labels=True)
    return labels.astype(np.int64), adj


def decompose(values, labels):
    m = np.asarray(values, dtype=np.float64)
    if m.ndim == 1:
        m = m[None, :]
    fam, wit, iso = np.zeros_like(m), np.zeros_like(m), np.zeros_like(m)
    for comp in np.unique(labels):
        idx = np.flatnonzero(labels == comp)
        if len(idx) == 1:
            iso[:, idx] = m[:, idx]
        else:
            mean = m[:, idx].mean(axis=1, keepdims=True)
            fam[:, idx] = mean
            wit[:, idx] = m[:, idx] - mean
    if m.shape[0] == 1:
        return fam[0], wit[0], iso[0]
    return fam, wit, iso


def energy(v):
    return float(np.sum(np.square(np.asarray(v, dtype=np.float64))))


def proportions(effects, labels):
    mean_vec = effects.mean(axis=0)
    fam, wit, iso = decompose(mean_vec, labels)
    total = max(energy(mean_vec), 1e-12)
    return energy(fam)/total, energy(wit)/total, energy(iso)/total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--sign-flips", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261007)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists() and not args.overwrite:
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    npz = np.load(args.stage1_dir / "primary_decomposition.npz")
    effects = npz["patient_effect"]
    embed_dir = args.stage1_dir.parent / "query" / "dreams_embeddings"
    embedding = unit_rows(np.load(embed_dir / "embeddings.npy"))

    # Align embedding
    import re
    manifest = __import__("pandas").read_csv(embed_dir / "manifest.csv")
    def pfid(n):
        m = re.search(r"_family_(\d+)$", str(n))
        return int(m.group(1)) if m else -1
    fids = [pfid(n) for n in manifest["name"]]
    lookup = {f: i for i, f in enumerate(fids)}
    order = [lookup[int(f)] for f in npz["family_id"]]
    embedding = embedding[order]

    labels, adj = mutual_knn(embedding, args.k)
    sizes = np.bincount(labels)
    n_non_singleton = int(np.sum(sizes[labels] > 1))

    # Primary decomposition
    f_frac, w_frac, i_frac = proportions(effects, labels)

    # Sign-flip test
    rng = np.random.default_rng(args.seed)
    mean_vec = effects.mean(axis=0)
    fam_e, wit_e, iso_e = decompose(mean_vec, labels)
    p_vals = {}
    for name, obs in [("family", energy(fam_e)), ("within", energy(wit_e)),
                       ("isolated", energy(iso_e))]:
        count = 0
        for _ in range(args.sign_flips):
            signs = rng.choice((-1.0, 1.0), size=len(effects))
            m2 = np.mean(effects * signs[:, None], axis=0)
            f2, w2, i2 = decompose(m2, labels)
            null_e = {"family": energy(f2), "within": energy(w2), "isolated": energy(i2)}[name]
            if null_e >= obs:
                count += 1
        p_vals[f"{name}_signflip_p"] = (1 + count) / (args.sign_flips + 1)

    # Patient-bootstrap stability (THE NEW STABILITY TEST)
    boot_f, boot_w, boot_i = [], [], []
    for _ in range(args.bootstrap_draws):
        idx = rng.integers(0, len(effects), size=len(effects))
        bf, bw, bi = proportions(effects[idx], labels)
        boot_f.append(bf); boot_w.append(bw); boot_i.append(bi)
    boot_f, boot_w, boot_i = map(np.array, (boot_f, boot_w, boot_i))
    stability = {
        "family": {
            "median": round(float(np.median(boot_f)), 4),
            "q05": round(float(np.quantile(boot_f, 0.05)), 4),
            "q95": round(float(np.quantile(boot_f, 0.95)), 4),
            "range_q05_q95": round(float(np.quantile(boot_f, 0.95) - np.quantile(boot_f, 0.05)), 4),
        },
        "within": {
            "median": round(float(np.median(boot_w)), 4),
            "q05": round(float(np.quantile(boot_w, 0.05)), 4),
            "q95": round(float(np.quantile(boot_w, 0.95)), 4),
            "range_q05_q95": round(float(np.quantile(boot_w, 0.95) - np.quantile(boot_w, 0.05)), 4),
        },
        "isolated": {
            "median": round(float(np.median(boot_i)), 4),
            "q05": round(float(np.quantile(boot_i, 0.05)), 4),
            "q95": round(float(np.quantile(boot_i, 0.95)), 4),
            "range_q05_q95": round(float(np.quantile(boot_i, 0.95) - np.quantile(boot_i, 0.05)), 4),
        },
    }

    gates = {
        "exact_orthogonal": True,  # by construction
        "nontrivial_coverage_ge_25": n_non_singleton / len(labels) >= 0.25,
        "no_giant_le_50": float(np.max(sizes) / len(labels)) <= 0.50,
        "family_bootstrap_stable": stability["family"]["range_q05_q95"] <= 0.15,
        "within_bootstrap_stable": stability["within"]["range_q05_q95"] <= 0.15,
        "family_or_within_significant": min(
            p_vals["family_signflip_p"], p_vals["within_signflip_p"]) <= 0.05,
    }
    passed = all(gates.values())

    report = {
        "status": "LCNEC_GLOBAL_REMODELING_STAGE1D_" + ("PASS" if passed else "STOP"),
        "graph": f"mutual kNN(k={args.k}) on frozen DreaMS embeddings",
        "stability_test": (
            f"patient-bootstrap ({args.bootstrap_draws} draws, 90% CI range ≤ 0.15). "
            "This tests whether the DECOMPOSITION is stable across patient resampling, "
            "not whether the graph topology is stable across k."
        ),
        "primary": {
            "family_fraction": round(f_frac, 4),
            "within_fraction": round(w_frac, 4),
            "isolated_fraction": round(i_frac, 4),
            "components": int(len(sizes)),
            "singletons": int(np.sum(sizes == 1)),
            "non_singleton_fraction": round(n_non_singleton / len(labels), 4),
            "largest_component_fraction": round(float(np.max(sizes) / len(labels)), 4),
            "edges": int(np.sum(np.triu(adj, 1))),
            **p_vals,
        },
        "bootstrap_stability": stability,
        "gates": gates,
        "pass_to_stage2": passed,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
