"""Stage-1 falsification of LCNEC patient-level chemical-remodelling decomposition.

This analysis deliberately stops before transformation claims.  It tests whether
the 34 paired patient-effect vectors admit a non-trivial and graph-definition-
stable decomposition into family-wide, within-family and singleton components.
The graph is phenotype blind and built only from frozen DreaMS embeddings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)


def mutual_knn_labels(embedding: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(embedding)
    if not 1 <= k < n:
        raise ValueError(f"k must be in [1,{n - 1}], got {k}")
    similarity = embedding @ embedding.T
    np.fill_diagonal(similarity, -np.inf)
    neighbors = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    directed = np.zeros((n, n), dtype=bool)
    directed[np.arange(n)[:, None], neighbors] = True
    adjacency = directed & directed.T
    adjacency |= adjacency.T
    n_components, labels = connected_components(
        csr_matrix(adjacency.astype(np.int8)), directed=False, return_labels=True
    )
    if n_components < 1:
        raise RuntimeError("mutual-kNN graph has no components")
    return labels.astype(np.int64), adjacency


def decompose(values: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact orthogonal decomposition for rows x features."""
    matrix = np.asarray(values, dtype=np.float64)
    one_dimensional = matrix.ndim == 1
    if one_dimensional:
        matrix = matrix[None, :]
    if matrix.ndim != 2 or matrix.shape[1] != len(labels):
        raise ValueError("effect matrix and graph labels are not aligned")
    family = np.zeros_like(matrix)
    within = np.zeros_like(matrix)
    isolated = np.zeros_like(matrix)
    for component in np.unique(labels):
        index = np.flatnonzero(labels == component)
        if len(index) == 1:
            isolated[:, index] = matrix[:, index]
            continue
        mean = matrix[:, index].mean(axis=1, keepdims=True)
        family[:, index] = mean
        within[:, index] = matrix[:, index] - mean
    if one_dimensional:
        return family[0], within[0], isolated[0]
    return family, within, isolated


def energy(values: np.ndarray) -> float:
    return float(np.sum(np.square(np.asarray(values, dtype=np.float64))))


def component_summary(vector: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    family, within, isolated = decompose(vector, labels)
    total = energy(vector)
    parts = {"family": energy(family), "within": energy(within), "isolated": energy(isolated)}
    identity_error = abs(total - sum(parts.values())) / max(total, 1e-12)
    return {
        "total_energy": total,
        **{f"{name}_energy": value for name, value in parts.items()},
        **{f"{name}_fraction": value / max(total, 1e-12) for name, value in parts.items()},
        "orthogonality_relative_error": identity_error,
    }


def sign_flip_null(
    patient_effects: np.ndarray, labels: np.ndarray, draws: int, seed: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    observed = component_summary(patient_effects.mean(axis=0), labels)
    rng = np.random.default_rng(seed)
    null = {name: np.empty(draws, dtype=np.float64) for name in ("family", "within", "isolated")}
    for draw in range(draws):
        signs = rng.choice((-1.0, 1.0), size=len(patient_effects))
        mean = np.mean(patient_effects * signs[:, None], axis=0)
        parts = decompose(mean, labels)
        for name, part in zip(null, parts, strict=True):
            null[name][draw] = energy(part)
    for name, values in null.items():
        obs = observed[f"{name}_energy"]
        observed[f"{name}_signflip_p"] = float((1 + np.sum(values >= obs)) / (draws + 1))
        observed[f"{name}_null_median"] = float(np.median(values))
        observed[f"{name}_null_q95"] = float(np.quantile(values, 0.95))
    return observed, null


def parse_family_id(name: object) -> int:
    match = re.search(r"_family_(\d+)$", str(name))
    if not match:
        raise RuntimeError(f"cannot recover family id from embedding manifest name: {name!r}")
    return int(match.group(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k-values", default="2,3,5,8")
    parser.add_argument("--primary-k", type=int, default=3)
    parser.add_argument("--sign-flips", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    k_values = tuple(int(value) for value in args.k_values.split(","))
    if len(set(k_values)) != len(k_values) or tuple(sorted(k_values)) != k_values:
        raise RuntimeError("k-values must be unique and sorted")
    if args.primary_k not in k_values:
        raise RuntimeError("primary-k must be included in k-values")
    if args.sign_flips < 1000:
        raise RuntimeError("at least 1,000 patient-level sign flips are required")
    embedding_path = args.embedding_dir / "embeddings.npy"
    manifest_path = args.embedding_dir / "manifest.csv"
    required = (args.matrix, embedding_path, manifest_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"LCNEC_REMODELING_PREFLIGHT_MISSING: {missing}")
    if args.output_dir.exists():
        if not args.overwrite:
            raise RuntimeError(f"refusing to overwrite {args.output_dir}")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    cache = np.load(args.matrix, allow_pickle=False)
    required_arrays = {
        "area", "family_id", "sample_id", "sample_code", "group_code", "file_class", "mz", "rt_sec"
    }
    if not required_arrays.issubset(cache.files):
        raise RuntimeError(f"matrix is missing arrays: {sorted(required_arrays - set(cache.files))}")
    area = np.asarray(cache["area"], dtype=np.float64)
    family_id = np.asarray(cache["family_id"], dtype=np.int64)
    sample_code = np.asarray(cache["sample_code"]).astype(str)
    group_code = np.asarray(cache["group_code"]).astype(str)
    file_class = np.asarray(cache["file_class"]).astype(str)
    mz = np.asarray(cache["mz"], dtype=np.float64)
    rt_sec = np.asarray(cache["rt_sec"], dtype=np.float64)
    if area.shape != (85, 263) or len(np.unique(family_id)) != 263:
        raise RuntimeError(f"expected 85 x 263 complete matrix, observed {area.shape}")

    manifest = pd.read_csv(manifest_path)
    embedding = unit_rows(np.load(embedding_path, allow_pickle=False))
    if len(manifest) != 263 or embedding.shape[0] != 263 or not np.isfinite(embedding).all():
        raise RuntimeError("expected exactly 263 embedding rows")
    embedded_id = np.asarray([parse_family_id(value) for value in manifest["name"]], dtype=np.int64)
    if len(np.unique(embedded_id)) != 263:
        raise RuntimeError("embedding manifest family ids are not unique")
    lookup = {int(value): index for index, value in enumerate(embedded_id)}
    if set(lookup) != set(map(int, family_id)):
        raise RuntimeError("embedding and EIC family-id universes differ")
    embedding = embedding[[lookup[int(value)] for value in family_id]]

    study = np.flatnonzero(file_class == "study")
    pairs: dict[str, dict[str, int]] = {}
    for index in study:
        pairs.setdefault(sample_code[index], {})[group_code[index]] = int(index)
    if len(pairs) != 34 or any(set(value) != {"TU", "NG"} for value in pairs.values()):
        raise RuntimeError("matrix does not contain exactly 34 complete TU/NG pairs")
    positive_or_inf = np.where(area[study] > 0, area[study], np.inf)
    pseudocount = np.min(positive_or_inf, axis=0) / 2.0
    pseudocount = np.where(np.isfinite(pseudocount) & (pseudocount > 0), pseudocount, 1.0)
    pair_names = sorted(pairs)
    tumour = np.stack([area[pairs[name]["TU"]] for name in pair_names])
    adjacent = np.stack([area[pairs[name]["NG"]] for name in pair_names])
    effects = np.log2(tumour + pseudocount) - np.log2(adjacent + pseudocount)
    if effects.shape != (34, 263) or not np.isfinite(effects).all():
        raise RuntimeError("non-finite or malformed patient effect matrix")

    graph_rows: list[dict[str, object]] = []
    patient_rows: list[dict[str, object]] = []
    primary_labels: np.ndarray | None = None
    primary_adjacency: np.ndarray | None = None
    primary_summary: dict[str, float] | None = None
    summaries: dict[int, dict[str, float]] = {}
    for k in k_values:
        labels, adjacency = mutual_knn_labels(embedding, k)
        sizes = np.bincount(labels)
        summary, _null = sign_flip_null(effects, labels, args.sign_flips, args.seed + k)
        summary.update({
            "k": k,
            "components": int(len(sizes)),
            "singletons": int(np.sum(sizes == 1)),
            "non_singleton_features": int(np.sum(sizes[sizes > 1])),
            "non_singleton_fraction": float(np.mean(sizes[labels] > 1)),
            "largest_component": int(np.max(sizes)),
            "largest_component_fraction": float(np.max(sizes) / len(labels)),
            "mutual_edges": int(np.sum(np.triu(adjacency, 1))),
        })
        summaries[k] = summary
        graph_rows.append(summary)
        components = decompose(effects, labels)
        for patient_index, patient in enumerate(pair_names):
            values = [energy(component[patient_index]) for component in components]
            total = sum(values)
            patient_rows.append({
                "k": k, "patient": patient,
                "total_energy": total,
                "family_fraction": values[0] / max(total, 1e-12),
                "within_fraction": values[1] / max(total, 1e-12),
                "isolated_fraction": values[2] / max(total, 1e-12),
            })
        if k == args.primary_k:
            primary_labels, primary_adjacency, primary_summary = labels, adjacency, summary

    assert primary_labels is not None and primary_adjacency is not None and primary_summary is not None
    pd.DataFrame(graph_rows).to_csv(args.output_dir / "graph_sensitivity.csv", index=False)
    pd.DataFrame(patient_rows).to_csv(args.output_dir / "patient_component_fractions.csv", index=False)
    np.savez_compressed(
        args.output_dir / "primary_decomposition.npz",
        family_id=family_id,
        mz=mz,
        rt_sec=rt_sec,
        patient=np.asarray(pair_names),
        patient_effect=effects,
        graph_component=primary_labels,
        graph_adjacency=primary_adjacency,
    )

    mean_effect = effects.mean(axis=0)
    sizes = np.bincount(primary_labels)
    family_rows: list[dict[str, object]] = []
    for component in np.flatnonzero(sizes > 1):
        index = np.flatnonzero(primary_labels == component)
        family_rows.append({
            "component": int(component), "size": int(len(index)),
            "mean_effect": float(np.mean(mean_effect[index])),
            "within_effect_sd": float(np.std(mean_effect[index], ddof=0)),
            "family_ids": ";".join(map(str, family_id[index])),
        })
    family_frame = pd.DataFrame(
        family_rows,
        columns=("component", "size", "mean_effect", "within_effect_sd", "family_ids"),
    )
    if len(family_frame):
        family_frame = family_frame.sort_values(
            "mean_effect", key=lambda values: np.abs(values), ascending=False
        )
    family_frame.to_csv(args.output_dir / "primary_family_effects.csv", index=False)

    edge_i, edge_j = np.where(np.triu(primary_adjacency, 1))
    edge_rows = pd.DataFrame({
        "family_id_u": family_id[edge_i], "family_id_v": family_id[edge_j],
        "dreams_cosine": np.sum(embedding[edge_i] * embedding[edge_j], axis=1),
        "mean_effect_u": mean_effect[edge_i], "mean_effect_v": mean_effect[edge_j],
        "effect_contrast_v_minus_u": mean_effect[edge_j] - mean_effect[edge_i],
        "delta_mz_abs": np.abs(mz[edge_j] - mz[edge_i]),
    })
    if len(edge_rows):
        edge_rows = edge_rows.reindex(
            edge_rows["effect_contrast_v_minus_u"].abs().sort_values(ascending=False).index
        )
    edge_rows.to_csv(args.output_dir / "primary_edge_contrasts_descriptive.csv", index=False)

    family_range = float(np.ptp([summaries[k]["family_fraction"] for k in k_values]))
    within_range = float(np.ptp([summaries[k]["within_fraction"] for k in k_values]))
    gates = {
        "exact_orthogonal_decomposition": bool(
            max(summaries[k]["orthogonality_relative_error"] for k in k_values) <= 1e-10
        ),
        "primary_nontrivial_coverage": bool(primary_summary["non_singleton_fraction"] >= 0.25),
        "primary_not_giant_component": bool(primary_summary["largest_component_fraction"] <= 0.50),
        "family_fraction_range_le_0_15": bool(family_range <= 0.15),
        "within_fraction_range_le_0_15": bool(within_range <= 0.15),
        "family_or_within_signal_p_le_0_05": bool(
            min(primary_summary["family_signflip_p"], primary_summary["within_signflip_p"]) <= 0.05
        ),
    }
    pass_to_stage2 = bool(all(gates.values()))
    report = {
        "status": "LCNEC_GLOBAL_REMODELING_STAGE1_PASS" if pass_to_stage2 else "LCNEC_GLOBAL_REMODELING_STAGE1_STOP",
        "formal": True,
        "analysis_universe": {"patients": 34, "families": 263, "primary_k": args.primary_k},
        "graph_definition": (
            "mutual k-nearest-neighbour graph on unit-normalized frozen official DreaMS embeddings; "
            "no phenotype labels used in graph construction"
        ),
        "primary": primary_summary,
        "sensitivity": {
            "k_values": list(k_values),
            "family_fraction_range": family_range,
            "within_fraction_range": within_range,
        },
        "gates": gates,
        "pass_to_stage2_lipn_and_known_increment": pass_to_stage2,
        "not_performed": {
            "cross_platform_replication": True,
            "known_only_vs_all_features": True,
            "transformation_identification": True,
            "flux_or_enzyme_activity": True,
        },
        "provenance": {
            "matrix_sha256": sha256(args.matrix),
            "embeddings_sha256": sha256(embedding_path),
            "manifest_sha256": sha256(manifest_path),
        },
        "claim_limit": (
            "Stage-1 decomposition falsification only. Passing licenses a frozen LIPn replication "
            "and known-only/all-feature increment test; it is not a cross-platform, dark-metabolome, "
            "transformation, mechanism, flux or publication result."
        ),
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
