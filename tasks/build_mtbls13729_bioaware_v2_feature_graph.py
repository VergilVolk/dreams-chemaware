#!/usr/bin/env python
"""Build the phenotype-blind experimental layer for BioAware v2.

This stage does not score chemical candidates and does not use phenotype
labels.  It maps every cached MS2 embedding to an aligned MS1 feature, forms an
equal-sample-weight feature embedding, and freezes a k-nearest-neighbour graph
with independent abundance/co-detection diagnostics.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


EPS = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, EPS)


def normalize_rt_seconds(rt: pd.Series) -> np.ndarray:
    values = pd.to_numeric(rt, errors="coerce").to_numpy(float)
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise RuntimeError("manifest has no finite retention times")
    # Existing DreaMS manifests store minutes. Keep the conversion explicit and
    # fail-safe for future manifests already exported in seconds.
    return values * 60.0 if np.quantile(finite, 0.99) < 100.0 else values


def map_ms2_to_features(
    manifest: pd.DataFrame,
    targets: pd.DataFrame,
    ppm: float,
    rt_sec: float,
) -> pd.DataFrame:
    required_manifest = {"file_name", "precursor_mz", "RT"}
    required_targets = {"feature_id", "mz", "rt_sec"}
    if not required_manifest <= set(manifest):
        raise RuntimeError(f"manifest missing {sorted(required_manifest-set(manifest))}")
    if not required_targets <= set(targets):
        raise RuntimeError(f"targets missing {sorted(required_targets-set(targets))}")

    target = targets.copy()
    target["mz"] = pd.to_numeric(target["mz"], errors="coerce")
    target["rt_sec"] = pd.to_numeric(target["rt_sec"], errors="coerce")
    valid_target = (
        np.isfinite(target["mz"].to_numpy(float))
        & np.isfinite(target["rt_sec"].to_numpy(float))
        & (target["mz"].to_numpy(float) > 0)
    )
    target = target.loc[valid_target].sort_values(
        ["mz", "rt_sec", "feature_id"]
    ).reset_index(drop=True)
    if target.empty:
        raise RuntimeError("targets contain no finite positive m/z and RT rows")
    target_mz = pd.to_numeric(target["mz"], errors="coerce").to_numpy(float)
    if not np.all(np.diff(target_mz[np.isfinite(target_mz)]) >= 0):
        raise RuntimeError("target m/z values are not sorted")

    precursor = pd.to_numeric(manifest["precursor_mz"], errors="coerce").to_numpy(float)
    observed_rt = normalize_rt_seconds(manifest["RT"])
    rows: list[dict] = []
    for index, (mz, rt) in enumerate(zip(precursor, observed_rt, strict=True)):
        if not np.isfinite(mz) or not np.isfinite(rt) or mz <= 0:
            continue
        tolerance = mz * ppm * 1e-6
        lo = bisect.bisect_left(target_mz, mz - tolerance)
        hi = bisect.bisect_right(target_mz, mz + tolerance)
        if lo == hi:
            continue
        candidates = target.iloc[lo:hi]
        dmz = np.abs(candidates["mz"].to_numpy(float) - mz)
        dppm = dmz / candidates["mz"].to_numpy(float) * 1e6
        drt = np.abs(candidates["rt_sec"].to_numpy(float) - rt)
        eligible = np.flatnonzero(drt <= rt_sec)
        if not eligible.size:
            continue
        cost = (dppm[eligible] / ppm) ** 2 + (drt[eligible] / rt_sec) ** 2
        local = int(eligible[int(np.argmin(cost))])
        chosen = candidates.iloc[local]
        rows.append(
            {
                "embedding_row": index,
                "sample_name": str(manifest.iloc[index]["file_name"]),
                "feature_id": int(chosen.feature_id),
                "feature_mz": float(chosen.mz),
                "feature_rt_sec": float(chosen.rt_sec),
                "ms2_precursor_mz": float(mz),
                "ms2_rt_sec": float(rt),
                "dppm": float(dppm[local]),
                "drt_sec": float(drt[local]),
                "link_cost": float(cost[int(np.argmin(cost))]),
            }
        )
    return pd.DataFrame(rows)


def aggregate_feature_embeddings(
    embeddings: np.ndarray,
    mapping: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Average spectra within sample, then samples within feature.

    This prevents features with many repeated MS2 acquisitions in one sample
    from dominating the feature representation.
    """
    sample_vectors: list[np.ndarray] = []
    sample_rows: list[dict] = []
    grouped = mapping.groupby(["feature_id", "sample_name"], sort=True)
    for (feature_id, sample_name), group in grouped:
        indices = group["embedding_row"].to_numpy(np.int64)
        vectors = np.asarray(embeddings[indices], dtype=np.float32)
        vector = l2_normalize(vectors).mean(axis=0)
        vector = l2_normalize(vector[None, :])[0]
        sample_vectors.append(vector.astype(np.float32))
        sample_rows.append(
            {
                "feature_id": int(feature_id),
                "sample_name": str(sample_name),
                "n_ms2": int(len(indices)),
            }
        )
    if not sample_vectors:
        raise RuntimeError("no MS2 spectra mapped to MS1 features")
    sample_matrix = np.stack(sample_vectors)
    sample_table = pd.DataFrame(sample_rows)

    feature_vectors: list[np.ndarray] = []
    feature_rows: list[dict] = []
    for feature_id, group in sample_table.groupby("feature_id", sort=True):
        positions = group.index.to_numpy(np.int64)
        vector = l2_normalize(sample_matrix[positions]).mean(axis=0)
        vector = l2_normalize(vector[None, :])[0]
        feature_vectors.append(vector.astype(np.float32))
        feature_rows.append(
            {
                "feature_id": int(feature_id),
                "n_ms2": int(group["n_ms2"].sum()),
                "n_ms2_samples": int(len(group)),
            }
        )
    return np.stack(feature_vectors), pd.DataFrame(feature_rows)


def normalized_log_intensities(matrix: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    sample_columns = [column for column in matrix.columns if column != "feature_id"]
    values = matrix[sample_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    values[~np.isfinite(values) | (values <= 0)] = np.nan
    medians = np.nanmedian(values, axis=0)
    valid = np.isfinite(medians) & (medians > 0)
    if not np.all(valid):
        bad = [sample_columns[i] for i in np.flatnonzero(~valid)]
        raise RuntimeError(f"samples have no positive MS1 intensities: {bad}")
    values = np.log1p(values / medians[None, :])
    return values, sample_columns


def paired_spearman(a: np.ndarray, b: np.ndarray, minimum_joint: int) -> tuple[float, int, float]:
    da = np.isfinite(a)
    db = np.isfinite(b)
    joint = da & db
    union = da | db
    n_joint = int(joint.sum())
    jaccard = float(n_joint / union.sum()) if union.any() else np.nan
    if n_joint < minimum_joint:
        return np.nan, n_joint, jaccard
    ra = rankdata(a[joint], method="average")
    rb = rankdata(b[joint], method="average")
    if np.std(ra) <= EPS or np.std(rb) <= EPS:
        return np.nan, n_joint, jaccard
    rho = float(np.corrcoef(ra, rb)[0, 1])
    return rho, n_joint, jaccard


def knn_edges(
    feature_ids: np.ndarray,
    embeddings: np.ndarray,
    intensities: np.ndarray,
    top_k: int,
    minimum_joint: int,
    block_size: int,
    device: str,
) -> pd.DataFrame:
    n = len(feature_ids)
    if n < 2:
        raise RuntimeError("at least two embedded features are required")
    k = min(top_k, n - 1)
    seen: set[tuple[int, int]] = set()
    rows: list[dict] = []
    embeddings = l2_normalize(np.asarray(embeddings, dtype=np.float32))
    torch_embeddings = None
    if device != "cpu":
        import torch

        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"requested {device}, but CUDA is unavailable")
        torch_embeddings = torch.from_numpy(embeddings).to(device)
    for start in range(0, n, block_size):
        stop = min(n, start + block_size)
        if torch_embeddings is None:
            similarities = embeddings[start:stop] @ embeddings.T
        else:
            similarities = (
                torch_embeddings[start:stop] @ torch_embeddings.T
            ).detach().cpu().numpy()
        for local, index in enumerate(range(start, stop)):
            similarities[local, index] = -np.inf
        neighbours = np.argpartition(similarities, -k, axis=1)[:, -k:]
        for local, index in enumerate(range(start, stop)):
            ordered = neighbours[local][np.argsort(similarities[local, neighbours[local]])[::-1]]
            for other in ordered:
                left, right = sorted((int(index), int(other)))
                if (left, right) in seen:
                    continue
                seen.add((left, right))
                rho, joint, jaccard = paired_spearman(
                    intensities[left], intensities[right], minimum_joint
                )
                rows.append(
                    {
                        "feature_id_a": int(feature_ids[left]),
                        "feature_id_b": int(feature_ids[right]),
                        "dreams_cosine": float(embeddings[left] @ embeddings[right]),
                        "co_detection_jaccard": jaccard,
                        "abundance_spearman": rho,
                        "n_joint_detected": joint,
                    }
                )
        print(f"[feature-graph] {stop:,}/{n:,} nodes", flush=True)
    edges = pd.DataFrame(rows)
    # These are preregistered diagnostics, not labels and not a ranking score.
    edges["metdna_spectral_edge"] = edges["dreams_cosine"] >= 0.50
    edges["abundance_supported"] = (
        (edges["n_joint_detected"] >= minimum_joint)
        & (edges["abundance_spearman"] >= 0.60)
    )
    edges["dual_data_support"] = edges["metdna_spectral_edge"] & (
        edges["abundance_supported"] | (edges["co_detection_jaccard"] >= 0.50)
    )
    return edges.sort_values(
        ["feature_id_a", "dreams_cosine", "feature_id_b"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def build_panel(args: argparse.Namespace, panel: str, out: Path) -> dict:
    manifest_path = args.embedding_root / panel / "manifest.csv"
    embedding_path = args.embedding_root / panel / "embeddings.npy"
    targets_path = args.consensus_dir / f"{panel}__requantification_targets.csv.gz"
    intensity_path = args.consensus_dir / f"{panel}__discovery_intensity_matrix.csv.gz"
    required = [manifest_path, embedding_path, targets_path, intensity_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{panel} missing inputs: {missing}")

    manifest = pd.read_csv(manifest_path)
    embeddings = np.load(embedding_path, mmap_mode="r")
    if len(manifest) != embeddings.shape[0]:
        raise RuntimeError(
            f"{panel} manifest/embedding mismatch: {len(manifest)} vs {embeddings.shape[0]}"
        )
    if args.max_embedding_rows:
        limit = min(args.max_embedding_rows, len(manifest))
        manifest = manifest.iloc[:limit].copy()
        embeddings = embeddings[:limit]
    targets = pd.read_csv(targets_path)
    intensity = pd.read_csv(intensity_path)
    if intensity["feature_id"].duplicated().any():
        raise RuntimeError(f"{panel} intensity matrix has duplicate feature IDs")

    mapping = map_ms2_to_features(manifest, targets, args.ppm, args.rt_sec)
    if len(mapping) and (
        mapping["embedding_row"].min() < 0
        or mapping["embedding_row"].max() >= len(embeddings)
    ):
        raise RuntimeError(f"{panel} mapped embedding row lies outside embedding cache")
    feature_embeddings, feature_counts = aggregate_feature_embeddings(embeddings, mapping)
    feature_ids = feature_counts["feature_id"].to_numpy(np.int64)
    target_meta = targets.drop_duplicates("feature_id").set_index("feature_id")
    nodes = feature_counts.join(target_meta, on="feature_id", validate="one_to_one")

    log_intensity, sample_columns = normalized_log_intensities(intensity)
    intensity_index = {int(feature): i for i, feature in enumerate(intensity["feature_id"])}
    missing_intensity = [int(feature) for feature in feature_ids if int(feature) not in intensity_index]
    if missing_intensity:
        raise RuntimeError(f"{panel} embedded features missing intensity rows: {missing_intensity[:10]}")
    aligned_intensity = log_intensity[[intensity_index[int(feature)] for feature in feature_ids]]
    edges = knn_edges(
        feature_ids,
        feature_embeddings,
        aligned_intensity,
        args.top_k,
        args.minimum_joint_samples,
        args.block_size,
        args.device,
    )

    prefix = out / panel
    mapping_path = prefix.with_name(prefix.name + "__ms2_feature_mapping.csv.gz")
    nodes_path = prefix.with_name(prefix.name + "__nodes.csv.gz")
    edges_path = prefix.with_name(prefix.name + "__edges.csv.gz")
    embeddings_path = prefix.with_name(prefix.name + "__feature_embeddings.npz")
    mapping.to_csv(mapping_path, index=False)
    nodes.to_csv(nodes_path, index=False)
    edges.to_csv(edges_path, index=False)
    np.savez_compressed(
        embeddings_path,
        feature_id=feature_ids,
        embedding=feature_embeddings.astype(np.float32),
    )
    report = {
        "panel": panel,
        "manifest_rows": int(len(manifest)),
        "mapped_ms2": int(len(mapping)),
        "mapped_fraction": float(len(mapping) / len(manifest)) if len(manifest) else 0.0,
        "feature_nodes": int(len(nodes)),
        "samples": int(len(sample_columns)),
        "edges": int(len(edges)),
        "metdna_spectral_edges": int(edges["metdna_spectral_edge"].sum()),
        "abundance_supported_edges": int(edges["abundance_supported"].sum()),
        "dual_data_supported_edges": int(edges["dual_data_support"].sum()),
        "median_ms2_samples_per_feature": float(nodes["n_ms2_samples"].median()),
        "provenance": {
            "manifest_sha256": sha256(manifest_path),
            "embedding_report_sha256": sha256(args.embedding_root / panel / "report.json"),
            "targets_sha256": sha256(targets_path),
            "intensity_sha256": sha256(intensity_path),
            "mapping_sha256": sha256(mapping_path),
            "nodes_sha256": sha256(nodes_path),
            "edges_sha256": sha256(edges_path),
            "feature_embeddings_sha256": sha256(embeddings_path),
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-root", type=Path, default=Path("data/mtbls13729/embeddings"))
    parser.add_argument("--consensus-dir", type=Path, default=Path("data/mtbls13729/ms1_consensus"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/mtbls13729/bioaware_v2_feature_graph"))
    parser.add_argument("--panels", nargs="+", default=["neg_rp", "pos_rp"])
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--rt-sec", type=float, default=20.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--minimum-joint-samples", type=int, default=10)
    parser.add_argument("--block-size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-embedding-rows", type=int, default=0)
    args = parser.parse_args()
    if args.ppm <= 0 or args.rt_sec <= 0 or args.top_k <= 0:
        raise ValueError("ppm, rt-sec and top-k must be positive")

    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    reports = [build_panel(args, panel, out) for panel in args.panels]
    report = {
        "status": "mtbls13729_bioaware_v2_feature_graph_complete",
        "formal": args.max_embedding_rows == 0,
        "phenotype_labels_used": False,
        "candidate_identities_used": False,
        "ranking_outcomes_used": False,
        "panels": reports,
        "parameters": {
            "ppm": args.ppm,
            "rt_sec": args.rt_sec,
            "top_k": args.top_k,
            "minimum_joint_samples": args.minimum_joint_samples,
            "block_size": args.block_size,
            "device": args.device,
            "max_embedding_rows": args.max_embedding_rows,
        },
        "contract": (
            "Experimental data layer only. Edge diagnostics are frozen before "
            "chemical candidate scoring and cannot establish identity."
        ),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
