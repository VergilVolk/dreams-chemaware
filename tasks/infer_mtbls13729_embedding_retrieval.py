"""Mass-constrained MTBLS13729 retrieval for one frozen embedding checkpoint.

This is the embedding-only arm.  It deliberately contains no P2b/raw-feature
fusion.  Query and library caches must declare the same checkpoint SHA256 and
must be row-aligned with the official protocol caches.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tasks"))

from g8r_p2_rank_fusion_core import grouped_max, unique_top_index  # noqa: E402
from infer_mtbls13729_p2b_vs_dreams import (  # noqa: E402
    nearest_target_links,
    resolve_query_embeddings,
    summarize_features,
)
from shared_dreams_inference import sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=["neg_rp", "pos_rp"], required=True)
    parser.add_argument("--method", default="e6_fixed_v2_sw2")
    parser.add_argument("--query-embedding-dir", type=Path, required=True)
    parser.add_argument("--official-query-dir", type=Path)
    parser.add_argument("--library-cache", type=Path, required=True)
    parser.add_argument("--official-library-cache", type=Path, required=True)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--rt-seconds", type=float, default=20.0)
    parser.add_argument("--max-queries", type=int, default=0)
    return parser.parse_args()


def model_metadata(directory: Path) -> dict:
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    metadata = report.get("model")
    if not isinstance(metadata, dict) or not metadata.get("checkpoint_sha256"):
        raise RuntimeError(f"cache does not declare a frozen model hash: {directory}")
    return metadata


def assert_query_alignment(experimental: pd.DataFrame, official: pd.DataFrame) -> None:
    required = ["file_name", "scan_number", "precursor_mz", "row_in_file"]
    for column in required:
        if column not in experimental or column not in official:
            raise RuntimeError(f"missing query alignment column: {column}")
    if len(experimental) != len(official):
        raise RuntimeError("official/experimental query counts differ")
    for column in ("file_name", "scan_number", "row_in_file"):
        if not np.array_equal(experimental[column].to_numpy(), official[column].to_numpy()):
            raise RuntimeError(f"official/experimental query order differs at {column}")
    if not np.allclose(
        pd.to_numeric(experimental.precursor_mz, errors="coerce"),
        pd.to_numeric(official.precursor_mz, errors="coerce"),
        rtol=0.0, atol=1e-8, equal_nan=True,
    ):
        raise RuntimeError("official/experimental query precursor masses differ")


def main() -> None:
    args = parse_args()
    if not args.method.replace("_", "").isalnum():
        raise ValueError("method must contain only letters, digits and underscores")
    official_query = args.official_query_dir or ROOT / f"data/mtbls13729/embeddings/{args.panel}"
    target_path = args.targets or ROOT / f"data/mtbls13729/ms1_consensus/{args.panel}__requantification_targets.csv.gz"
    required = [
        args.query_embedding_dir / "report.json", args.query_embedding_dir / "manifest.csv",
        official_query / "manifest.csv", args.library_cache / "report.json",
        args.library_cache / "manifest.csv", args.library_cache / "embeddings.npy",
        args.official_library_cache / "manifest.csv", target_path,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    query_model = model_metadata(args.query_embedding_dir)
    library_model = model_metadata(args.library_cache)
    if query_model["checkpoint_sha256"] != library_model["checkpoint_sha256"]:
        raise RuntimeError("query and library were encoded by different checkpoints")
    if query_model.get("kind") != "experimental_noise_shared_embedding":
        raise RuntimeError("this arm expects an experimental shared embedding")

    query_manifest = pd.read_csv(args.query_embedding_dir / "manifest.csv")
    official_query_manifest = pd.read_csv(official_query / "manifest.csv")
    assert_query_alignment(query_manifest, official_query_manifest)
    query_embeddings = np.load(resolve_query_embeddings(args.query_embedding_dir), mmap_mode="r")
    if len(query_embeddings) != len(query_manifest):
        raise RuntimeError("query embedding/manifest mismatch")

    library_manifest = pd.read_csv(args.library_cache / "manifest.csv")
    official_library_manifest = pd.read_csv(args.official_library_cache / "manifest.csv")
    if len(library_manifest) != len(official_library_manifest):
        raise RuntimeError("official/experimental library counts differ")
    for column in ("library_row", "ik14"):
        if not np.array_equal(
            library_manifest[column].fillna("").to_numpy(),
            official_library_manifest[column].fillna("").to_numpy(),
        ):
            raise RuntimeError(f"official/experimental library order differs at {column}")
    if not np.allclose(library_manifest.precursor_mz, official_library_manifest.precursor_mz, rtol=0, atol=1e-8):
        raise RuntimeError("official/experimental library masses differ")
    library_embeddings = np.load(args.library_cache / "embeddings.npy", mmap_mode="r")
    if len(library_embeddings) != len(library_manifest):
        raise RuntimeError("library embedding/manifest mismatch")

    targets = pd.read_csv(target_path)
    raw_links = nearest_target_links(official_query_manifest, targets, args.ppm, args.rt_seconds)
    raw_links["query_file"] = official_query_manifest.iloc[raw_links.query_idx.to_numpy(int)].file_name.to_numpy()
    raw_links["link_cost"] = (
        (raw_links.feature_dppm / args.ppm) ** 2
        + (raw_links.feature_drt_sec / args.rt_seconds) ** 2
    )
    links = (
        raw_links.sort_values(["feature_id", "query_file", "link_cost", "query_idx"], kind="stable")
        .groupby(["feature_id", "query_file"], sort=False, as_index=False).head(1)
        .sort_values("query_idx", kind="stable").reset_index(drop=True)
    )
    if args.max_queries:
        links = links.iloc[: args.max_queries].copy()
    if links.empty:
        raise RuntimeError("no query spectra link to MS1 targets")

    lib_mass = pd.to_numeric(library_manifest.precursor_mz, errors="coerce").to_numpy(np.float64)
    lib_ik14 = library_manifest.ik14.fillna("").astype(str).to_numpy(object)
    lib_inchikey = library_manifest.inchikey.fillna("").astype(str).to_numpy(object)
    lib_name = library_manifest["name"].fillna("").astype(str).to_numpy(object)
    lib_smiles = library_manifest.smiles.fillna("").astype(str).to_numpy(object)
    valid_identity = np.asarray([len(value) == 14 for value in lib_ik14], dtype=bool)
    order = np.argsort(lib_mass, kind="stable")
    sorted_mass = lib_mass[order]
    link_lookup = links.set_index("query_idx").to_dict("index")
    records = []
    for completed, query_idx in enumerate(links.query_idx.astype(int), 1):
        query = official_query_manifest.iloc[query_idx]
        mass = float(query.precursor_mz)
        tolerance = mass * args.ppm * 1e-6
        left = int(np.searchsorted(sorted_mass, mass - tolerance, side="left"))
        right = int(np.searchsorted(sorted_mass, mass + tolerance, side="right"))
        candidate_rows = order[left:right]
        candidate_rows = candidate_rows[valid_identity[candidate_rows]]
        if not len(candidate_rows):
            continue
        grouped: dict[str, list[int]] = defaultdict(list)
        for row in candidate_rows:
            grouped[str(lib_ik14[int(row)])].append(int(row))
        identities = sorted(grouped)
        pair_rows, ptr, ordered_library_rows = [], [0], []
        query_embedding = np.asarray(query_embeddings[query_idx], dtype=np.float32)
        for identity in identities:
            rows = sorted(grouped[identity])
            pair_rows.extend(
                float(query_embedding @ np.asarray(library_embeddings[row], dtype=np.float32))
                for row in rows
            )
            ordered_library_rows.extend(rows)
            ptr.append(len(pair_rows))
        pair_scores = np.asarray(pair_rows, dtype=np.float64)
        ptr_array = np.asarray(ptr, dtype=np.int64)
        molecule_scores = grouped_max(pair_scores, ptr_array)
        top = unique_top_index(molecule_scores)
        link = link_lookup[query_idx]
        record = {
            "panel": args.panel, "query_idx": query_idx,
            "query_file": str(query.file_name), "query_scan": int(query.scan_number),
            "query_precursor_mz": mass, "query_rt_sec": float(link["query_rt_sec"]),
            "feature_id": int(link["feature_id"]),
            "feature_dppm": float(link["feature_dppm"]),
            "feature_drt_sec": float(link["feature_drt_sec"]),
            "n_candidate_spectra": int(len(pair_scores)),
            "n_candidate_molecules": int(len(identities)),
            f"{args.method}_abstained_tie": top is None,
        }
        if top is None:
            for field in ("ik14", "inchikey", "name", "smiles"):
                record[f"{args.method}_{field}"] = ""
            record[f"{args.method}_score"] = math.nan
            record[f"{args.method}_dreams_similarity"] = math.nan
        else:
            start, stop = int(ptr_array[top]), int(ptr_array[top + 1])
            spectrum_position = start + int(np.argmax(pair_scores[start:stop]))
            library_row = ordered_library_rows[spectrum_position]
            record.update({
                f"{args.method}_ik14": identities[top],
                f"{args.method}_inchikey": str(lib_inchikey[library_row]),
                f"{args.method}_name": str(lib_name[library_row]),
                f"{args.method}_smiles": str(lib_smiles[library_row]),
                f"{args.method}_score": float(molecule_scores[top]),
                f"{args.method}_dreams_similarity": float(pair_scores[spectrum_position]),
            })
        records.append(record)
        if completed % 2000 == 0 or completed == len(links):
            print(f"[{args.panel} {args.method}] {completed:,}/{len(links):,}", flush=True)
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("no mass-constrained query was scored")

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    per_query_path = out / f"{args.panel}__{args.method}__per_query.csv.gz"
    feature_path = out / f"{args.panel}__{args.method}__feature_annotations.csv.gz"
    report_path = out / f"{args.panel}__{args.method}__report.json"
    if any(path.exists() for path in (per_query_path, feature_path, report_path)):
        raise RuntimeError(f"refusing to overwrite embedding retrieval output in {out}")
    frame.to_csv(per_query_path, index=False, compression="gzip")
    features = summarize_features(frame, args.method)
    features.to_csv(feature_path, index=False, compression="gzip")
    report = {
        "status": "mtbls13729_embedding_retrieval_complete",
        "formal": False,
        "method": args.method,
        "panel": args.panel,
        "model": query_model,
        "selected_query_spectra": int(len(links)),
        "scored_query_spectra": int(len(frame)),
        "linked_features": int(links.feature_id.nunique()),
        "annotated_features": int(len(features)),
        "protocol": "same 10 ppm mass graph and same one-MS2-per-feature/sample policy as official/P2b arm",
        "provenance": {
            "query_embeddings_sha256": sha256_file(resolve_query_embeddings(args.query_embedding_dir)),
            "library_embeddings_sha256": sha256_file(args.library_cache / "embeddings.npy"),
            "targets_sha256": sha256_file(target_path),
        },
        "claim_limit": "experimental one-fold shared embedding application; annotation changes are not correctness without standards",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

