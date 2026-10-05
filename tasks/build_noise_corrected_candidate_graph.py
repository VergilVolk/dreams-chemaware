"""Add frozen official DreaMS scores to the corrected P3-disjoint candidate graph.

The source manifest is metadata-only. This builder computes query-candidate
cosines from the already frozen official embedding cache, restores MCES strata,
and emits the legacy CandidateGraph schema needed by noise direct fine-tuning.
It performs no optimization and consumes no action outcomes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np

from noise_final_core import sha256_file, strict_rank


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--embedding-cache-dir", type=Path, required=True)
    parser.add_argument("--mces-pairs", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--chunk-pairs", type=int, default=100_000)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def row_positions(cache_rows: np.ndarray, requested: np.ndarray, name: str) -> np.ndarray:
    positions = np.searchsorted(cache_rows, requested)
    valid = positions < len(cache_rows)
    if not np.all(valid) or not np.array_equal(cache_rows[positions], requested):
        raise RuntimeError(f"official cache does not cover every {name} row")
    return positions.astype(np.int64, copy=False)


def load_mces(path: Path) -> dict[tuple[str, str], int]:
    body = json.loads(path.read_text(encoding="utf-8"))
    lookup: dict[tuple[str, str], int] = {}
    for grade, label in enumerate(("near", "mid", "far")):
        for row in body.get(label, []):
            key = tuple(sorted((str(row["ik_a"]), str(row["ik_b"]))))
            if key in lookup and lookup[key] != grade:
                raise RuntimeError(f"conflicting MCES grade for {key}")
            lookup[key] = grade
    return lookup


def main() -> None:
    args = arguments()
    started = time.time()
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite {args.output_dir}")
    if args.chunk_pairs < 1:
        raise ValueError("chunk-pairs must be positive")
    manifest_path = args.manifest_dir / "manifest.npz"
    manifest_report_path = args.manifest_dir / "manifest.json"
    cache_report_path = args.embedding_cache_dir / "report.json"
    rows_path = args.embedding_cache_dir / "rows.npy"
    embeddings_path = args.embedding_cache_dir / "official_embeddings_f32.npy"
    required = (
        manifest_path, manifest_report_path, cache_report_path, rows_path,
        embeddings_path, args.mces_pairs, args.official_checkpoint,
    )
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)
    manifest_report = json.loads(manifest_report_path.read_text(encoding="utf-8"))
    cache_report = json.loads(cache_report_path.read_text(encoding="utf-8"))
    if (
        manifest_report.get("data_contract") != "train_primary_all_p3_disjoint_v1"
        or manifest_report.get("simulation_challenge_semantics")
        != "spectrum-simulation benchmark subset membership; not provenance; not filtered"
    ):
        raise RuntimeError("corrected manifest data contract is not frozen")
    manifest_sha = sha256_file(manifest_path)
    official_sha = sha256_file(args.official_checkpoint)
    if (
        manifest_report.get("provenance", {}).get("manifest_sha256") != manifest_sha
        or cache_report.get("provenance", {}).get("manifest_sha256") != manifest_sha
        or cache_report.get("provenance", {}).get("official_checkpoint_sha256") != official_sha
    ):
        raise RuntimeError("manifest, cache and official checkpoint provenance disagree")

    with np.load(manifest_path, allow_pickle=False) as body:
        arrays = {name: body[name] for name in body.files}
    query_row = np.asarray(arrays["query_row"], dtype=np.int64)
    query_ptr = np.asarray(arrays["query_ptr"], dtype=np.int64)
    molecule_ptr = np.asarray(arrays["molecule_ptr"], dtype=np.int64)
    molecule_query = np.asarray(arrays["molecule_query"], dtype=np.int64)
    pair_candidate_row = np.asarray(arrays["pair_candidate_row"], dtype=np.int64)
    molecule_ik14 = np.asarray(arrays["molecule_ik14"], dtype=str)
    query_ik14 = np.asarray(arrays["query_ik14"], dtype=str)
    cache_rows = np.load(rows_path, mmap_mode="r")
    embeddings = np.load(embeddings_path, mmap_mode="r")
    if (
        cache_rows.ndim != 1 or not np.all(np.diff(cache_rows) > 0)
        or embeddings.shape != (len(cache_rows), 1024)
    ):
        raise RuntimeError("official embedding cache schema is invalid")
    sample = np.asarray(embeddings[:: max(1, len(embeddings) // 2048)], dtype=np.float32)
    sample_norm = np.linalg.norm(sample, axis=1)
    if not np.all(np.isfinite(sample_norm)) or np.max(np.abs(sample_norm - 1.0)) > 2e-3:
        raise RuntimeError("official embeddings are not finite unit vectors")

    query_position = row_positions(cache_rows, query_row, "query")
    candidate_position = row_positions(cache_rows, pair_candidate_row, "candidate")
    pair_query = np.repeat(molecule_query, np.diff(molecule_ptr)).astype(np.int64)
    if len(pair_query) != len(pair_candidate_row):
        raise RuntimeError("molecule-to-pair expansion is inconsistent")
    pair_score = np.empty(len(pair_candidate_row), dtype=np.float32)
    for left in range(0, len(pair_score), args.chunk_pairs):
        right = min(left + args.chunk_pairs, len(pair_score))
        q = np.asarray(embeddings[query_position[pair_query[left:right]]], dtype=np.float32)
        c = np.asarray(embeddings[candidate_position[left:right]], dtype=np.float32)
        pair_score[left:right] = np.einsum("ij,ij->i", q, c, optimize=True)
        if right % 1_000_000 < args.chunk_pairs or right == len(pair_score):
            print(f"[corrected official scores] {right:,}/{len(pair_score):,}", flush=True)
    if not np.all(np.isfinite(pair_score)) or np.any((pair_score < -1.0001) | (pair_score > 1.0001)):
        raise RuntimeError("official pair scores are invalid")

    mces = load_mces(args.mces_pairs)
    molecule_grade = np.full(len(molecule_ik14), -2, dtype=np.int8)
    for molecule in range(len(molecule_ik14)):
        query = int(molecule_query[molecule])
        if int(arrays["molecule_label"][molecule]) == 1:
            molecule_grade[molecule] = -1
        else:
            molecule_grade[molecule] = mces.get(
                tuple(sorted((str(query_ik14[query]), str(molecule_ik14[molecule])))), -2,
            )
    query_has_near = np.asarray([
        np.any(molecule_grade[int(left):int(right)] == 0)
        for left, right in zip(query_ptr[:-1], query_ptr[1:])
    ], dtype=bool)

    molecule_score = np.maximum.reduceat(pair_score, molecule_ptr[:-1])
    ranks = np.asarray([
        strict_rank(molecule_score[int(left):int(right)])
        for left, right in zip(query_ptr[:-1], query_ptr[1:])
    ], dtype=np.int16)
    baseline = {
        "queries": int(len(ranks)),
        "recall1": float(np.mean(ranks == 1)),
        "mrr": float(np.mean(1.0 / ranks)),
        "errors": int(np.sum(ranks != 1)),
        "near_queries": int(query_has_near.sum()),
        "near_recall1": float(np.mean(ranks[query_has_near] == 1))
        if np.any(query_has_near) else None,
    }
    graph_arrays = {
        "feature_names": np.asarray(["dreams_similarity"]),
        "features": pair_score[:, None],
        "pair_candidate_row": pair_candidate_row,
        "query_ptr": query_ptr,
        "molecule_ptr": molecule_ptr,
        "molecule_label": np.asarray(arrays["molecule_label"], dtype=np.int8),
        "molecule_ik14": molecule_ik14,
        "molecule_formula": np.asarray(arrays["molecule_formula"], dtype=str),
        "molecule_mces_grade": molecule_grade,
        "query_row": query_row,
        "query_ik14": query_ik14,
        "query_formula": np.asarray(arrays["query_formula"], dtype=str),
        "query_has_near": query_has_near,
    }

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".noise_corrected_graph_", dir=args.output_dir.parent))
    try:
        graph_path = staging / "candidate_graph.npz"
        cache_path = staging / "official_embeddings.npz"
        np.savez_compressed(graph_path, **graph_arrays)
        np.savez_compressed(
            cache_path, rows=np.asarray(cache_rows, dtype=np.int64),
            embeddings=np.asarray(embeddings, dtype=np.float32),
        )
        report = {
            "status": "noise_corrected_candidate_graph_complete",
            "formal_training_authorized": True,
            "data_contract": "train_primary_all_p3_disjoint_v1",
            "candidate_contract": manifest_report["candidate_contract"],
            "official_baseline": baseline,
            "counts": {
                "queries": int(len(query_row)),
                "query_identities": int(len(np.unique(query_ik14))),
                "query_formulas": int(len(np.unique(arrays["query_formula"]))),
                "candidate_molecules": int(len(molecule_ik14)),
                "candidate_spectrum_edges": int(len(pair_score)),
                "reachable_spectra": int(len(cache_rows)),
            },
            "contracts": {
                "official_score_is_query_candidate_cosine": True,
                "molecule_score_is_spectrum_max": True,
                "positive_molecule_unique_and_first": True,
                "formula_folds_must_be_frozen_before_training": True,
                "action_outcomes_consumed": False,
                "P2b": "forbidden",
                "P3_consumed": False,
            },
            "provenance": {
                "source_manifest_sha256": manifest_sha,
                "source_manifest_report_sha256": sha256_file(manifest_report_path),
                "source_cache_report_sha256": sha256_file(cache_report_path),
                "source_rows_sha256": sha256_file(rows_path),
                "source_official_embeddings_sha256": sha256_file(embeddings_path),
                "official_checkpoint_sha256": official_sha,
                "mces_pairs_sha256": sha256_file(args.mces_pairs),
                "candidate_graph_sha256": sha256_file(graph_path),
                "embedding_cache_sha256": sha256_file(cache_path),
                "script_sha256": sha256_file(Path(__file__)),
            },
            "runtime_seconds": time.time() - started,
            "claim_limit": (
                "Corrected train-side development graph; not a P3 test result and not "
                "evidence that any fine-tuned encoder improves retrieval."
            ),
        }
        (staging / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
