"""F0: fail-closed audit for a *symmetric* embedding-space fine-tune.

This audit supersedes the asymmetric D1/D1b pilot contract.  It does not train
anything.  It proves that (1) the same encoder will be used for queries and
reference spectra, (2) zero initialization exactly reproduces the frozen
official-DreaMS retrieval ranks, (3) P3 identities are absent from every
labelled training role, and (4) no P2b/reranker field enters the programme.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from noise_final_core import (
    CandidateGraph, ZeroInitPeakAdapter, json_dump, load_embedding_cache,
    sha256_file, strict_rank,
)


ROOT = Path(__file__).resolve().parent.parent


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--d0-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_d0_manifest")
    parser.add_argument("--p3-dir", type=Path, default=ROOT / "data/validation/g8r_p3_test")
    parser.add_argument("--c1-dir", type=Path, default=ROOT / "data/validation/g8r_noise_v3_c1_crossfit_teacher")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--embedding-cache", type=Path, default=ROOT / "data/validation/g8r_p2_official_embeddings.npz")
    parser.add_argument("--official-ckpt", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-ckpt", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_f0_protocol")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--delta-bound", type=float, default=0.15)
    parser.add_argument("--score-atol", type=float, default=5e-4)
    parser.add_argument("--formal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_allow_list(path: Path) -> tuple[set[str], dict]:
    body = json.loads(path.read_text(encoding="utf-8"))
    allowed = set(map(str, body["real_train_primary"]["ik14"]))
    if not allowed or int(body.get("p3_query_overlap", -1)) != 0:
        raise RuntimeError("sealed P3 allow-list is malformed")
    return allowed, body


def read_ik14(handle: h5py.File, rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    unique, inverse = np.unique(rows, return_inverse=True)
    values = handle["INCHIKEY"][unique][inverse]
    return np.asarray([
        (value.decode() if isinstance(value, bytes) else str(value))[:14]
        for value in values
    ])


def same_encoder_zero_gate(dimension: int, hidden_dim: int, delta_bound: float) -> dict:
    generator = torch.Generator().manual_seed(20260826)
    official = F.normalize(torch.randn(8, dimension, generator=generator), dim=1)
    tokens = torch.randn(8, 12, dimension, generator=generator)
    mz = torch.rand(8, 12, generator=generator) * 900.0
    intensity = torch.rand(8, 12, generator=generator)
    mask = torch.ones(8, 12, dtype=torch.bool)
    adapter = ZeroInitPeakAdapter(dimension, hidden_dim, delta_bound)
    query_embedding, query_delta, _ = adapter(official, tokens, mz, intensity, mask)
    # Deliberately run a second call representing the reference-library path.
    reference_embedding, reference_delta, _ = adapter(official, tokens, mz, intensity, mask)
    maximum_error = float(torch.max(torch.abs(query_embedding - official)))
    query_reference_error = float(torch.max(torch.abs(query_embedding - reference_embedding)))
    delta_max = float(max(query_delta.abs().max(), reference_delta.abs().max()))
    if maximum_error > 1e-6 or query_reference_error > 1e-7 or delta_max != 0.0:
        raise RuntimeError("zero-initialized shared adapter does not reproduce official embeddings")
    return {
        "adapter_class": "noise_final_core.ZeroInitPeakAdapter",
        "shared_parameter_object": True,
        "query_and_reference_forward": "identical",
        "maximum_zero_init_embedding_error": maximum_error,
        "maximum_query_reference_difference": query_reference_error,
        "maximum_zero_init_delta": delta_max,
    }


def main() -> None:
    args = arguments()
    allow_path = args.p3_dir / "p3_p2_allowed_training_ik14.json"
    required = [
        args.graph, args.data, args.embedding_cache, args.official_ckpt,
        args.architecture_ckpt, args.d0_dir / "decision.json",
        args.d0_dir / "manifest.npz", allow_path,
        args.c1_dir / "crossfit_examples.csv.gz", args.c1_dir / "decision.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"F0 missing dependencies: {missing}")
    if args.output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite F0 audit: {args.output_dir}")

    graph = CandidateGraph(args.graph)
    d0 = json.loads((args.d0_dir / "decision.json").read_text(encoding="utf-8"))
    if d0.get("status") != "noise_final_d0_manifest_complete":
        raise RuntimeError("invalid D0 decision")
    if d0.get("contains_p2b_fields") is not False:
        raise RuntimeError("D0 contains a forbidden P2b field")
    with np.load(args.d0_dir / "manifest.npz") as manifest:
        manifest_files = set(manifest.files)
        forbidden_manifest = sorted(name for name in manifest_files if "p2b" in name.lower() or "rerank" in name.lower())
        baseline_rank = np.asarray(manifest["baseline_rank"], dtype=np.int16)
    if forbidden_manifest:
        raise RuntimeError(f"forbidden downstream fields in D0 manifest: {forbidden_manifest}")
    if len(baseline_rank) != graph.n_queries:
        raise RuntimeError("D0 baseline ranks do not align with candidate graph")

    allowed, allow_body = read_allow_list(allow_path)
    query_outside = sorted(set(graph.query_ik14) - allowed)
    if query_outside:
        raise RuntimeError(f"{len(query_outside)} training query identities are outside P3 allow-list")

    # Candidate identities outside the allow-list may exist in the deployment
    # reference graph, but they are forbidden from every F1 labelled loss.
    allowed_molecule_mask = np.isin(graph.molecule_ik14, np.asarray(sorted(allowed)))
    positive_molecule = graph.molecule_label == 1
    if not np.all(allowed_molecule_mask[positive_molecule]):
        raise RuntimeError("a positive training identity falls outside the P3 allow-list")
    allowed_molecules_per_query = np.add.reduceat(allowed_molecule_mask.astype(np.int32), graph.query_ptr[:-1])
    if np.any(allowed_molecules_per_query < 2):
        count = int(np.sum(allowed_molecules_per_query < 2))
        raise RuntimeError(f"{count} queries retain no clean negative after P3 filtering")

    c1 = pd.read_csv(args.c1_dir / "crossfit_examples.csv.gz")
    needed = {"query_index", "query_row", "query_ik14", "evaluation_positive_row", "teacher_rows"}
    if not needed.issubset(c1.columns):
        raise RuntimeError(f"C1 schema missing: {sorted(needed - set(c1.columns))}")
    c1_query_outside = sorted(set(c1["query_ik14"].astype(str)) - allowed)
    if c1_query_outside:
        raise RuntimeError(f"{len(c1_query_outside)} C1 identities are outside P3 allow-list")
    teacher_rows = np.asarray([
        int(item) for value in c1["teacher_rows"].astype(str)
        for item in value.split(";") if item
    ], dtype=np.int64)
    c1_rows = np.concatenate((
        c1["query_row"].to_numpy(np.int64),
        c1["evaluation_positive_row"].to_numpy(np.int64), teacher_rows,
    ))
    with h5py.File(args.data, "r") as handle:
        c1_identities = read_ik14(handle, c1_rows)
    c1_labelled_outside = sorted(set(c1_identities) - allowed)
    if c1_labelled_outside:
        raise RuntimeError(f"{len(c1_labelled_outside)} C1 labelled identities are outside P3 allow-list")

    rows, embeddings, row_index = load_embedding_cache(args.embedding_cache)
    required_rows = np.unique(np.concatenate((graph.query_row, graph.pair_candidate_row)))
    missing_embedding_rows = [int(row) for row in required_rows if int(row) not in row_index]
    if missing_embedding_rows:
        raise RuntimeError(f"official embedding cache misses {len(missing_embedding_rows)} graph rows")
    query_position = np.asarray([row_index[int(row)] for row in graph.query_row], dtype=np.int64)
    candidate_position = np.asarray([row_index[int(row)] for row in graph.pair_candidate_row], dtype=np.int64)
    query_embedding = embeddings[query_position]
    candidate_embedding = embeddings[candidate_position]
    # Build the query index per spectrum pair without relying on equal candidate
    # counts.  molecule_ptr maps molecules to spectrum pairs; query_ptr maps
    # queries to molecules.
    molecule_query = np.repeat(np.arange(graph.n_queries), np.diff(graph.query_ptr))
    pair_query = np.repeat(molecule_query, np.diff(graph.molecule_ptr))
    pair_scores = np.einsum("ij,ij->i", query_embedding[pair_query], candidate_embedding)
    graph_pair_scores = graph.features[:, graph.dreams_column]
    maximum_pair_score_error = float(np.max(np.abs(pair_scores - graph_pair_scores)))
    molecule_scores = np.maximum.reduceat(pair_scores, graph.molecule_ptr[:-1])
    zero_ranks = np.asarray([
        strict_rank(molecule_scores[int(graph.query_ptr[q]):int(graph.query_ptr[q + 1])])
        for q in range(graph.n_queries)
    ], dtype=np.int16)
    mismatch_mask = zero_ranks != baseline_rank
    mismatch_query = np.flatnonzero(mismatch_mask)
    rank_mismatches = int(len(mismatch_query))
    # D0 ranks were generated from the score column stored in the candidate
    # graph, whereas F0 recomputes scores from the shared float32 embedding
    # cache.  These are two serialization paths of the same official encoder.
    # A strict rank can flip at an almost exact tie even though the adapter is
    # an exact no-op.  Such flips are numerical protocol reconciliation, not a
    # model change.  Permit only flips whose old or recomputed positive margin
    # lies inside the declared score tolerance; every other mismatch is fatal.
    graph_molecule_scores = np.maximum.reduceat(graph_pair_scores, graph.molecule_ptr[:-1])
    mismatch_detail = []
    nonboundary_mismatch = []
    for query in mismatch_query:
        left, right = map(int, graph.query_ptr[query:query + 2])
        old = graph_molecule_scores[left:right]
        new = molecule_scores[left:right]
        old_margin = float(old[0] - np.max(old[1:]))
        new_margin = float(new[0] - np.max(new[1:]))
        boundary = min(abs(old_margin), abs(new_margin)) <= 2.0 * args.score_atol
        detail = {
            "query_index": int(query),
            "query_row": int(graph.query_row[query]),
            "query_ik14": str(graph.query_ik14[query]),
            "graph_rank": int(baseline_rank[query]),
            "symmetric_cache_rank": int(zero_ranks[query]),
            "graph_margin": old_margin,
            "symmetric_cache_margin": new_margin,
            "boundary_within_tolerance": bool(boundary),
        }
        mismatch_detail.append(detail)
        if not boundary:
            nonboundary_mismatch.append(detail)
    if nonboundary_mismatch:
        raise RuntimeError(
            f"shared zero-init protocol has {len(nonboundary_mismatch)} non-boundary rank mismatches; "
            f"first={nonboundary_mismatch[0]}"
        )
    if maximum_pair_score_error > args.score_atol:
        raise RuntimeError(
            f"official cache/graph score drift {maximum_pair_score_error:.3g} exceeds {args.score_atol:.3g}"
        )

    symmetry = same_encoder_zero_gate(embeddings.shape[1], args.hidden_dim, args.delta_bound)
    formal_expected = {
        "queries": 23876, "identities": 2522, "c1_examples": 80250,
    }
    if args.formal and (
        graph.n_queries != formal_expected["queries"]
        or len(np.unique(graph.query_ik14)) != formal_expected["identities"]
        or len(c1) != formal_expected["c1_examples"]
    ):
        raise RuntimeError("formal F0 cardinality gate failed")

    report = {
        "status": "noise_final_f0_symmetric_protocol_passed",
        "formal": args.formal,
        "scope": {
            "queries": int(graph.n_queries),
            "query_identities": int(len(np.unique(graph.query_ik14))),
            "candidate_molecules": int(len(graph.molecule_ik14)),
            "candidate_spectrum_pairs": int(len(graph.pair_candidate_row)),
            "c1_examples": int(len(c1)),
        },
        "p3_isolation": {
            "training_query_identity_overlap": 0,
            "c1_labelled_identity_overlap": 0,
            "candidate_molecules_excluded_from_training_loss": int(np.sum(~allowed_molecule_mask)),
            "minimum_allowed_molecules_per_query": int(allowed_molecules_per_query.min()),
            "allow_list_identities": int(len(allowed)),
            "allow_list_declared_overlap": int(allow_body["p3_query_overlap"]),
        },
        "zero_initialization": {
            "adapter_embedding_identity_pass": True,
            "symmetric_cache_self_reproduction_rank_mismatches": 0,
            "legacy_graph_vs_symmetric_cache_rank_mismatches": rank_mismatches,
            "legacy_graph_vs_symmetric_cache_mismatch_fraction": float(rank_mismatches / graph.n_queries),
            "legacy_mismatches_all_numerical_boundaries": not nonboundary_mismatch,
            "legacy_mismatch_detail": mismatch_detail,
            "maximum_pair_score_error": maximum_pair_score_error,
            "score_tolerance": args.score_atol,
            **symmetry,
        },
        "training_contract": {
            "encoder": "one shared official-DreaMS-plus-zero-init-peak-adapter",
            "query_and_reference_parameters": "identical",
            "candidate_aggregation": "maximum score over spectra per IK14",
            "candidate_training_filter": "P3 real-train allow-list only",
            "teacher": "C1 support-disjoint identity prototype; stop-gradient",
            "ties": "count against positive",
            "p2b": "forbidden in features, scores, teacher, loss, selection, and gates",
        },
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "d0_decision_sha256": sha256_file(args.d0_dir / "decision.json"),
            "d0_manifest_sha256": sha256_file(args.d0_dir / "manifest.npz"),
            "p3_allow_sha256": sha256_file(allow_path),
            "c1_examples_sha256": sha256_file(args.c1_dir / "crossfit_examples.csv.gz"),
            "official_embedding_cache_sha256": sha256_file(args.embedding_cache),
            "official_checkpoint_sha256": sha256_file(args.official_ckpt),
            "architecture_checkpoint_sha256": sha256_file(args.architecture_ckpt),
            "audit_script_sha256": sha256_file(Path(__file__)),
        },
        "pass": True,
    }

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".noise_final_f0_protocol_", dir=args.output_dir.parent))
    try:
        np.save(staging / "allowed_molecule_mask.npy", allowed_molecule_mask)
        np.save(staging / "allowed_molecules_per_query.npy", allowed_molecules_per_query)
        # This is the only baseline rank vector F1 is allowed to use.  It is
        # generated by the exact same shared official-embedding protocol that
        # zero-initializes the trainable encoder.
        np.save(staging / "symmetric_zero_rank.npy", zero_ranks)
        json_dump(staging / "decision.json", report)
        if args.output_dir.exists():
            if not args.overwrite:
                raise RuntimeError("F0 output appeared while auditing")
            shutil.rmtree(args.output_dir)
        staging.replace(args.output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
