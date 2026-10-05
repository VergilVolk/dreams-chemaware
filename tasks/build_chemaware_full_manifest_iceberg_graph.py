"""Build a train-formula ICEBERG teacher graph from the corrected manifest.

The prior 922-query diagnostic graph has no query identities in common with
the 83k training manifest.  This builder therefore selects identity-distinct
training queries from the full manifest, excludes inner/outer-formula candidate
molecules, and materializes the small feature-bearing graph expected by the
existing ICEBERG teacher audit.  It is a biased teacher-development cohort,
never an independent benchmark.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from chemaware_iceberg_direct_core import stable_formula_folds  # noqa: E402
from noise_final_core import sha256_file  # noqa: E402
from train_chemaware_full_candidate_alignment import official_outcomes  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/chemaware_full_manifest_iceberg_graph_v1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--inner-fold", type=int, default=3)
    parser.add_argument("--outer-fold", type=int, default=4)
    parser.add_argument("--max-identities", type=int, default=2048)
    parser.add_argument("--error-fraction", type=float, default=0.5)
    return parser.parse_args()


def identity_representatives(
    pool: np.ndarray, identities: np.ndarray, error: np.ndarray,
    margin: np.ndarray, limit: int, error_fraction: float,
) -> np.ndarray:
    by_identity: dict[str, list[int]] = {}
    for query in np.asarray(pool, dtype=np.int64):
        by_identity.setdefault(str(identities[query]), []).append(int(query))
    error_examples = []
    clean_examples = []
    for identity, values in by_identity.items():
        index = np.asarray(values, dtype=np.int64)
        failed = index[error[index]]
        if len(failed):
            query = int(failed[np.argmin(margin[failed])])
            error_examples.append((float(margin[query]), identity, query))
        else:
            query = int(index[np.argmin(np.abs(margin[index]))])
            clean_examples.append((float(abs(margin[query])), identity, query))
    error_examples.sort(key=lambda value: (value[0], value[1]))
    clean_examples.sort(key=lambda value: (value[0], value[1]))
    target = min(limit, len(by_identity))
    n_error = min(len(error_examples), int(round(target * error_fraction)))
    n_clean = min(len(clean_examples), target - n_error)
    if n_error + n_clean < target:
        n_error = min(len(error_examples), target - n_clean)
    selected = [value[2] for value in error_examples[:n_error]]
    selected.extend(value[2] for value in clean_examples[:n_clean])
    return np.asarray(sorted(selected), dtype=np.int64)


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite output: {args.output}")
    if not 0 <= args.error_fraction <= 1 or args.max_identities < 2:
        raise ValueError("invalid selection parameters")
    required = [args.manifest, args.token_dir / "report.json",
                args.token_dir / "rows.npy", args.token_dir / "official_embeddings_f32.npy"]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)
    token_report = json.loads((args.token_dir / "report.json").read_text(encoding="utf-8"))
    if token_report.get("status") != "chemaware_corrected_manifest_token_cache_complete":
        raise RuntimeError("requires the complete official embedding cache")
    with np.load(args.manifest) as loaded:
        body = {key: loaded[key] for key in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    query_fold = stable_formula_folds(body["query_formula"], args.folds, args.fold_seed)
    molecule_fold = stable_formula_folds(body["molecule_formula"], args.folds, args.fold_seed)
    molecule_allowed = (molecule_fold != args.inner_fold) & (molecule_fold != args.outer_fold)
    allowed_count = np.add.reduceat(molecule_allowed.astype(np.int32), body["query_ptr"][:-1])
    train_pool = np.flatnonzero(
        (query_fold != args.inner_fold) & (query_fold != args.outer_fold)
        & (allowed_count >= 2)
    )
    error, margin = official_outcomes(
        body, train_pool, official, row_position, molecule_allowed,
    )
    selected = identity_representatives(
        train_pool, body["query_ik14"], error, margin,
        args.max_identities, args.error_fraction,
    )
    if len(np.unique(body["query_ik14"][selected].astype(str))) != len(selected):
        raise RuntimeError("teacher graph selection is not identity-distinct")

    query_ptr = [0]
    molecule_ptr = [0]
    pair_rows = []
    features = []
    molecule_label = []
    molecule_ik14 = []
    molecule_formula = []
    for query in selected:
        qpos = row_position[int(body["query_row"][query])]
        left, right = map(int, body["query_ptr"][query:query + 2])
        kept = [molecule for molecule in range(left, right) if molecule_allowed[molecule]]
        if len(kept) < 2 or int(body["molecule_label"][kept[0]]) != 1:
            raise RuntimeError("selected query lost its true or negative candidate")
        for molecule in kept:
            rleft, rright = map(int, body["molecule_ptr"][molecule:molecule + 2])
            candidate_rows = body["pair_candidate_row"][rleft:rright].astype(np.int64)
            positions = np.asarray([row_position[int(row)] for row in candidate_rows], dtype=np.int64)
            score = np.asarray(official[positions] @ official[qpos], dtype=np.float32)
            pair_rows.extend(map(int, candidate_rows))
            features.extend(score.reshape(-1, 1))
            molecule_ptr.append(len(pair_rows))
            molecule_label.append(int(body["molecule_label"][molecule]))
            molecule_ik14.append(str(body["molecule_ik14"][molecule]))
            molecule_formula.append(str(body["molecule_formula"][molecule]))
        query_ptr.append(len(molecule_label))
    feature_array = np.asarray(features, dtype=np.float32).reshape(-1, 1)
    args.output.mkdir(parents=True)
    graph_path = args.output / "graph.npz"
    np.savez_compressed(
        graph_path,
        feature_names=np.asarray(["dreams_similarity"], dtype=object),
        features=feature_array,
        pair_candidate_row=np.asarray(pair_rows, dtype=np.int64),
        query_ptr=np.asarray(query_ptr, dtype=np.int64),
        molecule_ptr=np.asarray(molecule_ptr, dtype=np.int64),
        molecule_label=np.asarray(molecule_label, dtype=np.int8),
        molecule_ik14=np.asarray(molecule_ik14, dtype=object),
        molecule_formula=np.asarray(molecule_formula, dtype=object),
        query_row=body["query_row"][selected].astype(np.int64),
        query_ik14=body["query_ik14"][selected],
        query_formula=body["query_formula"][selected],
        query_has_near=np.zeros(len(selected), dtype=bool),
    )
    np.save(args.output / "source_query_index.npy", selected)
    report = {
        "status": "chemaware_full_manifest_iceberg_graph_complete",
        "scope": {
            "training_formula_only": True,
            "biased_teacher_development_cohort": True,
            "independent_benchmark": False,
            "inner_outer_candidate_formulas_excluded": True,
        },
        "counts": {
            "queries": int(len(selected)),
            "official_errors": int(np.sum(error[selected])),
            "official_correct": int(np.sum(~error[selected])),
            "unique_identities": int(len(np.unique(body["query_ik14"][selected].astype(str)))),
            "unique_formulas": int(len(np.unique(body["query_formula"][selected].astype(str)))),
            "candidate_molecules": int(len(molecule_label)),
            "candidate_spectrum_edges": int(len(pair_rows)),
        },
        "provenance": {
            "manifest_sha256": sha256_file(args.manifest),
            "official_checkpoint_sha256": token_report["provenance"]["official_checkpoint_sha256"],
            "graph_sha256": sha256_file(graph_path),
        },
        "selection": {
            "max_identities": args.max_identities,
            "error_fraction": args.error_fraction,
            "error_query": "most-negative official margin per error identity",
            "clean_query": "smallest-absolute official margin per clean identity",
        },
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
