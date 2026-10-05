#!/usr/bin/env python
"""Freeze the exact BioAware-routed candidate graph for B4 direct training.

The network expert is used only to route labelled training examples.  Reaction
neighbours are never positives.  Each positive molecule is the query truth and
all other exact benchmark candidates remain negatives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_bioaware_metdna3_negative_loso_ablation import ABLATIONS  # noqa: E402
from develop_bioaware_metdna3_negative_loso_ranker import evaluate_fold  # noqa: E402


UNITS = (
    "BV2cell__hilic", "BV2cell__rplc", "Mouse_brain__hilic",
    "Mouse_brain__rplc", "Mouse_liver__hilic", "Mouse_liver__rplc",
    "NIST_plasma__hilic", "NIST_plasma__rplc",
)


def stable_fold(value: str, folds: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}|{value}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % folds


def sha256_file(path: Path, block: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.lower().eq("true").to_numpy(bool)


def load_mona_reference_tensors(
    mgf_path: Path,
    manifest_path: Path,
    selected_rows: np.ndarray,
    n_highest_peaks: int,
) -> np.ndarray:
    """Load selected MoNA rows using the original embedding-cache row order.

    External-negative ``library_row`` values index the MoNA-negative MGF and
    its embedding manifest.  They must never be interpreted as MassSpecGym
    HDF5 row numbers.
    """
    library = pd.read_csv(manifest_path)
    selected_rows = np.asarray(selected_rows, dtype=np.int64)
    if (selected_rows.ndim != 1 or not len(selected_rows)
            or int(selected_rows[0]) < 0 or int(selected_rows[-1]) >= len(library)
            or np.any(np.diff(selected_rows) <= 0)):
        raise RuntimeError("selected MoNA reference rows are invalid")
    wanted = set(map(int, selected_rows))
    tensors: dict[int, np.ndarray] = {}
    record_index = -1
    current: dict[str, str | float] | None = None
    peaks: list[tuple[float, float]] = []

    def finish_record() -> None:
        nonlocal record_index
        # Exact inclusion rule used by encode_mona_neg_library.py.
        if current is None or not peaks or not current.get("precursor_mz"):
            return
        record_index += 1
        if record_index not in wanted:
            return
        precursor = float(current["precursor_mz"])
        raw = np.asarray(peaks, dtype=np.float32).T
        highest = np.argsort(raw[1], kind="stable")[-n_highest_peaks:]
        highest = np.sort(highest)
        chosen = raw[:, highest].T.astype(np.float32, copy=True)
        if len(chosen) < n_highest_peaks:
            chosen = np.pad(chosen, ((0, n_highest_peaks - len(chosen)), (0, 0)))
        maximum = float(chosen[:, 1].max())
        if maximum > 0:
            chosen[:, 1] /= maximum
        tensor = np.vstack((
            np.asarray([[precursor, 1.1]], dtype=np.float32), chosen,
        )).astype(np.float32)

        expected = library.iloc[record_index]
        if not np.isclose(
            precursor, float(expected["precursor_mz"]), rtol=0, atol=1e-6,
        ):
            raise RuntimeError(f"MoNA MGF/manifest precursor mismatch at row {record_index}")
        if str(current.get("inchikey", "")).upper() != str(expected["inchikey"]).upper():
            raise RuntimeError(f"MoNA MGF/manifest identity mismatch at row {record_index}")
        tensors[record_index] = tensor

    with mgf_path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line == "BEGIN IONS":
                current, peaks = {}, []
            elif line == "END IONS":
                finish_record()
                current, peaks = None, []
            elif current is not None and "=" in line:
                key, value = line.split("=", 1)
                value = value.strip()
                if key == "PEPMASS":
                    # Preserve the original parser's truth-value semantics:
                    # PEPMASS=0 is a float zero and therefore not a record.
                    current["precursor_mz"] = float(value.split()[0])
                elif key == "INCHIKEY":
                    current["inchikey"] = value
            elif current is not None:
                fields = line.split()
                if len(fields) >= 2:
                    try:
                        peaks.append((float(fields[0]), float(fields[1])))
                    except ValueError:
                        pass
    if current is not None:
        raise RuntimeError("unterminated MoNA MGF record")
    if record_index + 1 != len(library):
        raise RuntimeError(
            f"MoNA MGF/manifest record count mismatch: {record_index + 1} != {len(library)}"
        )
    missing = wanted - set(tensors)
    if missing:
        raise RuntimeError(f"MoNA MGF misses {len(missing)} selected reference rows")
    return np.stack([tensors[int(row)] for row in selected_rows]).astype(np.float32)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-dir", type=Path,
        default=ROOT / "data/validation/bioaware_metdna3_external_negative_dreams_v2_chemically_filtered",
    )
    parser.add_argument(
        "--transition-dir", type=Path,
        default=ROOT / "data/validation/bioaware_metdna3_external_negative_loso_ablation_v3_chemically_filtered",
    )
    parser.add_argument(
        "--unit-dir", type=Path,
        default=ROOT / "data/validation/bioaware_metdna3_external_negative_units_v2",
    )
    parser.add_argument(
        "--library-mgf", type=Path,
        default=ROOT / "data/models/mona_neg_full.mgf",
    )
    parser.add_argument(
        "--library-manifest", type=Path,
        default=ROOT / "data/models/mona_neg_dreams_emb/manifest.csv",
    )
    parser.add_argument(
        "--library-embeddings", type=Path,
        default=ROOT / "data/models/mona_neg_dreams_emb/embeddings.npy",
    )
    parser.add_argument(
        "--official-checkpoint", type=Path,
        default=ROOT / "data/e1/official_embedding_slim.pt",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/validation/bioaware_b4_direct_manifest_v1",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.folds < 3:
        raise ValueError("at least three formula folds are required")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {args.output_dir}")
    paths = {
        "queries": args.benchmark_dir / "queries.csv.gz",
        "query_embeddings": args.benchmark_dir / "query_embeddings.npz",
        "scores": args.benchmark_dir / "candidate_scores.csv.gz",
        "references": args.benchmark_dir / "candidate_references.csv.gz",
        "safe": args.transition_dir / "full_bioaware__transitions.csv.gz",
        "recall": args.transition_dir / "full_no_edge_gate__transitions.csv.gz",
        "library_mgf": args.library_mgf,
        "library_manifest": args.library_manifest,
        "library_embeddings": args.library_embeddings,
        "official": args.official_checkpoint,
        "library_integrity": (
            ROOT / "data/validation/mona_negative_library_chemical_integrity_v1/"
            "library_row_integrity.csv.gz"
        ),
        "candidate_features": (
            ROOT / "data/validation/bioaware_metdna3_external_negative_loso_ranker_v4_chemically_filtered/"
            "candidate_features.csv.gz"
        ),
    }
    for unit in UNITS:
        paths[f"unit_queries:{unit}"] = args.unit_dir / unit / "queries.csv.gz"
        paths[f"unit_tensors:{unit}"] = args.unit_dir / unit / "query_tensors.npz"
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    source_queries = pd.read_csv(paths["queries"])
    source_query_embedding = np.load(
        paths["query_embeddings"], allow_pickle=False,
    )["query_embedding"].astype(np.float32)
    if (len(source_queries) != len(source_query_embedding)
            or source_queries["query_id"].duplicated().any()):
        raise RuntimeError("frozen query embedding rows are not aligned")
    query_embedding_by_id = dict(zip(
        source_queries["query_id"].astype(str), source_query_embedding, strict=True,
    ))
    queries = source_queries.sort_values("query_id").reset_index(drop=True)
    scores = pd.read_csv(paths["scores"])
    references = pd.read_csv(paths["references"])
    safe = pd.read_csv(paths["safe"])
    recall = pd.read_csv(paths["recall"])
    if len(queries) != 548 or queries["query_id"].nunique() != 548:
        raise RuntimeError("B4 requires the frozen 548-query benchmark")
    query_ids = set(queries["query_id"].astype(str))
    for name, frame in (("scores", scores), ("references", references),
                        ("safe", safe), ("recall", recall)):
        if set(frame["query_id"].astype(str)) != query_ids:
            raise RuntimeError(f"{name} query set differs from the frozen benchmark")

    tensor_by_query: dict[str, np.ndarray] = {}
    for unit in UNITS:
        unit_queries = pd.read_csv(paths[f"unit_queries:{unit}"])
        tensors = np.load(paths[f"unit_tensors:{unit}"], allow_pickle=False)["query_tensor"]
        if len(unit_queries) != len(tensors):
            raise RuntimeError(f"{unit}: query/tensor length mismatch")
        if unit_queries["query_id"].duplicated().any():
            raise RuntimeError(f"{unit}: duplicate query id")
        for query_id, tensor in zip(unit_queries["query_id"].astype(str), tensors, strict=True):
            if query_id in tensor_by_query:
                raise RuntimeError(f"duplicate tensor for {query_id}")
            tensor_by_query[query_id] = np.asarray(tensor, dtype=np.float32)
    # The unit caches contain the pre-chemical-integrity 595-query universe;
    # the frozen benchmark is its audited 548-query subset.  A missing frozen
    # query is fatal, while the 47 rejected source tensors must remain unused.
    if missing_query_tensors := query_ids - set(tensor_by_query):
        raise RuntimeError(
            f"unit tensor union misses {len(missing_query_tensors)} frozen queries"
        )

    safe = safe.set_index("query_id").loc[queries["query_id"]].reset_index()
    recall = recall.set_index("query_id").loc[queries["query_id"]].reset_index()
    if not np.array_equal(safe["truth_candidate_id"].astype(str), queries["truth_ik14"].astype(str)):
        raise RuntimeError("safe transition truths do not align")
    if not np.array_equal(recall["truth_candidate_id"].astype(str), queries["truth_ik14"].astype(str)):
        raise RuntimeError("recall transition truths do not align")

    query_ptr = [0]
    molecule_id: list[str] = []
    molecule_best_row: list[int] = []
    molecule_formula: list[str] = []
    molecule_official_score: list[float] = []
    reference_ptr = [0]
    reference_rows: list[int] = []
    baseline_rank: list[int] = []
    needed_rows = sorted(set(references["library_row"].astype(int)))
    integrity = pd.read_csv(
        paths["library_integrity"],
        usecols=["library_row", "calculated_formula", "approved_m_h_reference"],
    )
    integrity = integrity[integrity["library_row"].astype(int).isin(needed_rows)].copy()
    approved = integrity["approved_m_h_reference"].astype(str).str.lower().eq("true")
    if len(integrity) != len(needed_rows) or not approved.all():
        raise RuntimeError("candidate references are not all chemically approved [M-H]- rows")
    formula_by_row = dict(zip(
        integrity["library_row"].astype(int),
        integrity["calculated_formula"].astype(str), strict=True,
    ))

    for query in queries.itertuples(index=False):
        qid = str(query.query_id)
        local_scores = scores[scores["query_id"].astype(str).eq(qid)].copy()
        local_scores["is_positive"] = local_scores["candidate_id"].astype(str).eq(str(query.truth_ik14))
        if int(local_scores["is_positive"].sum()) != 1 or len(local_scores) < 2:
            raise RuntimeError(f"{qid}: expected one truth and at least one negative")
        local_scores = pd.concat((
            local_scores[local_scores["is_positive"]],
            local_scores[~local_scores["is_positive"]].sort_values(
                ["spectral_score", "candidate_id"], ascending=[False, True]
            ),
        ), ignore_index=True)
        positive_score = float(local_scores.iloc[0]["spectral_score"])
        negative_score = local_scores.iloc[1:]["spectral_score"].to_numpy(float)
        baseline_rank.append(1 + int(np.sum(negative_score >= positive_score)))
        for candidate in local_scores.itertuples(index=False):
            cid = str(candidate.candidate_id)
            local_refs = references[
                references["query_id"].astype(str).eq(qid)
                & references["candidate_id"].astype(str).eq(cid)
            ]["library_row"].astype(int).drop_duplicates().sort_values().tolist()
            if not local_refs or int(candidate.best_library_row) not in local_refs:
                raise RuntimeError(f"{qid}/{cid}: missing best reference")
            best_formula = formula_by_row[int(candidate.best_library_row)]
            if not best_formula or best_formula.lower() == "nan":
                raise RuntimeError(f"{qid}/{cid}: reference formula is missing")
            if any(formula_by_row[int(row)] != best_formula for row in local_refs):
                raise RuntimeError(f"{qid}/{cid}: inconsistent reference formula")
            if bool(candidate.is_positive) and best_formula != str(query.truth_formula):
                raise RuntimeError(
                    f"{qid}/{cid}: truth/reference formula mismatch "
                    f"({query.truth_formula} != {best_formula})"
                )
            molecule_id.append(cid)
            molecule_best_row.append(int(candidate.best_library_row))
            molecule_formula.append(best_formula)
            molecule_official_score.append(float(candidate.spectral_score))
            reference_rows.extend(local_refs)
            reference_ptr.append(len(reference_rows))
        query_ptr.append(len(molecule_id))

    # Materialise the actual MoNA spectra addressed by ``library_row``.  The
    # prior implementation accidentally read the same integers from the
    # unrelated MassSpecGym HDF5, which makes exact official replay impossible.
    unique_reference_rows = np.asarray(sorted(set(reference_rows)), dtype=np.int64)
    reference_tensor = load_mona_reference_tensors(
        args.library_mgf, args.library_manifest, unique_reference_rows, 100,
    )
    frozen_library_embeddings = np.load(args.library_embeddings, mmap_mode="r")
    if frozen_library_embeddings.shape != (36663, 1024):
        raise RuntimeError(
            f"unexpected frozen MoNA embedding shape: {frozen_library_embeddings.shape}"
        )
    reference_official_embedding = np.asarray(
        frozen_library_embeddings[unique_reference_rows], dtype=np.float32,
    )
    norm_error = float(np.max(np.abs(
        np.linalg.norm(reference_official_embedding, axis=1) - 1.0
    )))
    if not np.isfinite(reference_official_embedding).all() or norm_error > 1e-5:
        raise RuntimeError("selected frozen MoNA embeddings are invalid")

    baseline_rank_array = np.asarray(baseline_rank, dtype=np.int16)
    if not np.array_equal(baseline_rank_array == 1, as_bool(safe["baseline_correct"])):
        raise RuntimeError("strict benchmark rank does not reproduce transition baseline")
    safe_corrected = as_bool(safe["corrected"])
    safe_introduced = as_bool(safe["introduced"])
    recall_corrected = as_bool(recall["corrected"])
    recall_introduced = as_bool(recall["introduced"])
    if safe_introduced.any() or int(safe_corrected.sum()) != 17:
        raise RuntimeError("safe route no longer reproduces 17/0")
    if int(recall_corrected.sum()) != 24 or int(recall_introduced.sum()) != 2:
        raise RuntimeError("high-recall route no longer reproduces 24/2")

    folds = np.asarray([
        stable_fold(str(value), args.folds, args.fold_seed)
        for value in queries["truth_formula"].astype(str)
    ], dtype=np.int8)
    for formula, group in queries.assign(fold=folds).groupby("truth_formula"):
        if group["fold"].nunique() != 1:
            raise RuntimeError(f"formula split failed for {formula}")

    # Re-cross-fit each training router without the current outer formula fold.
    # The archived 8-unit OOF transitions above remain an implementation-
    # compatibility gate only and never route B4 training gradients.
    candidate_features = pd.read_csv(paths["candidate_features"])
    candidate_features = candidate_features[
        candidate_features["query_id"].astype(str).isin(query_ids)
    ].copy()
    if candidate_features["query_id"].nunique() != len(queries):
        raise RuntimeError("candidate feature table does not cover the 548-query benchmark")
    fold_by_query = dict(zip(queries["query_id"].astype(str), folds, strict=True))
    candidate_features["formula_fold"] = candidate_features["query_id"].astype(str).map(fold_by_query)
    query_position = {value: index for index, value in enumerate(queries["query_id"].astype(str))}
    nested_routes: dict[str, dict[str, np.ndarray]] = {}
    nested_reports: dict[str, dict] = {}
    for arm, ablation_name in (
        ("safe", "full_bioaware"),
        ("recall", "full_no_edge_gate"),
    ):
        spec = ABLATIONS[ablation_name]
        corrected_by_outer = np.zeros((args.folds, len(queries)), dtype=bool)
        introduced_by_outer = np.zeros_like(corrected_by_outer)
        arm_report = {}
        for outer in range(args.folds):
            parts = []
            held_outer = candidate_features["formula_fold"].eq(outer)
            for inner in range(args.folds):
                if inner == outer:
                    continue
                test = candidate_features[candidate_features["formula_fold"].eq(inner)].copy()
                inner_truth = set(test["truth_candidate_id"].astype(str))
                train_router = candidate_features[
                    (~held_outer)
                    & (~candidate_features["formula_fold"].eq(inner))
                    & (~candidate_features["truth_candidate_id"].astype(str).isin(inner_truth))
                ].copy()
                if not len(test) or train_router["query_id"].nunique() < 100:
                    raise RuntimeError(f"router fold too small: arm={arm} outer={outer} inner={inner}")
                transition, _ = evaluate_fold(
                    train_router, test, f"formula_outer_{outer}_inner_{inner}",
                    features=spec["features"],
                    require_raw_step0_edge=spec["require_raw_step0_edge"],
                )
                parts.append(transition)
            crossfit = pd.concat(parts, ignore_index=True)
            expected = set(
                queries.loc[folds != outer, "query_id"].astype(str)
            )
            if (crossfit["query_id"].duplicated().any()
                    or set(crossfit["query_id"].astype(str)) != expected):
                raise RuntimeError(f"nested router coverage failed: arm={arm} outer={outer}")
            for row in crossfit.itertuples(index=False):
                position = query_position[str(row.query_id)]
                corrected_by_outer[outer, position] = bool(row.corrected)
                introduced_by_outer[outer, position] = bool(row.introduced)
            arm_report[str(outer)] = {
                "training_queries_crossfit_routed": int(len(crossfit)),
                "corrective": int(crossfit["corrected"].sum()),
                "known_harm": int(crossfit["introduced"].sum()),
                "held_formula_queries_never_scored_for_training": int(np.sum(folds == outer)),
            }
        nested_routes[arm] = {
            "corrected": corrected_by_outer,
            "introduced": introduced_by_outer,
        }
        nested_reports[arm] = arm_report

    arrays = {
        "query_id": np.asarray(queries["query_id"].astype(str).tolist(), dtype=str),
        "unit_id": np.asarray(queries["unit_id"].astype(str).tolist(), dtype=str),
        "query_ik14": np.asarray(queries["truth_ik14"].astype(str).tolist(), dtype=str),
        "query_formula": np.asarray(queries["truth_formula"].astype(str).tolist(), dtype=str),
        "query_tensor": np.stack([tensor_by_query[qid] for qid in queries["query_id"].astype(str)]),
        "query_official_embedding": np.stack([
            query_embedding_by_id[qid] for qid in queries["query_id"].astype(str)
        ]).astype(np.float32),
        "formula_fold": folds,
        "query_ptr": np.asarray(query_ptr, dtype=np.int64),
        "molecule_id": np.asarray(molecule_id),
        "molecule_best_row": np.asarray(molecule_best_row, dtype=np.int64),
        "molecule_formula": np.asarray(molecule_formula),
        "molecule_official_score": np.asarray(molecule_official_score, dtype=np.float32),
        "reference_ptr": np.asarray(reference_ptr, dtype=np.int64),
        "reference_rows": np.asarray(reference_rows, dtype=np.int64),
        "reference_tensor_rows": unique_reference_rows,
        "reference_tensor": reference_tensor,
        "reference_official_embedding": reference_official_embedding,
        "baseline_rank": baseline_rank_array,
        "safe_corrected": safe_corrected,
        "safe_introduced": safe_introduced,
        "recall_corrected": recall_corrected,
        "recall_introduced": recall_introduced,
        "safe_corrected_by_outer": nested_routes["safe"]["corrected"],
        "safe_introduced_by_outer": nested_routes["safe"]["introduced"],
        "recall_corrected_by_outer": nested_routes["recall"]["corrected"],
        "recall_introduced_by_outer": nested_routes["recall"]["introduced"],
    }
    args.output_dir.mkdir(parents=True)
    manifest = args.output_dir / "manifest.npz"
    np.savez_compressed(manifest, **arrays)
    report = {
        "status": "bioaware_b4_direct_manifest_frozen",
        "formal": False,
        "protocol": "opened-cohort engineering pilot; five truth-formula folds",
        "queries": int(len(queries)),
        "source_tensor_queries": int(len(tensor_by_query)),
        "chemically_rejected_source_tensors_unused": int(len(tensor_by_query) - len(queries)),
        "truth_identities": int(queries["truth_ik14"].nunique()),
        "truth_formulas": int(queries["truth_formula"].nunique()),
        "candidate_molecules": int(len(molecule_id)),
        "candidate_reference_rows": int(len(reference_rows)),
        "unique_candidate_reference_spectra": int(len(unique_reference_rows)),
        "official_errors": int(np.sum(baseline_rank_array != 1)),
        "routes": {
            "full_bioaware_safe": {
                "corrective": int(safe_corrected.sum()),
                "known_harm": int(safe_introduced.sum()),
            },
            "full_no_edge_high_recall": {
                "corrective": int(recall_corrected.sum()),
                "known_harm": int(recall_introduced.sum()),
            },
        },
        "nested_formula_crossfit_training_routes": nested_reports,
        "formula_fold_counts": {
            str(fold): int(np.sum(folds == fold)) for fold in range(args.folds)
        },
        "contracts": {
            "reaction_neighbours_are_positives": False,
            "positive_is_query_truth_identity": True,
            "P2b_used": False,
            "phenotype_used": False,
            "network_context_is_training_router_only": True,
            "outer_formula_fold_excluded_from_router_fit": True,
            "training_routes_are_inner_formula_crossfit": True,
            "inference_clean_spectrum_only": True,
            "candidate_reference_source": "MoNA-negative MGF row",
            "cross_database_row_aliasing_forbidden": True,
        },
        "provenance": {
            **{key: sha256_file(path) for key, path in paths.items()},
            "manifest_sha256": sha256_file(manifest),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "claim_limit": "Frozen opened-cohort training manifest; not an embedding result or independent validation.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
