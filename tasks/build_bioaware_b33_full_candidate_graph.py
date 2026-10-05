#!/usr/bin/env python
"""Materialise the complete 860-query candidate graph for B33 training.

The B32 canary evaluated only supervised truth/action pairs.  B33 needs every
candidate from the frozen six-source benchmark so that a shared-encoder update
can be scored for hidden corrections and hidden introductions.  This builder
does not fit a model and does not change any action label.
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

from audit_bioaware_b15_action_spectrum_support import (  # noqa: E402
    INTERNAL_SOURCES,
    KGM_SOURCE,
    ST_SOURCE,
    candidate_reference_map,
    load_mona_reference_tensors,
)


def sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b32-dir", type=Path, required=True)
    parser.add_argument(
        "--internal-candidates", type=Path,
        default=ROOT / "data/validation/bioaware_b3_reaction_coabundance_local_20260906/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--internal-references", type=Path,
        default=ROOT / "data/validation/bioaware_metdna3_external_negative_dreams_v2_chemically_filtered/candidate_references.csv.gz",
    )
    parser.add_argument(
        "--st-candidates", type=Path,
        default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--kgmn-candidates", type=Path,
        default=ROOT / "data/validation/bioaware_kgmn200std_hidden_seed_v1/candidate_features.csv.gz",
    )
    parser.add_argument(
        "--st-manifest-dir", type=Path,
        default=ROOT / "data/validation/bioaware_st001154_hilic_extension_manifest_v1",
    )
    parser.add_argument(
        "--st-evaluation-dir", type=Path,
        default=ROOT / "data/validation/bioaware_st001154_hilic_extension_v3_evaluation_v1",
    )
    parser.add_argument(
        "--kgmn-manifest-dir", type=Path,
        default=ROOT / "data/validation/bioaware_kgmn200std_confirmation_manifest_v2",
    )
    parser.add_argument("--ppm", type=float, default=10.0)
    parser.add_argument("--mona-mgf", type=Path, default=ROOT / "data/models/mona_neg_full.mgf")
    parser.add_argument(
        "--mona-manifest", type=Path,
        default=ROOT / "data/models/mona_neg_dreams_emb/manifest.csv",
    )
    parser.add_argument(
        "--mona-embeddings", type=Path,
        default=ROOT / "data/models/mona_neg_dreams_emb/embeddings.npy",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def collapse_repeated_candidate_rows(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Collapse evaluation repeats only when their official score is identical.

    KGMN contains three hidden-seed repeats for each physical query.  Those are
    three evaluation units, not three candidate molecules.  The shared encoder
    sees one physical spectrum, so its candidate graph must contain one row per
    candidate while retaining the repeated query units in B32.
    """
    required = {"query_id", "candidate_id", "spectral_score"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"{label} candidate table lacks {sorted(required - set(frame.columns))}")
    score = pd.to_numeric(frame["spectral_score"], errors="raise")
    if not np.isfinite(score.to_numpy(float)).all():
        raise RuntimeError(f"{label} contains a non-finite official score")
    keyed = frame.assign(spectral_score=score).groupby(
        ["query_id", "candidate_id"], sort=False, dropna=False,
    )["spectral_score"]
    spread = keyed.max() - keyed.min()
    if len(spread) and float(spread.max()) > 1e-9:
        raise RuntimeError(f"{label} repeated candidate rows disagree in official score")
    return frame.drop_duplicates(["query_id", "candidate_id"], keep="first").copy()


def main() -> None:
    args = arguments()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output: {args.output_dir}")
    b32_report_path = args.b32_dir / "report.json"
    b32_table_path = args.b32_dir / "direct_actions.csv.gz"
    b32_manifest_path = args.b32_dir / "direct_action_manifest.npz"
    required = (
        b32_report_path, b32_table_path, b32_manifest_path,
        args.internal_candidates, args.internal_references,
        args.st_candidates, args.kgmn_candidates,
        args.st_manifest_dir / "queries.csv.gz",
        args.st_manifest_dir / "candidate_references.csv.gz",
        args.st_evaluation_dir / "query_embeddings.npy",
        args.kgmn_manifest_dir / "candidate_scores.csv.gz",
        args.kgmn_manifest_dir / "queries.csv.gz",
        args.mona_mgf, args.mona_manifest, args.mona_embeddings,
    )
    for path in required:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(b32_report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "bioaware_b32_comprehensive_action_bank_complete"
        or report.get("pass_to_direct_gradient_canary") is not True
        or report["provenance"].get("manifest_sha256") != sha256(b32_manifest_path)
    ):
        raise RuntimeError("B33 requires an intact passing B32 bank")
    table = pd.read_csv(b32_table_path)
    with np.load(b32_manifest_path, allow_pickle=False) as handle:
        b32 = {name: handle[name] for name in handle.files}
    if len(table) != 860 or len(b32["query_id"]) != 860:
        raise RuntimeError("B32 query coverage changed")
    if not table["query_id"].astype(str).equals(pd.Series(b32["query_id"].astype(str))):
        raise RuntimeError("B32 table/manifest order mismatch")

    internal = pd.read_csv(args.internal_candidates)
    internal = internal.loc[
        internal["polarity"].eq("negative")
        & internal["query_id"].astype(str).isin(set(table["source_query_id"].astype(str)))
    ].copy()
    internal = collapse_repeated_candidate_rows(internal, "internal")
    st = collapse_repeated_candidate_rows(pd.read_csv(args.st_candidates), "ST001154")
    kgmn = collapse_repeated_candidate_rows(pd.read_csv(args.kgmn_candidates), "KGMN")
    internal_references = pd.read_csv(args.internal_references)
    st_references = pd.read_csv(args.st_manifest_dir / "candidate_references.csv.gz")
    if not {"query_id", "candidate_id", "library_row"}.issubset(internal_references):
        raise RuntimeError("internal reference table contract changed")
    if not {"query_id", "candidate_id", "library_row"}.issubset(st_references):
        raise RuntimeError("ST001154 reference table contract changed")
    internal_reference_map = {
        (str(query), str(candidate)): np.asarray(sorted(set(group["library_row"].astype(int))), dtype=np.int64)
        for (query, candidate), group in internal_references.groupby(
            ["query_id", "candidate_id"], sort=False,
        )
    }
    st_reference_map = {
        (str(query), str(candidate)): np.asarray(sorted(set(group["library_row"].astype(int))), dtype=np.int64)
        for (query, candidate), group in st_references.groupby(
            ["query_id", "candidate_id"], sort=False,
        )
    }
    mona_manifest = pd.read_csv(args.mona_manifest)
    if not {"inchikey", "precursor_mz"}.issubset(mona_manifest):
        raise RuntimeError("MoNA manifest contract changed")
    mona_ik14 = mona_manifest["inchikey"].astype(str).str[:14].to_numpy()
    mona_precursor_mz = pd.to_numeric(
        mona_manifest["precursor_mz"], errors="coerce",
    ).to_numpy(float)
    kgmn_queries = pd.read_csv(args.kgmn_manifest_dir / "queries.csv.gz")
    if not {"query_id", "mz"}.issubset(kgmn_queries):
        raise RuntimeError("KGMN query table contract changed")
    kgmn_query_mz = dict(zip(
        kgmn_queries["query_id"].astype(str),
        pd.to_numeric(kgmn_queries["mz"], errors="raise").astype(float),
        strict=True,
    ))
    lookup_args = argparse.Namespace(
        internal_candidates=args.internal_candidates,
        st_manifest_dir=args.st_manifest_dir,
        st_candidate_scores=args.st_candidates,
        st_evaluation_dir=args.st_evaluation_dir,
        mona_embeddings=args.mona_embeddings,
        kgmn_manifest_dir=args.kgmn_manifest_dir,
    )
    row_map = candidate_reference_map(lookup_args)

    candidate_ptr = [0]
    candidate_id: list[str] = []
    candidate_row: list[int] = []
    candidate_all_reference_ptr = [0]
    candidate_all_reference_row: list[int] = []
    official_score: list[float] = []
    baseline_rank: list[int] = []
    baseline_top1: list[str] = []
    candidate_count: list[int] = []
    action_candidate_position = np.full(len(table), -1, dtype=np.int64)
    for query_index, row in enumerate(table.itertuples(index=False)):
        source = str(row.source)
        raw_query = str(row.source_query_id)
        if source in INTERNAL_SOURCES:
            local = internal.loc[
                internal["source"].astype(str).eq(source)
                & internal["query_id"].astype(str).eq(raw_query),
                ["candidate_id", "spectral_score"],
            ].copy()
        elif source == ST_SOURCE:
            local = st.loc[
                st["query_id"].astype(str).eq(raw_query),
                ["candidate_id", "spectral_score"],
            ].copy()
        elif source == KGM_SOURCE:
            local = kgmn.loc[
                kgmn["query_id"].astype(str).eq(raw_query),
                ["candidate_id", "spectral_score"],
            ].copy()
        else:
            raise RuntimeError(f"unexpected B33 source: {source}")
        if local.empty or local["candidate_id"].astype(str).duplicated().any():
            raise RuntimeError(f"invalid candidate group: {row.query_id}")
        truth = str(row.truth_candidate_id)
        if int(local["candidate_id"].astype(str).eq(truth).sum()) != 1 or len(local) < 2:
            raise RuntimeError(f"query lacks one truth and a negative: {row.query_id}")
        local["candidate_id"] = local["candidate_id"].astype(str)
        local["spectral_score"] = pd.to_numeric(local["spectral_score"], errors="raise")
        if not np.isfinite(local["spectral_score"].to_numpy(float)).all():
            raise RuntimeError(f"non-finite official score: {row.query_id}")
        local = pd.concat((
            local.loc[local["candidate_id"].eq(truth)],
            local.loc[~local["candidate_id"].eq(truth)].sort_values(
                ["spectral_score", "candidate_id"], ascending=[False, True],
            ),
        ), ignore_index=True)
        truth_score = float(local.iloc[0]["spectral_score"])
        negative_scores = local.iloc[1:]["spectral_score"].to_numpy(float)
        rank = 1 + int(np.sum(negative_scores >= truth_score))
        top = local.sort_values(
            ["spectral_score", "candidate_id"], ascending=[False, True],
        ).iloc[0]
        if str(top["candidate_id"]) != str(row.baseline_candidate_id):
            raise RuntimeError(f"frozen baseline candidate mismatch: {row.query_id}")
        left = len(candidate_id)
        for item in local.itertuples(index=False):
            key = (source, raw_query, str(item.candidate_id))
            if key not in row_map:
                raise RuntimeError(f"candidate lacks exact reference row: {key}")
            if source in INTERNAL_SOURCES:
                all_rows = internal_reference_map.get((raw_query, str(item.candidate_id)))
            elif source == ST_SOURCE:
                all_rows = st_reference_map.get((raw_query, str(item.candidate_id)))
            else:
                if raw_query not in kgmn_query_mz:
                    raise RuntimeError(f"KGMN query mass unavailable: {raw_query}")
                query_mz = float(kgmn_query_mz[raw_query])
                ppm_error = np.abs(mona_precursor_mz - query_mz) / query_mz * 1e6
                all_rows = np.flatnonzero(
                    (mona_ik14 == str(item.candidate_id))
                    & np.isfinite(ppm_error)
                    & (ppm_error <= args.ppm)
                ).astype(np.int64)
            if all_rows is None or len(all_rows) == 0:
                raise RuntimeError(f"candidate has no complete reference set: {key}")
            all_rows = np.asarray(sorted(set(np.asarray(all_rows, dtype=np.int64))), dtype=np.int64)
            if int(row_map[key]) not in set(all_rows.tolist()):
                raise RuntimeError(f"official best reference missing from complete set: {key}")
            candidate_id.append(str(item.candidate_id))
            candidate_row.append(int(row_map[key]))
            official_score.append(float(item.spectral_score))
            candidate_all_reference_row.extend(int(value) for value in all_rows)
            candidate_all_reference_ptr.append(len(candidate_all_reference_row))
        action_id = str(row.training_candidate_id)
        matches = np.flatnonzero(local["candidate_id"].astype(str).to_numpy() == action_id)
        if len(matches) != 1:
            raise RuntimeError(f"B32 action candidate outside full graph: {row.query_id}/{action_id}")
        action_candidate_position[query_index] = left + int(matches[0])
        candidate_ptr.append(len(candidate_id))
        candidate_count.append(len(local))
        baseline_rank.append(rank)
        baseline_top1.append(str(top["candidate_id"]))

    unique_rows = np.asarray(sorted(set(candidate_all_reference_row)), dtype=np.int64)
    reference_tensor = load_mona_reference_tensors(
        args.mona_mgf, args.mona_manifest, unique_rows,
    )
    row_position = {int(value): index for index, value in enumerate(unique_rows)}
    candidate_reference_position = np.asarray(
        [row_position[int(value)] for value in candidate_row], dtype=np.int64,
    )
    candidate_all_reference_position = np.asarray(
        [row_position[int(value)] for value in candidate_all_reference_row], dtype=np.int64,
    )
    candidate_ptr_array = np.asarray(candidate_ptr, dtype=np.int64)
    truth_position = candidate_ptr_array[:-1].copy()
    body = dict(b32)
    body.update({
        "candidate_ptr": candidate_ptr_array,
        "candidate_id": np.asarray(candidate_id, dtype=str),
        "candidate_reference_row": np.asarray(candidate_row, dtype=np.int64),
        "candidate_reference_position": candidate_reference_position,
        "candidate_all_reference_ptr": np.asarray(candidate_all_reference_ptr, dtype=np.int64),
        "candidate_all_reference_row": np.asarray(candidate_all_reference_row, dtype=np.int64),
        "candidate_all_reference_position": candidate_all_reference_position,
        "candidate_reference_rows_unique": unique_rows,
        "candidate_reference_tensor": reference_tensor.astype(np.float32),
        "candidate_official_score": np.asarray(official_score, dtype=np.float32),
        "truth_candidate_position": truth_position.astype(np.int64),
        "action_candidate_position": action_candidate_position.astype(np.int64),
        "baseline_rank_full_graph": np.asarray(baseline_rank, dtype=np.int64),
        "baseline_top1_full_graph": np.asarray(baseline_top1, dtype=str),
    })
    observed_correct = np.asarray(baseline_rank) == 1
    if not np.array_equal(observed_correct, np.asarray(b32["baseline_correct"], dtype=bool)):
        raise RuntimeError("full graph does not reproduce B32 baseline correctness")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    graph_path = args.output_dir / "full_candidate_graph.npz"
    np.savez_compressed(graph_path, **body)
    gates = {
        "queries_860": len(table) == 860,
        "candidate_groups_all_have_truth_and_negative": bool(min(candidate_count) >= 2),
        "baseline_correctness_exact": bool(
            np.array_equal(observed_correct, np.asarray(b32["baseline_correct"], dtype=bool))
        ),
        "all_candidates_have_exact_spectra": bool(
            len(candidate_id) == len(candidate_reference_position)
            and len(candidate_all_reference_ptr) == len(candidate_id) + 1
            and np.all(np.diff(np.asarray(candidate_all_reference_ptr)) >= 1)
        ),
        "all_B32_actions_inside_candidate_graph": bool(np.all(action_candidate_position >= 0)),
        "corrective_63_safety_14": bool(
            int(np.sum(b32["direct_corrective"])) == 63
            and int(np.sum(b32["direct_safety"])) == 14
        ),
    }
    output_report = {
        "status": "bioaware_b33_full_candidate_graph_complete",
        "formal": True,
        "queries": 860,
        "physical_queries": int(table.drop_duplicates(["source", "physical_query_id"]).shape[0]),
        "candidate_rows": int(len(candidate_id)),
        "candidate_identities": int(pd.Series(candidate_id).nunique()),
        "reference_spectra": int(len(unique_rows)),
        "candidate_reference_links": int(len(candidate_all_reference_row)),
        "candidate_count": {
            "minimum": int(np.min(candidate_count)),
            "median": float(np.median(candidate_count)),
            "p90": float(np.quantile(candidate_count, 0.9)),
            "maximum": int(np.max(candidate_count)),
        },
        "official": {
            "recall1": float(np.mean(observed_correct)),
            "mrr": float(np.mean(1.0 / np.asarray(baseline_rank, dtype=float))),
            "errors": int(np.sum(~observed_correct)),
        },
        "gates": gates,
        "pass_to_full_graph_bridge": bool(all(gates.values())),
        "contracts": {
            "model_fitted": False,
            "all_candidate_spectra_materialised": True,
            "action_outcomes_not_used_to_build_candidate_graph": True,
            "P2b_used": False,
            "phenotype_used": False,
        },
        "provenance": {
            "B32_report_sha256": sha256(b32_report_path),
            "B32_manifest_sha256": sha256(b32_manifest_path),
            "internal_candidates_sha256": sha256(args.internal_candidates),
            "internal_references_sha256": sha256(args.internal_references),
            "st_candidates_sha256": sha256(args.st_candidates),
            "kgmn_candidates_sha256": sha256(args.kgmn_candidates),
            "mona_manifest_sha256": sha256(args.mona_manifest),
            "graph_sha256": sha256(graph_path),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": "A complete opened candidate graph; no shared-embedding performance claim.",
    }
    atomic_json(args.output_dir / "report.json", output_report)
    print(json.dumps(output_report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
