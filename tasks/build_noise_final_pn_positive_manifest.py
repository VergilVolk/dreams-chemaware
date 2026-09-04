"""Build the leakage-free positive arm for direct P/N/S embedding fine-tuning.

The positive arm contains only real, same-identity, same-adduct spectrum pairs
whose acquisition conditions differ (instrument differs, or finite collision
energies differ by at least 10 units).  Pairs are recovered from the frozen
strict-10ppm candidate graph, so every query keeps its real deployment
candidate set.  No teacher score, model outcome, P2b feature, or post-hoc
action label is written to this manifest.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

from noise_final_core import CandidateGraph, json_dump, sha256_file, stable_fold  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=ROOT / "data/validation/g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--p3-allow", type=Path, default=ROOT / "data/validation/g8r_p3_test/p3_p2_allowed_training_ik14.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/g8r_noise_final_pn_positive_manifest")
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--minimum-ce-delta", type=float, default=10.0)
    parser.add_argument("--maximum-pairs-per-query", type=int, default=12)
    parser.add_argument("--max-queries", type=int, default=0, help="Diagnostic only; zero means the full graph.")
    return parser.parse_args()


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ], dtype=object)


def read_rows(handle: h5py.File, name: str, rows: np.ndarray) -> np.ndarray:
    """Read arbitrary HDF5 rows while obeying h5py's sorted-index contract."""
    rows = np.asarray(rows, dtype=np.int64)
    unique, inverse = np.unique(rows, return_inverse=True)
    values = handle[name][unique]
    return np.asarray(values)[inverse]


def load_allowed(path: Path) -> set[str]:
    body = json.loads(path.read_text(encoding="utf-8"))
    if body.get("p3_query_overlap", 0) != 0:
        raise RuntimeError("P3 allow-list does not certify zero query overlap")
    primary = body.get("real_train_primary", {})
    values = primary.get("ik14") if isinstance(primary, dict) else None
    if not isinstance(values, list) or not values:
        raise RuntimeError("P3 allow-list is missing real_train_primary.ik14")
    return set(map(str, values))


def main() -> None:
    args = arguments()
    required = [args.graph, args.data, args.p3_allow]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite frozen P-arm manifest: {args.output_dir}")
    if args.maximum_pairs_per_query < 1:
        raise ValueError("maximum-pairs-per-query must be positive")

    graph = CandidateGraph(args.graph)
    allowed = load_allowed(args.p3_allow)
    n_queries = graph.n_queries if args.max_queries <= 0 else min(graph.n_queries, args.max_queries)

    positive_rows: list[np.ndarray] = []
    for query in range(n_queries):
        pair_slice, candidate_rows, local_ptr, _ = graph.query_block(query)
        del pair_slice
        positive_rows.append(np.asarray(candidate_rows[: int(local_ptr[1])], dtype=np.int64))
    all_rows = np.unique(np.concatenate([
        graph.query_row[:n_queries], *positive_rows,
    ])).astype(np.int64)

    with h5py.File(args.data, "r") as handle:
        required_fields = {
            "INCHIKEY", "FORMULA", "fold", "SIMULATION_CHALLENGE",
            "INSTRUMENT_TYPE", "COLLISION_ENERGY", "adduct",
        }
        absent = required_fields - set(handle.keys())
        if absent:
            raise RuntimeError(f"HDF5 missing P-arm metadata: {sorted(absent)}")
        local = {int(row): index for index, row in enumerate(all_rows)}
        ik = decode(read_rows(handle, "INCHIKEY", all_rows))
        formula = decode(read_rows(handle, "FORMULA", all_rows))
        fold = decode(read_rows(handle, "fold", all_rows))
        simulation = decode(read_rows(handle, "SIMULATION_CHALLENGE", all_rows))
        instrument = decode(read_rows(handle, "INSTRUMENT_TYPE", all_rows))
        ce = np.asarray(read_rows(handle, "COLLISION_ENERGY", all_rows), dtype=np.float64)
        adduct = decode(read_rows(handle, "adduct", all_rows))

    records: list[dict] = []
    relation_counts: dict[str, int] = {}
    rejected = {"not_allowed": 0, "not_real_train": 0, "metadata_mismatch": 0, "same_condition": 0}
    for query in range(n_queries):
        query_row = int(graph.query_row[query])
        q = local[query_row]
        qik14 = str(ik[q])[:14]
        if qik14 not in allowed:
            rejected["not_allowed"] += 1
            continue
        if fold[q] != "train" or str(simulation[q]).lower() != "false":
            rejected["not_real_train"] += 1
            continue
        candidates: list[dict] = []
        for positive_row in positive_rows[query]:
            positive_row = int(positive_row)
            if positive_row == query_row:
                continue
            p = local[positive_row]
            if fold[p] != "train" or str(simulation[p]).lower() != "false":
                continue
            if str(ik[p])[:14] != qik14 or str(adduct[p]) != str(adduct[q]):
                rejected["metadata_mismatch"] += 1
                continue
            inst_q, inst_p = str(instrument[q]), str(instrument[p])
            inst_known = inst_q.lower() not in {"", "nan", "none"} and inst_p.lower() not in {"", "nan", "none"}
            inst_diff = inst_known and inst_q != inst_p
            ce_known = np.isfinite(ce[q]) and np.isfinite(ce[p])
            ce_delta = abs(float(ce[q]) - float(ce[p])) if ce_known else np.nan
            if inst_diff:
                relation = "cross_instrument"
            elif ce_known and ce_delta >= args.minimum_ce_delta:
                relation = "same_instrument_cross_ce"
            else:
                rejected["same_condition"] += 1
                continue
            candidates.append({
                "query_index": query,
                "query_row": query_row,
                "positive_row": positive_row,
                "query_ik14": qik14,
                "query_formula": str(graph.query_formula[query]),
                "formula_fold": stable_fold(str(graph.query_formula[query]), 5, args.formula_fold_seed),
                "relation": relation,
                "adduct": str(adduct[q]),
                "query_instrument": inst_q,
                "positive_instrument": inst_p,
                "query_ce": None if not np.isfinite(ce[q]) else float(ce[q]),
                "positive_ce": None if not np.isfinite(ce[p]) else float(ce[p]),
                "ce_delta": None if not np.isfinite(ce_delta) else float(ce_delta),
            })
        candidates.sort(key=lambda item: (
            item["relation"] != "cross_instrument",
            -(item["ce_delta"] if item["ce_delta"] is not None else -1.0),
            item["positive_row"],
        ))
        records.extend(candidates[: args.maximum_pairs_per_query])

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise RuntimeError("no strict cross-condition positive pairs were recovered")
    if frame.duplicated(["query_index", "positive_row"]).any():
        raise RuntimeError("duplicate directed positive pair")
    if not frame["query_ik14"].isin(allowed).all():
        raise RuntimeError("P3-disallowed identity entered P-arm")
    if not np.array_equal(
        frame["formula_fold"].to_numpy(np.int8),
        frame["query_formula"].map(lambda value: stable_fold(str(value), 5, args.formula_fold_seed)).to_numpy(np.int8),
    ):
        raise RuntimeError("formula fold drift in P-arm")

    relation_counts = frame["relation"].value_counts().astype(int).to_dict()
    formal = args.max_queries <= 0
    gates = {
        "pairs_ge_5000": len(frame) >= 5000,
        "identities_ge_750": frame["query_ik14"].nunique() >= 750,
        "formulas_ge_500": frame["query_formula"].nunique() >= 500,
        "every_fold_ge_100_pairs": bool((frame.groupby("formula_fold").size() >= 100).all()),
        "all_relations_strict": bool(frame["relation"].isin({"cross_instrument", "same_instrument_cross_ce"}).all()),
    }
    if formal and not all(gates.values()):
        raise RuntimeError(f"formal P-arm cardinality/contract gate failed: {gates}")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = args.output_dir / "positive_pairs.csv.gz"
    frame.to_csv(manifest_path, index=False, compression="gzip")
    report = {
        "status": "noise_final_pn_positive_manifest_complete",
        "formal": formal,
        "pairs": int(len(frame)),
        "query_rows": int(frame["query_row"].nunique()),
        "identities": int(frame["query_ik14"].nunique()),
        "formulas": int(frame["query_formula"].nunique()),
        "relations": relation_counts,
        "formula_folds": frame["formula_fold"].value_counts().sort_index().astype(int).to_dict(),
        "rejected": rejected,
        "gates": gates,
        "pass_to_pn_training": bool(all(gates.values())),
        "contracts": {
            "real_train_spectra_only": True,
            "same_identity_same_adduct": True,
            "strict_cross_condition_only": True,
            "P3_identity_overlap": 0,
            "teacher": "forbidden",
            "P2b": "forbidden",
            "outcome_columns": "absent",
        },
        "provenance": {
            "graph_sha256": sha256_file(args.graph),
            "hdf5_sha256": sha256_file(args.data),
            "p3_allow_sha256": sha256_file(args.p3_allow),
            "positive_pairs_sha256": sha256_file(manifest_path),
            "script_sha256": sha256_file(Path(__file__)),
        },
        "parameters": {
            key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
        },
        "claim_limit": "A frozen positive-arm training manifest; no embedding gain is claimed here.",
    }
    json_dump(args.output_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
