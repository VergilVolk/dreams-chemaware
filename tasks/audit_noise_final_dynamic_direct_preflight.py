"""Fail-closed artifact and split audit for dynamic direct noise fine-tuning.

This stage is intentionally model-free: it validates every input, alignment,
formula split and provenance edge before a 117M-parameter model can be loaded.
It never consumes P3 or P2b and never interprets historical held outcomes as
training labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from noise_final_dynamic_direct_core import N_CELLS, validate_n_cells


ROOT = Path(__file__).resolve().parents[1]
GRAPH_ARRAYS = {
    "feature_names", "features", "pair_candidate_row", "query_ptr",
    "molecule_ptr", "molecule_label", "molecule_ik14", "molecule_formula",
    "molecule_mces_grade", "query_row", "query_ik14", "query_formula",
    "query_has_near",
}
N_KEYS = ["query_index", "selector", "attenuation", "step"]
FORBIDDEN_TRAINING_COLUMNS = {
    "corrected", "introduced", "target_rank", "target_margin",
    "positive_guided_oracle_recoverable", "transfer_oracle_recoverable",
    "new_beyond_pn", "new_beyond_pn_and_intensity_matrix",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    validation = ROOT / "data/validation"
    parser.add_argument("--graph", type=Path, default=validation / "g8r_error_atlas_listwise_cache.npz")
    parser.add_argument("--embedding-cache", type=Path, default=validation / "g8r_p2_official_embeddings.npz")
    parser.add_argument("--data", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--official-checkpoint", type=Path, default=ROOT / "data/e1/official_embedding_slim.pt")
    parser.add_argument("--architecture-checkpoint", type=Path, default=ROOT / "dreams/models/pretrained/ssl_model_server.pt")
    parser.add_argument("--r0-dir", type=Path, default=validation / "g8r_noise_final_r0_faithful_s3a")
    parser.add_argument("--l0-dir", type=Path, default=validation / "g8r_noise_final_l0_action_learnability_ledger")
    parser.add_argument("--l1-dir", type=Path, default=validation / "g8r_noise_final_l1_clean_action_learnability")
    parser.add_argument("--token-dir", type=Path, default=validation / "g8r_noise_final_f1_full_tokens")
    parser.add_argument("--p-intensity-dir", type=Path, default=validation / "g8r_noise_final_positive_guided_matrix")
    parser.add_argument("--p-transfer-dir", type=Path, default=validation / "g8r_noise_final_positive_peak_transfer")
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--formula-fold-seed", type=int, default=20260825)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hash-large-files", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_fold(value: str, folds: int, seed: int) -> int:
    payload = f"{seed}|{value}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % folds


def load_report(path: Path, status: str) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != status or report.get("formal") is not True:
        raise RuntimeError(f"wrong status/formal contract in {path}: {report.get('status')}")
    return report


def p2b_forbidden(report: dict[str, object]) -> bool:
    contracts = report.get("contracts", {})
    return bool(
        contracts.get("P2b") == "forbidden"
        or contracts.get("P2b_forbidden") is True
        or report.get("feature_contract", {}).get("P2b") == "forbidden"
    )


def inventory(
    paths: list[Path], hash_large: bool, known_hashes: dict[Path, str] | None = None,
) -> list[dict[str, object]]:
    known_hashes = known_hashes or {}
    rows: list[dict[str, object]] = []
    for path in paths:
        size = int(path.stat().st_size)
        rows.append({
            "path": str(path.resolve()),
            "bytes": size,
            "sha256": (
                known_hashes[path]
                if path in known_hashes
                else sha256_file(path) if hash_large or size < 256 * 1024 * 1024 else None
            ),
        })
    return rows


def main() -> None:
    args = arguments()
    if args.outer_fold not in range(5):
        raise ValueError("outer-fold must be in 0..4")
    required = [
        args.graph, args.embedding_cache, args.data, args.official_checkpoint,
        args.architecture_checkpoint, args.initial_checkpoint,
        args.initial_checkpoint.parent / "decision.json",
        args.r0_dir / "report.json", args.r0_dir / "training_actions.csv.gz",
        args.l0_dir / "report.json", args.l0_dir / "action_labels.csv.gz",
        args.l1_dir / "report.json", args.l1_dir / "action_oof_predictions.csv.gz",
        args.l1_dir / "clean_query_features.npz",
        args.token_dir / "report.json", args.token_dir / "rows.npy",
        args.token_dir / "tokens_f16.npy", args.token_dir / "mz_f32.npy",
        args.token_dir / "intensity_f32.npy", args.token_dir / "valid.npy",
        args.p_intensity_dir / "report.json", args.p_intensity_dir / "action_manifest.csv.gz",
        args.p_intensity_dir / "matrix_results.h5",
        args.p_transfer_dir / "report.json", args.p_transfer_dir / "action_manifest.csv.gz",
        args.p_transfer_dir / "matrix_results.h5",
    ]
    missing = [str(path.resolve()) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("dynamic-direct preflight missing exact files:\n" + "\n".join(missing))
    if args.output_dir.exists():
        raise RuntimeError(f"refusing to overwrite preflight output: {args.output_dir}")

    with np.load(args.graph, allow_pickle=True) as body:
        absent = GRAPH_ARRAYS - set(body.files)
        if absent:
            raise RuntimeError(f"candidate graph misses arrays: {sorted(absent)}")
        query_ptr = np.asarray(body["query_ptr"], dtype=np.int64)
        molecule_ptr = np.asarray(body["molecule_ptr"], dtype=np.int64)
        molecule_label = np.asarray(body["molecule_label"], dtype=np.int8)
        pair_candidate_row = np.asarray(body["pair_candidate_row"], dtype=np.int64)
        query_row = np.asarray(body["query_row"], dtype=np.int64)
        query_ik14 = np.asarray(body["query_ik14"], dtype=str)
        query_formula = np.asarray(body["query_formula"], dtype=str)
    n_queries = len(query_ptr) - 1
    if n_queries != 23876 or len(query_row) != n_queries:
        raise RuntimeError(f"formal graph must contain 23,876 aligned queries, observed {n_queries}")
    if query_ptr[0] != 0 or query_ptr[-1] != len(molecule_label):
        raise RuntimeError("query pointer does not span molecule labels")
    if molecule_ptr[0] != 0 or molecule_ptr[-1] != len(pair_candidate_row):
        raise RuntimeError("molecule pointer does not span candidate spectra")
    for left, right in zip(query_ptr[:-1], query_ptr[1:]):
        labels = molecule_label[int(left):int(right)]
        if len(labels) < 2 or labels[0] != 1 or int(labels.sum()) != 1:
            raise RuntimeError("candidate graph violates unique-positive-first molecule contract")

    with np.load(args.embedding_cache) as body:
        embedding_rows = np.asarray(body["rows"], dtype=np.int64)
        embeddings = body["embeddings"]
        embedding_shape = list(embeddings.shape)
    reachable = np.unique(np.concatenate([query_row, pair_candidate_row]))
    if len(embedding_rows) != len(np.unique(embedding_rows)):
        raise RuntimeError("official embedding cache has duplicate rows")
    if set(map(int, reachable)) - set(map(int, embedding_rows)):
        raise RuntimeError("official embedding cache does not cover the full reachable graph")

    r0_report = load_report(args.r0_dir / "report.json", "noise_final_r0_faithful_s3a_manifest_complete")
    l0_report = load_report(args.l0_dir / "report.json", "noise_final_l0_action_learnability_ledger_complete")
    l1_report = load_report(args.l1_dir / "report.json", "noise_final_l1_clean_action_learnability_complete")
    token_report = load_report(args.token_dir / "report.json", "noise_final_f1_full_token_cache_complete")
    intensity_report = load_report(args.p_intensity_dir / "report.json", "noise_final_positive_guided_matrix_complete")
    transfer_report = load_report(args.p_transfer_dir / "report.json", "noise_final_positive_peak_transfer_complete")
    for name, report in (("R0", r0_report), ("L0", l0_report), ("L1", l1_report),
                         ("P-intensity", intensity_report), ("P-transfer", transfer_report)):
        if not p2b_forbidden(report):
            raise RuntimeError(f"{name} does not explicitly forbid P2b")
    graph_hash = sha256_file(args.graph)
    data_hash = sha256_file(args.data)
    initial_hash = sha256_file(args.initial_checkpoint)
    embedding_hash = sha256_file(args.embedding_cache)
    official_hash = sha256_file(args.official_checkpoint)
    if l0_report.get("provenance", {}).get("graph_sha256") != graph_hash:
        raise RuntimeError("L0 candidate graph provenance differs from the requested graph")
    if l0_report.get("provenance", {}).get("hdf5_sha256") != data_hash:
        raise RuntimeError("L0 HDF5 provenance differs from the requested data")
    if l0_report.get("provenance", {}).get("clean_checkpoint_sha256") != initial_hash:
        raise RuntimeError(
            "initial checkpoint is not the exact clean geometry used to define L0/L1 action labels"
        )
    if l1_report.get("provenance", {}).get("l0_report_sha256") != sha256_file(args.l0_dir / "report.json"):
        raise RuntimeError("L1 does not descend from the supplied L0 report")
    if l0_report.get("provenance", {}).get("r0_actions_sha256") != sha256_file(args.r0_dir / "training_actions.csv.gz"):
        raise RuntimeError("L0 does not descend from the supplied R0 action ledger")
    if l1_report.get("provenance", {}).get("l0_labels_sha256") != sha256_file(args.l0_dir / "action_labels.csv.gz"):
        raise RuntimeError("L1 does not descend from the supplied L0 action labels")
    token_provenance = token_report.get("provenance", {})
    if (token_provenance.get("graph_sha256") != graph_hash
            or token_provenance.get("hdf5_sha256") != data_hash
            or token_provenance.get("official_checkpoint_sha256") != official_hash):
        raise RuntimeError("contextual token cache provenance differs from graph/data/official checkpoint")
    for name, report in (("P-intensity", intensity_report), ("P-transfer", transfer_report)):
        provenance = report.get("provenance", {})
        if (provenance.get("graph_sha256") != graph_hash
                or provenance.get("embedding_cache_sha256") != embedding_hash
                or provenance.get("hdf5_sha256") != data_hash):
            raise RuntimeError(f"{name} provenance differs from the frozen graph geometry")

    actions = pd.read_csv(args.r0_dir / "training_actions.csv.gz", low_memory=False)
    labels = pd.read_csv(args.l0_dir / "action_labels.csv.gz", low_memory=False)
    predictions = pd.read_csv(args.l1_dir / "action_oof_predictions.csv.gz", low_memory=False)
    for name, frame in (("R0", actions), ("L0", labels), ("L1", predictions)):
        missing_columns = set(N_KEYS + ["query_row", "query_ik14", "query_formula", "formula_fold"]) - set(frame.columns)
        if missing_columns:
            raise RuntimeError(f"{name} N ledger misses columns: {sorted(missing_columns)}")
        if frame.duplicated(N_KEYS).any():
            raise RuntimeError(f"{name} N ledger contains duplicate action keys")
    validate_n_cells(actions[["selector", "attenuation", "step"]])
    canonical = actions[N_KEYS].sort_values(N_KEYS, kind="stable").reset_index(drop=True)
    for name, frame in (("L0", labels), ("L1", predictions)):
        observed = frame[N_KEYS].sort_values(N_KEYS, kind="stable").reset_index(drop=True)
        if not canonical.equals(observed):
            raise RuntimeError(f"{name} does not align one-to-one with R0 actions")
    if len(actions) != int(r0_report.get("training_action_rows", -1)):
        raise RuntimeError("R0 action count differs from its report")
    if len(labels) != int(l0_report.get("actions", -1)) or len(predictions) != len(labels):
        raise RuntimeError("L0/L1 action count is incomplete")
    prediction_columns = {"clean_pred_gain", "clean_p_positive", "clean_p_harmful"}
    if prediction_columns - set(predictions.columns):
        raise RuntimeError("L1 clean-visible OOF predictions are incomplete")
    if not np.isfinite(predictions[list(prediction_columns)].to_numpy(float)).all():
        raise RuntimeError("L1 clean-visible OOF predictions contain non-finite values")
    expected_folds = np.asarray([
        stable_fold(value, 5, args.formula_fold_seed) for value in actions["query_formula"].astype(str)
    ], dtype=np.int8)
    if not np.array_equal(expected_folds, actions["formula_fold"].to_numpy(np.int8)):
        raise RuntimeError("R0 formula folds do not reproduce from formula and frozen seed")
    if not np.array_equal(actions["query_row"].to_numpy(np.int64), query_row[actions["query_index"].to_numpy(np.int64)]):
        raise RuntimeError("R0 query rows disagree with the candidate graph")

    token_rows = np.load(args.token_dir / "rows.npy", mmap_mode="r")
    token_arrays = {
        name: np.load(args.token_dir / filename, mmap_mode="r")
        for name, filename in (("tokens", "tokens_f16.npy"), ("mz", "mz_f32.npy"),
                               ("intensity", "intensity_f32.npy"), ("valid", "valid.npy"))
    }
    if any(array.shape[:2] != (len(token_rows), int(token_report["tokens_per_spectrum"]))
           for array in token_arrays.values()):
        raise RuntimeError("contextual token arrays have inconsistent leading dimensions")
    if set(map(int, reachable)) - set(map(int, token_rows)):
        raise RuntimeError("contextual token cache does not cover the full reachable graph")

    p_intensity = pd.read_csv(args.p_intensity_dir / "action_manifest.csv.gz", low_memory=False)
    p_transfer = pd.read_csv(args.p_transfer_dir / "action_manifest.csv.gz", low_memory=False)
    for name, frame in (("P-intensity", p_intensity), ("P-transfer", p_transfer)):
        required_columns = {"query_index", "query_row", "query_ik14", "query_formula"}
        if required_columns - set(frame.columns) or len(frame) != n_queries or frame["query_index"].duplicated().any():
            raise RuntimeError(f"{name} manifest is not one-to-one with the formal graph")
        ordered = frame.sort_values("query_index", kind="stable").reset_index(drop=True)
        if not np.array_equal(ordered["query_index"].to_numpy(np.int64), np.arange(n_queries)):
            raise RuntimeError(f"{name} manifest query order is incomplete")
        if not np.array_equal(ordered["query_row"].to_numpy(np.int64), query_row):
            raise RuntimeError(f"{name} manifest rows disagree with graph")
        if not np.array_equal(ordered["query_ik14"].astype(str).to_numpy(), query_ik14):
            raise RuntimeError(f"{name} manifest identities disagree with graph")
        if not np.array_equal(ordered["query_formula"].astype(str).to_numpy(), query_formula):
            raise RuntimeError(f"{name} manifest formulas disagree with graph")
    if "positive_reference_rows" not in p_intensity.columns:
        raise RuntimeError("P-intensity manifest lacks real same-identity reference rows")
    historical_columns = sorted((set(p_intensity.columns) | set(p_transfer.columns)) & FORBIDDEN_TRAINING_COLUMNS)

    matrix_shapes: dict[str, object] = {}
    import h5py  # Loaded only after every exact input path has passed.
    for name, path, expected_cells in (
        ("P-intensity", args.p_intensity_dir / "matrix_results.h5", 24),
        ("P-transfer", args.p_transfer_dir / "matrix_results.h5", 18),
    ):
        with h5py.File(path, "r") as handle:
            shape = tuple(map(int, handle["result_rank"].shape))
            if shape != (n_queries, expected_cells):
                raise RuntimeError(f"{name} result matrix shape drifted: {shape}")
            matrix_shapes[name] = {"result_rank": list(shape), "attributes": sorted(handle.attrs.keys())}

    initial_decision = json.loads((args.initial_checkpoint.parent / "decision.json").read_text(encoding="utf-8"))
    configuration = initial_decision.get("configuration", {})
    if initial_decision.get("status") != "noise_final_e4a_direct_augmentation_complete" or initial_decision.get("formal") is not True:
        raise RuntimeError("initial checkpoint decision is not a formal mature E4-A result")
    if int(configuration.get("outer_fold", -1)) != args.outer_fold:
        raise RuntimeError("initial checkpoint and requested outer fold do not match")
    if int(configuration.get("formula_fold_seed", -1)) != args.formula_fold_seed:
        raise RuntimeError("initial checkpoint formula-fold seed does not match")
    if configuration.get("causal_arm") != "clean_duplicate":
        raise RuntimeError("dynamic-direct initialization must be the L0 clean-duplicate geometry")

    held_formulas = sorted(set(query_formula[np.asarray([
        stable_fold(value, 5, args.formula_fold_seed) == args.outer_fold for value in query_formula
    ], dtype=bool)]))
    held_mask = np.isin(query_formula, held_formulas)
    if set(query_formula[~held_mask]) & set(query_formula[held_mask]):
        raise RuntimeError("outer train and held formula sets overlap")

    report = {
        "status": "noise_final_dynamic_direct_preflight_complete",
        "formal": True,
        "outer_formula_fold": int(args.outer_fold),
        "formula_fold_seed": int(args.formula_fold_seed),
        "graph": {
            "queries": int(n_queries), "reachable_spectra": int(len(reachable)),
            "held_queries": int(held_mask.sum()), "held_formulas": int(len(held_formulas)),
            "embedding_shape": embedding_shape,
        },
        "N": {
            "actions": int(len(actions)), "queries": int(actions["query_index"].nunique()),
            "identities": int(actions["query_ik14"].nunique()),
            "formulas": int(actions["query_formula"].nunique()),
            "mature_cells": len(N_CELLS),
        },
        "P": {
            "intensity_queries": int(len(p_intensity)), "transfer_queries": int(len(p_transfer)),
            "matrix_shapes": matrix_shapes,
            "historical_outcome_columns_quarantined": historical_columns,
            "training_contract": "recipe/reference/control only; outer-held outcomes forbidden",
        },
        "tokens": {name: list(map(int, array.shape)) for name, array in token_arrays.items()},
        "initialization": {
            "checkpoint": str(args.initial_checkpoint.resolve()),
            "outer_fold": int(configuration["outer_fold"]),
            "seed": int(configuration["seed"]), "sha256": initial_hash,
        },
        "phase_a_arms": ["clean_continuation", "matched_random", "static_target", "dynamic_np"],
        "contracts": {
            "model_loaded": False, "P2b": "forbidden", "P3_consumed": False,
            "outer_held_outcome_used_for_training": False,
            "full_candidate_graph_preserved": True,
        },
        "inventory": inventory(required, args.hash_large_files, {
            args.graph: graph_hash, args.data: data_hash,
            args.embedding_cache: embedding_hash,
            args.official_checkpoint: official_hash,
            args.initial_checkpoint: initial_hash,
        }),
        "pass_to_gpu_replay": True,
    }
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".dynamic_direct_preflight_", dir=args.output_dir.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        pd.DataFrame({
            "query_index": np.arange(n_queries), "query_formula": query_formula,
            "formula_fold": [stable_fold(value, 5, args.formula_fold_seed) for value in query_formula],
            "split": np.where(held_mask, "held", "train"),
        }).to_csv(temporary / "formula_split.csv.gz", index=False, compression="gzip")
        temporary.replace(args.output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
