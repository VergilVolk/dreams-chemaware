"""Audit whether an ICEBERG structure-conditioned teacher has useful headroom.

This is deliberately a *teacher-only* diagnostic.  It does not train or alter
the DreaMS spectrum encoder.  Every candidate in a query is simulated under the
query spectrum's acquisition metadata and ranked against the same experimental
spectrum.  Correct structure predictions are compared with two negative
controls that preserve nuisance distributions:

1. candidate-swapped: cyclically relabel predictions inside each query;
2. peak-permuted: retain every predicted peak location and intensity marginal,
   but permute intensities among the non-zero locations.

The local candidate graph is an enriched diagnostic set and the public
MassSpecGym ICEBERG checkpoint may overlap its source.  Passing this audit can
therefore justify a distillation pilot, never a generalization claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from rdkit import Chem


@dataclass(frozen=True)
class QueryRecord:
    query_index: int
    query_row: int
    formula: str
    identity: str
    official_rank: int
    official_margin: float
    candidate_molecules: tuple[int, ...]
    candidate_rows: tuple[int, ...]
    smiles: tuple[str, ...]
    collision_energy: float
    precursor_mz: float
    adduct: str
    instrument: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("data/validation/chemaware_shared_v2_cached_real_diagnostic/graph.npz"),
    )
    parser.add_argument(
        "--hdf5", type=Path, default=Path("data/models/MassSpecGym_MurckoHist_split.hdf5")
    )
    parser.add_argument(
        "--source-root", type=Path, default=Path("data/external/ms-pred-iceberg2-src")
    )
    parser.add_argument(
        "--gen-checkpoint",
        type=Path,
        default=Path("data/external/iceberg_msg_weights_v1/msg_gen.ckpt"),
    )
    parser.add_argument(
        "--inten-checkpoint",
        type=Path,
        default=Path("data/external/iceberg_msg_weights_v1/msg_inten_cosine.ckpt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/validation/chemaware_iceberg_teacher_headroom_inner_v1"),
    )
    parser.add_argument("--max-queries", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-nodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument(
        "--reuse-output",
        type=Path,
        default=None,
        help="Reuse query predictions from a compatible earlier audit output.",
    )
    return parser.parse_args()


def decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def git_head(source_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def official_query_values(graph: np.lib.npyio.NpzFile, query: int) -> tuple[int, float]:
    feature_names = [str(value) for value in graph["feature_names"].tolist()]
    try:
        column = feature_names.index("dreams_similarity")
    except ValueError as error:
        raise RuntimeError("candidate graph lacks dreams_similarity") from error
    q_left, q_right = graph["query_ptr"][query : query + 2]
    scores = []
    for molecule in range(int(q_left), int(q_right)):
        left, right = graph["molecule_ptr"][molecule : molecule + 2]
        scores.append(float(np.max(graph["features"][int(left) : int(right), column])))
    scores_array = np.asarray(scores, dtype=np.float64)
    if len(scores_array) < 2:
        raise RuntimeError("retrieval query must contain at least two molecules")
    rank = 1 + int(np.sum(scores_array[1:] >= scores_array[0]))
    margin = float(scores_array[0] - np.max(scores_array[1:]))
    return rank, margin


def representative_row(graph: np.lib.npyio.NpzFile, molecule: int) -> int:
    left, right = graph["molecule_ptr"][molecule : molecule + 2]
    if int(right) <= int(left):
        raise RuntimeError(f"molecule {molecule} has no reference spectrum")
    return int(graph["pair_candidate_row"][int(left)])


def true_spectrum(h5: h5py.File, row: int) -> np.ndarray:
    raw = np.asarray(h5["spectrum"][row], dtype=np.float64)
    if raw.shape[0] == 2:
        raw = raw.T
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise RuntimeError(f"unexpected spectrum shape at row {row}: {raw.shape}")
    valid = np.isfinite(raw).all(axis=1) & (raw[:, 0] > 0) & (raw[:, 1] > 0)
    return raw[valid]


def build_eligible_records(
    graph: np.lib.npyio.NpzFile,
    h5: h5py.File,
    valid_adducts: set[str],
    valid_instruments: set[str],
) -> tuple[list[QueryRecord], dict[str, int]]:
    counters: dict[str, int] = {
        "total": 0,
        "invalid_metadata": 0,
        "unsupported_adduct": 0,
        "unsupported_instrument": 0,
        "invalid_smiles": 0,
        "empty_spectrum": 0,
        "eligible": 0,
    }
    records: list[QueryRecord] = []
    for query in range(len(graph["query_row"])):
        counters["total"] += 1
        row = int(graph["query_row"][query])
        ce = float(h5["COLLISION_ENERGY"][row])
        precursor = float(h5["precursor_mz"][row])
        adduct = decode(h5["adduct"][row])
        instrument = decode(h5["INSTRUMENT_TYPE"][row])
        if not math.isfinite(ce) or not math.isfinite(precursor) or precursor <= 0:
            counters["invalid_metadata"] += 1
            continue
        if adduct not in valid_adducts:
            counters["unsupported_adduct"] += 1
            continue
        if instrument not in valid_instruments:
            counters["unsupported_instrument"] += 1
            continue
        if len(true_spectrum(h5, row)) == 0:
            counters["empty_spectrum"] += 1
            continue
        q_left, q_right = graph["query_ptr"][query : query + 2]
        molecules = tuple(range(int(q_left), int(q_right)))
        rows = tuple(representative_row(graph, molecule) for molecule in molecules)
        smiles = tuple(decode(h5["smiles"][candidate_row]) for candidate_row in rows)
        if any(Chem.MolFromSmiles(value) is None for value in smiles):
            counters["invalid_smiles"] += 1
            continue
        rank, margin = official_query_values(graph, query)
        records.append(
            QueryRecord(
                query_index=query,
                query_row=row,
                formula=str(graph["query_formula"][query]),
                identity=str(graph["query_ik14"][query]),
                official_rank=rank,
                official_margin=margin,
                candidate_molecules=molecules,
                candidate_rows=rows,
                smiles=smiles,
                collision_energy=ce,
                precursor_mz=precursor,
                adduct=adduct,
                instrument=instrument,
            )
        )
        counters["eligible"] += 1
    return records, counters


def take_unique_identity(records: list[QueryRecord], count: int) -> list[QueryRecord]:
    selected: list[QueryRecord] = []
    seen: set[str] = set()
    for record in records:
        if record.identity in seen:
            continue
        selected.append(record)
        seen.add(record.identity)
        if len(selected) == count:
            break
    return selected


def select_balanced(records: list[QueryRecord], max_queries: int) -> list[QueryRecord]:
    if max_queries < 2:
        raise ValueError("--max-queries must be at least 2")
    n_error = max_queries // 2
    n_correct = max_queries - n_error
    # Near-boundary cases maximize sensitivity for this headroom diagnostic.
    # The report labels this enrichment and it must not be read as population performance.
    errors = sorted(
        (record for record in records if record.official_rank > 1),
        key=lambda record: (abs(record.official_margin), record.query_index),
    )
    correct = sorted(
        (record for record in records if record.official_rank == 1),
        key=lambda record: (abs(record.official_margin), record.query_index),
    )
    selected_errors = take_unique_identity(errors, n_error)
    selected_correct = take_unique_identity(correct, n_correct)
    if len(selected_errors) != n_error or len(selected_correct) != n_correct:
        raise RuntimeError(
            f"not enough identity-distinct eligible records: errors={len(selected_errors)}/{n_error}, "
            f"correct={len(selected_correct)}/{n_correct}"
        )
    return sorted(selected_errors + selected_correct, key=lambda record: record.query_index)


def bin_true_spectrum(common: object, h5: h5py.File, row: int) -> np.ndarray:
    return np.asarray(common.bin_spectra([true_spectrum(h5, row)])[0], dtype=np.float32)


def entropy_distances(predicted: np.ndarray, true: np.ndarray) -> np.ndarray:
    eps = 1e-22
    predicted = np.maximum(np.asarray(predicted, dtype=np.float64), 0.0)
    true = np.maximum(np.asarray(true, dtype=np.float64), 0.0)
    pred_norm = predicted / (predicted.sum(axis=1, keepdims=True) + eps)
    true_norm = true / (true.sum() + eps)

    def entropy(value: np.ndarray) -> np.ndarray:
        return -np.sum(value * np.log(value + eps), axis=-1)

    result = (
        2.0 * entropy((pred_norm + true_norm[None, :]) / 2.0)
        - entropy(pred_norm)
        - entropy(true_norm)
    ) / np.log(4.0)
    result[predicted.sum(axis=1) == 0] = 1.0
    return result.astype(np.float32)


def peak_permute(predicted: np.ndarray, seed: int) -> np.ndarray:
    output = predicted.copy()
    for index in range(len(output)):
        nonzero = np.flatnonzero(output[index] > 0)
        if len(nonzero) < 2:
            continue
        # A candidate-specific generator makes the null deterministic independent
        # of batching and preserves the exact support plus intensity multiset.
        rng = np.random.default_rng(seed + 1_000_003 * (index + 1))
        values = output[index, nonzero].copy()
        permutation = rng.permutation(len(nonzero))
        if np.array_equal(permutation, np.arange(len(nonzero))):
            permutation = np.roll(permutation, 1)
        output[index, nonzero] = values[permutation]
    return output


def load_reusable_predictions(
    reuse_output: Path | None,
    selected: list[QueryRecord],
    query_ptr: np.ndarray,
    graph_sha256: str,
    gen_sha256: str,
    inten_sha256: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    predictions = np.zeros((int(query_ptr[-1]), 15_000), dtype=np.float32)
    populated = np.zeros(int(query_ptr[-1]), dtype=bool)
    summary = {"queries": 0, "candidate_spectra": 0}
    if reuse_output is None:
        return predictions, populated, summary
    report_path = reuse_output / "report.json"
    required = [
        report_path,
        reuse_output / "selected_queries.npy",
        reuse_output / "query_ptr.npy",
        reuse_output / "iceberg_predictions_f16.npy",
    ]
    if any(not path.exists() for path in required):
        raise FileNotFoundError(f"incomplete reusable output: {reuse_output}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    prior_inputs = report["inputs"]
    expected = {
        "graph_sha256": graph_sha256,
        "gen_checkpoint_sha256": gen_sha256,
        "inten_checkpoint_sha256": inten_sha256,
    }
    for key, value in expected.items():
        if prior_inputs.get(key) != value:
            raise RuntimeError(f"reusable output {key} mismatch")
    prior_queries = np.load(reuse_output / "selected_queries.npy").astype(np.int64)
    prior_ptr = np.load(reuse_output / "query_ptr.npy").astype(np.int64)
    prior_predictions = np.load(reuse_output / "iceberg_predictions_f16.npy").astype(np.float32)
    if prior_ptr.shape != (len(prior_queries) + 1,) or prior_ptr[-1] != len(prior_predictions):
        raise RuntimeError("reusable query pointers/predictions are misaligned")
    prior_lookup = {int(query): index for index, query in enumerate(prior_queries)}
    for new_index, record in enumerate(selected):
        prior_index = prior_lookup.get(record.query_index)
        if prior_index is None:
            continue
        old_left, old_right = prior_ptr[prior_index : prior_index + 2]
        new_left, new_right = query_ptr[new_index : new_index + 2]
        if int(old_right - old_left) != int(new_right - new_left):
            raise RuntimeError(f"candidate count changed for reusable query {record.query_index}")
        predictions[int(new_left) : int(new_right)] = prior_predictions[int(old_left) : int(old_right)]
        populated[int(new_left) : int(new_right)] = True
        summary["queries"] += 1
        summary["candidate_spectra"] += int(new_right - new_left)
    return predictions, populated, summary


def ranks_and_margins(scores: np.ndarray, query_ptr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ranks = np.empty(len(query_ptr) - 1, dtype=np.int64)
    margins = np.empty(len(query_ptr) - 1, dtype=np.float32)
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:])):
        values = scores[int(left) : int(right)]
        ranks[query] = 1 + int(np.sum(values[1:] <= values[0]))
        margins[query] = float(np.min(values[1:]) - values[0])
    return ranks, margins


def summarize(ranks: np.ndarray, margins: np.ndarray) -> dict[str, float | int]:
    return {
        "queries": int(len(ranks)),
        "hit1": float(np.mean(ranks == 1)),
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
        "mean_positive_margin": float(np.mean(margins)),
        "median_positive_margin": float(np.median(margins)),
    }


def cluster_bootstrap(
    difference: np.ndarray,
    clusters: np.ndarray,
    seed: int,
    draws: int,
) -> dict[str, object]:
    unique = np.unique(clusters)
    cluster_values = np.asarray(
        [np.mean(difference[clusters == cluster]) for cluster in unique], dtype=np.float64
    )
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        estimates[draw] = np.mean(
            cluster_values[rng.integers(0, len(cluster_values), size=len(cluster_values))]
        )
    return {
        "formula_macro_advantage": float(np.mean(cluster_values)),
        "formula_cluster_bootstrap_95ci": [
            float(np.quantile(estimates, 0.025)),
            float(np.quantile(estimates, 0.975)),
        ],
        "formula_clusters": int(len(unique)),
        "draws": int(draws),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.max_nodes < 1 or args.bootstrap_draws < 1:
        raise ValueError("batch-size, max-nodes and bootstrap-draws must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    source_python = args.source_root / "src"
    if not source_python.exists():
        raise FileNotFoundError(source_python)
    sys.path.insert(0, str(source_python.resolve()))
    from ms_pred import common  # noqa: PLC0415
    from ms_pred.dag_pred import joint_model  # noqa: PLC0415

    torch.set_num_threads(args.torch_threads)
    np.random.seed(args.seed)
    graph = np.load(args.graph, allow_pickle=True)
    with h5py.File(args.hdf5, "r") as h5:
        eligible, exclusions = build_eligible_records(
            graph,
            h5,
            set(common.ion2onehot_pos),
            set(common.instrument2onehot_pos),
        )
        selected = select_balanced(eligible, args.max_queries)
        print(
            f"eligible={len(eligible)}/{exclusions['total']} selected={len(selected)} "
            f"official_errors={sum(record.official_rank > 1 for record in selected)}",
            flush=True,
        )

        query_ptr = [0]
        flat_smiles: list[str] = []
        flat_ce: list[float] = []
        flat_precursor: list[float] = []
        flat_adduct: list[str] = []
        flat_instrument: list[str] = []
        flat_molecule: list[int] = []
        flat_candidate_row: list[int] = []
        true_binned = []
        for record in selected:
            flat_smiles.extend(record.smiles)
            flat_ce.extend([record.collision_energy] * len(record.smiles))
            flat_precursor.extend([record.precursor_mz] * len(record.smiles))
            flat_adduct.extend([record.adduct] * len(record.smiles))
            flat_instrument.extend([record.instrument] * len(record.smiles))
            flat_molecule.extend(record.candidate_molecules)
            flat_candidate_row.extend(record.candidate_rows)
            query_ptr.append(len(flat_smiles))
            true_binned.append(bin_true_spectrum(common, h5, record.query_row))

    graph_sha256 = sha256_file(args.graph)
    gen_sha256 = sha256_file(args.gen_checkpoint)
    inten_sha256 = sha256_file(args.inten_checkpoint)
    predictions, populated, reuse_summary = load_reusable_predictions(
        args.reuse_output,
        selected,
        np.asarray(query_ptr, dtype=np.int64),
        graph_sha256,
        gen_sha256,
        inten_sha256,
    )
    missing = np.flatnonzero(~populated)
    model_start = time.perf_counter()
    print(
        f"reused {reuse_summary['candidate_spectra']} candidate spectra; "
        f"{len(missing)} require inference",
        flush=True,
    )
    model = None
    if len(missing):
        print("loading ICEBERG generator and intensity checkpoints", flush=True)
        model = joint_model.JointModel.from_checkpoints(
            str(args.gen_checkpoint), str(args.inten_checkpoint)
        )
        model.eval()
    print(
        f"predicting {len(missing)} query-conditioned candidate spectra "
        f"in batches of {args.batch_size}",
        flush=True,
    )
    prediction_start = time.perf_counter()
    for left in range(0, len(missing), args.batch_size):
        right = min(left + args.batch_size, len(missing))
        index = missing[left:right]
        with torch.inference_mode():
            output = model.predict_mol(
                smi=[flat_smiles[value] for value in index],
                collision_eng=[flat_ce[value] for value in index],
                precursor_mz=[flat_precursor[value] for value in index],
                adduct=[flat_adduct[value] for value in index],
                instrument=[flat_instrument[value] for value in index],
                threshold=0.0,
                device="cpu",
                max_nodes=args.max_nodes,
                binned_out=True,
                canonical_root_smi=False,
                adduct_shift=True,
            )
        batch_predictions = np.asarray(output["spec"], dtype=np.float32)
        if batch_predictions.shape != (len(index), 15_000):
            raise RuntimeError(
                f"unexpected ICEBERG output shape {batch_predictions.shape}, "
                f"expected {(len(index), 15_000)}"
            )
        predictions[index] = batch_predictions
        print(
            f"predicted {right}/{len(missing)} missing candidates "
            f"({time.perf_counter() - prediction_start:.1f}s)",
            flush=True,
        )

    # Quantize every arm uniformly to the persisted teacher-ledger precision so
    # reused and newly inferred candidates cannot differ merely by cache precision.
    predictions = predictions.astype(np.float16).astype(np.float32)

    query_ptr_array = np.asarray(query_ptr, dtype=np.int64)
    true_binned_array = np.asarray(true_binned, dtype=np.float32)
    correct_scores = np.empty(len(flat_smiles), dtype=np.float32)
    swapped_scores = np.empty(len(flat_smiles), dtype=np.float32)
    permuted_scores = np.empty(len(flat_smiles), dtype=np.float32)
    permuted_predictions = peak_permute(predictions, args.seed + 41)
    for query, (left, right) in enumerate(zip(query_ptr_array[:-1], query_ptr_array[1:])):
        left, right = int(left), int(right)
        true = true_binned_array[query]
        correct_scores[left:right] = entropy_distances(predictions[left:right], true)
        swapped_scores[left:right] = entropy_distances(
            np.roll(predictions[left:right], 1, axis=0), true
        )
        permuted_scores[left:right] = entropy_distances(
            permuted_predictions[left:right], true
        )

    official_ranks = np.asarray([record.official_rank for record in selected], dtype=np.int64)
    official_margins = np.asarray(
        [record.official_margin for record in selected], dtype=np.float32
    )
    correct_ranks, correct_margins = ranks_and_margins(correct_scores, query_ptr_array)
    swapped_ranks, swapped_margins = ranks_and_margins(swapped_scores, query_ptr_array)
    permuted_ranks, permuted_margins = ranks_and_margins(permuted_scores, query_ptr_array)
    formulas = np.asarray([record.formula for record in selected], dtype=object)
    error_mask = official_ranks > 1
    correct_mask = ~error_mask

    hit_correct = (correct_ranks == 1).astype(np.float64)
    hit_swapped = (swapped_ranks == 1).astype(np.float64)
    hit_permuted = (permuted_ranks == 1).astype(np.float64)
    correct_vs_swapped = cluster_bootstrap(
        hit_correct - hit_swapped,
        formulas,
        args.seed + 101,
        args.bootstrap_draws,
    )
    correct_vs_permuted = cluster_bootstrap(
        hit_correct - hit_permuted,
        formulas,
        args.seed + 103,
        args.bootstrap_draws,
    )
    margin_vs_swapped = cluster_bootstrap(
        correct_margins - swapped_margins,
        formulas,
        args.seed + 107,
        args.bootstrap_draws,
    )
    margin_vs_permuted = cluster_bootstrap(
        correct_margins - permuted_margins,
        formulas,
        args.seed + 109,
        args.bootstrap_draws,
    )

    ci_hit_swap = correct_vs_swapped["formula_cluster_bootstrap_95ci"]
    ci_hit_perm = correct_vs_permuted["formula_cluster_bootstrap_95ci"]
    ci_margin_swap = margin_vs_swapped["formula_cluster_bootstrap_95ci"]
    ci_margin_perm = margin_vs_permuted["formula_cluster_bootstrap_95ci"]
    passed = bool(
        ci_hit_swap[0] > 0
        and ci_hit_perm[0] > 0
        and ci_margin_swap[0] > 0
        and ci_margin_perm[0] > 0
    )

    selection_array = np.asarray([record.query_index for record in selected], dtype=np.int64)
    report = {
        "status": "PASS" if passed else "FAIL",
        "decision": (
            "Teacher candidate-specificity gate passed; a tightly controlled distillation pilot is justified."
            if passed
            else "Teacher candidate-specificity gate failed; do not distill this checkpoint into DreaMS."
        ),
        "scope": {
            "teacher_only": True,
            "dreaMS_weights_changed": False,
            "diagnostic_not_formal_generalization": True,
            "selection": (
                "Equal counts of identity-distinct near-boundary official errors and official-correct "
                "queries, ranked by absolute official margin. This is deliberately enriched and biased."
            ),
            "checkpoint_overlap_warning": (
                "The ICEBERG weights were trained on MassSpecGym and this local graph is derived from "
                "MassSpecGym. Results measure mechanism/headroom only, not independent transfer."
            ),
            "stereochemistry_boundary": (
                "ICEBERG canonicalization removes stereochemistry; this audit measures connectivity-level "
                "structure discrimination and cannot support stereoisomer claims."
            ),
        },
        "inputs": {
            "graph": str(args.graph.resolve()),
            "graph_sha256": graph_sha256,
            "hdf5": str(args.hdf5.resolve()),
            "gen_checkpoint": str(args.gen_checkpoint.resolve()),
            "gen_checkpoint_sha256": gen_sha256,
            "inten_checkpoint": str(args.inten_checkpoint.resolve()),
            "inten_checkpoint_sha256": inten_sha256,
            "source_root": str(args.source_root.resolve()),
            "source_git_head": git_head(args.source_root),
        },
        "protocol": {
            "seed": args.seed,
            "max_queries": args.max_queries,
            "selected_query_sha256": sha256_array(selection_array),
            "batch_size": args.batch_size,
            "max_nodes": args.max_nodes,
            "binned_bins": 15_000,
            "binned_upper_mz": 1_500,
            "teacher_cache_precision": "float16, uniformly quantized before scoring",
            "distance": "normalized spectral entropy distance, official ICEBERG retrieval definition",
            "rank_ties": "strict pessimistic: each negative with distance <= positive increments rank",
            "candidate_conditions": "query collision energy, precursor m/z, adduct, and instrument",
            "candidate_structure_row": "first available reference row for each molecule node",
            "candidate_swapped_control": "one-position cyclic prediction relabeling within each query",
            "peak_permuted_control": (
                "deterministic within-prediction permutation of intensities over non-zero bins; "
                "peak support and intensity multiset are preserved"
            ),
            "pass_gate": (
                "formula-cluster bootstrap 95% lower bound > 0 for correct-minus-control hit1 and "
                "positive-margin advantages against both controls"
            ),
        },
        "eligibility": exclusions,
        "selected": {
            "queries": len(selected),
            "official_errors": int(np.sum(error_mask)),
            "official_correct": int(np.sum(correct_mask)),
            "unique_identities": len(set(record.identity for record in selected)),
            "unique_formulas": len(set(record.formula for record in selected)),
            "candidate_spectra_predicted": len(flat_smiles),
            "reused_queries": reuse_summary["queries"],
            "reused_candidate_spectra": reuse_summary["candidate_spectra"],
            "new_candidate_spectra": int(len(missing)),
            "candidate_count_min": int(np.min(np.diff(query_ptr_array))),
            "candidate_count_median": float(np.median(np.diff(query_ptr_array))),
            "candidate_count_max": int(np.max(np.diff(query_ptr_array))),
        },
        "metrics": {
            "official_dreams": summarize(official_ranks, official_margins),
            "iceberg_correct": summarize(correct_ranks, correct_margins),
            "candidate_swapped": summarize(swapped_ranks, swapped_margins),
            "peak_permuted": summarize(permuted_ranks, permuted_margins),
            "official_error_subset": {
                "queries": int(np.sum(error_mask)),
                "iceberg_rescued": int(np.sum(correct_ranks[error_mask] == 1)),
                "candidate_swapped_hit1": int(np.sum(swapped_ranks[error_mask] == 1)),
                "peak_permuted_hit1": int(np.sum(permuted_ranks[error_mask] == 1)),
            },
            "official_correct_subset": {
                "queries": int(np.sum(correct_mask)),
                "iceberg_preserved": int(np.sum(correct_ranks[correct_mask] == 1)),
                "candidate_swapped_hit1": int(np.sum(swapped_ranks[correct_mask] == 1)),
                "peak_permuted_hit1": int(np.sum(permuted_ranks[correct_mask] == 1)),
            },
            "paired_formula_cluster_bootstrap": {
                "hit1_correct_minus_candidate_swapped": correct_vs_swapped,
                "hit1_correct_minus_peak_permuted": correct_vs_permuted,
                "margin_correct_minus_candidate_swapped": margin_vs_swapped,
                "margin_correct_minus_peak_permuted": margin_vs_permuted,
            },
        },
        "runtime_seconds": float(time.perf_counter() - model_start),
    }

    np.save(args.output / "selected_queries.npy", selection_array)
    np.save(args.output / "query_ptr.npy", query_ptr_array)
    np.save(args.output / "candidate_molecule_index.npy", np.asarray(flat_molecule, dtype=np.int64))
    np.save(args.output / "candidate_hdf5_row.npy", np.asarray(flat_candidate_row, dtype=np.int64))
    np.save(args.output / "true_binned_f16.npy", true_binned_array.astype(np.float16))
    np.save(args.output / "iceberg_predictions_f16.npy", predictions.astype(np.float16))
    np.savez_compressed(
        args.output / "scores_and_ranks.npz",
        official_rank=official_ranks,
        official_margin=official_margins,
        correct_score=correct_scores,
        correct_rank=correct_ranks,
        correct_margin=correct_margins,
        candidate_swapped_score=swapped_scores,
        candidate_swapped_rank=swapped_ranks,
        candidate_swapped_margin=swapped_margins,
        peak_permuted_score=permuted_scores,
        peak_permuted_rank=permuted_ranks,
        peak_permuted_margin=permuted_margins,
        formula=formulas,
    )
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["metrics"], indent=2, ensure_ascii=False), flush=True)
    print(f"decision={report['status']} report={args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
