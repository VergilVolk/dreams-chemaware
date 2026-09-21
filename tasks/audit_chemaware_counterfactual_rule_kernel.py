"""Audit local-counterfactual and co-occurrence rule kernels on the frozen graph.

Every feature is computed independently from one clean spectrum.  For each
curated fragment/neutral-loss mass ``m`` we compare response at ``m`` with
responses at the matched fake masses ``m-d`` and ``m+d``.  This subtracts a
local mass-density background without candidate structures.  A second channel
hashes co-occurring rule responses within the same spectrum, following the
Mass2Motif idea that conjunctions of fragments/losses are more specific than
isolated peaks.

The result is a three-fold formula-held retrospective diagnostic.  It does not
update DreaMS and does not authorize a training run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger", type=Path,
        default=ROOT / "data/validation/chemaware_iceberg_corrective_residual_ledger_v2/ledger.npz",
    )
    parser.add_argument(
        "--graph-dir", type=Path,
        default=ROOT / "data/validation/chemaware_full_manifest_iceberg_graph_v1",
    )
    parser.add_argument(
        "--token-dir", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1",
    )
    parser.add_argument(
        "--rule-library", type=Path,
        default=ROOT / "dreams/models/chem_aware/chem_rules_data.json",
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/validation/chemaware_counterfactual_rule_kernel_v1",
    )
    parser.add_argument("--top-peaks", type=int, default=32)
    parser.add_argument("--intensity-power", type=float, default=0.5)
    parser.add_argument("--rule-tolerance", type=float, default=0.02)
    parser.add_argument("--counterfactual-offset", type=float, default=0.137)
    parser.add_argument("--motif-dimension", type=int, default=1024)
    parser.add_argument("--motif-top-rules", type=int, default=12)
    parser.add_argument(
        "--beta", type=float, nargs="+",
        default=(0.0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.80),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260912)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, 1e-12)


def hash_pair(left: int, right: int, dimension: int) -> tuple[int, float]:
    key = (int(left) + 1) * 1_000_003 + (int(right) + 1) * 97_409
    value = (key * 2_654_435_761 + 2_246_822_519) & 0xFFFFFFFF
    sign_value = (value * 3_266_489_917 + 668_265_263) & 0xFFFFFFFF
    return int(value % dimension), (1.0 if sign_value & 1 else -1.0)


def motif_sketch(
    response: np.ndarray,
    *,
    dimension: int,
    top_rules: int,
) -> np.ndarray:
    output = np.zeros((len(response), dimension), dtype=np.float32)
    for row, values in enumerate(np.asarray(response, dtype=np.float32)):
        nonzero = np.flatnonzero(values > 0)
        if len(nonzero) > top_rules:
            nonzero = nonzero[np.argsort(-values[nonzero], kind="stable")[:top_rules]]
        for first in range(len(nonzero)):
            for second in range(first + 1, len(nonzero)):
                left, right = sorted((int(nonzero[first]), int(nonzero[second])))
                index, sign = hash_pair(left, right, dimension)
                output[row, index] += sign * float(np.sqrt(values[left] * values[right]))
    return normalize_rows(output)


def build_feature_views(
    node_rows: np.ndarray,
    cache_position: dict[int, int],
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    payload = json.loads(args.rule_library.read_text(encoding="utf-8"))
    if "channels" in payload:
        records = [
            {"category": x["category"], "match_type": x["match_type"], "value": x["value_da"]}
            for x in payload["channels"]
        ]
    else:
        records = payload["rules"]
    neutral_loss = np.asarray([
        float(x["value"]) for x in records
        if x.get("category") == "NL" and x.get("match_type") == "mass_diff"
    ], dtype=np.float64)
    fragment = np.asarray([
        float(x["value"]) for x in records
        if x.get("category") == "CF" and x.get("match_type") == "peak_mz"
    ], dtype=np.float64)
    targets = np.concatenate((neutral_loss, fragment))
    kinds = np.concatenate((np.zeros(len(neutral_loss), dtype=np.int8), np.ones(len(fragment), dtype=np.int8)))
    mz_all = np.load(args.token_dir / "mz_f32.npy", mmap_mode="r")
    intensity_all = np.load(args.token_dir / "intensity_f32.npy", mmap_mode="r")
    valid_all = np.load(args.token_dir / "valid.npy", mmap_mode="r")
    precursor_all = np.load(args.token_dir / "precursor_mz_f32.npy", mmap_mode="r")
    raw = np.zeros((len(node_rows), len(targets)), dtype=np.float32)
    lower = np.zeros_like(raw); upper = np.zeros_like(raw)
    tolerance = float(args.rule_tolerance)
    offset = float(args.counterfactual_offset)

    def response(values: np.ndarray, channel_targets: np.ndarray, weight: np.ndarray) -> np.ndarray:
        if not len(values) or not len(channel_targets):
            return np.zeros(len(channel_targets), dtype=np.float32)
        distance = np.abs(channel_targets[:, None] - values[None, :])
        return np.max(np.maximum(0.0, 1.0 - distance / tolerance) * weight[None, :], axis=1)

    for index, spectrum_row in enumerate(node_rows):
        position = cache_position[int(spectrum_row)]
        available = np.flatnonzero(np.asarray(valid_all[position], dtype=bool))
        order = available[
            np.argsort(-np.asarray(intensity_all[position, available]), kind="stable")[:args.top_peaks]
        ]
        mz = np.asarray(mz_all[position, order], dtype=np.float64)
        intensity = np.maximum(np.asarray(intensity_all[position, order], dtype=np.float64), 0.0)
        intensity /= max(float(np.sum(intensity)), 1e-12)
        weight = np.power(intensity, args.intensity_power)
        observed = np.where(kinds[:, None] == 0, float(precursor_all[position]) - mz, mz)
        # The two channel kinds share the same peak weights but use different observed masses.
        for destination, shift in ((raw, 0.0), (lower, -offset), (upper, offset)):
            distance = np.abs((targets + shift)[:, None] - observed)
            destination[index] = np.max(
                np.maximum(0.0, 1.0 - distance / tolerance) * weight[None, :], axis=1,
            ) if len(mz) else 0.0
        if (index + 1) % 4096 == 0:
            print(f"counterfactual rule features {index + 1}/{len(node_rows)}", flush=True)

    local_background = 0.5 * (lower + upper)
    signed = raw - local_background
    positive = np.maximum(signed, 0.0)
    row_permuted = np.empty_like(raw)
    for index, spectrum_row in enumerate(node_rows):
        row_permuted[index] = np.roll(raw[index], 1 + int(spectrum_row) % (raw.shape[1] - 1))
    nl_count = len(neutral_loss)
    raw_nl = raw[:, :nl_count]
    raw_cf = raw[:, nl_count:]
    background_nl = local_background[:, :nl_count]
    background_cf = local_background[:, nl_count:]
    # Preserve the NL/CF block boundary in the negative control.  A rotation
    # across that boundary would confound wrong rule identities with a change
    # in evidence type.
    permuted_nl = np.empty_like(raw_nl)
    permuted_cf = np.empty_like(raw_cf)
    for index, spectrum_row in enumerate(node_rows):
        if raw_nl.shape[1] > 1:
            permuted_nl[index] = np.roll(
                raw_nl[index], 1 + int(spectrum_row) % (raw_nl.shape[1] - 1),
            )
        if raw_cf.shape[1] > 1:
            permuted_cf[index] = np.roll(
                raw_cf[index], 1 + int(spectrum_row) % (raw_cf.shape[1] - 1),
            )
    raw_motif = motif_sketch(raw, dimension=args.motif_dimension, top_rules=args.motif_top_rules)
    background_motif = motif_sketch(
        local_background, dimension=args.motif_dimension, top_rules=args.motif_top_rules,
    )
    row_permuted_motif = motif_sketch(
        row_permuted, dimension=args.motif_dimension, top_rules=args.motif_top_rules,
    )
    positive_motif = motif_sketch(positive, dimension=args.motif_dimension, top_rules=args.motif_top_rules)
    views = {
        "raw_rule": normalize_rows(raw),
        "local_background": normalize_rows(local_background),
        "row_permuted_rule": normalize_rows(row_permuted),
        "signed_counterfactual": normalize_rows(signed),
        "positive_counterfactual": normalize_rows(positive),
        "raw_motif_only": raw_motif,
        "local_background_motif_only": background_motif,
        "row_permuted_rule_motif_only": row_permuted_motif,
        "counterfactual_motif_only": positive_motif,
        "raw_rule_motif": normalize_rows(np.concatenate((normalize_rows(raw), raw_motif), axis=1)),
        "local_background_motif": normalize_rows(
            np.concatenate((normalize_rows(local_background), background_motif), axis=1)
        ),
        "row_permuted_rule_motif": normalize_rows(
            np.concatenate((normalize_rows(row_permuted), row_permuted_motif), axis=1)
        ),
        "counterfactual_motif": normalize_rows(
            np.concatenate((normalize_rows(positive), positive_motif), axis=1)
        ),
        "raw_neutral_loss": normalize_rows(raw_nl),
        "raw_fragment_ion": normalize_rows(raw_cf),
        "background_neutral_loss": normalize_rows(background_nl),
        "background_fragment_ion": normalize_rows(background_cf),
        "row_permuted_neutral_loss": normalize_rows(permuted_nl),
        "row_permuted_fragment_ion": normalize_rows(permuted_cf),
        # Scalar self-observability channels for localized kernel gating.  They
        # are never used as pairwise chemical kernels and intentionally retain
        # magnitude information discarded by cosine normalization.
        "gate_raw_neutral_loss_energy": np.linalg.norm(raw_nl, axis=1, keepdims=True),
        "gate_raw_fragment_ion_energy": np.linalg.norm(raw_cf, axis=1, keepdims=True),
        "gate_background_neutral_loss_energy": np.linalg.norm(
            background_nl, axis=1, keepdims=True,
        ),
        "gate_background_fragment_ion_energy": np.linalg.norm(
            background_cf, axis=1, keepdims=True,
        ),
        "gate_raw_neutral_loss_count": np.sum(raw_nl > 0, axis=1, keepdims=True).astype(np.float32),
        "gate_raw_fragment_ion_count": np.sum(raw_cf > 0, axis=1, keepdims=True).astype(np.float32),
    }
    return views, {
        "neutral_loss_rules": int(len(neutral_loss)),
        "fragment_rules": int(len(fragment)),
        "counterfactual_offset_da": offset,
        "motif_dimension": args.motif_dimension,
        "motif_top_rules": args.motif_top_rules,
        "raw_nonzero_fraction": float(np.mean(np.linalg.norm(raw, axis=1) > 0)),
        "positive_counterfactual_nonzero_fraction": float(np.mean(np.linalg.norm(positive, axis=1) > 0)),
    }


def rank_queries(scores: np.ndarray, query_ptr: np.ndarray, labels: np.ndarray) -> np.ndarray:
    result = np.empty(len(query_ptr) - 1, dtype=np.int64)
    for query, (left, right) in enumerate(zip(query_ptr[:-1], query_ptr[1:], strict=True)):
        local_score = scores[int(left) : int(right)]
        local_label = labels[int(left) : int(right)]
        positive = float(local_score[np.flatnonzero(local_label)[0]])
        result[query] = 1 + int(np.sum(local_score[~local_label] >= positive))
    return result


def paired_dot(
    feature: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    batch_size: int = 8192,
) -> np.ndarray:
    """Row-wise dot products without materializing two huge indexed tensors."""

    feature = np.asarray(feature)
    left = np.asarray(left, dtype=np.int64); right = np.asarray(right, dtype=np.int64)
    if left.shape != right.shape:
        raise ValueError("paired-dot indices are not aligned")
    output = np.empty(len(left), dtype=np.float64)
    for start in range(0, len(left), batch_size):
        stop = min(start + batch_size, len(left))
        output[start:stop] = np.einsum(
            "ij,ij->i", feature[left[start:stop]], feature[right[start:stop]],
        )
    return output


def retrieval(old_rank: np.ndarray, new_rank: np.ndarray) -> dict[str, object]:
    old_rank = np.asarray(old_rank); new_rank = np.asarray(new_rank)
    result: dict[str, object] = {
        "queries": int(len(old_rank)),
        "baseline_mrr": float(np.mean(1.0 / old_rank)),
        "mrr": float(np.mean(1.0 / new_rank)),
        "delta_mrr": float(np.mean(1.0 / new_rank - 1.0 / old_rank)),
        "corrected_at_1": int(np.sum((old_rank > 1) & (new_rank == 1))),
        "introduced_at_1": int(np.sum((old_rank == 1) & (new_rank > 1))),
        "rank_improved": int(np.sum(new_rank < old_rank)),
        "rank_worsened": int(np.sum(new_rank > old_rank)),
    }
    for k in (1, 3, 5, 10, 20, 50):
        result[f"baseline_recall{k}"] = float(np.mean(old_rank <= k))
        result[f"recall{k}"] = float(np.mean(new_rank <= k))
        result[f"delta_recall{k}"] = float(np.mean(new_rank <= k) - np.mean(old_rank <= k))
    result["risk_utility_at_1"] = int(result["corrected_at_1"]) - 2 * int(result["introduced_at_1"])
    return result


def bootstrap(
    formula: np.ndarray, old_rank: np.ndarray, new_rank: np.ndarray,
    draws: int, seed: int,
) -> list[float]:
    unique, inverse = np.unique(np.asarray(formula, dtype=str), return_inverse=True)
    delta = (new_rank == 1).astype(float) - (old_rank == 1).astype(float)
    sums = np.bincount(inverse, weights=delta); counts = np.bincount(inverse)
    rng = np.random.default_rng(seed); output = np.empty(draws, dtype=np.float64)
    for left in range(0, draws, 500):
        right = min(left + 500, draws)
        selected = rng.integers(0, len(unique), size=(right - left, len(unique)))
        output[left:right] = sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
    return [float(value) for value in np.quantile(output, (0.025, 0.975))]


def main() -> None:
    args = arguments()
    if args.counterfactual_offset <= args.rule_tolerance or args.motif_dimension <= 0:
        raise ValueError("counterfactual offset must exceed tolerance and motif dimension must be positive")
    if args.motif_top_rules < 2 or any(beta < 0 for beta in args.beta):
        raise ValueError("invalid motif or beta setting")
    graph_path = args.graph_dir / "graph.npz"
    required = [
        args.ledger, graph_path, args.token_dir / "rows.npy",
        args.token_dir / "official_embeddings_f32.npy", args.rule_library,
    ]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)
    ledger = np.load(args.ledger, allow_pickle=False)
    graph = np.load(graph_path, allow_pickle=True)
    if not np.array_equal(ledger["query_row"], graph["query_row"]):
        raise RuntimeError("ledger and graph query order drifted")
    fold = np.asarray(ledger["query_formula_fold"], dtype=np.int64)
    if not np.array_equal(np.unique(fold), np.asarray([0, 1, 2])):
        raise RuntimeError("expected the frozen three formula folds")
    formula = np.asarray(ledger["query_formula"]).astype(str)

    cache_rows = np.load(args.token_dir / "rows.npy", mmap_mode="r")
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    cache_position = {int(row): index for index, row in enumerate(cache_rows)}
    graph_rows = np.unique(np.concatenate((graph["query_row"], graph["pair_candidate_row"]))).astype(np.int64)
    graph_position = {int(row): index for index, row in enumerate(graph_rows)}
    embedding = np.asarray(official[[cache_position[int(row)] for row in graph_rows]], dtype=np.float32)
    views, feature_report = build_feature_views(graph_rows, cache_position, args)

    query_ptr = np.asarray(graph["query_ptr"], dtype=np.int64)
    molecule_ptr = np.asarray(graph["molecule_ptr"], dtype=np.int64)
    labels = np.asarray(graph["molecule_label"], dtype=bool)
    molecule_query = np.repeat(np.arange(len(fold)), np.diff(query_ptr))
    pair_molecule = np.repeat(np.arange(len(labels)), np.diff(molecule_ptr))
    pair_query = molecule_query[pair_molecule]
    query_node = np.asarray([graph_position[int(row)] for row in graph["query_row"]], dtype=np.int64)
    reference_node = np.asarray([graph_position[int(row)] for row in graph["pair_candidate_row"]], dtype=np.int64)
    recomputed_official_pair_score = paired_dot(embedding, query_node[pair_query], reference_node)
    # The frozen graph stores these same dot products in float32.  Use its
    # values as the exact baseline so beta=0 is bit-for-bit rank preserving;
    # harmless BLAS accumulation differences can otherwise flip exact ties.
    official_pair_score = np.asarray(graph["features"][:, 0], dtype=np.float64)
    official_molecule_score = np.maximum.reduceat(official_pair_score, molecule_ptr[:-1])
    baseline_rank = rank_queries(official_molecule_score, query_ptr, labels)
    maximum_official_recompute_error = float(np.max(
        np.abs(recomputed_official_pair_score - official_pair_score)
    ))
    if maximum_official_recompute_error > 1e-5:
        raise RuntimeError("official embedding cache and frozen graph scores materially drifted")

    reports = []
    oof_ranks = {}
    for view_index, (name, feature) in enumerate(views.items()):
        kernel_pair = paired_dot(feature, query_node[pair_query], reference_node)
        ranks_by_beta = {}
        for beta in args.beta:
            molecule_score = np.maximum.reduceat(
                official_pair_score + float(beta) * kernel_pair, molecule_ptr[:-1],
            )
            ranks_by_beta[float(beta)] = rank_queries(molecule_score, query_ptr, labels)
        selected_by_fold = {}
        oof = np.full(len(fold), -1, dtype=np.int64)
        for held in np.unique(fold):
            train = fold != held; held_mask = fold == held
            candidates = []
            for beta in args.beta:
                metric = retrieval(baseline_rank[train], ranks_by_beta[float(beta)][train])
                candidates.append((float(beta), metric))
            selected_beta, selected_metric = max(
                candidates,
                key=lambda item: (
                    int(item[1]["risk_utility_at_1"]),
                    -int(item[1]["introduced_at_1"]),
                    float(item[1]["delta_mrr"]),
                    -item[0],
                ),
            )
            oof[held_mask] = ranks_by_beta[selected_beta][held_mask]
            selected_by_fold[str(int(held))] = {
                "beta": selected_beta, "training_formula_metric": selected_metric,
            }
        if np.any(oof < 1):
            raise RuntimeError("OOF rank coverage is incomplete")
        oof_ranks[name] = oof
        reports.append({
            "variant": name,
            "dimension": int(feature.shape[1]),
            "selected_by_fold": selected_by_fold,
            "formula_held_oof": retrieval(baseline_rank, oof),
            "formula_cluster_bootstrap_delta_recall1_ci95": bootstrap(
                formula, baseline_rank, oof, args.bootstrap_draws, args.seed + view_index,
            ),
            "full_beta_grid_retrospective": [
                {"beta": float(beta), **retrieval(baseline_rank, ranks_by_beta[float(beta)])}
                for beta in args.beta
            ],
        })
        print(f"completed rule-kernel variant {name}", flush=True)

    report = {
        "status": "CHEMAWARE_COUNTERFACTUAL_RULE_KERNEL_RETROSPECTIVE_COMPLETE",
        "formal_training_authorized": False,
        "weights_updated": False,
        "claim_limit": (
            "Formula-held OOF on the already consumed 2048-query diagnostic graph; it can reject "
            "feature ideas but cannot serve as an outer or external model-performance claim."
        ),
        "method": {
            "one_clean_spectrum_one_feature_vector": True,
            "candidate_structure_used": False,
            "candidate_set_used_in_feature_map": False,
            "shared_psd_product_kernel": True,
            "counterfactual_definition": "response(m)-0.5*[response(m-offset)+response(m+offset)]",
            "motif_definition": "hashed within-spectrum products of top co-occurring rule responses",
            "beta_selected_without_held_formula_fold": True,
        },
        "data": {
            "queries": int(len(fold)),
            "formulas": int(len(np.unique(formula))),
            "candidate_molecules": int(len(labels)),
            "reference_pairs": int(len(reference_node)),
            "unique_spectrum_rows": int(len(graph_rows)),
            "baseline": retrieval(baseline_rank, baseline_rank),
            "maximum_official_pair_score_recompute_error": maximum_official_recompute_error,
        },
        "features": feature_report,
        "variants": reports,
        "provenance": {
            "ledger_sha256": sha256(args.ledger),
            "graph_sha256": sha256(graph_path),
            "official_embeddings_sha256": sha256(args.token_dir / "official_embeddings_f32.npy"),
            "rule_library_sha256": sha256(args.rule_library),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_counterfactual_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "oof_ranks.npz", baseline_rank=baseline_rank, formula=formula, **oof_ranks,
        )
        if args.output.exists():
            shutil.rmtree(args.output)
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    best = max(
        reports,
        key=lambda item: (
            int(item["formula_held_oof"]["risk_utility_at_1"]),
            -int(item["formula_held_oof"]["introduced_at_1"]),
            float(item["formula_held_oof"]["delta_mrr"]),
        ),
    )
    print(json.dumps({
        "status": report["status"], "best": best,
        "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
