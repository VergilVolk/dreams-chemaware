"""Test whether P2b neutral-loss evidence is compatible with a shared embedding.

The exact P2b neutral-loss score uses tolerance-based greedy peak matching.  A
generic pair matcher is not automatically the inner product of two independently
computed spectrum vectors.  This audit compares it with a positive-semidefinite
Gaussian set kernel over each spectrum's neutral-loss peaks:

    K(A, B) = sum_ij sqrt(p_i p_j) exp(-(loss_i-loss_j)^2 / (2 sigma^2)).

After self-normalization, K is a cosine in a (possibly implicit) shared feature
space.  Sigma is selected only by agreement with the frozen exact pair score;
identity labels and retrieval outcomes are not used for selection.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tasks"))

from audit_e0_observability_residual import pair_features  # noqa: E402
from g8r_p2_rank_fusion_core import (  # noqa: E402
    fuse_one_query,
    normalize_pair_features,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        type=Path,
        default=ROOT / "data/validation/chemaware_shared_v2_cached_real_diagnostic/graph.npz",
    )
    parser.add_argument(
        "--hdf5",
        type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/validation/chemaware_neutral_loss_shared_kernel_v1/report.json",
    )
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument(
        "--sigma",
        type=float,
        nargs="+",
        default=[0.005, 0.01, 0.02],
        help="Physically tied soft-matching widths; selection never uses labels.",
    )
    parser.add_argument("--psd-subset", type=int, default=96)
    return parser.parse_args()


def read_rows(handle: h5py.File, key: str, rows: np.ndarray) -> np.ndarray:
    order = np.argsort(rows, kind="stable")
    inverse = np.argsort(order, kind="stable")
    return np.asarray(handle[key][rows[order]])[inverse]


def neutral_loss_measure(spectrum: np.ndarray, precursor: float) -> tuple[np.ndarray, np.ndarray]:
    mz = np.asarray(spectrum[0], dtype=np.float64)
    intensity = np.asarray(spectrum[1], dtype=np.float64)
    keep = (
        np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) &
        (intensity > 0) & (precursor - mz > 0)
    )
    losses = precursor - mz[keep]
    intensity = intensity[keep]
    if len(losses) == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    order = np.argsort(losses, kind="stable")
    losses = losses[order]
    probability = intensity[order] / max(float(np.sum(intensity)), 1e-12)
    return losses, np.sqrt(probability)


def greedy_score(
    left_loss: np.ndarray,
    left_weight: np.ndarray,
    right_loss: np.ndarray,
    right_weight: np.ndarray,
    tolerance: float,
) -> float:
    candidates = []
    for i, value in enumerate(left_loss):
        lo = int(np.searchsorted(right_loss, value - tolerance, side="left"))
        hi = int(np.searchsorted(right_loss, value + tolerance, side="right"))
        candidates.extend((abs(value - right_loss[j]), i, j) for j in range(lo, hi))
    used_left: set[int] = set()
    used_right: set[int] = set()
    score = 0.0
    for _, i, j in sorted(candidates):
        if i not in used_left and j not in used_right:
            used_left.add(i)
            used_right.add(j)
            score += float(left_weight[i] * right_weight[j])
    return score


def gaussian_raw(
    left_loss: np.ndarray,
    left_weight: np.ndarray,
    right_loss: np.ndarray,
    right_weight: np.ndarray,
    sigma: float,
) -> float:
    if len(left_loss) == 0 or len(right_loss) == 0:
        return 0.0
    distance = left_loss[:, None] - right_loss[None, :]
    kernel = np.exp(-0.5 * np.square(distance / sigma))
    return float(left_weight @ kernel @ right_weight)


def strict_rank(scores: np.ndarray) -> int:
    return 1 + int(np.sum(np.asarray(scores[1:]) >= float(scores[0])))


def molecule_scores(pair_scores: np.ndarray, molecule_ptr: np.ndarray) -> np.ndarray:
    return np.maximum.reduceat(pair_scores, molecule_ptr[:-1])


def query_ranks(
    molecule_score: np.ndarray,
    query_ptr: np.ndarray,
) -> np.ndarray:
    return np.asarray([
        strict_rank(molecule_score[left:right])
        for left, right in zip(query_ptr[:-1], query_ptr[1:])
    ], dtype=np.int32)


def top_identity_indices(molecule_score: np.ndarray, query_ptr: np.ndarray) -> np.ndarray:
    result = []
    for left, right in zip(query_ptr[:-1], query_ptr[1:]):
        values = molecule_score[left:right]
        top = int(np.argmax(values))
        result.append(top if int(np.sum(values == values[top])) == 1 else -1)
    return np.asarray(result, dtype=np.int32)


def frozen_p2b_ranks(
    features: np.ndarray,
    query_ptr: np.ndarray,
    molecule_ptr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the already-frozen P2b recipe independently to every query."""
    ranks = []
    top = []
    intervention = []
    for molecule_left, molecule_right in zip(query_ptr[:-1], query_ptr[1:]):
        pair_left = int(molecule_ptr[molecule_left])
        pair_right = int(molecule_ptr[molecule_right])
        local_ptr = molecule_ptr[molecule_left:molecule_right + 1] - pair_left
        values = features[pair_left:pair_right]
        normalized = normalize_pair_features(
            values, np.asarray([0, len(values)], dtype=np.int64), "absolute"
        )
        score, used, _ = fuse_one_query(
            normalized,
            values[:, 0],
            local_ptr,
            np.asarray([0.1, 0.0, 0.1, 0.8], dtype=np.float64),
            (1, 2, 3),
            1,
            0.0,
        )
        ranks.append(strict_rank(score))
        winner = int(np.argmax(score))
        top.append(winner if int(np.sum(score == score[winner])) == 1 else -1)
        intervention.append(bool(used))
    return (
        np.asarray(ranks, dtype=np.int32),
        np.asarray(top, dtype=np.int32),
        np.asarray(intervention, dtype=bool),
    )


def gram_diagnostics(matrix: np.ndarray) -> dict[str, float | int]:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = max(float(np.max(np.abs(eigenvalues))), 1e-12)
    threshold = -1e-8 * scale
    negative = eigenvalues[eigenvalues < threshold]
    return {
        "size": int(len(matrix)),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "negative_eigenvalues_below_relative_1e-8": int(len(negative)),
        "negative_eigenvalue_mass": float(-np.sum(negative)),
    }


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.tolerance <= 0 or args.psd_subset < 8 or any(value <= 0 for value in args.sigma):
        raise ValueError("invalid tolerance, sigma, or PSD subset")
    with np.load(args.graph, allow_pickle=True) as body:
        query_rows = np.asarray(body["query_row"], dtype=np.int64)
        candidate_rows = np.asarray(body["pair_candidate_row"], dtype=np.int64)
        query_ptr = np.asarray(body["query_ptr"], dtype=np.int64)
        molecule_ptr = np.asarray(body["molecule_ptr"], dtype=np.int64)
        official_pair = np.asarray(body["features"][:, 0], dtype=np.float64)
        near = np.asarray(body["query_has_near"], dtype=bool)
    query_pair_ptr = molecule_ptr[query_ptr]
    pair_query_rows = np.repeat(query_rows, np.diff(query_pair_ptr))
    if len(pair_query_rows) != len(candidate_rows):
        raise RuntimeError("graph query/pair pointers are not aligned")

    reachable = np.unique(np.concatenate((query_rows, candidate_rows)))
    with h5py.File(args.hdf5, "r") as handle:
        spectra = read_rows(handle, "spectrum", reachable)
        precursor = read_rows(handle, "precursor_mz", reachable).astype(np.float64)
    position = {int(row): index for index, row in enumerate(reachable)}
    measures = [
        neutral_loss_measure(spectrum, float(parent))
        for spectrum, parent in zip(spectra, precursor)
    ]

    unique_edge: dict[tuple[int, int], int] = {}
    edge_keys = []
    for query, candidate in zip(pair_query_rows, candidate_rows):
        key = tuple(sorted((position[int(query)], position[int(candidate)])))
        if key not in unique_edge:
            unique_edge[key] = len(edge_keys)
            edge_keys.append(key)
    edge_index = np.asarray([
        unique_edge[tuple(sorted((position[int(query)], position[int(candidate)])))]
        for query, candidate in zip(pair_query_rows, candidate_rows)
    ], dtype=np.int64)

    authoritative_raw = []
    manual_exact = []
    for left, right in edge_keys:
        raw = pair_features(
            spectra[left], float(precursor[left]),
            spectra[right], float(precursor[right]), args.tolerance,
        )
        authoritative_raw.append([
            raw["sqrt_cosine"], raw["entropy_similarity"],
            raw["neutral_loss_sqrt_cosine"],
        ])
        manual_exact.append(
            greedy_score(*measures[left], *measures[right], args.tolerance)
        )
    authoritative_raw = np.asarray(authoritative_raw, dtype=np.float64)
    exact_unique = authoritative_raw[:, 2]
    manual_exact = np.asarray(manual_exact, dtype=np.float64)
    exact_pair = exact_unique[edge_index]
    raw_pair = authoritative_raw[edge_index]
    exact_molecule = molecule_scores(exact_pair, molecule_ptr)
    exact_rank = query_ranks(exact_molecule, query_ptr)
    exact_top = top_identity_indices(exact_molecule, query_ptr)

    variants = []
    soft_by_sigma: dict[float, np.ndarray] = {}
    for sigma in args.sigma:
        self_raw = np.asarray([
            gaussian_raw(*measure, *measure, sigma) for measure in measures
        ], dtype=np.float64)
        soft_unique = []
        for left, right in edge_keys:
            raw = gaussian_raw(*measures[left], *measures[right], sigma)
            denominator = np.sqrt(max(self_raw[left] * self_raw[right], 1e-24))
            soft_unique.append(raw / denominator)
        soft_pair = np.asarray(soft_unique, dtype=np.float64)[edge_index]
        soft_by_sigma[float(sigma)] = soft_pair
        soft_molecule = molecule_scores(soft_pair, molecule_ptr)
        soft_rank = query_ranks(soft_molecule, query_ptr)
        soft_top = top_identity_indices(soft_molecule, query_ptr)
        soft_features = np.column_stack((official_pair, raw_pair[:, :2], soft_pair))
        soft_p2b_rank, soft_p2b_top, soft_intervention = frozen_p2b_ranks(
            soft_features, query_ptr, molecule_ptr
        )
        variants.append({
            "sigma": float(sigma),
            "selection_metrics_without_labels": {
                "spearman_vs_exact_pair_score": float(spearmanr(exact_pair, soft_pair).statistic),
                "pearson_vs_exact_pair_score": float(pearsonr(exact_pair, soft_pair).statistic),
                "rmse_vs_exact_pair_score": float(np.sqrt(np.mean(np.square(exact_pair - soft_pair)))),
            },
            "post_selection_descriptive_retrieval": {
                "recall1": float(np.mean(soft_rank == 1)),
                "near_recall1": float(np.mean(soft_rank[near] == 1)),
                "top_candidate_agreement_with_exact": float(np.mean(soft_top == exact_top)),
                "rank_agreement_with_exact": float(np.mean(soft_rank == exact_rank)),
                "frozen_p2b_with_soft_nl_recall1": float(np.mean(soft_p2b_rank == 1)),
                "frozen_p2b_with_soft_nl_near_recall1": float(
                    np.mean(soft_p2b_rank[near] == 1)
                ),
                "frozen_p2b_top_agreement_with_exact_nl_recipe": None,
                "frozen_p2b_intervention_rate": float(np.mean(soft_intervention)),
            },
            "_soft_p2b_top": soft_p2b_top,
        })
    chosen = max(
        variants,
        key=lambda item: item["selection_metrics_without_labels"]["spearman_vs_exact_pair_score"],
    )
    chosen_sigma = float(chosen["sigma"])

    exact_features = np.column_stack((official_pair, raw_pair))
    exact_p2b_rank, exact_p2b_top, exact_intervention = frozen_p2b_ranks(
        exact_features, query_ptr, molecule_ptr
    )
    for variant in variants:
        soft_top = variant.pop("_soft_p2b_top")
        variant["post_selection_descriptive_retrieval"][
            "frozen_p2b_top_agreement_with_exact_nl_recipe"
        ] = float(np.mean(soft_top == exact_p2b_top))

    subset_size = min(args.psd_subset, len(measures))
    subset = np.linspace(0, len(measures) - 1, subset_size, dtype=np.int64)
    exact_gram = np.eye(subset_size, dtype=np.float64)
    soft_gram = np.eye(subset_size, dtype=np.float64)
    selected_measures = [measures[index] for index in subset]
    self_raw = np.asarray([
        gaussian_raw(*measure, *measure, chosen_sigma) for measure in selected_measures
    ])
    for i in range(subset_size):
        for j in range(i):
            exact = greedy_score(
                *selected_measures[i], *selected_measures[j], args.tolerance
            )
            raw = gaussian_raw(
                *selected_measures[i], *selected_measures[j], chosen_sigma
            )
            soft = raw / np.sqrt(max(self_raw[i] * self_raw[j], 1e-24))
            exact_gram[i, j] = exact_gram[j, i] = exact
            soft_gram[i, j] = soft_gram[j, i] = soft

    official_molecule = molecule_scores(official_pair, molecule_ptr)
    official_rank = query_ranks(official_molecule, query_ptr)
    output = {
        "status": "chemaware_neutral_loss_shared_kernel_audit_complete",
        "training_was_run": False,
        "graph": str(args.graph),
        "hdf5": str(args.hdf5),
        "queries": int(len(query_rows)),
        "reachable_spectra": int(len(reachable)),
        "pair_edges": int(len(candidate_rows)),
        "unique_undirected_pair_edges": int(len(edge_keys)),
        "exact_p2b_neutral_loss": {
            "tolerance_da": float(args.tolerance),
            "recall1": float(np.mean(exact_rank == 1)),
            "near_recall1": float(np.mean(exact_rank[near] == 1)),
            "psd_subset": gram_diagnostics(exact_gram),
            "implementation_crosscheck_max_abs_error": float(
                np.max(np.abs(exact_unique - manual_exact))
            ),
        },
        "frozen_p2b_recipe_on_local_graph": {
            "recall1": float(np.mean(exact_p2b_rank == 1)),
            "near_recall1": float(np.mean(exact_p2b_rank[near] == 1)),
            "intervention_rate": float(np.mean(exact_intervention)),
        },
        "official_dreams": {
            "recall1": float(np.mean(official_rank == 1)),
            "near_recall1": float(np.mean(official_rank[near] == 1)),
        },
        "gaussian_shared_kernel_variants": variants,
        "selection": {
            "uses_identity_labels": False,
            "criterion": "maximum Spearman agreement with frozen exact pair score",
            "chosen_sigma": chosen_sigma,
            "chosen_variant": chosen,
        },
        "chosen_gaussian_psd_subset": gram_diagnostics(soft_gram),
        "interpretation": (
            "The Gaussian set kernel is a valid shared-spectrum similarity. "
            "The exact greedy tolerance matcher is compatible with a single cosine only "
            "to the extent shown by the label-free approximation and PSD audits."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
