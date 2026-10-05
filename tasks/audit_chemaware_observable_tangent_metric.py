"""Formula-held audit of a clean-observable ChemAware tangent metric field.

This audit never loads or updates the DreaMS backbone.  Candidate-conditioned
ICEBERG residuals define training-only positive-versus-negative margin targets.
The deployed map receives one clean spectrum through its frozen official
embedding and optional fixed chemical-rule responses, and returns one shared
embedding.  Correct, structure-swapped, and peak-permuted targets are fitted by
the identical convex procedure.

The setting grid is retrospective and is reported in full.  Its best row is a
diagnostic, not an unbiased performance estimate and never authorizes training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from chemaware_observable_tangent_metric_core import (
    apply_tangent_field,
    edge_design,
    fit_clean_gates,
    fit_tangent_fields,
    margin_design,
    tangent_basis_from_edges,
    unit_rows,
)


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("correct", "structure_swapped", "peak_permuted")
RESIDUAL = {
    "correct": "centered_residual",
    "structure_swapped": "structure_swapped_centered_residual",
    "peak_permuted": "peak_permuted_centered_residual",
}


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
        default=ROOT / "data/validation/chemaware_observable_tangent_metric_v1",
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--embedding-gates", type=int, default=4)
    parser.add_argument("--rule-gates", type=int, default=4)
    parser.add_argument(
        "--gate-modes", nargs="+",
        choices=("constant", "embedding", "rule", "combined"),
        default=("constant", "embedding", "rule", "combined"),
    )
    parser.add_argument("--ridge", type=float, nargs="+", default=(0.1, 1.0, 10.0))
    parser.add_argument("--dose", type=float, default=0.5)
    parser.add_argument("--top-peaks", type=int, default=32)
    parser.add_argument("--intensity-power", type=float, default=0.5)
    parser.add_argument("--rule-tolerance", type=float, default=0.02)
    parser.add_argument("--bootstrap-draws", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=20260912)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rule_response_features(
    node_rows: np.ndarray,
    cache_position: dict[int, int],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build the same clean NL/CF response vector used by the rule-mass kernel."""

    payload = json.loads(args.rule_library.read_text(encoding="utf-8"))
    if "channels" in payload:
        records = [
            {
                "category": item["category"], "match_type": item["match_type"],
                "value": item["value_da"],
            }
            for item in payload["channels"]
        ]
    else:
        records = payload["rules"]
    neutral_loss = np.asarray([
        float(item["value"]) for item in records
        if item.get("category") == "NL" and item.get("match_type") == "mass_diff"
    ], dtype=np.float64)
    characteristic = np.asarray([
        float(item["value"]) for item in records
        if item.get("category") == "CF" and item.get("match_type") == "peak_mz"
    ], dtype=np.float64)
    mz_all = np.load(args.token_dir / "mz_f32.npy", mmap_mode="r")
    intensity_all = np.load(args.token_dir / "intensity_f32.npy", mmap_mode="r")
    valid_all = np.load(args.token_dir / "valid.npy", mmap_mode="r")
    precursor_all = np.load(args.token_dir / "precursor_mz_f32.npy", mmap_mode="r")
    output = np.zeros((len(node_rows), len(neutral_loss) + len(characteristic)), dtype=np.float32)
    tolerance = float(args.rule_tolerance)

    def response(values: np.ndarray, targets: np.ndarray, weight: np.ndarray) -> np.ndarray:
        if not len(values) or not len(targets):
            return np.zeros(len(targets), dtype=np.float32)
        distance = np.abs(targets[:, None] - values[None, :])
        return np.max(np.maximum(0.0, 1.0 - distance / tolerance) * weight[None, :], axis=1)

    for index, row in enumerate(node_rows):
        position = cache_position[int(row)]
        available = np.flatnonzero(np.asarray(valid_all[position], dtype=bool))
        order = available[
            np.argsort(-np.asarray(intensity_all[position, available]), kind="stable")[:args.top_peaks]
        ]
        mz = np.asarray(mz_all[position, order], dtype=np.float64)
        intensity = np.maximum(np.asarray(intensity_all[position, order], dtype=np.float64), 0.0)
        intensity /= max(float(np.sum(intensity)), 1e-12)
        weight = np.power(intensity, args.intensity_power)
        precursor = float(precursor_all[position])
        vector = np.concatenate((
            response(precursor - mz, neutral_loss, weight),
            response(mz, characteristic, weight),
        ))
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        output[index] = vector
        if (index + 1) % 4096 == 0:
            print(f"clean rule responses {index + 1}/{len(node_rows)}", flush=True)
    return output, {
        "neutral_loss_channels": int(len(neutral_loss)),
        "characteristic_fragment_channels": int(len(characteristic)),
        "dimension": int(output.shape[1]),
        "nonzero_spectra": int(np.sum(np.linalg.norm(output, axis=1) > 0)),
    }


def rank(scores: np.ndarray, labels: np.ndarray) -> int:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    positive = float(scores[np.flatnonzero(labels)[0]])
    return 1 + int(np.sum(scores[~labels] >= positive))


def evaluate_queries(
    embeddings: np.ndarray,
    query_nodes: np.ndarray,
    query_ptr: np.ndarray,
    molecule_ptr: np.ndarray,
    molecule_labels: np.ndarray,
    reference_nodes: np.ndarray,
    queries: np.ndarray,
) -> tuple[np.ndarray, int]:
    ranks = []
    switches = 0
    for query in np.asarray(queries, dtype=np.int64):
        query_value = embeddings[query_nodes[query]]
        left, right = query_ptr[query : query + 2]
        scores = []
        for molecule in range(int(left), int(right)):
            ref_left, ref_right = molecule_ptr[molecule : molecule + 2]
            values = embeddings[reference_nodes[ref_left:ref_right]] @ query_value
            scores.append(float(np.max(values)))
            switches += int(np.argmax(values) != 0)  # overwritten by caller-independent exact count below
        ranks.append(rank(np.asarray(scores), molecule_labels[left:right]))
    return np.asarray(ranks, dtype=np.int64), int(switches)


def retrieval(old_rank: np.ndarray, new_rank: np.ndarray) -> dict[str, object]:
    old_rank = np.asarray(old_rank, dtype=np.int64)
    new_rank = np.asarray(new_rank, dtype=np.int64)
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


def formula_bootstrap(
    formulas: np.ndarray,
    old_rank: np.ndarray,
    new_rank: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> list[float]:
    formula, inverse = np.unique(np.asarray(formulas, dtype=str), return_inverse=True)
    delta = (np.asarray(new_rank) == 1).astype(float) - (np.asarray(old_rank) == 1).astype(float)
    sums = np.bincount(inverse, weights=delta)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=np.float64)
    for left in range(0, draws, 500):
        right = min(left + 500, draws)
        selected = rng.integers(0, len(formula), size=(right - left, len(formula)))
        values[left:right] = sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
    return [float(value) for value in np.quantile(values, (0.025, 0.975))]


def balanced_pair_weights(
    pair_query: np.ndarray,
    selected: np.ndarray,
    active: np.ndarray,
    safe: np.ndarray,
) -> np.ndarray:
    selected = np.asarray(selected, dtype=bool)
    pair_query = np.asarray(pair_query, dtype=np.int64)
    active_queries = np.flatnonzero(selected & active)
    safe_queries = np.flatnonzero(selected & safe)
    if not len(active_queries) or not len(safe_queries):
        raise RuntimeError("each training split needs corrective and clean-safety queries")
    pair_count = np.bincount(pair_query, minlength=len(selected))
    query_weight = np.zeros(len(selected), dtype=np.float64)
    query_weight[active_queries] = 0.5 / len(active_queries)
    query_weight[safe_queries] = 0.5 / len(safe_queries)
    weight = query_weight[pair_query] / np.maximum(pair_count[pair_query], 1)
    mask = selected[pair_query] & (active[pair_query] | safe[pair_query])
    weight = weight[mask]
    weight *= len(weight) / np.sum(weight)
    return weight


def main() -> None:
    args = arguments()
    if args.rank <= 0 or args.embedding_gates <= 0 or args.rule_gates <= 0:
        raise ValueError("rank and gate counts must be positive")
    if not 0 < args.dose <= 1 or any(value <= 0 for value in args.ridge):
        raise ValueError("dose and ridge values must be positive")
    graph_path = args.graph_dir / "graph.npz"
    required = [
        args.ledger, graph_path, args.token_dir / "rows.npy",
        args.token_dir / "official_embeddings_f32.npy", args.rule_library,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)

    ledger = np.load(args.ledger, allow_pickle=False)
    graph = np.load(graph_path, allow_pickle=True)
    if not np.array_equal(ledger["query_row"], graph["query_row"]):
        raise RuntimeError("ledger and graph query rows drifted")
    if not np.array_equal(
        np.asarray(ledger["molecule_ik14"]).astype(str),
        np.asarray(graph["molecule_ik14"]).astype(str),
    ):
        raise RuntimeError("ledger and graph candidate identities drifted")

    cache_rows = np.load(args.token_dir / "rows.npy", mmap_mode="r")
    official_cache = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    cache_position = {int(row): index for index, row in enumerate(cache_rows)}
    graph_rows = np.unique(np.concatenate((graph["query_row"], graph["pair_candidate_row"]))).astype(np.int64)
    absent = [int(row) for row in graph_rows if int(row) not in cache_position]
    if absent:
        raise RuntimeError(f"{len(absent)} graph rows are absent from the official cache")
    positions = np.asarray([cache_position[int(row)] for row in graph_rows], dtype=np.int64)
    embeddings = unit_rows(np.asarray(official_cache[positions], dtype=np.float64))
    node_position = {int(row): index for index, row in enumerate(graph_rows)}

    query_rows = np.asarray(graph["query_row"], dtype=np.int64)
    query_nodes = np.asarray([node_position[int(row)] for row in query_rows], dtype=np.int64)
    reference_rows = np.asarray(graph["pair_candidate_row"], dtype=np.int64)
    reference_nodes = np.asarray([node_position[int(row)] for row in reference_rows], dtype=np.int64)
    query_ptr = np.asarray(graph["query_ptr"], dtype=np.int64)
    molecule_ptr = np.asarray(graph["molecule_ptr"], dtype=np.int64)
    labels = np.asarray(graph["molecule_label"], dtype=bool)
    molecule_query = np.repeat(np.arange(len(query_rows)), np.diff(query_ptr))

    molecule_best_ref = np.empty(len(labels), dtype=np.int64)
    molecule_score = np.empty(len(labels), dtype=np.float64)
    official_reference_winner = np.empty(len(labels), dtype=np.int64)
    for molecule, query in enumerate(molecule_query):
        left, right = molecule_ptr[molecule : molecule + 2]
        values = embeddings[reference_nodes[left:right]] @ embeddings[query_nodes[query]]
        winner = int(np.argmax(values))
        molecule_best_ref[molecule] = reference_nodes[int(left) + winner]
        molecule_score[molecule] = float(values[winner])
        official_reference_winner[molecule] = winner

    baseline_rank = np.empty(len(query_rows), dtype=np.int64)
    positive_molecule = np.empty(len(query_rows), dtype=np.int64)
    pair_query = []
    pair_positive = []
    pair_negative = []
    for query in range(len(query_rows)):
        left, right = query_ptr[query : query + 2]
        local_labels = labels[left:right]
        positive = int(left) + int(np.flatnonzero(local_labels)[0])
        positive_molecule[query] = positive
        baseline_rank[query] = rank(molecule_score[left:right], local_labels)
        for negative in np.flatnonzero(~local_labels) + int(left):
            pair_query.append(query); pair_positive.append(positive); pair_negative.append(int(negative))
    pair_query = np.asarray(pair_query, dtype=np.int64)
    pair_positive = np.asarray(pair_positive, dtype=np.int64)
    pair_negative = np.asarray(pair_negative, dtype=np.int64)

    active = np.asarray(ledger["active_query"], dtype=bool)
    formula = np.asarray(ledger["query_formula"]).astype(str)
    formula_fold = np.asarray(ledger["query_formula_fold"], dtype=np.int64)
    folds = np.unique(formula_fold)
    if not np.array_equal(folds, np.asarray([0, 1, 2])):
        raise RuntimeError(f"expected three frozen formula folds, observed {folds.tolist()}")
    if np.any(baseline_rank[active] == 1):
        raise RuntimeError("strict corrective query is official-correct")
    safe = baseline_rank == 1

    pair_target = {}
    for arm, name in RESIDUAL.items():
        residual = np.asarray(ledger[name], dtype=np.float64)
        pair_target[arm] = args.dose * (residual[pair_positive] - residual[pair_negative])
    rule_features, rule_report = rule_response_features(graph_rows, cache_position, args)

    settings: dict[tuple[str, float], dict[str, object]] = {}
    for mode in args.gate_modes:
        for ridge in args.ridge:
            settings[(mode, float(ridge))] = {
                "mode": mode, "ridge": float(ridge),
                "ranks": {arm: np.full(len(query_rows), -1, dtype=np.int64) for arm in ARMS},
                "fold_reports": [],
            }

    all_molecule_edge = np.arange(len(labels), dtype=np.int64)
    for held_fold in folds:
        held_query_mask = formula_fold == held_fold
        train_query_mask = ~held_query_mask
        allowed_train = train_query_mask & (active | safe)
        train_molecules = allowed_train[molecule_query]
        train_nodes = np.unique(np.concatenate((
            query_nodes[allowed_train], molecule_best_ref[train_molecules],
        )))
        held_molecules = held_query_mask[molecule_query]
        held_nodes = np.unique(np.concatenate((
            query_nodes[held_query_mask], reference_nodes[np.concatenate([
                np.arange(molecule_ptr[m], molecule_ptr[m + 1])
                for m in np.flatnonzero(held_molecules)
            ])],
        )))
        basis = tangent_basis_from_edges(
            embeddings,
            query_nodes[molecule_query[train_molecules]],
            molecule_best_ref[train_molecules],
            rank=args.rank,
            seed=args.seed + int(held_fold),
        )
        embedding_gate, embedding_gate_report = fit_clean_gates(
            embeddings[train_nodes], embeddings,
            components=args.embedding_gates, seed=args.seed + 100 + int(held_fold),
        )
        rule_gate, rule_gate_report = fit_clean_gates(
            rule_features[train_nodes], rule_features,
            components=args.rule_gates, seed=args.seed + 200 + int(held_fold),
        )
        ones = np.ones((len(graph_rows), 1), dtype=np.float64)
        gate_by_mode = {
            "constant": ones,
            "embedding": np.concatenate((ones, embedding_gate), axis=1),
            "rule": np.concatenate((ones, rule_gate), axis=1),
            "combined": np.concatenate((ones, embedding_gate, rule_gate), axis=1),
        }
        train_pair_mask = allowed_train[pair_query]
        weight = balanced_pair_weights(pair_query, train_query_mask, active, safe)
        targets = np.stack([
            np.where(active[pair_query], pair_target[arm], 0.0)[train_pair_mask]
            for arm in ARMS
        ], axis=1)
        held_queries = np.flatnonzero(held_query_mask)

        for mode in args.gate_modes:
            gates = gate_by_mode[mode]
            edge = edge_design(
                embeddings, basis, gates,
                query_nodes[molecule_query], molecule_best_ref,
            )
            pair = margin_design(edge, pair_positive, pair_negative)
            train_design = pair[train_pair_mask]
            for ridge in args.ridge:
                fits = fit_tangent_fields(
                    train_design, targets, weight, basis=basis,
                    gates=gates.shape[1], ridge=float(ridge),
                )
                setting = settings[(mode, float(ridge))]
                fold_arm = {}
                for arm, fit in zip(ARMS, fits, strict=True):
                    updated, adapter_audit = apply_tangent_field(embeddings, gates, fit)
                    ranks, _unused = evaluate_queries(
                        updated, query_nodes, query_ptr, molecule_ptr, labels,
                        reference_nodes, held_queries,
                    )
                    setting["ranks"][arm][held_queries] = ranks
                    fold_arm[arm] = {
                        "training_rms": fit.training_rms,
                        "training_weighted_rms": fit.training_weighted_rms,
                        "effective_columns": fit.effective_columns,
                        "held": retrieval(baseline_rank[held_queries], ranks),
                        "adapter": adapter_audit,
                    }
                setting["fold_reports"].append({
                    "held_fold": int(held_fold),
                    "train_queries": int(np.sum(train_query_mask)),
                    "train_corrective_queries": int(np.sum(train_query_mask & active)),
                    "train_clean_safety_queries": int(np.sum(train_query_mask & safe)),
                    "held_queries": int(len(held_queries)),
                    "train_held_formula_overlap": int(len(
                        set(formula[train_query_mask]) & set(formula[held_query_mask])
                    )),
                    "train_held_spectrum_node_overlap": int(len(set(train_nodes) & set(held_nodes))),
                    "basis_rank": int(basis.shape[1]),
                    "gates": int(gates.shape[1]),
                    "embedding_gate_fit": embedding_gate_report,
                    "rule_gate_fit": rule_gate_report,
                    "arms": fold_arm,
                })
            del edge, pair, train_design
        print(f"completed held formula fold {held_fold}", flush=True)

    setting_reports = []
    for index, ((mode, ridge), setting) in enumerate(settings.items()):
        arm_reports = {}
        for arm in ARMS:
            ranks = setting["ranks"][arm]
            if np.any(ranks < 1):
                raise RuntimeError(f"OOF rank coverage is incomplete for {mode}/{ridge}/{arm}")
            arm_reports[arm] = {
                "all_queries": retrieval(baseline_rank, ranks),
                "strict_corrective_queries": retrieval(baseline_rank[active], ranks[active]),
                "baseline_correct_safety_queries": retrieval(baseline_rank[safe], ranks[safe]),
                "formula_cluster_bootstrap_delta_recall1_ci95": formula_bootstrap(
                    formula, baseline_rank, ranks,
                    draws=args.bootstrap_draws, seed=args.seed + 1000 + 10 * index + ARMS.index(arm),
                ),
            }
        correct_delta = float(arm_reports["correct"]["all_queries"]["delta_recall1"])
        setting_reports.append({
            "mode": mode,
            "ridge": ridge,
            "parameters_per_fold": int(
                setting["fold_reports"][0]["gates"] * args.rank * (args.rank + 1) // 2
            ),
            "arms": arm_reports,
            "specificity": {
                "correct_minus_structure_swapped_delta_recall1": correct_delta - float(
                    arm_reports["structure_swapped"]["all_queries"]["delta_recall1"]
                ),
                "correct_minus_peak_permuted_delta_recall1": correct_delta - float(
                    arm_reports["peak_permuted"]["all_queries"]["delta_recall1"]
                ),
            },
            "fold_reports": setting["fold_reports"],
        })
    best = max(
        setting_reports,
        key=lambda item: (
            int(item["arms"]["correct"]["all_queries"]["risk_utility_at_1"]),
            -int(item["arms"]["correct"]["all_queries"]["introduced_at_1"]),
            float(item["arms"]["correct"]["all_queries"]["delta_mrr"]),
            -int(item["parameters_per_fold"]),
        ),
    )
    best_summary = {
        key: best[key] for key in ("mode", "ridge", "parameters_per_fold", "arms", "specificity")
    }
    report = {
        "status": "CHEMAWARE_OBSERVABLE_TANGENT_METRIC_RETROSPECTIVE_COMPLETE",
        "formal_training_authorized": False,
        "weights_updated": False,
        "claim_limit": (
            "Three-fold formula-held OOF is a clean-observability diagnostic.  The same OOF grid "
            "was inspected to name the best setting, so its value is not an unbiased model-performance claim."
        ),
        "method": {
            "name": "conditional low-rank observable tangent metric field",
            "shared_embedding": "normalize(z + (I-zzT) Q [sum_r psi_r(x) B_r] QT z)",
            "training_target": "positive-minus-each-negative ICEBERG residual margin",
            "objective": "balanced corrective/safety weighted ridge on first-order candidate margins",
            "convex_given_basis_and_gates": True,
            "basis_target_free_and_fit_on_training_formula_rows_only": True,
            "gates_target_free_and_fit_on_training_formula_rows_only": True,
            "candidate_structure_available_at_inference": False,
            "candidate_set_available_at_inference": False,
            "correct_and_controls_use_identical_optimizer": True,
        },
        "data": {
            "queries": int(len(query_rows)),
            "formula_clusters": int(len(np.unique(formula))),
            "strict_corrective_queries": int(np.sum(active)),
            "baseline_correct_safety_queries": int(np.sum(safe)),
            "candidate_molecules": int(len(labels)),
            "positive_negative_margin_pairs": int(len(pair_query)),
            "unique_graph_spectrum_rows": int(len(graph_rows)),
            "baseline": retrieval(baseline_rank, baseline_rank),
        },
        "rule_features": rule_report,
        "grid": {
            "rank": args.rank,
            "embedding_gates": args.embedding_gates,
            "rule_gates": args.rule_gates,
            "gate_modes": list(args.gate_modes),
            "ridge": list(map(float, args.ridge)),
            "dose": args.dose,
        },
        "settings": setting_reports,
        "retrospective_best_diagnostic": best_summary,
        "provenance": {
            "ledger": str(args.ledger.resolve()),
            "ledger_sha256": sha256(args.ledger),
            "graph": str(graph_path.resolve()),
            "graph_sha256": sha256(graph_path),
            "official_embeddings_sha256": sha256(args.token_dir / "official_embeddings_f32.npy"),
            "rule_library_sha256": sha256(args.rule_library),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_otmf_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "oof_ranks.npz",
            baseline_rank=baseline_rank, active_query=active, formula=formula,
            **{
                f"{item['mode']}_ridge_{item['ridge']}_{arm}": settings[(item["mode"], item["ridge"])]["ranks"][arm]
                for item in setting_reports for arm in ARMS
            },
        )
        if args.output.exists():
            shutil.rmtree(args.output)
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"],
        "best_mode": best["mode"],
        "best_ridge": best["ridge"],
        "best_correct_all": best["arms"]["correct"]["all_queries"],
        "specificity": best["specificity"],
        "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
