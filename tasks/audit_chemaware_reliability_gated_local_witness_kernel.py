"""Audit a chemistry-reliability gate on the centered local-witness kernel.

The gate is a frozen monotone function of one spectrum's rule-center
specificity.  It never sees a candidate set, identity, formula or outcome at
inference.  Multiplying the chemical feature block by this nonnegative scalar
therefore remains an ordinary shared PSD embedding.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from audit_chemaware_counterfactual_rule_kernel import bootstrap, rank_queries, retrieval
from audit_chemaware_counterfactual_rule_kernel_natural import (
    paired_rank_comparison,
    query_reference_dot,
    subset_graph,
)
from audit_chemaware_mass_kernel_embedding import KernelCache
from chemaware_iceberg_direct_core import stable_formula_folds
from chemaware_reliability_gated_kernel_core import (
    centered_reliability,
    gated_pair_score,
    monotone_gate,
    reliability_gated_embedding,
)
from noise_final_core import sha256_file
from train_chemaware_full_candidate_alignment import identity_balanced_queries


ROOT = Path(__file__).resolve().parents[1]
CENTER_NAMES = ("true", "local_a", "local_b", "local_c")
CACHE_VARIANTS = (
    "mass", "rule_response", "rule_response_local_background_a",
    "rule_response_local_background_b", "rule_response_local_background_c",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz",
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
        default=ROOT / "data/validation/chemaware_reliability_gated_local_witness_kernel_v1",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--inner-fold", type=int, default=3)
    parser.add_argument("--outer-fold", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--discovery-natural-identities", type=int, default=2048)
    parser.add_argument("--max-inner-identities", type=int, default=0)
    parser.add_argument("--top-peaks", type=int, default=32)
    parser.add_argument("--kernel-dim", type=int, default=2048)
    parser.add_argument("--bin-width", type=float, default=0.02)
    parser.add_argument("--grid-offsets", type=int, default=4)
    parser.add_argument("--intensity-power", type=float, default=0.5)
    parser.add_argument("--mass-shift-da", type=float, default=0.137)
    parser.add_argument("--pair-weight", type=float, default=0.25)
    parser.add_argument("--multi-bin-widths", type=float, nargs="+", default=(0.01, 0.02, 0.05))
    parser.add_argument("--uniform-channel-weight", type=float, default=1.0)
    parser.add_argument("--rule-tolerance", type=float, default=0.02)
    parser.add_argument("--rule-channel-weight", type=float, default=1.0)
    parser.add_argument(
        "--beta", type=float, nargs="+",
        default=(0.0, 0.025, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60),
    )
    parser.add_argument("--gate-floor", type=float, nargs="+", default=(0.0, 0.25, 0.50, 0.75))
    parser.add_argument("--gate-power", type=float, nargs="+", default=(0.5, 1.0, 2.0, 4.0))
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def normalize_rows(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    return np.divide(value, norm, out=np.zeros_like(value), where=norm > 1e-12)


def centered_arms(centers: list[np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    feature: dict[str, np.ndarray] = {}
    reliability: dict[str, np.ndarray] = {}
    for index, name in enumerate(CENTER_NAMES):
        background = tuple(value for other, value in enumerate(centers) if other != index)
        residual = centers[index] - np.mean(np.stack(background, axis=0), axis=0)
        feature[name] = normalize_rows(residual)
        reliability[name] = centered_reliability(centers[index], background)
    return feature, reliability


def make_pair_layout(compact: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    molecule_query = np.repeat(np.arange(len(compact["query_row"])), np.diff(compact["query_ptr"]))
    pair_molecule = np.repeat(np.arange(len(compact["molecule_label"])), np.diff(compact["molecule_ptr"]))
    return pair_molecule, molecule_query[pair_molecule]


def ranks_from_pairs(
    official_pair: np.ndarray,
    mass_pair: np.ndarray,
    chemical_pair: np.ndarray,
    query_gate_pair: np.ndarray,
    reference_gate: np.ndarray,
    compact: dict[str, np.ndarray],
    *,
    mass_beta: float,
    chemical_beta: float,
    query_limit: int | None = None,
) -> np.ndarray:
    if query_limit is None:
        query_limit = len(compact["query_row"])
    molecule_end = int(compact["query_ptr"][query_limit])
    pair_end = int(compact["molecule_ptr"][molecule_end])
    pair_score = gated_pair_score(
        official_pair[:pair_end], mass_pair[:pair_end], chemical_pair[:pair_end],
        query_gate_pair[:pair_end], reference_gate[:pair_end],
        mass_beta=mass_beta, chemical_beta=chemical_beta,
    )
    molecule_score = np.maximum.reduceat(pair_score, compact["molecule_ptr"][:molecule_end])
    return rank_queries(
        molecule_score, compact["query_ptr"][:query_limit + 1],
        compact["molecule_label"][:molecule_end],
    )


def selection_key(row: dict[str, object]) -> tuple[float, ...]:
    return (
        int(row["risk_utility_at_1"]), -int(row["introduced_at_1"]),
        float(row["delta_mrr"]), int(row["gate_kind"] == "constant"),
        -float(row["mass_beta"]), -float(row["chemical_beta"]),
    )


def select_arm(
    name: str,
    chemical_pair: np.ndarray,
    reliability: np.ndarray,
    official_pair: np.ndarray,
    mass_pair: np.ndarray,
    query_node_pair: np.ndarray,
    reference_node: np.ndarray,
    compact: dict[str, np.ndarray],
    baseline_discovery: np.ndarray,
    discovery_nodes: np.ndarray,
    discovery_count: int,
    args: argparse.Namespace,
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray, np.ndarray]:
    beta = tuple(map(float, args.beta))
    constant_gate = np.ones(len(reliability), dtype=np.float32)
    constant_rows: list[dict[str, object]] = []
    for mass_beta in beta:
        for chemical_beta in beta:
            rank = ranks_from_pairs(
                official_pair, mass_pair, chemical_pair,
                constant_gate[query_node_pair], constant_gate[reference_node], compact,
                mass_beta=mass_beta, chemical_beta=chemical_beta, query_limit=discovery_count,
            )
            constant_rows.append({
                "arm": name, "gate_kind": "constant", "floor": 1.0, "power": 1.0,
                "mass_beta": mass_beta, "chemical_beta": chemical_beta,
                **retrieval(baseline_discovery, rank),
            })
    constant_selected = max(constant_rows, key=selection_key)
    mass_index = beta.index(float(constant_selected["mass_beta"]))
    mass_candidates = tuple(dict.fromkeys(
        beta[max(0, min(len(beta) - 1, mass_index + offset))] for offset in (-1, 0, 1)
    ))
    calibration = reliability[discovery_nodes]
    rows = list(constant_rows)
    gate_cache: dict[tuple[float, float], np.ndarray] = {}
    for floor in map(float, args.gate_floor):
        for power in map(float, args.gate_power):
            gate = monotone_gate(reliability, calibration, floor=floor, power=power)
            gate_cache[(floor, power)] = gate
            qgate = gate[query_node_pair]
            rgate = gate[reference_node]
            for mass_beta in mass_candidates:
                for chemical_beta in beta:
                    rank = ranks_from_pairs(
                        official_pair, mass_pair, chemical_pair, qgate, rgate, compact,
                        mass_beta=mass_beta, chemical_beta=chemical_beta,
                        query_limit=discovery_count,
                    )
                    rows.append({
                        "arm": name, "gate_kind": "monotone_empirical_cdf",
                        "floor": floor, "power": power,
                        "mass_beta": mass_beta, "chemical_beta": chemical_beta,
                        **retrieval(baseline_discovery, rank),
                    })
    selected = max(rows, key=selection_key)
    if selected["gate_kind"] == "constant":
        selected_gate = constant_gate
    else:
        selected_gate = gate_cache[(float(selected["floor"]), float(selected["power"]))]
    return selected, rows, selected_gate, constant_gate


def main() -> None:
    args = arguments()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    if args.smoke:
        args.discovery_natural_identities = min(args.discovery_natural_identities, 192)
        args.max_inner_identities = 128
        args.bootstrap_draws = min(args.bootstrap_draws, 300)
        args.gate_floor = (0.0, 0.5)
        args.gate_power = (0.5, 2.0)
    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {key: np.asarray(loaded[key]) for key in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official_cache = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    fold = stable_formula_folds(body["query_formula"], args.folds, args.fold_seed)
    discovery_pool = np.flatnonzero((fold != args.inner_fold) & (fold != args.outer_fold))
    inner_pool = np.flatnonzero(fold == args.inner_fold)
    discovery = identity_balanced_queries(
        discovery_pool, body["query_ik14"], np.random.default_rng(args.seed + 72),
        args.discovery_natural_identities,
    )
    inner = identity_balanced_queries(
        inner_pool, body["query_ik14"], np.random.default_rng(args.seed + 19),
        args.max_inner_identities,
    )
    discovery_formula = body["query_formula"][discovery].astype(str)
    inner_formula = body["query_formula"][inner].astype(str)
    if set(discovery_formula) & set(inner_formula):
        raise RuntimeError("discovery and inner formulas overlap")
    selected_queries = np.concatenate((discovery, inner))
    compact = subset_graph(body, selected_queries)
    node_rows = np.unique(np.concatenate((compact["query_row"], compact["reference_row"]))).astype(np.int64)
    node_position = {int(row): index for index, row in enumerate(node_rows)}
    query_node = np.asarray([node_position[int(row)] for row in compact["query_row"]], dtype=np.int64)
    reference_node = np.asarray([node_position[int(row)] for row in compact["reference_row"]], dtype=np.int64)
    pair_molecule, pair_query = make_pair_layout(compact)
    query_node_pair = query_node[pair_query]
    positions = np.asarray([row_position[int(row)] for row in node_rows], dtype=np.int64)
    official = np.asarray(official_cache[positions], dtype=np.float32)

    cache_args = SimpleNamespace(**vars(args))
    cache = KernelCache(cache_args, row_position, variants=CACHE_VARIANTS)
    cached = [cache.get(int(row)) for row in node_rows]
    mass = np.stack([value["mass"] for value in cached])
    centers = [
        np.stack([value[name] for value in cached]).astype(np.float32)
        for name in (
            "rule_response", "rule_response_local_background_a",
            "rule_response_local_background_b", "rule_response_local_background_c",
        )
    ]
    chemical, reliability = centered_arms(centers)
    chemical["raw_rule"] = centers[0]
    official_pair = query_reference_dot(
        official, query_node, reference_node, compact["query_ptr"], compact["molecule_ptr"],
    )
    mass_pair = query_reference_dot(
        mass, query_node, reference_node, compact["query_ptr"], compact["molecule_ptr"],
    )
    chemical_pair = {
        name: query_reference_dot(
            feature, query_node, reference_node, compact["query_ptr"], compact["molecule_ptr"],
        )
        for name, feature in chemical.items()
    }
    baseline_rank = ranks_from_pairs(
        official_pair, mass_pair, chemical_pair["true"],
        np.ones(len(reference_node), dtype=np.float32),
        np.ones(len(reference_node), dtype=np.float32), compact,
        mass_beta=0.0, chemical_beta=0.0,
    )
    discovery_count = len(discovery)
    discovery_molecule_end = int(compact["query_ptr"][discovery_count])
    discovery_pair_end = int(compact["molecule_ptr"][discovery_molecule_end])
    discovery_nodes = np.unique(np.concatenate((
        query_node[:discovery_count], reference_node[:discovery_pair_end],
    )))

    selections: dict[str, dict[str, object]] = {}
    grids: dict[str, list[dict[str, object]]] = {}
    gates: dict[str, np.ndarray] = {}
    constant_gates: dict[str, np.ndarray] = {}
    for name in CENTER_NAMES:
        selected, grid, gate, constant = select_arm(
            name, chemical_pair[name], reliability[name], official_pair, mass_pair,
            query_node_pair, reference_node, compact, baseline_rank[:discovery_count],
            discovery_nodes, discovery_count, args,
        )
        selections[name] = selected; grids[name] = grid
        gates[name] = gate; constant_gates[name] = constant
        print(
            f"selected {name}: {selected['gate_kind']} floor={selected['floor']} "
            f"power={selected['power']} mass={selected['mass_beta']} chem={selected['chemical_beta']} ",
            flush=True,
        )

    # Raw-rule comparator deliberately has no center-specificity gate.
    raw_selected, raw_grid, raw_gate, _ = select_arm(
        "raw_rule", chemical_pair["raw_rule"], reliability["true"],
        official_pair, mass_pair, query_node_pair, reference_node, compact,
        baseline_rank[:discovery_count], discovery_nodes, discovery_count, args,
    )
    raw_constant_rows = [row for row in raw_grid if row["gate_kind"] == "constant"]
    raw_selected = max(raw_constant_rows, key=selection_key)
    raw_gate = np.ones(len(node_rows), dtype=np.float32)
    selections["raw_rule"] = raw_selected; grids["raw_rule"] = raw_constant_rows
    gates["raw_rule"] = raw_gate

    ranks: dict[str, np.ndarray] = {}
    for name in (*CENTER_NAMES, "raw_rule"):
        selected = selections[name]
        rank = ranks_from_pairs(
            official_pair, mass_pair, chemical_pair[name],
            gates[name][query_node_pair], gates[name][reference_node], compact,
            mass_beta=float(selected["mass_beta"]),
            chemical_beta=float(selected["chemical_beta"]),
        )
        ranks[name] = rank
    true_constant = max(
        (row for row in grids["true"] if row["gate_kind"] == "constant"), key=selection_key,
    )
    true_constant_rank = ranks_from_pairs(
        official_pair, mass_pair, chemical_pair["true"],
        constant_gates["true"][query_node_pair], constant_gates["true"][reference_node], compact,
        mass_beta=float(true_constant["mass_beta"]),
        chemical_beta=float(true_constant["chemical_beta"]),
    )

    inner_slice = slice(discovery_count, len(selected_queries))
    inner_baseline = baseline_rank[inner_slice]
    inner_results = {
        name: retrieval(inner_baseline, rank[inner_slice]) for name, rank in ranks.items()
    }
    inner_results["true_constant"] = retrieval(inner_baseline, true_constant_rank[inner_slice])
    comparisons = {
        "true_minus_true_constant": paired_rank_comparison(
            ranks["true"][inner_slice], true_constant_rank[inner_slice], inner_formula,
            draws=args.bootstrap_draws, seed=args.seed + 601,
        ),
        "true_minus_raw_rule": paired_rank_comparison(
            ranks["true"][inner_slice], ranks["raw_rule"][inner_slice], inner_formula,
            draws=args.bootstrap_draws, seed=args.seed + 602,
        ),
        **{
            f"true_minus_{name}": paired_rank_comparison(
                ranks["true"][inner_slice], ranks[name][inner_slice], inner_formula,
                draws=args.bootstrap_draws, seed=args.seed + 610 + index,
            )
            for index, name in enumerate(("local_a", "local_b", "local_c"))
        },
    }
    absolute_ci = bootstrap(
        inner_formula, inner_baseline, ranks["true"][inner_slice],
        args.bootstrap_draws, args.seed + 500,
    )
    true_result = inner_results["true"]

    # Explicitly verify the selected score against the claimed primal map on
    # sampled pairs without materializing the full high-dimensional embedding.
    audit_pair = np.linspace(0, len(reference_node) - 1, min(4096, len(reference_node)), dtype=np.int64)
    unique_nodes = np.unique(np.concatenate((query_node_pair[audit_pair], reference_node[audit_pair])))
    remap = {int(node): index for index, node in enumerate(unique_nodes)}
    phi = reliability_gated_embedding(
        official[unique_nodes], mass[unique_nodes], chemical["true"][unique_nodes],
        gates["true"][unique_nodes], mass_beta=float(selections["true"]["mass_beta"]),
        chemical_beta=float(selections["true"]["chemical_beta"]),
    )
    left = np.asarray([remap[int(node)] for node in query_node_pair[audit_pair]])
    right = np.asarray([remap[int(node)] for node in reference_node[audit_pair]])
    explicit = np.sum(phi[left] * phi[right], axis=1)
    direct = gated_pair_score(
        official_pair[audit_pair], mass_pair[audit_pair], chemical_pair["true"][audit_pair],
        gates["true"][query_node_pair[audit_pair]], gates["true"][reference_node[audit_pair]],
        mass_beta=float(selections["true"]["mass_beta"]),
        chemical_beta=float(selections["true"]["chemical_beta"]),
    )
    primal_error = float(np.max(np.abs(explicit - direct)))

    report_gates = {
        "absolute_formula_ci_positive": absolute_ci[0] > 0,
        "corrected_exceeds_twice_introduced": (
            int(true_result["corrected_at_1"]) > 2 * int(true_result["introduced_at_1"])
        ),
        "reliability_gate_selected": selections["true"]["gate_kind"] != "constant",
        "beats_constant_center_ci": comparisons["true_minus_true_constant"][
            "formula_cluster_bootstrap_delta_recall1_ci95"
        ][0] > 0,
        "beats_raw_rule_ci": comparisons["true_minus_raw_rule"][
            "formula_cluster_bootstrap_delta_recall1_ci95"
        ][0] > 0,
        **{
            f"beats_{name}_ci": comparisons[f"true_minus_{name}"][
                "formula_cluster_bootstrap_delta_recall1_ci95"
            ][0] > 0
            for name in ("local_a", "local_b", "local_c")
        },
        # The cached 2,048-D mass block is float16; different BLAS reduction
        # orders therefore agree to float16 accumulation precision, not 1e-5.
        "primal_dot_product_error_below_2e-4": primal_error < 2e-4,
        "outer_fold_untouched": True,
    }
    report = {
        "status": "CHEMAWARE_RELIABILITY_GATED_LOCAL_WITNESS_COMPLETE",
        "formal_training_authorized": False,
        "weights_updated": False,
        "scope": "natural folds 0-2 selection; used inner fold 3; outer fold 4 sealed",
        "claim_limit": (
            "Frozen shared-kernel development result on an already-used inner fold; "
            "not DreaMS parameter fine-tuning or external confirmation."
        ),
        "method": {
            "map": "[official, sqrt(mass_beta)*mass, sqrt(chemical_beta)*g(x)*centered_rule(x)]",
            "reliability": "L2 norm of center minus mean of the other three shared-coordinate centers",
            "gate": "floor + (1-floor)*frozen_empirical_CDF(reliability)^power",
            "same_map_for_query_and_reference": True,
            "candidate_formula_identity_free_at_inference": True,
            "symmetric_four_center_controls": True,
            "primal_dot_product_max_abs_error": primal_error,
        },
        "data": {
            "discovery_queries": int(len(discovery)), "inner_queries": int(len(inner)),
            "outer_queries_untouched": int(np.sum(fold == args.outer_fold)),
            "unique_spectrum_nodes": int(len(node_rows)), "formula_overlap": 0,
        },
        "selection_on_discovery": {
            name: {"selected": selections[name], "grid": grids[name]}
            for name in (*CENTER_NAMES, "raw_rule")
        },
        "true_constant_selected": true_constant,
        "held_inner": inner_results,
        "true_absolute_formula_cluster_bootstrap_delta_recall1_ci95": absolute_ci,
        "paired_inner": comparisons,
        "gates": report_gates,
        "provenance": {
            "manifest_sha256": sha256_file(args.manifest),
            "token_report_sha256": sha256_file(args.token_dir / "report.json"),
            "rule_library_sha256": sha256_file(args.rule_library),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_reliability_gate_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "inner_ranks.npz", query=inner, formula=inner_formula,
            baseline_rank=inner_baseline, true_rank=ranks["true"][inner_slice],
            true_constant_rank=true_constant_rank[inner_slice],
            raw_rule_rank=ranks["raw_rule"][inner_slice],
            local_a_rank=ranks["local_a"][inner_slice],
            local_b_rank=ranks["local_b"][inner_slice],
            local_c_rank=ranks["local_c"][inner_slice],
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"], "selected": selections,
        "true_constant_selected": true_constant, "held_inner": inner_results,
        "absolute_ci": absolute_ci, "paired_inner": comparisons,
        "gates": report_gates, "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
