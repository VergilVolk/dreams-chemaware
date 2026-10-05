"""Validate the strongest ChemAware action policy with structure fingerprints.

The formula-IDF policy supplies a high-recall candidate action.  A separately
fit, formula-disjoint spectrum-to-Morgan decoder may (1) veto that action when
the selected candidate lacks relative structural support, or (2) add its own
candidate only where the formula policy abstains.  This tests complementarity,
not a replacement ranker.  An exact-formula fingerprint permutation receives
the same decoder, calibration, and two-threshold policy capacity.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import sklearn
from rdkit import rdBase

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_chemaware_candidate_fingerprint_decoder import (
    fit_family,
    identity_embedding_table,
    morgan_matrix,
    reliability,
    weighted_support,
)
from audit_chemaware_counterfactual_rule_kernel import bootstrap, retrieval, sha256
from audit_chemaware_counterfactual_rule_kernel_natural import paired_rank_comparison
from audit_chemaware_hierarchical_predicate_support import (
    array_sha256,
    candidate_geometry,
    fused_proposal,
)
from chemaware_iceberg_direct_core import stable_formula_folds
from chemaware_predicate_support_core import stable_nontrivial_formula_permutation


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1")
    parser.add_argument("--structure-dir", type=Path, default=ROOT / "data/validation/chemaware_chemberta_teacher_v1")
    parser.add_argument("--formula-policy", type=Path, default=ROOT / "data/validation/chemaware_formula_gated_candidate_policy_v2_replay")
    parser.add_argument("--fingerprint-screen", type=Path, default=ROOT / "data/validation/chemaware_candidate_fingerprint_decoder_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/chemaware_fingerprint_validated_action_union_v1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--fingerprint-bits", type=int, default=512)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--minimum-total-formulas", type=int, default=50)
    parser.add_argument("--minimum-added-formulas", type=int, default=10)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    return parser.parse_args()


def candidate_support_delta(
    support: np.ndarray,
    selected_candidate: np.ndarray,
    baseline_candidate: np.ndarray,
) -> np.ndarray:
    output = np.full(len(support), np.nan, dtype=np.float32)
    for index in range(len(support)):
        selected = int(selected_candidate[index])
        baseline = int(baseline_candidate[index])
        if selected >= 0 and selected != baseline:
            values = np.asarray(support[index], dtype=np.float32)
            output[index] = values[selected] - values[baseline]
    return output


def threshold_grid(values: np.ndarray, points: int = 41) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.asarray([np.inf])
    return np.unique(np.r_[
        -np.inf,
        np.quantile(finite, np.linspace(0.0, 1.0, min(points, len(finite)))),
        np.nextafter(np.max(finite), np.inf),
        np.inf,
    ])


def apply_union(
    baseline_rank: np.ndarray,
    current_rank: np.ndarray,
    current_delta: np.ndarray,
    fingerprint_rank: np.ndarray,
    fingerprint_confidence: np.ndarray,
    keep_threshold: float,
    add_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current_action = current_rank != baseline_rank
    keep = current_action & np.isfinite(current_delta) & (current_delta >= keep_threshold)
    add = (~current_action) & np.isfinite(fingerprint_confidence) & (
        fingerprint_confidence >= add_threshold
    )
    rank = np.asarray(baseline_rank).copy()
    rank[keep] = current_rank[keep]
    rank[add] = fingerprint_rank[add]
    return rank.astype(np.int16), keep, add


def select_union(
    baseline_rank: np.ndarray,
    current_rank: np.ndarray,
    current_delta: np.ndarray,
    fingerprint_rank: np.ndarray,
    fingerprint_confidence: np.ndarray,
    formulas: np.ndarray,
    minimum_total_formulas: int,
    minimum_added_formulas: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for keep_threshold in threshold_grid(current_delta):
        for add_threshold in threshold_grid(fingerprint_confidence):
            rank, keep, add = apply_union(
                baseline_rank, current_rank, current_delta,
                fingerprint_rank, fingerprint_confidence,
                float(keep_threshold), float(add_threshold),
            )
            total = keep | add
            total_formulas = len(np.unique(formulas[total]))
            added_formulas = len(np.unique(formulas[add]))
            if np.any(total) and total_formulas < minimum_total_formulas:
                continue
            if np.any(add) and added_formulas < minimum_added_formulas:
                continue
            rows.append({
                "keep_threshold": float(keep_threshold),
                "add_threshold": float(add_threshold),
                "kept_current_actions": int(np.sum(keep)),
                "added_fingerprint_actions": int(np.sum(add)),
                "selected_formulas": int(total_formulas),
                "added_formulas": int(added_formulas),
                **retrieval(baseline_rank, rank),
            })
    if not rows:
        raise RuntimeError("no admissible union threshold pair")
    return max(rows, key=lambda row: (
        int(row["risk_utility_at_1"]), float(row["delta_recall1"]),
        float(row["delta_mrr"]), -int(row["kept_current_actions"]),
        -int(row["added_fingerprint_actions"]),
    )), rows


def prepare_policy(path: Path, expected_query: np.ndarray) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        body = {key: np.asarray(loaded[key]) for key in loaded.files}
    if not np.array_equal(body["query"], expected_query):
        raise RuntimeError(f"policy query set drifted: {path}")
    return body


def main() -> None:
    args = arguments()
    started = time.time()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.bootstrap_draws < 10_000 or args.minimum_total_formulas < 50:
        raise ValueError("union audit support contract was weakened")
    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {key: np.asarray(loaded[key]) for key in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    structure_identity = np.load(args.structure_dir / "identities.npy").astype(str)
    structure_formula = np.load(args.structure_dir / "formulas.npy").astype(str)
    structure_smiles = np.load(args.structure_dir / "canonical_smiles.npy").astype(str)
    structure_position = {value: index for index, value in enumerate(structure_identity)}
    fingerprint, fingerprint_report = morgan_matrix(
        structure_smiles, args.fingerprint_radius, args.fingerprint_bits,
    )
    permutation = stable_nontrivial_formula_permutation(
        structure_identity, structure_formula, args.seed + 2701,
    )
    candidates_by_arm = {
        "correct_fingerprint": fingerprint,
        "within_formula_fingerprint_permuted": fingerprint[permutation],
    }
    screen = json.loads((args.fingerprint_screen / "report.json").read_text(encoding="utf-8"))
    if screen.get("status") != "CHEMAWARE_CANDIDATE_FINGERPRINT_DECODER_COMPLETE":
        raise RuntimeError("fingerprint architecture screen is invalid")

    validation_policy = np.load(args.formula_policy / "validation_policy.npz", allow_pickle=False)
    inner_policy = np.load(args.formula_policy / "inner_policy.npz", allow_pickle=False)
    validation_query = np.asarray(validation_policy["query"], dtype=np.int64)
    inner_query = np.asarray(inner_policy["query"], dtype=np.int64)
    queries = np.concatenate((validation_query, inner_query))
    geometry_all = candidate_geometry(queries, body, official, row_position, structure_position)
    cut = len(validation_query)
    geometries = [
        {key: value[:cut] for key, value in geometry_all.items()},
        {key: value[cut:] for key, value in geometry_all.items()},
    ]
    query_feature = np.asarray(
        official[[row_position[int(body["query_row"][q])] for q in queries]],
        dtype=np.float32,
    )
    query_features = [query_feature[:cut], query_feature[cut:]]
    policies = [
        prepare_policy(args.formula_policy / "validation_policy.npz", validation_query),
        prepare_policy(args.formula_policy / "inner_policy.npz", inner_query),
    ]
    # Normalize the two file schemas.
    policies[1]["policy_rank"] = policies[1]["formula_idf_rank"]
    formulas = [body["query_formula"][query].astype(str) for query in (validation_query, inner_query)]

    identity_feature, identity_structure, identity_formula = identity_embedding_table(
        body, official, row_position, structure_position,
    )
    identity_fold = stable_formula_folds(identity_formula, args.folds, args.fold_seed)
    reliability_identity = np.flatnonzero(identity_fold == 2)
    arm_results: dict[str, dict[str, object]] = {}
    arm_arrays: dict[str, dict[str, np.ndarray]] = {}
    for arm_name, candidate_fingerprint in candidates_by_arm.items():
        selected_architecture = screen["arms"][arm_name]["architecture"]
        ridge = float(selected_architecture["ridge"])
        rank_value = int(selected_architecture["rank"])
        gamma = float(selected_architecture["gamma"])
        identity_target = candidate_fingerprint[identity_structure]
        family, scale, fit_report = fit_family(
            identity_feature, identity_target, identity_formula, identity_fold,
            (0, 1), (ridge,), (rank_value,),
        )
        coefficient = family[(ridge, rank_value)]
        prediction = (identity_feature[reliability_identity] / scale) @ coefficient
        bit_reliability, reliability_report = reliability(
            prediction, identity_target[reliability_identity],
            identity_formula[reliability_identity],
        )
        supports = [
            weighted_support(feature, geometry, candidate_fingerprint, coefficient, scale, bit_reliability)
            for feature, geometry in zip(query_features, geometries, strict=True)
        ]
        proposals = [
            fused_proposal(geometry, support, gamma)
            for geometry, support in zip(geometries, supports, strict=True)
        ]
        current_delta = [
            candidate_support_delta(
                support, policy["selected_candidate_index"], policy["baseline_candidate_index"],
            )
            for support, policy in zip(supports, policies, strict=True)
        ]
        selected, search = select_union(
            policies[0]["baseline_rank"], policies[0]["policy_rank"], current_delta[0],
            proposals[0]["rank"], proposals[0]["confidence"], formulas[0],
            args.minimum_total_formulas, args.minimum_added_formulas,
        )
        inner_rank, keep, add = apply_union(
            policies[1]["baseline_rank"], policies[1]["policy_rank"], current_delta[1],
            proposals[1]["rank"], proposals[1]["confidence"],
            float(selected["keep_threshold"]), float(selected["add_threshold"]),
        )
        arm_results[arm_name] = {
            "decoder_architecture": {"ridge": ridge, "rank": rank_value, "gamma": gamma},
            "decoder_fit": fit_report,
            "bit_reliability": reliability_report,
            "validation_selection": selected,
            "validation_search_pairs": int(len(search)),
            "held_inner": retrieval(policies[1]["baseline_rank"], inner_rank),
            "held_inner_formula_bootstrap_delta_recall1_ci95": bootstrap(
                formulas[1], policies[1]["baseline_rank"], inner_rank,
                args.bootstrap_draws, args.seed + 800,
            ),
            "held_inner_kept_current_actions": int(np.sum(keep)),
            "held_inner_added_fingerprint_actions": int(np.sum(add)),
            "coefficient_sha256": array_sha256(coefficient),
            "reliability_sha256": array_sha256(bit_reliability),
        }
        arm_arrays[arm_name] = {
            "rank": inner_rank, "keep": keep, "add": add,
            "current_support_delta": current_delta[1],
            "fingerprint_proposal_rank": proposals[1]["rank"],
            "fingerprint_confidence": proposals[1]["confidence"],
        }
        print(f"completed {arm_name}: {arm_results[arm_name]['held_inner']}", flush=True)

    baseline_rank = policies[1]["baseline_rank"]
    current_rank = policies[1]["policy_rank"]
    correct_rank = arm_arrays["correct_fingerprint"]["rank"]
    control_rank = arm_arrays["within_formula_fingerprint_permuted"]["rank"]
    paired = {
        "correct_union_minus_current_formula_policy": paired_rank_comparison(
            correct_rank, current_rank, formulas[1], draws=args.bootstrap_draws, seed=args.seed + 901,
        ),
        "correct_union_minus_permuted_union": paired_rank_comparison(
            correct_rank, control_rank, formulas[1], draws=args.bootstrap_draws, seed=args.seed + 902,
        ),
    }
    oracle_rank = np.minimum.reduce((
        baseline_rank, current_rank,
        arm_arrays["correct_fingerprint"]["fingerprint_proposal_rank"],
    ))
    report = {
        "status": "CHEMAWARE_FINGERPRINT_VALIDATED_ACTION_UNION_COMPLETE",
        "formal_training_authorized": False,
        "weights_updated": False,
        "candidate_conditioned": True,
        "shared_embedding_result": False,
        "scope": "existing formula policy plus fold01 decoder; fold2 two-threshold calibration; used fold3; fold4 sealed",
        "claim_limit": "Development action-composition audit, not shared-embedding or external-confirmation performance.",
        "method": {
            "primary_action": "frozen formula-IDF candidate policy",
            "chemical_validator": "formula-differential spectrum-to-Morgan support for selected candidate versus official top candidate",
            "union": "veto unsupported current actions; add high-confidence fingerprint action only where current policy abstains",
            "control": "identical pipeline with exact-formula fingerprint permutation",
            "threshold_dimensions": 2,
            "minimum_total_formulas": args.minimum_total_formulas,
            "minimum_added_formulas": args.minimum_added_formulas,
        },
        "data": {
            "validation_queries": int(len(validation_query)), "inner_queries": int(len(inner_query)),
            "outer_queries_untouched": int(np.sum(stable_formula_folds(body["query_formula"], args.folds, args.fold_seed) == 4)),
            "formula_overlap": int(len(set(formulas[0]) & set(formulas[1]))),
            "fingerprint": fingerprint_report,
        },
        "current_formula_policy_held_inner": retrieval(baseline_rank, current_rank),
        "arms": arm_results,
        "paired_inner": paired,
        "held_inner_joint_truth_oracle": {
            **retrieval(baseline_rank, oracle_rank),
            "claim_limit": "truth-selected headroom only",
        },
        "gates": {
            "absolute_formula_ci_positive": arm_results["correct_fingerprint"]["held_inner_formula_bootstrap_delta_recall1_ci95"][0] > 0,
            "increment_over_current_policy_ci_positive": paired["correct_union_minus_current_formula_policy"]["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
            "beats_permuted_union_ci": paired["correct_union_minus_permuted_union"]["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
            "at_least_four_pp": arm_results["correct_fingerprint"]["held_inner"]["delta_recall1"] >= 0.04,
            "outer_fold_untouched": True,
        },
        "replay_contract": {
            "arguments": {key: (str(value.resolve()) if isinstance(value, Path) else value) for key, value in vars(args).items()},
            "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "scikit_learn": sklearn.__version__, "rdkit": rdBase.rdkitVersion},
            "validation_query_sha256": array_sha256(validation_query),
            "inner_query_sha256": array_sha256(inner_query),
            "fingerprint_sha256": array_sha256(fingerprint),
            "permutation_sha256": array_sha256(permutation),
        },
        "provenance": {
            "manifest_sha256": sha256(args.manifest),
            "official_embeddings_sha256": sha256(args.token_dir / "official_embeddings_f32.npy"),
            "formula_policy_report_sha256": sha256(args.formula_policy / "report.json"),
            "fingerprint_screen_report_sha256": sha256(args.fingerprint_screen / "report.json"),
        },
        "runtime_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_validated_union_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "inner_policy.npz", query=inner_query, formula=formulas[1],
            baseline_rank=baseline_rank, current_policy_rank=current_rank,
            correct_union_rank=correct_rank, permuted_union_rank=control_rank,
            joint_truth_oracle_rank=oracle_rank,
            correct_keep=arm_arrays["correct_fingerprint"]["keep"],
            correct_add=arm_arrays["correct_fingerprint"]["add"],
            correct_current_support_delta=arm_arrays["correct_fingerprint"]["current_support_delta"],
            correct_fingerprint_confidence=arm_arrays["correct_fingerprint"]["fingerprint_confidence"],
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"], "current": report["current_formula_policy_held_inner"],
        "arms": arm_results, "paired_inner": paired,
        "joint_truth_oracle": report["held_inner_joint_truth_oracle"],
        "gates": report["gates"], "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
