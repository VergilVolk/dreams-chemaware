"""Audit a candidate-specific hierarchical ChemAware support score.

This experiment closes the main mismatch in the legacy spectrum-only rule
kernel: an observed fragment/loss matters only when a *candidate structure*
contains a compatible structural predicate.  A ridge-shrunk observation to
predicate map is learned from within-(formula, acquisition-domain) contrasts.
It never sees retrieval labels.  Candidate-set centring removes every
predicate shared by all same-formula candidates.

Protocol
--------
* formula fold 0: fit candidate-support maps;
* formula fold 1: choose ridge rank and fusion dose;
* folds 0-1: refit the frozen selected map;
* formula fold 2: freeze a one-dimensional abstention threshold;
* formula fold 3: used inner evaluation;
* formula fold 4: sealed and never evaluated.

The equal-capacity causal control cyclically permutes predicate fingerprints
within exact molecular formula before both map fitting and candidate scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import sklearn
from rdkit import Chem, rdBase

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_chemaware_counterfactual_rule_kernel import bootstrap, retrieval, sha256
from audit_chemaware_counterfactual_rule_kernel_natural import paired_rank_comparison
from audit_chemaware_mass_kernel_embedding import strict_rank
from chemaware_iceberg_direct_core import stable_formula_folds
from chemaware_predicate_support_core import (
    candidate_compatibility,
    choose_abstention_threshold,
    formula_domain_centered_design,
    ridge_support_family,
    stable_nontrivial_formula_permutation,
)
from train_chemaware_full_candidate_alignment import identity_balanced_queries


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz")
    parser.add_argument("--token-dir", type=Path, default=ROOT / "data/validation/chemaware_corrected_manifest_tokens_v1")
    parser.add_argument("--evidence", type=Path, default=ROOT / "data/validation/chemaware_domain_conditioned_action_rules_local_runtime_assets_20260907_v1/rules/identity_domain_evidence.npz")
    parser.add_argument("--predicates", type=Path, default=ROOT / "dreams/models/chem_aware/chem_parent_predicates_rdkit_v1.json")
    parser.add_argument("--observations", type=Path, default=ROOT / "dreams/models/chem_aware/chem_observation_channels_v2.json")
    parser.add_argument("--structure-dir", type=Path, default=ROOT / "data/validation/chemaware_chemberta_teacher_v1")
    parser.add_argument("--current-policy", type=Path, default=ROOT / "data/validation/chemaware_formula_gated_candidate_policy_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/chemaware_hierarchical_predicate_support_v1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--sampling-seed", type=int, default=20260905)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--architecture-identities", type=int, default=2048)
    parser.add_argument("--calibration-identities", type=int, default=2048)
    parser.add_argument("--max-inner-identities", type=int, default=0)
    parser.add_argument("--ridges", type=float, nargs="+", default=(1.0, 10.0, 100.0, 1000.0))
    parser.add_argument("--ranks", type=int, nargs="+", default=(4, 8, 16, 0))
    parser.add_argument("--gamma", type=float, nargs="+", default=(0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32))
    parser.add_argument("--intensity-threshold", type=float, default=0.01)
    parser.add_argument("--minimum-calibration-formulas", type=int, default=50)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    return parser.parse_args()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def build_predicate_matrix(
    identities: np.ndarray,
    smiles: np.ndarray,
    predicate_records: list[dict],
) -> tuple[np.ndarray, dict[str, object]]:
    molecules = [Chem.MolFromSmiles(str(value)) for value in smiles]
    valid = np.asarray([molecule is not None for molecule in molecules])
    matrix = np.zeros((len(identities), len(predicate_records)), dtype=bool)
    for column, record in enumerate(predicate_records):
        smarts = [Chem.MolFromSmarts(str(value)) for value in record["smarts_any"]]
        if any(value is None for value in smarts):
            raise RuntimeError(f"invalid SMARTS: {record.get('predicate_id')}")
        matrix[:, column] = [
            bool(molecule is not None and any(molecule.HasSubstructMatch(query) for query in smarts))
            for molecule in molecules
        ]
    return matrix, {
        "identities": int(len(identities)),
        "valid_structures": int(np.sum(valid)),
        "predicates": int(matrix.shape[1]),
        "predicate_prevalence_quantiles": np.quantile(matrix.mean(axis=0), [0, .25, .5, .75, 1]).tolist(),
    }


def nearest_detect(mz: np.ndarray, target: np.ndarray, tolerance: np.ndarray) -> np.ndarray:
    if not len(mz):
        return np.zeros(len(target), dtype=bool)
    mz = np.sort(np.asarray(mz, dtype=np.float64))
    position = np.searchsorted(mz, target)
    left = np.maximum(position - 1, 0)
    right = np.minimum(position, len(mz) - 1)
    distance = np.minimum(np.abs(mz[left] - target), np.abs(mz[right] - target))
    return distance <= tolerance


def query_observations(
    queries: np.ndarray,
    body: dict[str, np.ndarray],
    row_position: dict[int, int],
    mz_cache: np.ndarray,
    intensity_cache: np.ndarray,
    precursor_cache: np.ndarray,
    channels: list[dict],
    intensity_threshold: float,
) -> np.ndarray:
    target = np.asarray([value["value_da"] for value in channels], dtype=np.float64)
    tolerance = np.asarray([value["match_tolerance_da"] for value in channels], dtype=np.float64)
    neutral = np.asarray([value["match_type"] == "mass_diff" for value in channels])
    output = np.zeros((len(queries), len(channels)), dtype=np.float32)
    for out, query in enumerate(map(int, queries)):
        position = row_position[int(body["query_row"][query])]
        mz = np.asarray(mz_cache[position], dtype=np.float64)
        intensity = np.asarray(intensity_cache[position], dtype=np.float64)
        valid = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity >= intensity_threshold)
        mz = mz[valid]
        output[out, ~neutral] = nearest_detect(mz, target[~neutral], tolerance[~neutral])
        loss = float(precursor_cache[position]) - mz
        output[out, neutral] = nearest_detect(loss[loss > 0], target[neutral], tolerance[neutral])
    return output


def candidate_geometry(
    queries: np.ndarray,
    body: dict[str, np.ndarray],
    official: np.ndarray,
    row_position: dict[int, int],
    structure_position: dict[str, int],
) -> dict[str, np.ndarray]:
    score = np.empty(len(queries), dtype=object)
    structure = np.empty(len(queries), dtype=object)
    labels = np.empty(len(queries), dtype=object)
    baseline_rank = np.empty(len(queries), dtype=np.int16)
    baseline_candidate = np.empty(len(queries), dtype=np.int16)
    for out, query in enumerate(map(int, queries)):
        left, right = map(int, body["query_ptr"][query:query + 2])
        local_labels = body["molecule_label"][left:right].astype(bool)
        qposition = row_position[int(body["query_row"][query])]
        qembedding = np.asarray(official[qposition], dtype=np.float32)
        local_score = np.empty(right - left, dtype=np.float32)
        for offset, molecule in enumerate(range(left, right)):
            ref_left, ref_right = map(int, body["molecule_ptr"][molecule:molecule + 2])
            reference_rows = body["pair_candidate_row"][ref_left:ref_right]
            reference_position = [row_position[int(row)] for row in reference_rows]
            local_score[offset] = float(np.max(np.asarray(official[reference_position]) @ qembedding))
        local_identity = body["molecule_ik14"][left:right].astype(str)
        local_structure = np.asarray([structure_position[value] for value in local_identity], dtype=np.int64)
        score[out] = local_score
        structure[out] = local_structure
        labels[out] = local_labels
        baseline_rank[out] = strict_rank(local_score, local_labels)
        baseline_candidate[out] = int(np.argmax(local_score))
        if (out + 1) % 256 == 0:
            print(f"built official candidate geometry {out + 1}/{len(queries)}", flush=True)
    return {
        "score": score,
        "structure": structure,
        "labels": labels,
        "baseline_rank": baseline_rank,
        "baseline_candidate": baseline_candidate,
    }


def support_for_geometry(
    observation: np.ndarray,
    geometry: dict[str, np.ndarray],
    predicate: np.ndarray,
    weight: np.ndarray,
    scale: np.ndarray,
    support_only: bool = True,
) -> np.ndarray:
    output = np.empty(len(observation), dtype=object)
    for index in range(len(observation)):
        output[index] = candidate_compatibility(
            observation[index], predicate[np.asarray(geometry["structure"][index])],
            weight, scale, support_only,
        )
    return output


def fused_proposal(
    geometry: dict[str, np.ndarray], support: np.ndarray, gamma: float,
) -> dict[str, np.ndarray]:
    rank = np.empty(len(support), dtype=np.int16)
    candidate = np.empty(len(support), dtype=np.int16)
    confidence = np.full(len(support), -np.inf, dtype=np.float32)
    for index in range(len(support)):
        official = np.asarray(geometry["score"][index], dtype=np.float64)
        chemical = np.asarray(support[index], dtype=np.float64)
        fused = official + float(gamma) * chemical
        candidate[index] = int(np.argmax(fused))
        rank[index] = strict_rank(fused, np.asarray(geometry["labels"][index], dtype=bool))
        baseline = int(geometry["baseline_candidate"][index])
        if candidate[index] != baseline:
            confidence[index] = float(chemical[candidate[index]] - chemical[baseline])
    return {"rank": rank, "candidate": candidate, "confidence": confidence}


def select_architecture(
    family: dict[tuple[float, int], np.ndarray],
    scale: np.ndarray,
    observation: np.ndarray,
    geometry: dict[str, np.ndarray],
    predicate: np.ndarray,
    gamma_values: tuple[float, ...],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for (ridge, rank_value), weight in sorted(family.items()):
        support = support_for_geometry(observation, geometry, predicate, weight, scale)
        for gamma in gamma_values:
            proposal = fused_proposal(geometry, support, gamma)
            metric = retrieval(geometry["baseline_rank"], proposal["rank"])
            rows.append({
                "ridge": float(ridge), "rank": int(rank_value), "gamma": float(gamma),
                **metric,
            })
    selected = max(
        rows,
        key=lambda row: (
            int(row["risk_utility_at_1"]), float(row["delta_recall1"]),
            float(row["delta_mrr"]), -float(row["ridge"]), -int(row["rank"]),
        ),
    )
    return selected, rows


def fit_selected_map(
    observation: np.ndarray,
    predicate_unit: np.ndarray,
    unit_formula: np.ndarray,
    unit_domain: np.ndarray,
    unit_fold: np.ndarray,
    folds: tuple[int, ...],
    ridge: float,
    rank_value: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    positions = np.flatnonzero(np.isin(unit_fold, folds))
    x, y, design_report = formula_domain_centered_design(
        observation, predicate_unit, unit_formula, unit_domain, positions,
    )
    family, scale, ridge_report = ridge_support_family(
        x, y, [ridge], [rank_value], support_only=True,
    )
    return family[(float(ridge), int(rank_value))], scale, {
        "design": design_report, "ridge": ridge_report,
    }


def evaluate_arm(
    selected: dict[str, object],
    weight: np.ndarray,
    scale: np.ndarray,
    predicate: np.ndarray,
    calibration_observation: np.ndarray,
    calibration_geometry: dict[str, np.ndarray],
    calibration_formula: np.ndarray,
    inner_observation: np.ndarray,
    inner_geometry: dict[str, np.ndarray],
    inner_formula: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    calibration_support = support_for_geometry(
        calibration_observation, calibration_geometry, predicate, weight, scale,
    )
    calibration_proposal = fused_proposal(
        calibration_geometry, calibration_support, float(selected["gamma"]),
    )
    threshold = choose_abstention_threshold(
        calibration_proposal["confidence"], calibration_geometry["baseline_rank"],
        calibration_proposal["rank"], calibration_formula,
        args.minimum_calibration_formulas,
    )
    inner_support = support_for_geometry(
        inner_observation, inner_geometry, predicate, weight, scale,
    )
    inner_proposal = fused_proposal(
        inner_geometry, inner_support, float(selected["gamma"]),
    )
    selected_query = np.isfinite(inner_proposal["confidence"]) & (
        inner_proposal["confidence"] >= float(threshold["threshold"])
    )
    inner_rank = np.where(
        selected_query, inner_proposal["rank"], inner_geometry["baseline_rank"],
    ).astype(np.int16)
    metric = retrieval(inner_geometry["baseline_rank"], inner_rank)
    ci = bootstrap(
        inner_formula, inner_geometry["baseline_rank"], inner_rank,
        args.bootstrap_draws, args.seed + 810,
    )
    return {
        "architecture": selected,
        "calibration_threshold": threshold,
        "held_inner": metric,
        "held_inner_formula_bootstrap_delta_recall1_ci95": ci,
        "held_inner_selected_queries": int(np.sum(selected_query)),
        "held_inner_selected_formulas": int(len(np.unique(inner_formula[selected_query]))),
    }, {
        "rank": inner_rank,
        "proposal_rank": inner_proposal["rank"],
        "confidence": inner_proposal["confidence"],
        "selected": selected_query,
        "support": inner_support,
    }


def main() -> None:
    args = arguments()
    started = time.time()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.folds != 5 or args.bootstrap_draws < 10_000:
        raise ValueError("the five-fold/10k-bootstrap audit contract was weakened")
    if any(value <= 0 for value in args.ridges) or any(value < 0 for value in args.ranks):
        raise ValueError("invalid ridge/rank grid")
    if any(value <= 0 for value in args.gamma):
        raise ValueError("gamma must be strictly positive; no-op is handled explicitly")
    required = [
        args.manifest, args.evidence, args.predicates, args.observations,
        args.structure_dir / "identities.npy", args.structure_dir / "formulas.npy",
        args.structure_dir / "canonical_smiles.npy", args.current_policy / "inner_policy.npz",
        args.token_dir / "rows.npy", args.token_dir / "mz_f32.npy",
        args.token_dir / "intensity_f32.npy", args.token_dir / "precursor_mz_f32.npy",
        args.token_dir / "official_embeddings_f32.npy",
    ]
    if missing := [str(path) for path in required if not path.is_file()]:
        raise FileNotFoundError(missing)

    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {key: np.asarray(loaded[key]) for key in loaded.files}
    query_fold = stable_formula_folds(body["query_formula"], args.folds, args.fold_seed)
    if set(np.unique(query_fold)) != set(range(args.folds)):
        raise RuntimeError("formula folds are incomplete")
    pools = [np.flatnonzero(query_fold == value) for value in range(args.folds)]
    architecture_query = identity_balanced_queries(
        pools[1], body["query_ik14"], np.random.default_rng(args.sampling_seed + 11),
        args.architecture_identities,
    )
    calibration_query = identity_balanced_queries(
        pools[2], body["query_ik14"], np.random.default_rng(args.sampling_seed + 2),
        args.calibration_identities,
    )
    inner_query = identity_balanced_queries(
        pools[3], body["query_ik14"], np.random.default_rng(args.sampling_seed + 19),
        args.max_inner_identities,
    )
    queries = np.concatenate((architecture_query, calibration_query, inner_query))
    formulas = [body["query_formula"][value].astype(str) for value in (architecture_query, calibration_query, inner_query)]
    if any(set(formulas[i]) & set(formulas[j]) for i in range(3) for j in range(i + 1, 3)):
        raise RuntimeError("formula leakage across architecture/calibration/inner folds")

    structure_identity = np.load(args.structure_dir / "identities.npy").astype(str)
    structure_formula = np.load(args.structure_dir / "formulas.npy").astype(str)
    structure_smiles = np.load(args.structure_dir / "canonical_smiles.npy").astype(str)
    if len(np.unique(structure_identity)) != len(structure_identity):
        raise RuntimeError("candidate structure identities are not unique")
    structure_position = {value: index for index, value in enumerate(structure_identity)}
    manifest_identities = set(body["molecule_ik14"].astype(str))
    if manifest_identities != set(structure_identity):
        raise RuntimeError("candidate structure coverage is not exact")
    predicate_records = json.loads(args.predicates.read_text(encoding="utf-8"))["predicates"]
    predicate, predicate_report = build_predicate_matrix(
        structure_identity, structure_smiles, predicate_records,
    )
    permutation = stable_nontrivial_formula_permutation(
        structure_identity, structure_formula, args.seed + 1701,
    )
    permuted_predicate = predicate[permutation]

    with np.load(args.evidence, allow_pickle=False) as loaded:
        evidence = {key: np.asarray(loaded[key]) for key in loaded.files}
    evidence_position = np.asarray([structure_position[value] for value in evidence["identity"].astype(str)], dtype=np.int64)
    # The old evidence miner evaluated the first *query* SMILES for an IK14,
    # whereas the candidate asset evaluates the first *reference* SMILES.
    # IK14 collapses stereochemistry and source records can also differ in
    # tautomer/protonation representation.  Retrieval must use one consistent
    # candidate representation, so the candidate-side matrix is authoritative
    # for both fitting and scoring; the frozen mismatch is reported explicitly.
    frozen_predicate_delta = (
        predicate[evidence_position] != evidence["predicate_presence"]
    )
    unit_structure_position = evidence_position[evidence["unit_identity"]]
    unit_formula = evidence["unit_formula"].astype(str)
    unit_fold = stable_formula_folds(unit_formula, args.folds, args.fold_seed)
    valid_unit = evidence["valid_structure_unit"].astype(bool)
    if not np.all(valid_unit):
        unit_fold = np.where(valid_unit, unit_fold, -1)
    unit_observation = evidence["observation_prevalence"].astype(np.float32)
    unit_domain = evidence["unit_domain"].astype(str)

    channels = json.loads(args.observations.read_text(encoding="utf-8"))["channels"]
    if len(channels) != unit_observation.shape[1] or len(predicate_records) != predicate.shape[1]:
        raise RuntimeError("observation/predicate registries do not align with the evidence asset")
    cache_rows = np.load(args.token_dir / "rows.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(cache_rows)}
    mz_cache = np.load(args.token_dir / "mz_f32.npy", mmap_mode="r")
    intensity_cache = np.load(args.token_dir / "intensity_f32.npy", mmap_mode="r")
    precursor_cache = np.load(args.token_dir / "precursor_mz_f32.npy", mmap_mode="r")
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    observation_all = query_observations(
        queries, body, row_position, mz_cache, intensity_cache, precursor_cache,
        channels, args.intensity_threshold,
    )
    lengths = [len(architecture_query), len(calibration_query), len(inner_query)]
    split_at = np.cumsum([0, *lengths])
    observations = [observation_all[split_at[i]:split_at[i + 1]] for i in range(3)]
    geometry_all = candidate_geometry(
        queries, body, official, row_position, structure_position,
    )
    geometries = []
    for i in range(3):
        left, right = split_at[i:i + 2]
        geometries.append({key: value[left:right] for key, value in geometry_all.items()})

    arms: dict[str, dict[str, object]] = {}
    arm_arrays: dict[str, dict[str, np.ndarray]] = {}
    architecture_rows: dict[str, list[dict[str, object]]] = {}
    map_reports: dict[str, dict[str, object]] = {}
    for arm_name, candidate_predicate in (
        ("correct_structure", predicate),
        ("within_formula_predicate_permuted", permuted_predicate),
    ):
        predicate_unit = candidate_predicate[unit_structure_position]
        # Refit a family once, rather than repeatedly reconstructing the design.
        fit_positions = np.flatnonzero(unit_fold == 0)
        x, y, design_report = formula_domain_centered_design(
            unit_observation, predicate_unit, unit_formula, unit_domain, fit_positions,
        )
        family, architecture_scale, ridge_report = ridge_support_family(
            x, y, args.ridges, args.ranks, support_only=True,
        )
        selected, rows = select_architecture(
            family, architecture_scale, observations[0], geometries[0],
            candidate_predicate, tuple(map(float, args.gamma)),
        )
        architecture_rows[arm_name] = rows
        refit_weight, refit_scale, refit_report = fit_selected_map(
            unit_observation, predicate_unit, unit_formula, unit_domain, unit_fold,
            (0, 1), float(selected["ridge"]), int(selected["rank"]),
        )
        result, arrays = evaluate_arm(
            selected, refit_weight, refit_scale, candidate_predicate,
            observations[1], geometries[1], formulas[1],
            observations[2], geometries[2], formulas[2], args,
        )
        arms[arm_name] = result
        arm_arrays[arm_name] = arrays
        map_reports[arm_name] = {
            "architecture_design": design_report,
            "architecture_ridge": ridge_report,
            "refit": refit_report,
            "weight_sha256": array_sha256(refit_weight),
            "nonzero_weights": int(np.sum(np.abs(refit_weight) > 1e-12)),
            "positive_weights": int(np.sum(refit_weight > 1e-12)),
        }
        print(f"completed {arm_name}: {result['held_inner']}", flush=True)

    correct = arm_arrays["correct_structure"]["rank"]
    control = arm_arrays["within_formula_predicate_permuted"]["rank"]
    paired = {
        "correct_minus_within_formula_predicate_permuted": paired_rank_comparison(
            correct, control, formulas[2], draws=args.bootstrap_draws, seed=args.seed + 901,
        )
    }
    with np.load(args.current_policy / "inner_policy.npz", allow_pickle=False) as loaded:
        if not np.array_equal(loaded["query"], inner_query):
            raise RuntimeError("current policy comparator query set drifted")
        current_rank = np.asarray(loaded["formula_idf_rank"], dtype=np.int16)
    paired["correct_minus_current_formula_policy"] = paired_rank_comparison(
        correct, current_rank, formulas[2], draws=args.bootstrap_draws, seed=args.seed + 902,
    )
    baseline_rank = geometries[2]["baseline_rank"]
    proposal_matrix = np.stack((
        baseline_rank,
        arm_arrays["correct_structure"]["proposal_rank"],
        current_rank,
    ))
    oracle_rank = np.min(proposal_matrix, axis=0)

    group_size = np.bincount(
        np.unique(structure_formula, return_inverse=True)[1]
    )
    changed = np.any(predicate != permuted_predicate, axis=1)
    report = {
        "status": "CHEMAWARE_HIERARCHICAL_PREDICATE_SUPPORT_COMPLETE",
        "formal_training_authorized": False,
        "weights_updated": False,
        "candidate_conditioned": True,
        "shared_embedding_result": False,
        "scope": "fold 0 map fit; fold 1 architecture; folds 0-1 refit; fold 2 abstention; used fold 3 evaluation; fold 4 sealed",
        "claim_limit": "Development candidate-scoring audit. Any gain is not a shared-embedding result or external confirmation.",
        "method": {
            "equation": "official_similarity + gamma * cosine(candidate-centred predicates, ridge-decoded observed support)",
            "map": "formula/domain-centred, formula-balanced ridge map from 120 observed channels to 85 structure predicates",
            "low_rank": "truncated SVD of the positive ridge map; reconstruction clipped nonnegative",
            "support_only": True,
            "absence_or_conflict_action": False,
            "candidate_set_centring": True,
            "retrieval_labels_used_to_fit_map": False,
            "architecture_selection": "fold-1 risk utility corrected-2*introduced",
            "abstention": "one confidence threshold frozen on fold 2",
            "control": "same pipeline after deterministic nontrivial cyclic predicate permutation within exact formula",
        },
        "data": {
            "architecture_queries": int(lengths[0]),
            "calibration_queries": int(lengths[1]),
            "inner_queries": int(lengths[2]),
            "outer_queries_untouched": int(len(pools[4])),
            "formula_overlap": 0,
            "observed_channel_count": int(len(channels)),
            "candidate_predicate_count": int(len(predicate_records)),
            "candidate_structures": predicate_report,
            "query_vs_candidate_representative_predicate_drift": {
                "differing_bits": int(np.sum(frozen_predicate_delta)),
                "identities_with_any_difference": int(np.sum(np.any(frozen_predicate_delta, axis=1))),
                "predicates_with_any_difference": int(np.sum(np.any(frozen_predicate_delta, axis=0))),
                "policy": "candidate-reference representation used consistently for map fitting and retrieval scoring",
            },
            "candidate_formula_group_size_quantiles": np.quantile(group_size, [0, .25, .5, .75, 1]).tolist(),
            "identities_changed_by_control": int(np.sum(changed)),
            "identities_unchanged_by_control": int(np.sum(~changed)),
            "query_observation_density_quantiles": np.quantile(observation_all.mean(axis=1), [0, .25, .5, .75, 1]).tolist(),
        },
        "arms": arms,
        "map_audit": map_reports,
        "paired_inner": paired,
        "held_inner_joint_truth_oracle": {
            **retrieval(baseline_rank, oracle_rank),
            "choices": ["official", "unabstained predicate proposal", "current formula policy"],
            "claim_limit": "truth-selected headroom only",
        },
        "gates": {
            "absolute_formula_ci_positive": arms["correct_structure"]["held_inner_formula_bootstrap_delta_recall1_ci95"][0] > 0,
            "corrected_exceeds_twice_introduced": arms["correct_structure"]["held_inner"]["risk_utility_at_1"] > 0,
            "beats_structure_permutation_ci": paired["correct_minus_within_formula_predicate_permuted"]["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
            "beats_current_policy_ci": paired["correct_minus_current_formula_policy"]["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
            "outer_fold_untouched": True,
        },
        "replay_contract": {
            "arguments": {key: (str(value.resolve()) if isinstance(value, Path) else value) for key, value in vars(args).items()},
            "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "scikit_learn": sklearn.__version__, "rdkit": rdBase.rdkitVersion},
            "query_sha256": {name: array_sha256(value) for name, value in zip(("architecture", "calibration", "inner"), (architecture_query, calibration_query, inner_query), strict=True)},
            "predicate_sha256": array_sha256(predicate),
            "predicate_permutation_sha256": array_sha256(permutation),
            "observation_sha256": array_sha256(observation_all),
        },
        "provenance": {
            "manifest_sha256": sha256(args.manifest),
            "official_embeddings_sha256": sha256(args.token_dir / "official_embeddings_f32.npy"),
            "evidence_sha256": sha256(args.evidence),
            "predicate_registry_sha256": sha256(args.predicates),
            "observation_registry_sha256": sha256(args.observations),
            "structure_identity_sha256": sha256(args.structure_dir / "identities.npy"),
            "structure_smiles_sha256": sha256(args.structure_dir / "canonical_smiles.npy"),
            "current_policy_report_sha256": sha256(args.current_policy / "report.json"),
        },
        "runtime_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_predicate_support_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "inner_policy.npz",
            query=inner_query, formula=formulas[2], baseline_rank=baseline_rank,
            correct_rank=correct, control_rank=control, current_policy_rank=current_rank,
            correct_proposal_rank=arm_arrays["correct_structure"]["proposal_rank"],
            correct_confidence=arm_arrays["correct_structure"]["confidence"],
            correct_selected=arm_arrays["correct_structure"]["selected"],
            control_selected=arm_arrays["within_formula_predicate_permuted"]["selected"],
            joint_truth_oracle_rank=oracle_rank,
        )
        temporary.replace(args.output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": report["status"], "arms": arms, "paired_inner": paired,
        "joint_truth_oracle": report["held_inner_joint_truth_oracle"],
        "gates": report["gates"], "output": str((args.output / "report.json").resolve()),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
