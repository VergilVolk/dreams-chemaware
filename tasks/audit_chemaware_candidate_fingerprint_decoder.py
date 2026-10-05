"""Formula-disjoint candidate fingerprint decoder and late-fusion audit.

Unlike the failed symmetric ChemBERTa embedding append, this is an asymmetric
candidate-scoring experiment: a clean spectrum predicts formula-centred Morgan
substructure bits, and those predictions are compared with the deterministic
fingerprints of each same-formula candidate.  Fold-1 estimates per-bit transfer
reliability and chooses the ridge/rank/fusion architecture; fold-2 freezes a
single abstention threshold; fold-3 is the used inner evaluation.  Fold-4 is
never evaluated.
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
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from audit_chemaware_counterfactual_rule_kernel import bootstrap, retrieval, sha256
from audit_chemaware_counterfactual_rule_kernel_natural import paired_rank_comparison
from audit_chemaware_hierarchical_predicate_support import (
    array_sha256,
    candidate_geometry,
    fused_proposal,
    support_for_geometry,
)
from chemaware_iceberg_direct_core import stable_formula_folds
from chemaware_predicate_support_core import (
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
    parser.add_argument("--structure-dir", type=Path, default=ROOT / "data/validation/chemaware_chemberta_teacher_v1")
    parser.add_argument("--current-policy", type=Path, default=ROOT / "data/validation/chemaware_formula_gated_candidate_policy_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/validation/chemaware_candidate_fingerprint_decoder_v1")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int, default=20260905)
    parser.add_argument("--sampling-seed", type=int, default=20260905)
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument("--fingerprint-bits", type=int, default=512)
    parser.add_argument("--fingerprint-radius", type=int, default=2)
    parser.add_argument("--architecture-identities", type=int, default=2048)
    parser.add_argument("--calibration-identities", type=int, default=2048)
    parser.add_argument("--max-inner-identities", type=int, default=0)
    parser.add_argument("--ridges", type=float, nargs="+", default=(1.0, 10.0, 100.0, 1000.0))
    parser.add_argument("--ranks", type=int, nargs="+", default=(16, 32, 64, 128, 0))
    parser.add_argument("--gamma", type=float, nargs="+", default=(0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32))
    parser.add_argument("--minimum-calibration-formulas", type=int, default=50)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    return parser.parse_args()


def morgan_matrix(smiles: np.ndarray, radius: int, bits: int) -> tuple[np.ndarray, dict[str, object]]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    output = np.zeros((len(smiles), bits), dtype=np.float32)
    valid = 0
    for index, value in enumerate(smiles.astype(str)):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            continue
        valid += 1
        fingerprint = generator.GetFingerprint(molecule)
        DataStructs.ConvertToNumpyArray(fingerprint, output[index])
    prevalence = output.mean(axis=0)
    return output, {
        "identities": int(len(output)), "valid_structures": int(valid),
        "bits": int(bits), "radius": int(radius),
        "active_bits": int(np.sum(prevalence > 0)),
        "prevalence_quantiles": np.quantile(prevalence, [0, .25, .5, .75, 1]).tolist(),
    }


def identity_embedding_table(
    body: dict[str, np.ndarray],
    official: np.ndarray,
    row_position: dict[int, int],
    structure_position: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups: dict[str, list[int]] = {}
    identity_formula: dict[str, str] = {}
    for query, identity in enumerate(body["query_ik14"].astype(str)):
        groups.setdefault(identity, []).append(int(query))
        formula = str(body["query_formula"][query])
        if identity in identity_formula and identity_formula[identity] != formula:
            raise RuntimeError("one identity maps to multiple formulas")
        identity_formula[identity] = formula
    identity = np.asarray(sorted(groups))
    feature = np.empty((len(identity), official.shape[1]), dtype=np.float32)
    for out, value in enumerate(identity):
        positions = [row_position[int(body["query_row"][q])] for q in groups[value]]
        mean = np.asarray(official[positions], dtype=np.float32).mean(axis=0)
        feature[out] = mean / max(float(np.linalg.norm(mean)), 1e-12)
    structure = np.asarray([structure_position[value] for value in identity], dtype=np.int64)
    formula = np.asarray([identity_formula[value] for value in identity])
    return feature, structure, formula


def reliability(
    prediction: np.ndarray, target: np.ndarray, formulas: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Nonnegative per-bit held-formula correlation used as transfer weight."""
    x, y, _ = formula_domain_centered_design(
        prediction, target, formulas, np.repeat("all", len(formulas)), range(len(formulas)),
    )
    numerator = np.sum(x * y, axis=0)
    denominator = np.sqrt(np.sum(x * x, axis=0) * np.sum(y * y, axis=0))
    correlation = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)
    weight = np.maximum(correlation, 0.0).astype(np.float32)
    return weight, {
        "positive_reliability_bits": int(np.sum(weight > 0)),
        "reliability_quantiles": np.quantile(weight, [0, .25, .5, .75, .9, 1]).tolist(),
    }


def weighted_support(
    query_feature: np.ndarray,
    geometry: dict[str, np.ndarray],
    fingerprint: np.ndarray,
    coefficient: np.ndarray,
    scale: np.ndarray,
    bit_reliability: np.ndarray,
) -> np.ndarray:
    weighted_coefficient = coefficient * bit_reliability[None, :]
    return support_for_geometry(
        query_feature, geometry, fingerprint, weighted_coefficient, scale,
        support_only=False,
    )


def fit_family(
    identity_feature: np.ndarray,
    identity_fingerprint: np.ndarray,
    identity_formula: np.ndarray,
    identity_fold: np.ndarray,
    fit_folds: tuple[int, ...],
    ridges: tuple[float, ...],
    ranks: tuple[int, ...],
) -> tuple[dict[tuple[float, int], np.ndarray], np.ndarray, dict[str, object]]:
    positions = np.flatnonzero(np.isin(identity_fold, fit_folds))
    x, y, design = formula_domain_centered_design(
        identity_feature, identity_fingerprint, identity_formula,
        np.repeat("all", len(identity_formula)), positions,
    )
    family, scale, ridge = ridge_support_family(
        x, y, ridges, ranks, support_only=False,
    )
    return family, scale, {"design": design, "ridge": ridge}


def select_architecture(
    family: dict[tuple[float, int], np.ndarray],
    scale: np.ndarray,
    architecture_identity_feature: np.ndarray,
    architecture_identity_fingerprint: np.ndarray,
    architecture_identity_formula: np.ndarray,
    query_feature: np.ndarray,
    geometry: dict[str, np.ndarray],
    candidate_fingerprint: np.ndarray,
    gamma: tuple[float, ...],
) -> tuple[dict[str, object], list[dict[str, object]], np.ndarray]:
    rows: list[dict[str, object]] = []
    reliability_by_key: dict[tuple[float, int], np.ndarray] = {}
    for key, coefficient in sorted(family.items()):
        predicted = (architecture_identity_feature / scale) @ coefficient
        bit_weight, bit_report = reliability(
            predicted, architecture_identity_fingerprint, architecture_identity_formula,
        )
        reliability_by_key[key] = bit_weight
        support = weighted_support(
            query_feature, geometry, candidate_fingerprint, coefficient, scale, bit_weight,
        )
        for dose in gamma:
            proposal = fused_proposal(geometry, support, dose)
            rows.append({
                "ridge": float(key[0]), "rank": int(key[1]), "gamma": float(dose),
                "bit_reliability": bit_report,
                **retrieval(geometry["baseline_rank"], proposal["rank"]),
            })
    selected = max(rows, key=lambda row: (
        int(row["risk_utility_at_1"]), float(row["delta_recall1"]),
        float(row["delta_mrr"]), -float(row["ridge"]), -int(row["rank"]),
    ))
    key = (float(selected["ridge"]), int(selected["rank"]))
    return selected, rows, reliability_by_key[key]


def evaluate(
    selected: dict[str, object], coefficient: np.ndarray, scale: np.ndarray,
    bit_reliability: np.ndarray, fingerprint: np.ndarray,
    calibration_feature: np.ndarray, calibration_geometry: dict[str, np.ndarray],
    calibration_formula: np.ndarray, inner_feature: np.ndarray,
    inner_geometry: dict[str, np.ndarray], inner_formula: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    calibration_support = weighted_support(
        calibration_feature, calibration_geometry, fingerprint,
        coefficient, scale, bit_reliability,
    )
    calibration_proposal = fused_proposal(
        calibration_geometry, calibration_support, float(selected["gamma"]),
    )
    threshold = choose_abstention_threshold(
        calibration_proposal["confidence"], calibration_geometry["baseline_rank"],
        calibration_proposal["rank"], calibration_formula,
        args.minimum_calibration_formulas,
    )
    inner_support = weighted_support(
        inner_feature, inner_geometry, fingerprint,
        coefficient, scale, bit_reliability,
    )
    proposal = fused_proposal(inner_geometry, inner_support, float(selected["gamma"]))
    active = np.isfinite(proposal["confidence"]) & (
        proposal["confidence"] >= float(threshold["threshold"])
    )
    rank = np.where(active, proposal["rank"], inner_geometry["baseline_rank"]).astype(np.int16)
    return {
        "architecture": selected,
        "calibration_threshold": threshold,
        "held_inner": retrieval(inner_geometry["baseline_rank"], rank),
        "held_inner_formula_bootstrap_delta_recall1_ci95": bootstrap(
            inner_formula, inner_geometry["baseline_rank"], rank,
            args.bootstrap_draws, args.seed + 550,
        ),
        "selected_queries": int(np.sum(active)),
        "selected_formulas": int(len(np.unique(inner_formula[active]))),
    }, {"rank": rank, "proposal_rank": proposal["rank"], "confidence": proposal["confidence"], "active": active}


def main() -> None:
    args = arguments()
    started = time.time()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.folds != 5 or args.bootstrap_draws < 10_000 or args.fingerprint_bits < 64:
        raise ValueError("fingerprint audit contract was weakened")
    with np.load(args.manifest, allow_pickle=False) as loaded:
        body = {key: np.asarray(loaded[key]) for key in loaded.files}
    rows = np.load(args.token_dir / "rows.npy").astype(np.int64)
    official = np.load(args.token_dir / "official_embeddings_f32.npy", mmap_mode="r")
    row_position = {int(row): index for index, row in enumerate(rows)}
    structure_identity = np.load(args.structure_dir / "identities.npy").astype(str)
    structure_formula = np.load(args.structure_dir / "formulas.npy").astype(str)
    structure_smiles = np.load(args.structure_dir / "canonical_smiles.npy").astype(str)
    structure_position = {value: index for index, value in enumerate(structure_identity)}
    if set(body["molecule_ik14"].astype(str)) != set(structure_identity):
        raise RuntimeError("candidate structures do not cover the manifest exactly")
    fingerprint, fingerprint_report = morgan_matrix(
        structure_smiles, args.fingerprint_radius, args.fingerprint_bits,
    )
    permutation = stable_nontrivial_formula_permutation(
        structure_identity, structure_formula, args.seed + 2701,
    )
    permuted_fingerprint = fingerprint[permutation]
    identity_feature, identity_structure, identity_formula = identity_embedding_table(
        body, official, row_position, structure_position,
    )
    identity_fold = stable_formula_folds(identity_formula, args.folds, args.fold_seed)
    query_fold = stable_formula_folds(body["query_formula"], args.folds, args.fold_seed)
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
    lengths = [len(architecture_query), len(calibration_query), len(inner_query)]
    split = np.cumsum([0, *lengths])
    query_feature_all = np.asarray(official[[row_position[int(body["query_row"][q])] for q in queries]], dtype=np.float32)
    geometry_all = candidate_geometry(queries, body, official, row_position, structure_position)
    query_features = [query_feature_all[split[i]:split[i + 1]] for i in range(3)]
    geometries = [{key: value[split[i]:split[i + 1]] for key, value in geometry_all.items()} for i in range(3)]
    formulas = [body["query_formula"][value].astype(str) for value in (architecture_query, calibration_query, inner_query)]

    # Reliability is estimated on every fold-1 identity, independently of
    # which spectrum was sampled for the retrieval architecture screen.
    architecture_identity = np.flatnonzero(identity_fold == 1)
    arms: dict[str, dict[str, object]] = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    fit_reports: dict[str, object] = {}
    search_top: dict[str, object] = {}
    for name, candidate_fingerprint in (
        ("correct_fingerprint", fingerprint),
        ("within_formula_fingerprint_permuted", permuted_fingerprint),
    ):
        identity_target = candidate_fingerprint[identity_structure]
        family, scale, fit_report = fit_family(
            identity_feature, identity_target, identity_formula, identity_fold,
            (0,), tuple(map(float, args.ridges)), tuple(map(int, args.ranks)),
        )
        selected, search, selected_reliability = select_architecture(
            family, scale,
            identity_feature[architecture_identity], identity_target[architecture_identity],
            identity_formula[architecture_identity], query_features[0], geometries[0],
            candidate_fingerprint, tuple(map(float, args.gamma)),
        )
        refit_family, refit_scale, refit_report = fit_family(
            identity_feature, identity_target, identity_formula, identity_fold,
            (0, 1), (float(selected["ridge"]),), (int(selected["rank"]),),
        )
        coefficient = refit_family[(float(selected["ridge"]), int(selected["rank"]))]
        # Re-estimate bit reliability by out-of-fit fold-2 structural transfer;
        # this uses no retrieval outcome.  The same fold subsequently calibrates
        # only one action threshold, so both roles are reported explicitly.
        reliability_identity = np.flatnonzero(identity_fold == 2)
        prediction = (identity_feature[reliability_identity] / refit_scale) @ coefficient
        refit_reliability, reliability_report = reliability(
            prediction, identity_target[reliability_identity], identity_formula[reliability_identity],
        )
        result, arm_array = evaluate(
            selected, coefficient, refit_scale, refit_reliability,
            candidate_fingerprint, query_features[1], geometries[1], formulas[1],
            query_features[2], geometries[2], formulas[2], args,
        )
        arms[name] = result
        arrays[name] = arm_array
        fit_reports[name] = {
            "fold0_fit": fit_report, "fold01_refit": refit_report,
            "fold1_selected_reliability": selected["bit_reliability"],
            "fold2_refit_reliability": reliability_report,
            "coefficient_sha256": array_sha256(coefficient),
            "reliability_sha256": array_sha256(refit_reliability),
        }
        search_top[name] = sorted(
            search,
            key=lambda row: (int(row["risk_utility_at_1"]), float(row["delta_recall1"]), float(row["delta_mrr"])),
            reverse=True,
        )[:10]
        print(f"completed {name}: {result['held_inner']}", flush=True)

    baseline_rank = geometries[2]["baseline_rank"]
    correct_rank = arrays["correct_fingerprint"]["rank"]
    control_rank = arrays["within_formula_fingerprint_permuted"]["rank"]
    with np.load(args.current_policy / "inner_policy.npz", allow_pickle=False) as loaded:
        if not np.array_equal(loaded["query"], inner_query):
            raise RuntimeError("current policy comparator query set drifted")
        current_rank = np.asarray(loaded["formula_idf_rank"], dtype=np.int16)
    paired = {
        "correct_minus_fingerprint_permuted": paired_rank_comparison(
            correct_rank, control_rank, formulas[2], draws=args.bootstrap_draws, seed=args.seed + 701,
        ),
        "correct_minus_current_formula_policy": paired_rank_comparison(
            correct_rank, current_rank, formulas[2], draws=args.bootstrap_draws, seed=args.seed + 702,
        ),
    }
    oracle_rank = np.minimum.reduce((
        baseline_rank, current_rank, arrays["correct_fingerprint"]["proposal_rank"],
    ))
    report = {
        "status": "CHEMAWARE_CANDIDATE_FINGERPRINT_DECODER_COMPLETE",
        "formal_training_authorized": False,
        "weights_updated": False,
        "candidate_conditioned": True,
        "shared_embedding_result": False,
        "scope": "fold0 map; fold1 architecture and reliability; folds0-1 refit; fold2 reliability and threshold; used fold3; fold4 sealed",
        "claim_limit": "Development candidate reranking audit, not shared-embedding or external-confirmation performance.",
        "method": {
            "spectrum_input": "frozen official DreaMS clean-spectrum embedding",
            "structure_target": "formula-centred Morgan radius-2 binary substructure fingerprint",
            "candidate_score": "reliability-weighted cosine against candidate-set-centred deterministic fingerprints",
            "asymmetric_candidate_input": True,
            "candidate_or_structure_needed_for_shared_spectrum_embedding": False,
            "retrieval_labels_used_to_fit_decoder": False,
            "equal_capacity_control": "nontrivial cyclic fingerprint permutation within exact formula",
            "explicit_no_op": True,
        },
        "data": {
            "identity_spectra_for_decoder": int(len(identity_feature)),
            "architecture_queries": int(lengths[0]), "calibration_queries": int(lengths[1]),
            "inner_queries": int(lengths[2]), "outer_queries_untouched": int(len(pools[4])),
            "formula_overlap": 0, "fingerprint": fingerprint_report,
            "identities_changed_by_control": int(np.sum(np.any(fingerprint != permuted_fingerprint, axis=1))),
        },
        "arms": arms,
        "fit_audit": fit_reports,
        "architecture_top10": search_top,
        "paired_inner": paired,
        "held_inner_joint_truth_oracle": {
            **retrieval(baseline_rank, oracle_rank),
            "choices": ["official", "current formula policy", "unabstained fingerprint proposal"],
            "claim_limit": "truth-selected headroom only",
        },
        "gates": {
            "absolute_formula_ci_positive": arms["correct_fingerprint"]["held_inner_formula_bootstrap_delta_recall1_ci95"][0] > 0,
            "corrected_exceeds_twice_introduced": arms["correct_fingerprint"]["held_inner"]["risk_utility_at_1"] > 0,
            "beats_fingerprint_permutation_ci": paired["correct_minus_fingerprint_permuted"]["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
            "beats_current_policy_ci": paired["correct_minus_current_formula_policy"]["formula_cluster_bootstrap_delta_recall1_ci95"][0] > 0,
            "outer_fold_untouched": True,
        },
        "replay_contract": {
            "arguments": {key: (str(value.resolve()) if isinstance(value, Path) else value) for key, value in vars(args).items()},
            "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "scikit_learn": sklearn.__version__, "rdkit": rdBase.rdkitVersion},
            "query_sha256": {name: array_sha256(value) for name, value in zip(("architecture", "calibration", "inner"), (architecture_query, calibration_query, inner_query), strict=True)},
            "fingerprint_sha256": array_sha256(fingerprint),
            "permutation_sha256": array_sha256(permutation),
        },
        "provenance": {
            "manifest_sha256": sha256(args.manifest),
            "official_embeddings_sha256": sha256(args.token_dir / "official_embeddings_f32.npy"),
            "structure_smiles_sha256": sha256(args.structure_dir / "canonical_smiles.npy"),
            "current_policy_report_sha256": sha256(args.current_policy / "report.json"),
        },
        "runtime_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="chemaware_fingerprint_decoder_", dir=args.output.parent))
    try:
        (temporary / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        np.savez_compressed(
            temporary / "inner_policy.npz", query=inner_query, formula=formulas[2],
            baseline_rank=baseline_rank, correct_rank=correct_rank, control_rank=control_rank,
            current_policy_rank=current_rank,
            correct_proposal_rank=arrays["correct_fingerprint"]["proposal_rank"],
            correct_confidence=arrays["correct_fingerprint"]["confidence"],
            correct_active=arrays["correct_fingerprint"]["active"],
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
