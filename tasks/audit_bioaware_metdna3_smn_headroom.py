#!/usr/bin/env python
"""Audit structure-network headroom on the consumed MetDNA3 development set.

This is deliberately an evidence/headroom audit rather than a model-selection
script.  The SMN definition is frozen to the MetDNA3 benchmark setting:
Morgan fingerprints and Dice similarity >= 0.4.  Network construction never
uses query truth or correctness.  Truth is consulted only after every
candidate path and every fold-level decision has been frozen.

The audit asks a narrow question: can a structure-guided network contribute
error rescues that are independent of the existing known-reaction/raw-MS2
expert, and is the union large enough to make a +10 percentage-point target
mathematically possible on the 117-query development protocol?
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

try:
    from audit_bioaware_metdna3_recursive_headroom import nearest_feature
except ModuleNotFoundError:
    from tasks.audit_bioaware_metdna3_recursive_headroom import nearest_feature


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def canonical_smiles(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    molecule = Chem.MolFromSmiles(str(value))
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, isomericSmiles=False)


def load_smiles_by_ik14(
    hdf5_path: Path, wanted: set[str], seed_table: pd.DataFrame, chunk_size: int = 8192
) -> dict[str, str]:
    """Recover one outcome-blind canonical structure for each first-block key."""
    result: dict[str, str] = {}
    for row in seed_table.itertuples(index=False):
        key = str(row.ik14)
        if key not in wanted or key in result:
            continue
        smiles = canonical_smiles(row.smiles)
        if smiles is not None:
            result[key] = smiles
    remaining = wanted - set(result)
    if not remaining:
        return result
    with h5py.File(hdf5_path, "r") as handle:
        n_rows = len(handle["INCHIKEY"])
        for start in range(0, n_rows, chunk_size):
            stop = min(start + chunk_size, n_rows)
            keys = handle["INCHIKEY"][start:stop]
            structures = handle["smiles"][start:stop]
            for raw_key, raw_smiles in zip(keys, structures, strict=True):
                key = decode(raw_key)[:14]
                if key not in remaining:
                    continue
                smiles = canonical_smiles(decode(raw_smiles))
                if smiles is not None:
                    result[key] = smiles
                    remaining.remove(key)
            if not remaining:
                break
    return result


def build_fingerprints(smiles_by_id: dict[str, str]) -> dict[str, object]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints: dict[str, object] = {}
    for identity, smiles in sorted(smiles_by_id.items()):
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is not None:
            fingerprints[identity] = generator.GetFingerprint(molecule)
    return fingerprints


def structural_neighbors(
    query_ids: set[str], reference_ids: list[str], fingerprints: dict[str, object], threshold: float
) -> dict[str, list[tuple[str, float]]]:
    reference_ids = [identity for identity in reference_ids if identity in fingerprints]
    reference_fps = [fingerprints[identity] for identity in reference_ids]
    output: dict[str, list[tuple[str, float]]] = {}
    for identity in sorted(query_ids):
        if identity not in fingerprints:
            output[identity] = []
            continue
        similarities = DataStructs.BulkDiceSimilarity(fingerprints[identity], reference_fps)
        output[identity] = [
            (other, float(score))
            for other, score in zip(reference_ids, similarities, strict=True)
            if other != identity and score >= threshold
        ]
    return output


def structure_path_evidence(
    candidate: str,
    seeds: set[str],
    candidate_to_nodes: dict[str, list[tuple[str, float]]],
    node_to_seeds: dict[str, list[tuple[str, float]]],
) -> dict:
    """Return the best direct or one-observed-intermediate SMN path."""
    direct = [score for seed, score in candidate_to_nodes.get(candidate, []) if seed in seeds]
    if direct:
        best = max(direct)
        return {
            "path_available": True,
            "minimum_depth": 1,
            "best_bottleneck": best,
            "supporting_paths": int(sum(score == best for score in direct)),
        }
    best = -np.inf
    support = 0
    for intermediate, first_score in candidate_to_nodes.get(candidate, []):
        for seed, second_score in node_to_seeds.get(intermediate, []):
            if seed not in seeds or intermediate == seed:
                continue
            bottleneck = min(first_score, second_score)
            if bottleneck > best + 1e-12:
                best = bottleneck
                support = 1
            elif abs(bottleneck - best) <= 1e-12:
                support += 1
    if np.isfinite(best):
        return {
            "path_available": True,
            "minimum_depth": 2,
            "best_bottleneck": float(best),
            "supporting_paths": int(support),
        }
    return {
        "path_available": False,
        "minimum_depth": None,
        "best_bottleneck": None,
        "supporting_paths": 0,
    }


def evidence_key(row: pd.Series | dict) -> tuple:
    if not bool(row["path_available"]):
        return (0, 0, 0.0, 0)
    return (
        1,
        -int(row["minimum_depth"]),
        float(row["best_bottleneck"]),
        int(row["supporting_paths"]),
    )


def aggregate_majority(
    fold_transitions: pd.DataFrame, minimum_votes: int = 4
) -> pd.DataFrame:
    rows: list[dict] = []
    for query_id, group in fold_transitions.groupby("query_id", sort=True):
        baseline = str(group["baseline_candidate_id"].iloc[0])
        truth = str(group["truth_candidate_id"].iloc[0])
        formula = str(group["truth_formula"].iloc[0])
        alternatives = Counter(
            str(value) for value in group.loc[group["intervene"], "network_candidate_id"]
        )
        winner = baseline
        votes = 0
        if alternatives:
            ranked = alternatives.most_common()
            if ranked[0][1] >= minimum_votes and (
                len(ranked) == 1 or ranked[0][1] > ranked[1][1]
            ):
                winner, votes = ranked[0]
        baseline_correct = baseline == truth
        final_correct = winner == truth
        truth_advantage_votes = int(group["truth_strict_advantage"].sum())
        rows.append({
            "query_id": str(query_id),
            "truth_candidate_id": truth,
            "truth_formula": formula,
            "baseline_candidate_id": baseline,
            "final_candidate_id": winner,
            "baseline_correct": baseline_correct,
            "final_correct": final_correct,
            "corrected": (not baseline_correct) and final_correct,
            "introduced": baseline_correct and (not final_correct),
            "winning_vote_count": int(votes),
            "heldout_rotations": int(len(group)),
            "truth_strict_advantage_votes": truth_advantage_votes,
            "truth_headroom": (not baseline_correct) and truth_advantage_votes >= minimum_votes,
        })
    return pd.DataFrame(rows)


def bootstrap_delta(frame: pd.DataFrame, seed: int, resamples: int = 5000) -> dict:
    values = (frame["final_correct"].astype(float) - frame["baseline_correct"].astype(float)).to_numpy()
    formulas = frame["truth_formula"].astype(str).to_numpy()
    unique = np.unique(formulas)
    grouped = {formula: values[formulas == formula] for formula in unique}
    rng = np.random.default_rng(seed)
    boot = np.empty(resamples, float)
    for index in range(resamples):
        sampled = rng.choice(unique, len(unique), replace=True)
        boot[index] = np.mean(np.concatenate([grouped[item] for item in sampled]))
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "formulas": int(len(unique)),
        "resamples": int(resamples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=Path, default=Path("data/models/MassSpecGym_MurckoHist_split.hdf5"))
    parser.add_argument("--recursive-dir", type=Path, default=Path("data/validation/bioaware_metdna3_recursive_headroom_v1"))
    parser.add_argument("--development-dir", type=Path, default=Path("data/validation/bioaware_metdna3_development_v1"))
    parser.add_argument("--query-cache", type=Path, default=Path("data/validation/bioaware_metdna3_dreams_cache_v2/queries.csv.gz"))
    parser.add_argument("--candidate-scores", type=Path, default=Path("data/validation/bioaware_metdna3_dreams_official_v1/candidate_scores.csv.gz"))
    parser.add_argument("--baseline-transitions", type=Path, default=Path("data/validation/bioaware_metdna3_development_eval_v1/raw_transitions.csv.gz"))
    parser.add_argument("--mrn-decision-dir", type=Path, default=Path("data/validation/bioaware_metdna3_candidate_edge_decision_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/validation/bioaware_metdna3_smn_headroom_v1"))
    parser.add_argument("--dice-threshold", type=float, default=0.4)
    parser.add_argument("--ppm", type=float, default=15.0)
    parser.add_argument("--rt-sec", type=float, default=25.0)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--truth-name", default="development_level1.csv.gz")
    parser.add_argument("--scope", choices=("development", "internal_rplc", "external"), default="development")
    args = parser.parse_args()

    files = {
        "nodes": args.recursive_dir / "stable_ms1_feature_nodes.csv.gz",
        "assignments": args.recursive_dir / "feature_candidate_assignments.csv.gz",
        "truth": args.development_dir / args.truth_name,
        "splits": args.development_dir / "identity_splits.csv.gz",
        "queries": args.query_cache,
        "candidate_scores": args.candidate_scores,
        "baseline": args.baseline_transitions,
        "mrn_queries": args.mrn_decision_dir / "query_transitions.csv.gz",
        "mrn_report": args.mrn_decision_dir / "report.json",
        "hdf5": args.hdf5,
    }
    for path in files.values():
        if not path.exists():
            raise FileNotFoundError(path)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise RuntimeError(f"fail-closed: output is non-empty: {output}")

    truth = pd.read_csv(files["truth"])
    splits = pd.read_csv(files["splits"])
    queries = pd.read_csv(files["queries"])
    candidates = pd.read_csv(files["candidate_scores"])
    baseline_frame = pd.read_csv(files["baseline"])
    expected_queries = int(queries.query_id.nunique())
    if candidates["query_id"].nunique() != expected_queries:
        raise RuntimeError("frozen candidate query coverage changed")
    baseline_by_query = baseline_frame.groupby("query_id")["baseline_top_candidate"].first()
    if len(baseline_by_query) != expected_queries:
        raise RuntimeError("frozen baseline protocol changed")

    nodes = pd.read_csv(files["nodes"])
    assignments = pd.read_csv(files["assignments"])
    assignments = assignments[assignments["maximum_step"].eq(0)].copy()
    feature_candidates = assignments.groupby("feature_node")["candidate_ik14"].agg(set).to_dict()
    observed_ids = set(assignments["candidate_ik14"].astype(str))
    target_ids = set(candidates["candidate_id"].astype(str))
    seed_ids = set(truth["ik14"].astype(str))
    wanted = observed_ids | target_ids | seed_ids
    smiles_by_id = load_smiles_by_ik14(files["hdf5"], wanted, truth)
    fingerprints = build_fingerprints(smiles_by_id)
    structured_observed = sorted(observed_ids & set(fingerprints))
    structured_seeds = sorted(seed_ids & set(fingerprints))
    reference_ids = sorted(set(structured_observed) | set(structured_seeds))

    candidate_to_nodes = structural_neighbors(
        target_ids, reference_ids, fingerprints, args.dice_threshold
    )
    node_to_seeds = structural_neighbors(
        set(structured_observed), structured_seeds, fingerprints, args.dice_threshold
    )

    truth["feature_node"] = [
        nearest_feature(nodes, str(row.polarity), float(row.mz), float(row.rt), args.ppm, args.rt_sec)
        for row in truth.itertuples(index=False)
    ]
    recovered_seed_ids = set(truth.loc[truth["feature_node"].notna(), "ik14"].astype(str))
    queries["feature_node"] = [
        nearest_feature(nodes, str(row.polarity), float(row.feature_mz), float(row.feature_rt_sec), args.ppm, args.rt_sec)
        for row in queries.itertuples(index=False)
    ]
    query_meta = queries.set_index("query_id")

    evidence_rows: list[dict] = []
    transition_rows: list[dict] = []
    for fold in range(10):
        fold_seed_ids = set(
            splits[(splits["fold"].eq(fold)) & splits["role"].eq("seed")]["ik14"].astype(str)
        ) & recovered_seed_ids & set(fingerprints)
        heldout_ids = set(
            splits[(splits["fold"].eq(fold)) & splits["role"].eq("heldout")]["ik14"].astype(str)
        )
        fold_candidates = candidates[candidates["truth_candidate_id"].isin(heldout_ids)]
        for query_id, group in fold_candidates.groupby("query_id", sort=True):
            meta = query_meta.loc[str(query_id)]
            feature_node = None if pd.isna(meta.feature_node) else int(meta.feature_node)
            mass_candidates = set() if feature_node is None else feature_candidates.get(feature_node, set())
            frozen_rows: list[dict] = []
            for row in group.itertuples(index=False):
                candidate = str(row.candidate_id)
                evidence = structure_path_evidence(
                    candidate, fold_seed_ids, candidate_to_nodes, node_to_seeds
                ) if candidate in mass_candidates else {
                    "path_available": False,
                    "minimum_depth": None,
                    "best_bottleneck": None,
                    "supporting_paths": 0,
                }
                record = {
                    "fold": int(fold),
                    "query_id": str(query_id),
                    "candidate_id": candidate,
                    "truth_candidate_id": str(row.truth_candidate_id),
                    "truth_formula": str(row.truth_formula),
                    "spectral_score": float(row.spectral_score),
                    "feature_recovered": feature_node is not None,
                    "mass_candidate": candidate in mass_candidates,
                    **evidence,
                }
                evidence_rows.append(record)
                frozen_rows.append(record)
            frozen = pd.DataFrame(frozen_rows)
            baseline_id = str(baseline_by_query.loc[str(query_id)])
            baseline_rows = frozen[frozen["candidate_id"].eq(baseline_id)]
            truth_rows = frozen[frozen["candidate_id"].eq(frozen["truth_candidate_id"].iloc[0])]
            if len(baseline_rows) != 1 or len(truth_rows) != 1:
                raise RuntimeError(f"candidate contract failed for {query_id} fold {fold}")
            baseline = baseline_rows.iloc[0]
            truth_row = truth_rows.iloc[0]
            keys = [evidence_key(row) for _, row in frozen.iterrows()]
            best_key = max(keys)
            winners = frozen[[key == best_key for key in keys]]
            baseline_key = evidence_key(baseline)
            intervene = len(winners) == 1 and best_key > baseline_key
            network_candidate = str(winners.iloc[0].candidate_id) if intervene else baseline_id
            transition_rows.append({
                "fold": int(fold),
                "query_id": str(query_id),
                "truth_candidate_id": str(truth_row.candidate_id),
                "truth_formula": str(truth_row.truth_formula),
                "baseline_candidate_id": baseline_id,
                "network_candidate_id": network_candidate,
                "intervene": bool(intervene),
                "baseline_correct": baseline_id == str(truth_row.candidate_id),
                "network_correct": network_candidate == str(truth_row.candidate_id),
                "truth_strict_advantage": evidence_key(truth_row) > baseline_key,
                "wrong_strict_advantage": any(
                    evidence_key(row) > evidence_key(truth_row)
                    for _, row in frozen[frozen["candidate_id"].ne(truth_row.candidate_id)].iterrows()
                ),
            })

    evidence_frame = pd.DataFrame(evidence_rows)
    fold_frame = pd.DataFrame(transition_rows)
    query_frame = aggregate_majority(fold_frame)
    if len(query_frame) != expected_queries:
        raise RuntimeError(f"expected {expected_queries} query decisions, got {len(query_frame)}")
    evidence_path = output / "candidate_structural_evidence.csv.gz"
    fold_path = output / "fold_transitions.csv.gz"
    query_path = output / "query_transitions.csv.gz"
    evidence_frame.to_csv(evidence_path, index=False, compression="gzip")
    fold_frame.to_csv(fold_path, index=False, compression="gzip")
    query_frame.to_csv(query_path, index=False, compression="gzip")

    mrn_queries = pd.read_csv(files["mrn_queries"])
    mrn_depth3 = mrn_queries[mrn_queries["maximum_depth"].eq(3)]
    mrn_corrected = set(mrn_depth3.loc[(~mrn_depth3["baseline_correct"]) & mrn_depth3["final_correct"], "query_id"].astype(str))
    smn_corrected = set(query_frame.loc[query_frame["corrected"], "query_id"].astype(str))
    smn_headroom = set(query_frame.loc[query_frame["truth_headroom"], "query_id"].astype(str))
    baseline_errors = int((~query_frame["baseline_correct"]).sum())
    required = int(math.ceil(0.10 * len(query_frame)))
    actual_union = mrn_corrected | smn_corrected
    optimistic_union = mrn_corrected | smn_headroom
    report = {
        "status": "bioaware_metdna3_smn_headroom_complete",
        "formal": True,
        "scope": args.scope,
        "network": {
            "definition": "Morgan radius-2 2048-bit fingerprints; Dice >= 0.4",
            "threshold_source": "MetDNA3 SMN benchmark; not tuned here",
            "maximum_depth": 2,
            "intermediate_constraint": "must be an observed step-0 mass candidate with a recoverable structure",
            "structured_observed_nodes": int(len(structured_observed)),
            "structured_seed_nodes": int(len(structured_seeds)),
            "target_candidate_nodes": int(len(target_ids)),
            "target_candidates_with_structure": int(len(target_ids & set(fingerprints))),
        },
        "development": {
            "queries": int(len(query_frame)),
            "identities": int(query_frame["truth_candidate_id"].nunique()),
            "formulas": int(query_frame["truth_formula"].nunique()),
            "baseline_recall1": float(query_frame["baseline_correct"].mean()),
            "smn_recall1": float(query_frame["final_correct"].mean()),
            "delta_recall1": float(query_frame["final_correct"].mean() - query_frame["baseline_correct"].mean()),
            "corrected": int(query_frame["corrected"].sum()),
            "introduced": int(query_frame["introduced"].sum()),
            "truth_headroom_errors": int(query_frame["truth_headroom"].sum()),
            "formula_cluster_bootstrap": bootstrap_delta(query_frame, args.seed),
        },
        "ten_point_feasibility": {
            "required_net_corrections": required,
            "baseline_errors": baseline_errors,
            "existing_mrn_raw_ms2_corrected_queries": int(len(mrn_corrected)),
            "smn_corrected_queries": int(len(smn_corrected)),
            "smn_truth_headroom_queries": int(len(smn_headroom)),
            "actual_union_corrected_queries": int(len(actual_union)),
            "optimistic_union_headroom_queries": int(len(optimistic_union)),
            "actual_union_reaches_ten_points": len(actual_union) >= required,
            "optimistic_union_reaches_ten_points": len(optimistic_union) >= required,
        },
        "contracts": {
            "truth_used_to_construct_edges": False,
            "outcomes_used_to_select_threshold": False,
            "P2b_used": False,
            "external_test_opened": False,
            "SMN_is_a_specificity_control_not_an_automatic_upgrade": True,
        },
        "next_gate": (
            "Only promote SMN as an incremental layer if it contributes independent error rescues "
            "without introduced errors. Otherwise retain it as a negative-control topology and "
            "measure predicted-reaction, RT, and ion-form/global-feature layers next."
        ),
        "provenance": {name: sha256(path) for name, path in files.items()},
        "outputs": {
            "candidate_evidence_sha256": sha256(evidence_path),
            "fold_transitions_sha256": sha256(fold_path),
            "query_transitions_sha256": sha256(query_path),
        },
        "claim_limit": (
            "Consumed-development topology audit. It is neither an external validation nor a SOTA claim. "
            "Truth-headroom counts are feasibility bounds, not deployable gains."
        ),
    }
    atomic_json(output / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
