#!/usr/bin/env python
"""B0: falsify reaction signal in frozen official DreaMS embeddings.

This is deliberately a *probe*, not fine-tuning.  True Rhea edges are paired
with degree-preserving target permutations.  A global assignment minimizes
imbalance in molecular mass difference, Morgan similarity, heavy-atom
difference and same-formula status while preserving the source and target node
multisets exactly.  Formula-graph communities are assigned to outer folds;
cross-fold reaction edges are omitted, so neither endpoint identity nor formula
can cross train/test.

Reaction neighbours are never retrieval positives.  The only question is
whether a low-capacity readout can distinguish a true biochemical relation
from matched non-edges better than metadata alone.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import h5py
import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from annotation.bioaware_relations import _direction_label, noncurrency_signature  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode() if isinstance(value, (bytes, bytearray)) else str(value)
        for value in values
    ], dtype=str)


class UnionFind:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


@dataclass(frozen=True)
class ReactionEdge:
    source: str
    target: str
    relation_type: str
    reaction_ids: tuple[str, ...]


def fast_reaction_edges(
    participants: pd.DataFrame, identities: set[str]
) -> list[ReactionEdge]:
    """Extract eligible typed edges without grouping all 17k Rhea records."""
    frame = participants.copy()
    frame["ik14"] = frame["compound_id"].astype(str).str[:14]
    candidate_reactions = set(
        frame.loc[frame["ik14"].isin(identities), "reaction_id"].astype(str)
    )
    frame = frame[frame["reaction_id"].astype(str).isin(candidate_reactions)]
    aggregate: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for reaction, group in frame.groupby("reaction_id", sort=False):
        if noncurrency_signature(group, "left") == noncurrency_signature(group, "right"):
            continue
        left = sorted(set(group.loc[
            group["side"].astype(str).eq("left")
            & ~group["is_currency"].astype(bool), "ik14"
        ].astype(str)) & identities)
        right = sorted(set(group.loc[
            group["side"].astype(str).eq("right")
            & ~group["is_currency"].astype(bool), "ik14"
        ].astype(str)) & identities)
        if not left or not right:
            continue
        semantics = str(group["direction_semantics"].iloc[0])
        for source in left:
            for target in right:
                if source == target:
                    continue
                relation = _direction_label(semantics, "left", "right")
                a, b = source, target
                if relation in {"reaction_bidirectional", "reaction_direction_unknown"}:
                    a, b = sorted((a, b))
                aggregate[(a, b, relation)].add(str(reaction))
    return [
        ReactionEdge(a, b, relation, tuple(sorted(reactions)))
        for (a, b, relation), reactions in sorted(aggregate.items())
    ]


def assign_formula_community_folds(
    identities: list[str],
    formula: dict[str, str],
    edges: list[ReactionEdge],
    folds: int,
    seed: int = 20260901,
    resolution: float = 2.0,
) -> tuple[dict[str, int], dict]:
    """Partition formulas while omitting rather than leaking cross-fold edges."""
    if folds < 2:
        raise ValueError("folds must be at least two")
    graph = nx.Graph()
    all_formulas = sorted({formula[value] for value in identities})
    graph.add_nodes_from(all_formulas)
    for edge in edges:
        left, right = formula[edge.source], formula[edge.target]
        if left == right:
            continue
        previous = float(graph.get_edge_data(left, right, {}).get("weight", 0.0))
        graph.add_edge(left, right, weight=previous + 1.0)
    communities = [set(values) for values in nx.community.louvain_communities(
        graph, weight="weight", resolution=resolution, seed=seed,
    )]
    membership = {
        value: index for index, values in enumerate(communities) for value in values
    }
    internal_load = [0] * len(communities)
    for edge in edges:
        left, right = formula[edge.source], formula[edge.target]
        if membership[left] == membership[right]:
            internal_load[membership[left]] += 1
    order = sorted(
        range(len(communities)),
        key=lambda index: (-internal_load[index], -len(communities[index]),
                           min(communities[index])),
    )
    fold_loads = [0] * folds
    fold_formula_counts = [0] * folds
    formula_fold: dict[str, int] = {}
    for community_index in order:
        fold = min(
            range(folds),
            key=lambda index: (fold_loads[index], fold_formula_counts[index], index),
        )
        for value in communities[community_index]:
            formula_fold[value] = fold
        fold_loads[fold] += internal_load[community_index]
        fold_formula_counts[fold] += len(communities[community_index])
    output = {identity: formula_fold[formula[identity]] for identity in identities}
    retained = sum(
        formula_fold[formula[edge.source]] == formula_fold[formula[edge.target]]
        for edge in edges
    )
    audit = {
        "resolution": float(resolution),
        "communities": int(len(communities)),
        "largest_community_formulas": int(max(map(len, communities), default=0)),
        "fold_internal_edge_loads": [int(value) for value in fold_loads],
        "fold_formula_counts": [int(value) for value in fold_formula_counts],
        "reaction_edges_total": int(len(edges)),
        "reaction_edges_retained": int(retained),
        "reaction_edges_omitted_cross_fold": int(len(edges) - retained),
        "retained_fraction": float(retained / max(len(edges), 1)),
    }
    return output, audit


def assign_component_folds(
    identities: list[str], formula: dict[str, str], edges: list[ReactionEdge], folds: int
) -> dict[str, int]:
    """Compatibility wrapper for the superseded connected-component API."""
    return assign_formula_community_folds(identities, formula, edges, folds)[0]


def molecular_table(
    hdf5_path: Path, rows: np.ndarray, embeddings: np.ndarray
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, object]]:
    with h5py.File(hdf5_path, "r") as handle:
        ik14 = np.asarray([value[:14] for value in decode(handle["INCHIKEY"][rows])])
        formula = decode(handle["FORMULA"][rows])
        smiles = decode(handle["smiles"][rows])
    row_frame = pd.DataFrame({"row": rows, "ik14": ik14, "formula": formula, "smiles": smiles})
    row_frame["embedding_index"] = np.arange(len(row_frame))
    records = []
    prototypes: dict[str, np.ndarray] = {}
    fingerprints: dict[str, object] = {}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    for identity, group in row_frame.groupby("ik14", sort=True):
        if group["formula"].nunique() != 1:
            continue
        mol = None
        selected_smiles = ""
        for value in group["smiles"].astype(str):
            candidate = Chem.MolFromSmiles(value)
            if candidate is not None:
                mol, selected_smiles = candidate, value
                break
        if mol is None:
            continue
        values = embeddings[group["embedding_index"].to_numpy(int)].astype(np.float64)
        prototype = values.mean(axis=0)
        prototype /= max(np.linalg.norm(prototype), 1e-12)
        identity = str(identity)
        prototypes[identity] = prototype.astype(np.float32)
        fingerprints[identity] = generator.GetFingerprint(mol)
        records.append({
            "ik14": identity,
            "formula": str(group["formula"].iloc[0]),
            "smiles": selected_smiles,
            "exact_mass": float(Descriptors.ExactMolWt(mol)),
            "heavy_atoms": int(mol.GetNumHeavyAtoms()),
            "spectra": int(len(group)),
        })
    return pd.DataFrame(records), prototypes, fingerprints


def tanimoto(fingerprints: dict[str, object], left: str, right: str) -> float:
    return float(DataStructs.TanimotoSimilarity(fingerprints[left], fingerprints[right]))


def _edge_covariates(
    source: str,
    target: str,
    meta: dict[str, dict],
    fingerprints: dict[str, object],
    degree: dict[str, int],
) -> tuple[float, float, float, float, bool]:
    return (
        tanimoto(fingerprints, source, target),
        abs(meta[source]["exact_mass"] - meta[target]["exact_mass"]),
        abs(meta[source]["heavy_atoms"] - meta[target]["heavy_atoms"]),
        float(np.log1p(degree[target])),
        bool(meta[source]["formula"] == meta[target]["formula"]),
    )


def build_matched_pairs(
    molecules: pd.DataFrame,
    fingerprints: dict[str, object],
    edges: list[ReactionEdge],
    fold_by_identity: dict[str, int],
    controls_per_edge: int,
    seed: int = 20260901,
) -> pd.DataFrame:
    """Build exact source/target-degree controls by global target assignment."""
    meta = molecules.set_index("ik14").to_dict("index")
    degree = defaultdict(int)
    undirected_edges: set[tuple[str, str]] = set()
    for edge in edges:
        pair = tuple(sorted((edge.source, edge.target)))
        undirected_edges.add(pair)
        degree[edge.source] += 1
        degree[edge.target] += 1
    rows = []
    rng = np.random.default_rng(seed)
    retained_edges = [
        edge for edge in edges
        if fold_by_identity[edge.source] == fold_by_identity[edge.target]
    ]
    group_index = 0
    for fold in sorted(set(fold_by_identity.values())):
        fold_edges = [edge for edge in retained_edges if fold_by_identity[edge.source] == fold]
        if len(fold_edges) < 2:
            continue
        active = list(range(len(fold_edges)))
        while True:
            active_edges = [fold_edges[index] for index in active]
            n = len(active_edges)
            if n < max(2, controls_per_edge + 1):
                active_edges = []
                break
            sources = [edge.source for edge in active_edges]
            targets = [edge.target for edge in active_edges]
            truth_cov = [
                _edge_covariates(source, target, meta, fingerprints, degree)
                for source, target in zip(sources, targets)
            ]
            numeric = np.asarray([value[:4] for value in truth_cov], dtype=float)
            scale = np.maximum(
                np.std(numeric, axis=0), np.asarray([0.05, 5.0, 1.0, 0.25])
            )
            base_cost = np.full((n, n), 1e9, dtype=float)
            candidate_cov: dict[
                tuple[int, int], tuple[float, float, float, float, bool]
            ] = {}
            for row_index, source in enumerate(sources):
                for column_index, candidate in enumerate(targets):
                    if candidate == source:
                        continue
                    if tuple(sorted((source, candidate))) in undirected_edges:
                        continue
                    covariates = _edge_covariates(
                        source, candidate, meta, fingerprints, degree,
                    )
                    candidate_cov[(row_index, column_index)] = covariates
                    delta = np.abs(np.asarray(covariates[:4]) - numeric[row_index]) / scale
                    formula_penalty = 4.0 * float(
                        covariates[4] != truth_cov[row_index][4]
                    )
                    base_cost[row_index, column_index] = float(
                        3.0 * delta[0] + delta[1] + 0.5 * delta[2]
                        + 0.5 * delta[3] + formula_penalty
                    )
            used: set[tuple[str, str]] = set()
            assignments = []
            failed_rows: list[int] = []
            for _control_index in range(controls_per_edge):
                cost = base_cost.copy()
                for row_index, source in enumerate(sources):
                    for column_index, candidate in enumerate(targets):
                        if (source, candidate) in used:
                            cost[row_index, column_index] = 1e9
                finite = cost < 1e8
                cost[finite] += rng.uniform(0.0, 1e-5, size=int(finite.sum()))
                row_ind, column_ind = linear_sum_assignment(cost)
                invalid = cost[row_ind, column_ind] >= 1e8
                if len(row_ind) != n or np.any(invalid):
                    failed_rows = [int(value) for value in row_ind[invalid]]
                    if not failed_rows:
                        failed_rows = [int(np.argmin(finite.sum(axis=1)))]
                    break
                assignments.append((row_ind, column_ind))
                used.update((sources[i], targets[j]) for i, j in zip(row_ind, column_ind))
            if not failed_rows:
                break
            # Removing the same real edge removes one source occurrence and
            # its truth-target occurrence, retaining exact degree preservation
            # on the remaining feasible subgraph.
            drop_local = min(
                failed_rows,
                key=lambda index: (int((base_cost[index] < 1e8).sum()), index),
            )
            del active[drop_local]
        if not active_edges:
            continue
        fold_edges = active_edges
        controls_by_row: dict[int, list[tuple[str, tuple, float]]] = defaultdict(list)
        for row_ind, column_ind in assignments:
            for row_index, column_index in zip(row_ind, column_ind):
                controls_by_row[int(row_index)].append((
                    targets[int(column_index)],
                    candidate_cov[(int(row_index), int(column_index))],
                    float(base_cost[int(row_index), int(column_index)]),
                ))
        for row_index, edge in enumerate(fold_edges):
            source, truth = edge.source, edge.target
            truth_tanimoto, truth_mass, truth_heavy, truth_degree, same_formula = (
                truth_cov[row_index]
            )
            common = {
                "group_id": group_index, "source_identity": source, "fold": fold,
                "relation_type": edge.relation_type,
                "reaction_ids": ";".join(edge.reaction_ids),
            }
            rows.append({
                **common, "target_identity": truth, "label": 1, "match_cost": 0.0,
                "tanimoto": truth_tanimoto, "mass_delta": truth_mass,
                "heavy_atom_delta": truth_heavy, "target_log_degree": truth_degree,
                "same_formula": same_formula,
            })
            for candidate, covariates, cost in controls_by_row[row_index]:
                sim, mass, heavy, target_degree, candidate_same_formula = covariates
                rows.append({
                    **common, "target_identity": candidate, "label": 0,
                    "match_cost": cost, "tanimoto": sim, "mass_delta": mass,
                    "heavy_atom_delta": heavy, "target_log_degree": target_degree,
                    "same_formula": candidate_same_formula,
                })
            group_index += 1
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("no matched reaction groups")
    counts = frame.groupby("group_id")["label"].agg(["sum", "count"])
    expected = controls_per_edge + 1
    if not ((counts["sum"] == 1) & (counts["count"] == expected)).all():
        raise RuntimeError("matched groups are incomplete")
    return frame


def pair_features(
    pairs: pd.DataFrame, prototypes: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    left = np.stack([prototypes[value] for value in pairs["source_identity"].astype(str)])
    right = np.stack([prototypes[value] for value in pairs["target_identity"].astype(str)])
    embedding = np.concatenate((np.abs(left - right), left * right), axis=1)
    metadata = pairs[[
        "tanimoto", "mass_delta", "heavy_atom_delta", "target_log_degree", "same_formula"
    ]].to_numpy(float)
    return embedding, metadata


def prepare_fold_designs(
    features: np.ndarray, folds: np.ndarray, seed: int,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Fit each label-independent PCA once and cache its fold design matrices."""
    output = {}
    for fold in sorted(np.unique(folds)):
        train, test = folds != fold, folds == fold
        components = max(1, min(32, features.shape[1], int(train.sum()) - 2))
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(features[train])
        test_scaled = scaler.transform(features[test])
        pca = PCA(n_components=components, whiten=True, random_state=seed + int(fold))
        output[int(fold)] = (
            train, test, pca.fit_transform(train_scaled), pca.transform(test_scaled),
        )
    return output


def prepare_combined_fold_designs(
    embedding: np.ndarray, metadata: np.ndarray, folds: np.ndarray, seed: int,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Transform spectral and chemical covariates separately before fusion."""
    output = {}
    for fold in sorted(np.unique(folds)):
        train, test = folds != fold, folds == fold
        components = max(1, min(32, embedding.shape[1], int(train.sum()) - 2))
        embedding_scaler = StandardScaler()
        embedding_train = embedding_scaler.fit_transform(embedding[train])
        embedding_test = embedding_scaler.transform(embedding[test])
        pca = PCA(n_components=components, whiten=True, random_state=seed + int(fold))
        embedding_train = pca.fit_transform(embedding_train)
        embedding_test = pca.transform(embedding_test)
        metadata_scaler = StandardScaler()
        metadata_train = metadata_scaler.fit_transform(metadata[train])
        metadata_test = metadata_scaler.transform(metadata[test])
        output[int(fold)] = (
            train,
            test,
            np.concatenate((embedding_train, metadata_train), axis=1),
            np.concatenate((embedding_test, metadata_test), axis=1),
        )
    return output


def make_classifier(seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=0.1, fit_intercept=True, solver="lbfgs", max_iter=2000,
        class_weight="balanced", random_state=seed,
    )


def make_probe(n_samples: int, n_features: int, seed: int):
    """Legacy constructor retained for notebooks; formal code caches PCA by fold."""
    from sklearn.pipeline import Pipeline
    components = max(1, min(32, n_features, n_samples - 2))
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=components, whiten=True, random_state=seed)),
        ("logistic", make_classifier(seed)),
    ])


def crossfit_scores(
    features: np.ndarray, labels: np.ndarray, folds: np.ndarray, seed: int,
    designs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray:
    output = np.full(len(labels), np.nan, dtype=float)
    if designs is None:
        designs = prepare_fold_designs(features, folds, seed)
    for fold, (train, test, train_features, test_features) in designs.items():
        if len(np.unique(labels[train])) < 2 or len(np.unique(labels[test])) < 2:
            continue
        model = make_classifier(seed + int(fold))
        model.fit(train_features, labels[train])
        output[test] = model.predict_proba(test_features)[:, 1]
    return output


def metrics(labels: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> dict:
    valid = np.isfinite(scores)
    labels, scores, groups = labels[valid], scores[valid], groups[valid]
    wins = []
    for group in np.unique(groups):
        selected = groups == group
        positive = scores[selected & (labels == 1)]
        negative = scores[selected & (labels == 0)]
        if len(positive) == 1 and len(negative):
            wins.append(bool(positive[0] > np.max(negative)))
    return {
        "pairs": int(len(labels)),
        "groups": int(len(wins)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "groupwise_top1": float(np.mean(wins)),
    }


def groupwise_wins(
    labels: np.ndarray, scores: np.ndarray, groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    group_values, wins = [], []
    for group in np.unique(groups):
        selected = (groups == group) & np.isfinite(scores)
        positive = scores[selected & (labels == 1)]
        negative = scores[selected & (labels == 0)]
        if len(positive) == 1 and len(negative):
            group_values.append(group)
            wins.append(float(positive[0] > np.max(negative)))
    return np.asarray(group_values), np.asarray(wins, dtype=float)


def paired_group_bootstrap(
    labels: np.ndarray,
    baseline_scores: np.ndarray,
    candidate_scores: np.ndarray,
    groups: np.ndarray,
    repeats: int,
    seed: int,
) -> dict:
    baseline_groups, baseline_wins = groupwise_wins(labels, baseline_scores, groups)
    candidate_groups, candidate_wins = groupwise_wins(labels, candidate_scores, groups)
    if not np.array_equal(baseline_groups, candidate_groups):
        raise RuntimeError("paired group bootstrap has mismatched score groups")
    delta = candidate_wins - baseline_wins
    rng = np.random.default_rng(seed)
    boot = np.empty(repeats, dtype=float)
    for repeat in range(repeats):
        selected = rng.integers(0, len(delta), size=len(delta))
        boot[repeat] = float(delta[selected].mean())
    return {
        "groups": int(len(delta)),
        "mean_delta": float(delta.mean()),
        "ci_low": float(np.quantile(boot, 0.025)),
        "ci_high": float(np.quantile(boot, 0.975)),
        "corrected": int(np.sum(delta > 0)),
        "introduced": int(np.sum(delta < 0)),
    }


def permutation_pvalue(
    features: np.ndarray, labels: np.ndarray, folds: np.ndarray, groups: np.ndarray,
    observed_auc: float, repeats: int, seed: int,
    designs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> tuple[float, list[float]]:
    rng = np.random.default_rng(seed)
    null = []
    unique_groups = np.unique(groups)
    if designs is None:
        designs = prepare_fold_designs(features, folds, seed + 1000)
    for repeat in range(repeats):
        permuted = np.zeros_like(labels)
        for group in unique_groups:
            indices = np.flatnonzero(groups == group)
            permuted[int(rng.choice(indices))] = 1
        scores = crossfit_scores(
            features, permuted, folds, seed + 1000 + repeat, designs=designs,
        )
        valid = np.isfinite(scores)
        null.append(float(roc_auc_score(permuted[valid], scores[valid])))
    pvalue = (1.0 + sum(value >= observed_auc for value in null)) / (1.0 + len(null))
    return float(pvalue), null


def conditional_embedding_permutation_pvalue(
    labels: np.ndarray,
    folds: np.ndarray,
    groups: np.ndarray,
    metadata_scores: np.ndarray,
    combined_designs: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    observed_delta_auc: float,
    repeats: int,
    seed: int,
) -> tuple[float, list[float]]:
    """Shuffle only reduced embedding columns within candidate groups.

    Metadata columns and labels remain fixed.  This tests incremental frozen
    embedding information rather than rediscovering obvious reaction chemistry.
    """
    rng = np.random.default_rng(seed)
    null = []
    for repeat in range(repeats):
        output = np.full(len(labels), np.nan, dtype=float)
        for fold, (train_mask, test_mask, train_x, test_x) in combined_designs.items():
            train_indices = np.flatnonzero(train_mask)
            test_indices = np.flatnonzero(test_mask)
            train_perm, test_perm = train_x.copy(), test_x.copy()
            embedding_columns = train_x.shape[1] - 5
            for indices, design in ((train_indices, train_perm), (test_indices, test_perm)):
                local_position = {int(value): pos for pos, value in enumerate(indices)}
                for group in np.unique(groups[indices]):
                    original = np.flatnonzero(groups == group)
                    local = np.asarray([local_position[int(value)] for value in original])
                    shuffled = rng.permutation(local)
                    design[local, :embedding_columns] = design[
                        shuffled, :embedding_columns
                    ]
            model = make_classifier(seed + 10000 * (repeat + 1) + int(fold))
            model.fit(train_perm, labels[train_mask])
            output[test_mask] = model.predict_proba(test_perm)[:, 1]
        valid = np.isfinite(output) & np.isfinite(metadata_scores)
        null.append(float(
            roc_auc_score(labels[valid], output[valid])
            - roc_auc_score(labels[valid], metadata_scores[valid])
        ))
    pvalue = (1.0 + sum(value >= observed_delta_auc for value in null)) / (
        1.0 + len(null)
    )
    return float(pvalue), null


def standardised_imbalance(pairs: pd.DataFrame) -> dict[str, float]:
    output = {}
    for column in ("tanimoto", "mass_delta", "heavy_atom_delta", "target_log_degree"):
        positive = pairs.loc[pairs.label == 1, column].to_numpy(float)
        negative = pairs.loc[pairs.label == 0, column].to_numpy(float)
        pooled = np.sqrt((positive.var() + negative.var()) / 2.0)
        output[column] = float(abs(positive.mean() - negative.mean()) / max(pooled, 1e-12))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5")
    parser.add_argument("--embedding-cache", type=Path, default=ROOT / "data/validation/bioaware_metdna3_dreams_official_v1/embeddings.npz")
    parser.add_argument("--participants", type=Path, default=ROOT / "data/reference/bioaware_rhea_reactome_direction_20260830/rhea_participants.csv.gz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/validation/bioaware_b0_reaction_embedding_signal_local")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--community-resolution", type=float, default=2.0)
    parser.add_argument("--controls-per-edge", type=int, default=3)
    parser.add_argument("--permutations", type=int, default=20)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument(
        "--maximum-standardised-imbalance", type=float, default=0.10,
        help=(
            "Hard balance limit for each reaction-versus-decoy covariate. "
            "Without balanced decoys, a reaction signal is not interpretable."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--formal", action="store_true")
    args = parser.parse_args()
    if not (0.0 < args.maximum_standardised_imbalance <= 0.25):
        raise ValueError("maximum-standardised-imbalance must be in (0, 0.25]")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(f"fail-closed: non-empty output: {args.output_dir}")
    for path in (args.hdf5, args.embedding_cache, args.participants):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    cache = np.load(args.embedding_cache, allow_pickle=False)
    if {"reference_rows", "reference_embedding"}.issubset(cache.files):
        rows = np.asarray(cache["reference_rows"], dtype=np.int64)
        embeddings = np.asarray(cache["reference_embedding"], dtype=np.float32)
        embedding_cache_format = "reference_rows_reference_embedding"
    elif {"rows", "embeddings"}.issubset(cache.files):
        rows = np.asarray(cache["rows"], dtype=np.int64)
        embeddings = np.asarray(cache["embeddings"], dtype=np.float32)
        embedding_cache_format = "rows_embeddings"
    else:
        raise RuntimeError(
            "embedding cache must contain reference_rows/reference_embedding "
            "or rows/embeddings"
        )
    if len(rows) != len(embeddings) or embeddings.ndim != 2:
        raise RuntimeError("embedding row/cache shape mismatch")
    molecules, prototypes, fingerprints = molecular_table(args.hdf5, rows, embeddings)
    identities = set(molecules["ik14"].astype(str))
    participants = pd.read_csv(args.participants)
    edges = fast_reaction_edges(participants, identities)
    if len(edges) < (100 if args.formal else 25):
        raise RuntimeError(f"too few eligible reaction edges: {len(edges)}")
    formula = molecules.set_index("ik14")["formula"].astype(str).to_dict()
    fold_by_identity, partition_audit = assign_formula_community_folds(
        sorted(identities), formula, edges, args.folds, args.seed,
        args.community_resolution,
    )
    pairs = build_matched_pairs(
        molecules, fingerprints, edges, fold_by_identity, args.controls_per_edge,
        args.seed + 1,
    )
    embedding_features, metadata_features = pair_features(pairs, prototypes)
    labels = pairs["label"].to_numpy(np.int64)
    folds = pairs["fold"].to_numpy(np.int64)
    groups = pairs["group_id"].to_numpy(np.int64)
    metadata_scores = crossfit_scores(metadata_features, labels, folds, args.seed)
    embedding_designs = prepare_fold_designs(
        embedding_features, folds, args.seed + 100,
    )
    embedding_scores = crossfit_scores(
        embedding_features, labels, folds, args.seed + 100,
        designs=embedding_designs,
    )
    combined_designs = prepare_combined_fold_designs(
        embedding_features, metadata_features, folds, args.seed + 200,
    )
    combined_scores = crossfit_scores(
        np.empty((len(labels), 1)), labels, folds, args.seed + 200,
        designs=combined_designs,
    )
    reports = {
        "metadata_only": metrics(labels, metadata_scores, groups),
        "embedding_only": metrics(labels, embedding_scores, groups),
        "embedding_plus_metadata": metrics(labels, combined_scores, groups),
    }
    permutation_p, null = permutation_pvalue(
        embedding_features, labels, folds, groups,
        reports["embedding_only"]["roc_auc"], args.permutations, args.seed + 500,
        designs=embedding_designs,
    )
    valid_incremental = np.isfinite(combined_scores) & np.isfinite(metadata_scores)
    observed_incremental_auc = float(
        roc_auc_score(labels[valid_incremental], combined_scores[valid_incremental])
        - roc_auc_score(labels[valid_incremental], metadata_scores[valid_incremental])
    )
    conditional_p, conditional_null = conditional_embedding_permutation_pvalue(
        labels, folds, groups, metadata_scores, combined_designs,
        observed_incremental_auc, args.permutations, args.seed + 700,
    )
    groupwise_increment = paired_group_bootstrap(
        labels, metadata_scores, combined_scores, groups,
        args.bootstrap_resamples, args.seed + 800,
    )
    # Identity invariance must remain much stronger than biochemical relatedness.
    same_identity_cosine = []
    with h5py.File(args.hdf5, "r") as handle:
        cache_ik = np.asarray([value[:14] for value in decode(handle["INCHIKEY"][rows])])
    for identity in sorted(identities):
        selected = np.flatnonzero(cache_ik == identity)
        if len(selected) >= 2:
            same_identity_cosine.append(float(embeddings[selected[0]] @ embeddings[selected[1]]))
    reaction_cosine = [
        float(prototypes[edge.source] @ prototypes[edge.target]) for edge in edges
    ]
    fold_audit = {}
    for fold in range(args.folds):
        test_ids = set(pairs.loc[pairs.fold == fold, "source_identity"].astype(str)) | set(
            pairs.loc[pairs.fold == fold, "target_identity"].astype(str)
        )
        train_ids = set(pairs.loc[pairs.fold != fold, "source_identity"].astype(str)) | set(
            pairs.loc[pairs.fold != fold, "target_identity"].astype(str)
        )
        test_formula = {formula[value] for value in test_ids}
        train_formula = {formula[value] for value in train_ids}
        fold_audit[str(fold)] = {
            "groups": int(pairs.loc[pairs.fold == fold, "group_id"].nunique()),
            "identity_overlap": int(len(test_ids & train_ids)),
            "formula_overlap": int(len(test_formula & train_formula)),
        }
    imbalance = standardised_imbalance(pairs)
    gates = {
        "reaction_groups_ge_minimum": bool(
            pairs.group_id.nunique() >= (100 if args.formal else 25)
        ),
        "every_fold_has_reaction_groups": bool(
            all(value["groups"] >= 2 for value in fold_audit.values())
        ),
        "identity_and_formula_purged": bool(all(
            value["identity_overlap"] == 0 and value["formula_overlap"] == 0
            for value in fold_audit.values()
        )),
        "matched_decoy_imbalance_within_threshold": bool(all(
            value <= args.maximum_standardised_imbalance
            for value in imbalance.values()
        )),
        "metadata_auc_reported_not_used_as_success": True,
        "incremental_auc_ge_0_02": bool(observed_incremental_auc >= 0.02),
        "conditional_embedding_permutation_p_le_0_05": bool(conditional_p <= 0.05),
        "groupwise_top1_increment_ci_positive": bool(
            groupwise_increment["ci_low"] > 0.0
        ),
        "groupwise_corrected_gt_introduced": bool(
            groupwise_increment["corrected"] > groupwise_increment["introduced"]
        ),
        "same_identity_more_similar_than_reaction": bool(
            np.mean(same_identity_cosine) > np.mean(reaction_cosine) + 0.05
        ),
    }
    report = {
        "status": "bioaware_b0_reaction_embedding_signal_complete",
        "formal": bool(args.formal),
        "embedding_cache_scope": {
            "spectra": int(len(rows)), "identities": int(len(identities)),
            "dimension": int(embeddings.shape[1]), "format": embedding_cache_format,
        },
        "reaction_edges_available": int(len(edges)),
        "matched_groups": int(pairs.group_id.nunique()),
        "matched_pair_rows": int(len(pairs)),
        "relation_type_counts": {
            str(k): int(v) for k, v in pd.Series([edge.relation_type for edge in edges]).value_counts().items()
        },
        "probe_results": reports,
        "embedding_label_permutation": {
            "repeats": int(args.permutations), "empirical_p": permutation_p,
            "null_mean_auc": float(np.mean(null)),
            "null_p95_auc": float(np.quantile(null, 0.95)),
        },
        "incremental_embedding_evidence": {
            "combined_minus_metadata_auc": observed_incremental_auc,
            "conditional_permutation_repeats": int(args.permutations),
            "conditional_permutation_p": conditional_p,
            "conditional_null_mean_delta_auc": float(np.mean(conditional_null)),
            "conditional_null_p95_delta_auc": float(np.quantile(conditional_null, 0.95)),
            "paired_groupwise_top1_bootstrap": groupwise_increment,
            "bootstrap_resamples": int(args.bootstrap_resamples),
        },
        "matching_standardised_imbalance": imbalance,
        "matching_balance_contract": {
            "maximum_standardised_imbalance": float(args.maximum_standardised_imbalance),
            "interpretation": (
                "All reaction-versus-decoy covariates must pass before an "
                "embedding increment can be interpreted as reaction-specific."
            ),
        },
        "identity_vs_reaction_geometry": {
            "same_identity_pairs": int(len(same_identity_cosine)),
            "same_identity_mean_cosine": float(np.mean(same_identity_cosine)),
            "reaction_edges": int(len(reaction_cosine)),
            "reaction_mean_cosine": float(np.mean(reaction_cosine)),
        },
        "fold_audit": fold_audit,
        "formula_community_partition": partition_audit,
        "gates": gates,
        "pass_to_b1": bool(args.formal and all(gates.values())),
        "contracts": {
            "frozen_official_embedding": True,
            "reaction_neighbour_is_retrieval_positive": False,
            "decoys_are_non_edges": True,
            "phenotype_used": False,
            "P2b": "forbidden",
            "probe_only": True,
        },
        "provenance": {
            "hdf5_sha256": sha256(args.hdf5),
            "embedding_cache_sha256": sha256(args.embedding_cache),
            "participants_sha256": sha256(args.participants),
            "script_sha256": sha256(Path(__file__)),
        },
        "claim_limit": (
            "A frozen-representation mechanism probe. Even a passing formal result does not "
            "establish shared-embedding improvement; B1/B2 retrieval evaluation is required."
        ),
    }
    args.output_dir.mkdir(parents=True)
    pairs.to_csv(args.output_dir / "matched_pairs.csv.gz", index=False, compression="gzip")
    np.savez_compressed(
        args.output_dir / "oof_scores.npz", label=labels, fold=folds, group=groups,
        metadata=metadata_scores, embedding=embedding_scores, combined=combined_scores,
    )
    report["provenance"]["matched_pairs_sha256"] = sha256(args.output_dir / "matched_pairs.csv.gz")
    report["provenance"]["oof_scores_sha256"] = sha256(args.output_dir / "oof_scores.npz")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
