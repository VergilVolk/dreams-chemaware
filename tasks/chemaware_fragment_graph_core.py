"""Candidate-specific connected fragmentation-graph evidence for ChemAware.

This module intentionally contains no retrieval truth and no training code.  It
reuses the pinned, non-neural MAGMa enumerator already present in ``ms-pred`` to
construct atom-subset fragment DAGs.  A spectrum is explained by a rooted,
connected set of fragment nodes; each observed peak may be used once, each
fragment node may be used once, and every newly introduced parent-child edge
pays a complexity cost.

The result is a truth-blind candidate score suitable only for selecting native
DreaMS triplets after a separate formula-disjoint matched-control gate.
"""
from __future__ import annotations

import hashlib
import math
import sys
import types
from dataclasses import replace
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from rdkit import Chem

from audit_chemaware_candidate_differential_rules import (
    HYDROGEN,
    load_magma_module,
)


SUPPORTED_ADDUCTS = {"[M+H]+"}
REGULAR_SHIFT_PRIOR = 1.0
ALTERNATE_REGULAR_SHIFT_PRIOR = 0.75
SEMIRESOLVED_SHIFT_PRIOR = 0.10


def _fragmentation_module(source_root: str):
    """Load the pinned enumerator without requiring unrelated cache helpers."""
    if "platformdirs" not in sys.modules:
        try:
            __import__("platformdirs")
        except ModuleNotFoundError:
            shim = types.ModuleType("platformdirs")
            shim.user_cache_dir = lambda *args, **kwargs: str(Path(".cache").resolve())
            sys.modules["platformdirs"] = shim
    return load_magma_module(source_root)


@dataclass(frozen=True)
class FragmentNode:
    key: str
    atom_mask: int
    formula: str
    base_mass: float
    depth: int
    broken_bond_budget: int
    cut_score: float
    parent_key: str | None
    boundary_elements: tuple[str, ...]
    max_remove_h: int
    max_add_h: int
    edge_cost: float
    ring_fission: bool
    charge_migration: bool


@dataclass(frozen=True)
class FragmentGraph:
    smiles: str
    nodes: tuple[FragmentNode, ...]
    root_index: int
    parent_index: np.ndarray
    variant_mz: np.ndarray
    variant_node: np.ndarray
    variant_shift: np.ndarray
    variant_prior: np.ndarray
    variant_rule: tuple[str, ...]


@dataclass(frozen=True)
class PeakMatch:
    peak_index: int
    node_index: int
    hydrogen_shift: int
    rule: str
    theoretical_mz: float
    error_da: float
    scaled_error: float
    likelihood: float


@dataclass(frozen=True)
class ConnectedExplanation:
    score: float
    objective: float
    reward: float
    complexity_cost: float
    matched_peaks: int
    matched_nodes: int
    selected_edges: int
    explained_weight_fraction: float
    maximum_depth: int
    regular_matches: int
    semiresolved_matches: int
    ring_fission_edges: int
    charge_migration_edges: int
    observed_lineage_pairs: int
    shared_branch_pairs: int
    coherence_bonus: float
    peak_matches: tuple[PeakMatch, ...]


def stable_u64(*values: object, seed: int = 0) -> int:
    body = "\x1f".join(map(str, values)).encode("utf-8")
    digest = hashlib.blake2b(
        body, digest_size=8, person=f"fg{seed}".encode("utf-8"),
    ).digest()
    return int.from_bytes(digest, "little", signed=False)


def _atom_indices(mask: int, count: int) -> tuple[int, ...]:
    return tuple(index for index in range(count) if mask & (1 << index))


def _boundary_elements(molecule: Chem.Mol, child_mask: int, parent_mask: int) -> tuple[str, ...]:
    values: set[str] = set()
    for bond in molecule.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        left_child = bool(child_mask & (1 << left))
        right_child = bool(child_mask & (1 << right))
        if left_child == right_child:
            continue
        outside = right if left_child else left
        if not (parent_mask & (1 << outside)):
            continue
        inside = left if left_child else right
        values.add(molecule.GetAtomWithIdx(inside).GetSymbol())
    return tuple(sorted(values))


def _edge_properties(
    molecule: Chem.Mol,
    child_mask: int,
    parent_mask: int,
    child_broken: int,
    parent_broken: int,
    fixed_positive_mask: int,
) -> tuple[float, bool, bool]:
    cut_orders = []
    for bond in molecule.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if not (parent_mask & (1 << left) and parent_mask & (1 << right)):
            continue
        if bool(child_mask & (1 << left)) != bool(child_mask & (1 << right)):
            cut_orders.append(float(bond.GetBondTypeAsDouble()))
    ring_fission = len(cut_orders) > 1
    delta_broken = max(1.0, float(child_broken - parent_broken))
    bond_cost = max(delta_broken, sum(cut_orders) if cut_orders else delta_broken)
    charge_migration = bool(fixed_positive_mask and not (child_mask & fixed_positive_mask))
    return bond_cost + float(ring_fission) + float(charge_migration), ring_fission, charge_migration


def _regular_hydrogen_shifts(
    boundary_elements: Iterable[str], depth: int,
) -> dict[int, tuple[float, str]]:
    elements = set(boundary_elements)
    output: dict[int, tuple[float, str]] = {}
    if depth == 1:
        if elements.intersection({"C", "P", "S"}):
            output[-1] = (REGULAR_SHIFT_PRIOR, "MSFINDER_POSITIVE_P1")
        if elements.intersection({"N", "O"}):
            output[1] = (REGULAR_SHIFT_PRIOR, "MSFINDER_POSITIVE_P2")
        elif elements.intersection({"P", "S"}):
            output[1] = (
                ALTERNATE_REGULAR_SHIFT_PRIOR,
                "MSFINDER_POSITIVE_P2_ALTERNATE",
            )
    else:
        # P3 (+H relative to the neutralized child) and P4 (-H for an
        # initially protonated precursor) are both regular for later cleavage;
        # the boundary element only changes their empirical preference.
        plus_prior = (
            REGULAR_SHIFT_PRIOR
            if elements.intersection({"N", "O", "S"})
            else ALTERNATE_REGULAR_SHIFT_PRIOR
        )
        minus_prior = (
            REGULAR_SHIFT_PRIOR
            if elements.intersection({"C", "P"})
            else ALTERNATE_REGULAR_SHIFT_PRIOR
        )
        output[1] = (plus_prior, "MSFINDER_POSITIVE_P3")
        output[-1] = (minus_prior, "MSFINDER_POSITIVE_P4")
    if not output:
        output[-1] = (ALTERNATE_REGULAR_SHIFT_PRIOR, "EVEN_ELECTRON_UNTYPED_MINUS_H")
        output[1] = (ALTERNATE_REGULAR_SHIFT_PRIOR, "EVEN_ELECTRON_UNTYPED_PLUS_H")
    return output


def hydrogen_shift_hypotheses(node: FragmentNode) -> tuple[tuple[int, float, str], ...]:
    """Return regular HR variants plus lower-prior +/-2 H exceptions.

    The neutralized fragment formula supplied by MAGMa is the zero point.  The
    regular positive-mode MS-FINDER hypotheses are +/-1 H; all other reachable
    variants within +/-2 H are explicitly labelled semiresolved and downweighted.
    """
    regular = _regular_hydrogen_shifts(node.boundary_elements, node.depth)
    values: list[tuple[int, float, str]] = []
    low = max(-2, -node.max_remove_h)
    high = min(2, node.max_add_h)
    for shift in range(low, high + 1):
        if shift in regular:
            prior, rule = regular[shift]
            values.append((shift, prior, rule))
        else:
            values.append((shift, SEMIRESOLVED_SHIFT_PRIOR, "MSFINDER_SEMIRESOLVED_PM2H"))
    return tuple(values)


@lru_cache(maxsize=8192)
def build_fragment_graph(
    smiles: str,
    source_root: str,
    max_tree_depth: int = 3,
    max_broken_bonds: int = 6,
) -> FragmentGraph:
    if max_tree_depth < 1 or max_tree_depth > 4:
        raise ValueError("fragment tree depth must be in [1, 4]")
    if max_broken_bonds < 1 or max_broken_bonds > 8:
        raise ValueError("broken-bond budget must be in [1, 8]")
    fragmentation = _fragmentation_module(source_root)
    engine = fragmentation.FragmentEngine(
        smiles,
        max_tree_depth=max_tree_depth,
        max_broken_bonds=max_broken_bonds,
    )
    engine.generate_fragments()
    molecule = engine.mol
    if molecule is None:
        raise ValueError(f"invalid candidate SMILES: {smiles}")
    entries = engine.frag_to_entry
    roots = [key for key, body in entries.items() if int(body["tree_depth"]) == 0]
    if len(roots) != 1:
        raise RuntimeError("MAGMa graph does not contain exactly one root")
    root_key = roots[0]
    fixed_positive_mask = 0
    for atom in molecule.GetAtoms():
        if atom.GetFormalCharge() > 0:
            fixed_positive_mask |= 1 << atom.GetIdx()

    ordered_keys = sorted(
        entries,
        key=lambda key: (
            int(entries[key]["tree_depth"]),
            int(entries[key]["max_broken"]),
            str(key),
        ),
    )
    cumulative_cost: dict[object, float] = {root_key: 0.0}
    canonical_parent: dict[object, object | None] = {root_key: None}
    edge_properties: dict[object, tuple[float, bool, bool]] = {
        root_key: (0.0, False, False),
    }
    for key in ordered_keys:
        if key == root_key:
            continue
        body = entries[key]
        child_mask = int(body["frag"])
        candidates = []
        for parent in body.get("parent_hashes", []):
            if parent not in entries or parent not in cumulative_cost:
                continue
            parent_body = entries[parent]
            props = _edge_properties(
                molecule,
                child_mask,
                int(parent_body["frag"]),
                int(body["max_broken"]),
                int(parent_body["max_broken"]),
                fixed_positive_mask,
            )
            candidates.append((cumulative_cost[parent] + props[0], str(parent), parent, props))
        if not candidates:
            raise RuntimeError(f"fragment node {key} has no reachable parent")
        total, _text, parent, props = min(candidates)
        canonical_parent[key] = parent
        cumulative_cost[key] = total
        edge_properties[key] = props

    nodes: list[FragmentNode] = []
    for key in ordered_keys:
        body = entries[key]
        parent = canonical_parent[key]
        parent_mask = int(entries[parent]["frag"]) if parent is not None else int(body["frag"])
        props = edge_properties[key]
        nodes.append(FragmentNode(
            key=str(key),
            atom_mask=int(body["frag"]),
            formula=str(body["form"]),
            base_mass=float(body["base_mass"]),
            depth=int(body["tree_depth"]),
            broken_bond_budget=int(body["max_broken"]),
            cut_score=float(body["score"]),
            parent_key=None if parent is None else str(parent),
            boundary_elements=_boundary_elements(
                molecule, int(body["frag"]), parent_mask,
            ),
            max_remove_h=int(body["max_remove_hs"]),
            max_add_h=int(body["max_add_hs"]),
            edge_cost=float(props[0]),
            ring_fission=bool(props[1]),
            charge_migration=bool(props[2]),
        ))
    key_to_index = {node.key: index for index, node in enumerate(nodes)}
    parent_index = np.asarray([
        -1 if node.parent_key is None else key_to_index[node.parent_key]
        for node in nodes
    ], dtype=np.int32)
    root_index = key_to_index[str(root_key)]

    variant_mz: list[float] = []
    variant_node: list[int] = []
    variant_shift: list[int] = []
    variant_prior: list[float] = []
    variant_rule: list[str] = []
    for node_index, node in enumerate(nodes):
        if node_index == root_index:
            continue
        for shift, prior, rule in hydrogen_shift_hypotheses(node):
            mass = node.base_mass + shift * HYDROGEN
            if mass <= 0:
                continue
            variant_mz.append(float(mass))
            variant_node.append(node_index)
            variant_shift.append(shift)
            variant_prior.append(prior)
            variant_rule.append(rule)
    order = np.argsort(np.asarray(variant_mz), kind="stable")
    return FragmentGraph(
        smiles=engine.smiles,
        nodes=tuple(nodes),
        root_index=root_index,
        parent_index=parent_index,
        variant_mz=np.asarray(variant_mz, dtype=np.float64)[order],
        variant_node=np.asarray(variant_node, dtype=np.int32)[order],
        variant_shift=np.asarray(variant_shift, dtype=np.int8)[order],
        variant_prior=np.asarray(variant_prior, dtype=np.float64)[order],
        variant_rule=tuple(variant_rule[int(index)] for index in order),
    )


def _peak_match_candidates(
    graph: FragmentGraph,
    mz: np.ndarray,
    ppm: float,
    floor_da: float,
    maximum_per_peak: int = 8,
) -> tuple[tuple[PeakMatch, ...], ...]:
    output: list[tuple[PeakMatch, ...]] = []
    theory = graph.variant_mz
    for peak_index, observed in enumerate(np.asarray(mz, dtype=np.float64)):
        tolerance = max(float(floor_da), abs(float(observed)) * float(ppm) * 1e-6)
        left = int(np.searchsorted(theory, observed - tolerance, side="left"))
        right = int(np.searchsorted(theory, observed + tolerance, side="right"))
        best_by_node: dict[int, tuple[tuple[object, ...], PeakMatch]] = {}
        for variant in range(left, right):
            error = float(observed - theory[variant])
            scaled = error / tolerance
            prior = float(graph.variant_prior[variant])
            likelihood = prior * math.exp(-0.5 * scaled * scaled)
            node_index = int(graph.variant_node[variant])
            node = graph.nodes[node_index]
            key = (
                -likelihood,
                node.edge_cost,
                node.depth,
                abs(int(graph.variant_shift[variant])),
                node.key,
            )
            match = PeakMatch(
                    peak_index=peak_index,
                    node_index=node_index,
                    hydrogen_shift=int(graph.variant_shift[variant]),
                    rule=graph.variant_rule[variant],
                    theoretical_mz=float(theory[variant]),
                    error_da=error,
                    scaled_error=scaled,
                    likelihood=likelihood,
                )
            prior = best_by_node.get(node_index)
            if prior is None or key < prior[0]:
                best_by_node[node_index] = (key, match)
        ranked = sorted(best_by_node.values(), key=lambda value: value[0])
        output.append(tuple(value[1] for value in ranked[:maximum_per_peak]))
    return tuple(output)


def _best_peak_matches(
    graph: FragmentGraph,
    mz: np.ndarray,
    ppm: float,
    floor_da: float,
) -> tuple[PeakMatch | None, ...]:
    return tuple(
        values[0] if values else None
        for values in _peak_match_candidates(graph, mz, ppm, floor_da, 1)
    )


def candidate_peak_support(
    graph: FragmentGraph,
    mz: np.ndarray,
    ppm: float = 20.0,
    floor_da: float = 0.01,
) -> np.ndarray:
    return np.asarray([
        value is not None for value in _best_peak_matches(graph, mz, ppm, floor_da)
    ], dtype=bool)


def _path_to_root(graph: FragmentGraph, node_index: int) -> tuple[int, ...]:
    path = []
    seen = set()
    current = int(node_index)
    while current != graph.root_index:
        if current in seen or current < 0:
            raise RuntimeError("fragment graph parent relation is cyclic or disconnected")
        seen.add(current)
        path.append(current)
        current = int(graph.parent_index[current])
    return tuple(reversed(path))


def _coherence(graph: FragmentGraph, left: int, right: int) -> tuple[float, str]:
    if left == right:
        return 0.0, "none"
    left_path = _path_to_root(graph, left)
    right_path = _path_to_root(graph, right)
    left_position = {node: index for index, node in enumerate(left_path)}
    right_position = {node: index for index, node in enumerate(right_path)}
    if left in right_position:
        distance = len(right_path) - right_position[left]
        return 1.0 / max(distance, 1), "lineage"
    if right in left_position:
        distance = len(left_path) - left_position[right]
        return 1.0 / max(distance, 1), "lineage"
    common = set(left_path).intersection(right_path)
    if not common:
        return 0.0, "none"
    shared = max(common, key=lambda node: graph.nodes[node].depth)
    depth = graph.nodes[shared].depth
    if depth <= 0:
        return 0.0, "none"
    denominator = max(graph.nodes[left].depth, graph.nodes[right].depth, 1)
    return 0.5 * depth / denominator, "shared_branch"


def explain_candidate(
    graph: FragmentGraph,
    mz: np.ndarray,
    intensity: np.ndarray,
    peak_weight: np.ndarray | None = None,
    ppm: float = 20.0,
    floor_da: float = 0.01,
    edge_penalty_scale: float = 0.025,
    coherence_scale: float = 0.5,
    minimum_matched_peaks: int = 2,
) -> ConnectedExplanation:
    mz = np.asarray(mz, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if mz.shape != intensity.shape:
        raise ValueError("m/z and intensity arrays must have identical shape")
    if peak_weight is None:
        peak_weight = np.ones(len(mz), dtype=np.float64)
    peak_weight = np.asarray(peak_weight, dtype=np.float64)
    if peak_weight.shape != mz.shape or np.any(peak_weight < 0):
        raise ValueError("peak weights must be nonnegative and aligned")
    normalized = intensity / max(float(np.max(intensity)), 1e-12)
    base_reward = np.sqrt(np.clip(normalized, 0.0, None)) * peak_weight
    matches = _peak_match_candidates(graph, mz, ppm, floor_da)
    candidates = []
    for peak_matches in matches:
        for match in peak_matches:
            reward = float(base_reward[match.peak_index] * match.likelihood)
            if reward <= 0:
                continue
            candidates.append((match, reward, _path_to_root(graph, match.node_index)))

    selected_peaks: set[int] = set()
    selected_nodes: set[int] = {graph.root_index}
    selected_edges: set[int] = set()
    selected_matches: list[PeakMatch] = []
    selected_rewards: list[float] = []
    total_reward = 0.0
    total_cost = 0.0
    total_coherence = 0.0
    # A candidate is not credited merely because independent mass hypotheses
    # can each be connected back to the precursor through unobserved nodes.  A
    # seed must contain two observed peaks with an explicit lineage or a shared
    # non-root fragmentation branch.
    seed = None
    for left_index, left in enumerate(candidates):
        left_match, left_reward, left_path = left
        for right in candidates[left_index + 1:]:
            right_match, right_reward, right_path = right
            if left_match.peak_index == right_match.peak_index:
                continue
            if left_match.node_index == right_match.node_index:
                continue
            coherence, relation = _coherence(
                graph, left_match.node_index, right_match.node_index,
            )
            if coherence <= 0:
                continue
            edges = set(left_path).union(right_path)
            cost = edge_penalty_scale * sum(graph.nodes[node].edge_cost for node in edges)
            bonus = coherence_scale * math.sqrt(left_reward * right_reward) * coherence
            objective = left_reward + right_reward + bonus - cost
            key = (
                -objective,
                -coherence,
                left_match.peak_index,
                right_match.peak_index,
                graph.nodes[left_match.node_index].key,
                graph.nodes[right_match.node_index].key,
            )
            if objective > 0 and (seed is None or key < seed[0]):
                seed = (key, left, right, edges, cost, bonus, relation)
    if seed is not None:
        _key, left, right, edges, cost, bonus, _relation = seed
        for match, reward, path in (left, right):
            selected_peaks.add(match.peak_index)
            selected_nodes.update(path)
            selected_matches.append(match)
            selected_rewards.append(reward)
            total_reward += reward
        selected_edges.update(edges)
        total_cost += cost
        total_coherence += bonus

    while seed is not None:
        best = None
        for match, reward, path in candidates:
            if match.peak_index in selected_peaks or match.node_index in selected_nodes:
                continue
            new_edges = [node for node in path if node not in selected_edges]
            cost = edge_penalty_scale * sum(graph.nodes[node].edge_cost for node in new_edges)
            coherence_terms = []
            for prior_match, prior_reward in zip(
                selected_matches, selected_rewards, strict=True,
            ):
                coherence, _relation = _coherence(
                    graph, match.node_index, prior_match.node_index,
                )
                if coherence > 0:
                    coherence_terms.append(
                        math.sqrt(reward * prior_reward) * coherence
                    )
            bonus = coherence_scale * sum(coherence_terms)
            marginal = reward + bonus - cost
            key = (
                -marginal,
                -reward,
                cost,
                match.peak_index,
                graph.nodes[match.node_index].key,
            )
            if marginal > 0 and (best is None or key < best[0]):
                best = (key, match, reward, cost, bonus, path, new_edges)
        if best is None:
            break
        _key, match, reward, cost, bonus, path, new_edges = best
        selected_peaks.add(match.peak_index)
        selected_nodes.update(path)
        selected_edges.update(new_edges)
        selected_matches.append(match)
        selected_rewards.append(reward)
        total_reward += reward
        total_cost += cost
        total_coherence += bonus

    possible = float(np.sum(base_reward))
    objective = total_reward + total_coherence - total_cost
    if len(selected_matches) < minimum_matched_peaks or possible <= 0:
        score = 0.0
    else:
        score = max(0.0, objective / possible)
    selected_nonroot = selected_nodes.difference({graph.root_index})
    lineage_pairs = 0
    branch_pairs = 0
    for left_index, left in enumerate(selected_matches):
        for right in selected_matches[left_index + 1:]:
            value, relation = _coherence(graph, left.node_index, right.node_index)
            if value <= 0:
                continue
            lineage_pairs += int(relation == "lineage")
            branch_pairs += int(relation == "shared_branch")
    return ConnectedExplanation(
        score=float(score),
        objective=float(objective),
        reward=float(total_reward),
        complexity_cost=float(total_cost),
        matched_peaks=len(selected_matches),
        matched_nodes=len(selected_nonroot),
        selected_edges=len(selected_edges),
        explained_weight_fraction=(
            float(sum(base_reward[index] for index in selected_peaks) / possible)
            if possible > 0 else 0.0
        ),
        maximum_depth=max(
            (graph.nodes[index].depth for index in selected_nonroot), default=0,
        ),
        regular_matches=sum("SEMIRESOLVED" not in match.rule for match in selected_matches),
        semiresolved_matches=sum("SEMIRESOLVED" in match.rule for match in selected_matches),
        ring_fission_edges=sum(graph.nodes[index].ring_fission for index in selected_edges),
        charge_migration_edges=sum(
            graph.nodes[index].charge_migration for index in selected_edges
        ),
        observed_lineage_pairs=lineage_pairs,
        shared_branch_pairs=branch_pairs,
        coherence_bonus=float(total_coherence),
        peak_matches=tuple(selected_matches),
    )


def explain_candidate_set(
    graphs: Sequence[FragmentGraph],
    mz: np.ndarray,
    intensity: np.ndarray,
    ppm: float = 20.0,
    floor_da: float = 0.01,
    edge_penalty_scale: float = 0.025,
    coherence_scale: float = 0.5,
    minimum_matched_peaks: int = 2,
) -> tuple[ConnectedExplanation, ...]:
    """Score candidates with within-query inverse candidate-frequency peaks."""
    if not graphs:
        return ()
    support = np.vstack([
        candidate_peak_support(graph, mz, ppm, floor_da) for graph in graphs
    ])
    counts = np.sum(support, axis=0)
    peak_weight = np.log((len(graphs) + 1.0) / (counts + 1.0))
    peak_weight[counts == 0] = 0.0
    return tuple(
        explain_candidate(
            graph, mz, intensity, peak_weight,
            ppm=ppm, floor_da=floor_da,
            edge_penalty_scale=edge_penalty_scale,
            coherence_scale=coherence_scale,
            minimum_matched_peaks=minimum_matched_peaks,
        )
        for graph in graphs
    )


def fragment_assignment_control(graph: FragmentGraph, key: object) -> FragmentGraph:
    """Permute mass-to-node assignments while preserving graph and capacity.

    The theoretical mass multiset, hydrogen-shift labels, priors, node count,
    edge count and topology are unchanged.  Only the association between a mass
    hypothesis and an atom-subset/path node is destroyed.
    """
    count = len(graph.variant_node)
    if count < 2:
        return graph
    shift = 1 + stable_u64(key, graph.smiles, "fragment_assignment", seed=3407) % (count - 1)
    permuted = np.roll(graph.variant_node, int(shift)).copy()
    if np.array_equal(permuted, graph.variant_node):
        permuted = np.roll(graph.variant_node, 1).copy()
    return replace(graph, variant_node=permuted)


def fragment_topology_control(graph: FragmentGraph, key: object) -> FragmentGraph:
    """Permute parents within depth while preserving edge/node capacity."""
    parent = graph.parent_index.copy()
    changed = False
    depths = sorted({node.depth for node in graph.nodes if node.depth >= 2})
    for depth in depths:
        indices = np.asarray([
            index for index, node in enumerate(graph.nodes) if node.depth == depth
        ], dtype=np.int64)
        if len(indices) < 2:
            continue
        existing = parent[indices].copy()
        shift = 1 + stable_u64(key, graph.smiles, depth, seed=3407) % (len(indices) - 1)
        permuted = np.roll(existing, int(shift))
        parent[indices] = permuted
        changed |= not np.array_equal(existing, permuted)
    if not changed:
        return graph
    nodes = tuple(
        replace(
            node,
            parent_key=(
                None if int(parent[index]) < 0
                else graph.nodes[int(parent[index])].key
            ),
        )
        for index, node in enumerate(graph.nodes)
    )
    return replace(graph, nodes=nodes, parent_index=parent)


def intensity_rank_permutation(intensity: np.ndarray, query: int) -> np.ndarray:
    values = np.asarray(intensity, dtype=np.float64)
    if len(values) < 2:
        return values.copy()
    shift = 1 + stable_u64(query, "intensity", seed=3407) % (len(values) - 1)
    return np.roll(values, int(shift))


def mass_shift_control(mz: np.ndarray, precursor_mz: float, query: int) -> np.ndarray:
    """Cardinality-preserving deterministic mass perturbation control."""
    values = np.asarray(mz, dtype=np.float64)
    if not len(values):
        return values.copy()
    delta = 0.271 + (stable_u64(query, "mass", seed=3407) % 100) * 0.001
    direction = -1.0 if stable_u64(query, "direction", seed=3407) % 2 else 1.0
    shifted = values + direction * delta
    upper = max(1.0, float(precursor_mz) - 0.05)
    shifted = np.where(shifted > upper, values - delta, shifted)
    shifted = np.where(shifted <= 0.5, values + delta, shifted)
    return shifted


def source_hashes(source_root: Path) -> dict[str, str]:
    paths = {
        "fragmentation.py": source_root / "src/ms_pred/magma/fragmentation.py",
        "chem_utils.py": source_root / "src/ms_pred/common/chem_utils.py",
    }
    output = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        output[name] = digest.hexdigest()
    return output
