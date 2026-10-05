"""CPU contracts for candidate-specific connected fragment-graph evidence."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

import build_chemaware_fragment_graph_candidate_source_ledger as builder
import chemaware_fragment_graph_core as core


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/external/ms-pred-src"


def unique_variant_masses(
    primary: core.FragmentGraph,
    other: core.FragmentGraph,
    count: int,
) -> np.ndarray:
    values = []
    for mass in primary.variant_mz:
        if mass < 30 or mass > 500:
            continue
        if len(other.variant_mz) and float(np.min(np.abs(other.variant_mz - mass))) <= 0.02:
            continue
        if any(abs(value - mass) <= 0.02 for value in values):
            continue
        values.append(float(mass))
        if len(values) == count:
            break
    if len(values) != count:
        raise AssertionError("test molecules do not provide enough unique fragment masses")
    return np.asarray(values, dtype=np.float64)


def coherent_variant_masses(
    primary: core.FragmentGraph,
    other: core.FragmentGraph,
    count: int,
) -> np.ndarray:
    candidates = sorted(
        range(len(primary.nodes)),
        key=lambda index: primary.nodes[index].depth,
        reverse=True,
    )
    for terminal in candidates:
        path = core._path_to_root(primary, terminal)
        values = []
        for node in reversed(path):
            variants = np.flatnonzero(primary.variant_node == node)
            variants = sorted(
                variants,
                key=lambda index: (-primary.variant_prior[index], primary.variant_mz[index]),
            )
            for variant in variants:
                mass = float(primary.variant_mz[variant])
                if mass < 30 or mass > 500:
                    continue
                if len(other.variant_mz) and float(np.min(np.abs(other.variant_mz - mass))) <= 0.02:
                    continue
                if any(abs(value - mass) <= 0.02 for value in values):
                    continue
                values.append(mass)
                break
            if len(values) == count:
                return np.asarray(values, dtype=np.float64)
    raise AssertionError("test molecules do not provide a coherent unique mass path")


def main() -> None:
    hashes = core.source_hashes(SOURCE)
    assert hashes == builder.EXPECTED_SOURCE_HASHES

    first = core.build_fragment_graph(
        "CCOC(=O)NCC1=CC=CC=C1", str(SOURCE), 3, 6,
    )
    second = core.build_fragment_graph(
        "CC1=NC(=O)N(C)C(=O)N1C", str(SOURCE), 3, 6,
    )
    assert len(first.nodes) > 10 and len(first.variant_mz) > len(first.nodes)
    assert first.nodes[first.root_index].parent_key is None
    for index, node in enumerate(first.nodes):
        if index == first.root_index:
            continue
        parent = int(first.parent_index[index])
        assert parent >= 0
        assert first.nodes[parent].depth < node.depth
        assert node.edge_cost > 0

    typed_nodes = [node for node in first.nodes if node.boundary_elements]
    assert typed_nodes
    hypotheses = [core.hydrogen_shift_hypotheses(node) for node in typed_nodes]
    assert any(any(abs(shift) == 1 and "MSFINDER" in rule for shift, _, rule in body) for body in hypotheses)
    assert all(all(-2 <= shift <= 2 for shift, _, _ in body) for body in hypotheses)

    mz = coherent_variant_masses(first, second, 3)
    intensity = np.asarray([1.0, 0.8, 0.6], dtype=np.float64)
    explanations = core.explain_candidate_set(
        [first, second], mz, intensity,
        ppm=20.0, floor_da=0.01, edge_penalty_scale=0.001,
        minimum_matched_peaks=2,
    )
    assert explanations[0].matched_peaks >= 2
    assert explanations[0].selected_edges >= 1
    assert explanations[0].score > explanations[1].score
    shifted = core.mass_shift_control(mz, 600.0, query=19)
    shifted_explanation = core.explain_candidate_set(
        [first, second], shifted, intensity,
        ppm=20.0, floor_da=0.01, edge_penalty_scale=0.001,
        minimum_matched_peaks=2,
    )
    assert explanations[0].score > shifted_explanation[0].score
    permuted_graph = core.fragment_assignment_control(first, (19, 0))
    assert np.array_equal(permuted_graph.variant_mz, first.variant_mz)
    assert np.array_equal(permuted_graph.variant_shift, first.variant_shift)
    assert np.array_equal(permuted_graph.variant_prior, first.variant_prior)
    assert not np.array_equal(permuted_graph.variant_node, first.variant_node)
    assert permuted_graph.nodes == first.nodes
    topology_graph = core.fragment_topology_control(first, (19, 0))
    assert np.array_equal(topology_graph.variant_mz, first.variant_mz)
    assert len(topology_graph.nodes) == len(first.nodes)
    assert np.array_equal(
        np.sort(topology_graph.parent_index[topology_graph.parent_index >= 0]),
        np.sort(first.parent_index[first.parent_index >= 0]),
    )
    assert not np.array_equal(core.intensity_rank_permutation(intensity, 19), intensity)

    builder_source = inspect.getsource(builder)
    forbidden_reads = ("molecule_label", "old_rank", "baseline_rank", "dreams_score")
    assert not any(token in builder_source for token in forbidden_reads)
    assert "qualify_chemaware_candidate_source_ledger.py" in builder_source
    assert "fragmentation_scores_are_training_targets\": False" in builder_source
    assert "control_fragment_assignment_permuted_score" in builder_source
    assert "control_fragment_topology_permuted_score" in builder_source
    assert "control_mass_shifted_score" in builder_source
    assert "intensity_rank_permutation" not in builder_source
    assert "read_candidates" not in builder_source
    assert "read_query_registry" not in builder_source
    assert "molecule_ptr" in builder_source
    assert "pair_candidate_row" in builder_source
    print("PASS: ChemAware connected fragment-graph core and truth-blind ledger contracts")


if __name__ == "__main__":
    main()
