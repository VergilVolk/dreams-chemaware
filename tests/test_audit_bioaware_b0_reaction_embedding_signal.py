from pathlib import Path
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasks.audit_bioaware_b0_reaction_embedding_signal import (
    ReactionEdge,
    assign_component_folds,
    assign_formula_community_folds,
    build_matched_pairs,
    metrics,
)


def test_component_folds_purge_reaction_endpoints_and_formulas():
    identities = ["A", "B", "C", "D", "E"]
    formula = {"A": "F1", "B": "F2", "C": "F2", "D": "F3", "E": "F4"}
    edges = [ReactionEdge("A", "B", "reaction_forward", ("R1",))]
    folds = assign_component_folds(identities, formula, edges, folds=2)
    assert folds["B"] == folds["C"]


def test_formula_community_partition_reports_retained_edges():
    identities = ["A", "B", "C", "D", "E", "F"]
    formula = {value: f"F{value}" for value in identities}
    edges = [
        ReactionEdge("A", "B", "reaction_forward", ("R1",)),
        ReactionEdge("B", "C", "reaction_forward", ("R2",)),
        ReactionEdge("D", "E", "reaction_forward", ("R3",)),
    ]
    folds, audit = assign_formula_community_folds(
        identities, formula, edges, folds=2, seed=7, resolution=2.0,
    )
    assert set(folds) == set(identities)
    assert audit["reaction_edges_total"] == 3
    assert 0 < audit["reaction_edges_retained"] <= 3


def test_groupwise_top1_counts_ties_as_failure():
    labels = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.5, 0.5])
    groups = np.array([0, 0, 1, 1])
    result = metrics(labels, scores, groups)
    assert result["groupwise_top1"] == 0.5


def test_reaction_neighbour_contract_is_not_same_identity():
    edge = ReactionEdge("A", "B", "reaction_forward", ("R1",))
    assert edge.source != edge.target
    assert not edge.relation_type.startswith("same_identity")


def test_degree_preserving_control_keeps_source_and_target_multisets():
    identities = ["A", "B", "C", "D"]
    smiles = {"A": "CCO", "B": "CC=O", "C": "CCCO", "D": "CCC=O"}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=128)
    fingerprints = {
        key: generator.GetFingerprint(Chem.MolFromSmiles(value))
        for key, value in smiles.items()
    }
    molecules = pd.DataFrame([
        {
            "ik14": key, "formula": key, "smiles": value,
            "exact_mass": float(Descriptors.ExactMolWt(Chem.MolFromSmiles(value))),
            "heavy_atoms": int(Chem.MolFromSmiles(value).GetNumHeavyAtoms()), "spectra": 1,
        }
        for key, value in smiles.items()
    ])
    edges = [
        ReactionEdge("A", "B", "reaction_forward", ("R1",)),
        ReactionEdge("C", "D", "reaction_forward", ("R2",)),
    ]
    pairs = build_matched_pairs(
        molecules, fingerprints, edges, {key: 0 for key in identities},
        controls_per_edge=1, seed=7,
    )
    positive = pairs[pairs.label == 1]
    negative = pairs[pairs.label == 0]
    assert sorted(positive.source_identity) == sorted(negative.source_identity)
    assert sorted(positive.target_identity) == sorted(negative.target_identity)
