"""CPU contracts for qualified dynamic-reference native triplets."""
from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import numpy as np

import build_chemaware_dynamic_reference_native_triplets as dynamic
import build_chemaware_layered_10k_native_pool as layered
from build_chemaware_max_boundary_native_triplets import PoolWriter
from evaluate_chemaware_v2_direct_triplet import validate_formula_role_policy


def main() -> None:
    events = dynamic.active_reference_events(
        positive_rows=np.asarray([10, 11, 12]),
        positive_scores=np.asarray([0.90, 0.70, 0.50]),
        negative_rows=np.asarray([20, 21, 22]),
        negative_scores=np.asarray([0.85, 0.65, 0.20]),
        margin=0.1,
        maximum_positive=1,
        maximum_negative=1,
    )
    assert [event["positive_row"] for event in events] == [10]
    assert events[0]["negative_rows"].tolist() == [20]
    for event in events:
        assert np.all(np.asarray(event["hinges"]) > 0)
    try:
        dynamic.active_reference_events(
            np.asarray([10]), np.asarray([0.5]), np.asarray([20, 21]),
            np.asarray([0.6, 0.55]), 0.1, 1, 2,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("same-candidate reference multiplication was accepted")

    writer = PoolWriter()
    writer.append(1, [3, 2], [5, 4], 0, 1, 0, 0)
    writer.append(1, [2, 3], [4, 5], 0, 1, 0, 0)
    assert len(writer.anchor) == 1
    arrays = writer.arrays()
    assert arrays["positive_idx"].tolist() == [3, 2]
    assert arrays["negative_idx"].tolist() == [5, 4]

    with tempfile.TemporaryDirectory(prefix="chem_role_contract_") as raw:
        directory = Path(raw)
        canonical = {
            "query": np.asarray([2, 7]), "formula": np.asarray(["A", "B"]),
            "identity": np.asarray(["X", "Y"]), "baseline_rank": np.asarray([1, 2]),
        }
        np.savez(directory / "selection_triplet_evidence.npz", **canonical)
        np.savez(directory / "confirmation_triplet_evidence.npz", **{
            **canonical, "query": np.asarray([3, 8]),
        })
        assert validate_formula_role_policy(canonical, 2, directory).is_file()
        try:
            validate_formula_role_policy(canonical, 3, directory)
        except RuntimeError:
            pass
        else:
            raise AssertionError("role-2 policy was accepted as role 3")

    # A source family that has already passed formula-disjoint matched-control
    # qualification may supply a positive relation even when one arbitrary
    # per-pair null delta is larger.  Preserve that distinction explicitly:
    # strict remains the highest-confidence tier, not a second sample label.
    families = {
        "fragment_graph": {
            "scope": "all_candidates", "confidence_tier": "C_calibration_required",
        },
    }
    candidates = {
        0: {"fragment_graph": {
            "score": 0.8, "formula": "C2H6O", "scope": "all_candidates",
            "controls": True, "control_names": ("control_x_score",),
            "control_scores": (0.9,),
        }},
        1: {"fragment_graph": {
            "score": 0.6, "formula": "C2H6O", "scope": "all_candidates",
            "controls": True, "control_names": ("control_x_score",),
            "control_scores": (0.1,),
        }},
    }
    strict_support, _, _, strict_flag = dynamic.relation_evidence(
        candidates, families, 0, 1, "pairwise_dominance",
    )
    family_support, family_block, _, family_strict_flag = dynamic.relation_evidence(
        candidates, families, 0, 1, "family_qualified",
    )
    assert strict_support == [] and not strict_flag
    assert family_support == [("fragment_graph", 0.20000000000000007)]
    assert family_block == [] and not family_strict_flag

    source = inspect.getsource(dynamic)
    assert "pair_evidence" in source
    assert "family_qualified_pair_evidence" in source
    assert "copy_phasea" in source
    assert "chemical_score_used_as_training_target\": False" in source
    assert "minimum_native_hinge" in source
    assert "loss_weight" not in source
    layered_source = inspect.getsource(layered)
    assert "CHEMAWARE_DYNAMIC_REFERENCE_NATIVE_TRIPLETS_COMPLETE" in layered_source
    assert "dynamic_chemical_events_added" in layered_source
    assert "dynamic_chemical_queries" in layered_source
    assert "chemical_query_count_is_diagnostic_only" not in layered_source
    print("PASS: ChemAware dynamic-reference native-triplet contracts")


if __name__ == "__main__":
    main()
