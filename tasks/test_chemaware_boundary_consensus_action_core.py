"""CPU contracts for the ChemAware same-boundary action core."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tasks")]

from chemaware_boundary_consensus_action_core import (
    EvidenceProfile,
    PairEvidenceProfile,
    apply_action_plan,
    apply_action_to_arrays,
    build_pair_logratio_action_plan,
    build_peak_action_plan,
    capacity_matched_pair_logratio_plan,
    capacity_matched_plan,
    consensus_evidence,
    intensity_rank_permuted_pair_profile,
    intensity_rank_permuted_profile,
    invert_action_plan,
    pairwise_consensus_evidence,
    peak_bins,
    top_official_negative_positions,
)


def test_official_boundary_is_formula_isolated() -> None:
    score = np.asarray([0.70, 0.95, 0.90, 0.85, 0.80])
    formula = np.asarray(["C6H12O6", "C7H14O5", "C6H12O6", "C6H12O6", "C5H8O7"])
    selected = top_official_negative_positions(score, formula, "C6H12O6", top_k=5)
    assert selected.tolist() == [2, 3]
    assert (
        top_official_negative_positions(
            score[:3],
            formula[:3],
            "C6H12O6",
            top_k=5,
            minimum_negatives=2,
        ).size
        == 0
    )


def test_consensus_requires_boundary_and_negative_agreement() -> None:
    bins = 100
    mz = np.asarray([100.0, 200.0, 300.0])
    index = peak_bins(mz, bins)
    prediction = np.zeros((4, bins), dtype=np.float32)
    # Peak 0 is true-specific against every negative.
    prediction[0, index[0]] = 1.0
    prediction[1:, index[0]] = [0.04, 0.09, 0.16]
    # Peak 1 is conflict evidence against every negative.
    prediction[0, index[1]] = 0.04
    prediction[1:, index[1]] = [1.0, 0.81, 0.64]
    # Peak 2 disagrees: the official boundary is weaker but peers are stronger.
    prediction[0, index[2]] = 0.49
    prediction[1:, index[2]] = [0.25, 0.81, 0.81]
    profile = consensus_evidence(
        prediction,
        true_position=0,
        boundary_position=1,
        negative_positions=np.asarray([1, 2, 3]),
        observed_mz=mz,
        agreement_quantile=0.75,
        minimum_prediction=0.0,
    )
    assert profile.signed[0] > 0
    assert profile.signed[1] < 0
    assert profile.signed[2] == 0
    assert profile.agreement[0] == 1.0
    assert profile.agreement[1] == 1.0


def test_action_abstains_and_excludes_precursor_region() -> None:
    profile = EvidenceProfile(
        signed=np.asarray([0.30, -0.40, 0.01], dtype=np.float32),
        agreement=np.ones(3, dtype=np.float32),
        amplitude=np.ones(3, dtype=np.float32),
    )
    mz = np.asarray([499.5, 200.0, 300.0])
    intensity = np.asarray([0.8, 0.4, 1.0])
    plan = build_peak_action_plan(
        profile,
        mz,
        intensity,
        precursor_mz=500.0,
        mode="support_boost",
        strength=0.5,
        top_k=2,
        minimum_abs_evidence=0.05,
        precursor_exclusion_da=1.1,
    )
    assert plan.abstained  # The only support peak lies in the precursor region.
    conflict = build_peak_action_plan(
        profile,
        mz,
        intensity,
        precursor_mz=500.0,
        mode="conflict_attenuate",
        strength=0.5,
        top_k=2,
        minimum_abs_evidence=0.05,
        precursor_exclusion_da=1.1,
    )
    assert conflict.positions.tolist() == [1]


def test_bidirectional_and_controls_have_identical_capacity() -> None:
    profile = EvidenceProfile(
        signed=np.asarray([-0.5, -0.3, 0.4, 0.2, 0.0, 0.1], dtype=np.float32),
        agreement=np.ones(6, dtype=np.float32),
        amplitude=np.ones(6, dtype=np.float32),
    )
    mz = np.asarray([50, 75, 100, 125, 150, 175], dtype=float)
    intensity = np.asarray([0.7, 0.6, 0.65, 0.55, 1.0, 0.4], dtype=float)
    target = build_peak_action_plan(
        profile,
        mz,
        intensity,
        precursor_mz=250.0,
        mode="bidirectional_sharpen",
        strength=0.25,
        top_k=2,
        minimum_abs_evidence=0.15,
    )
    assert target.attenuated == 2 and target.boosted == 2
    permuted = intensity_rank_permuted_profile(profile, intensity, seed=7, groups=2)
    control = capacity_matched_plan(
        permuted,
        mz,
        intensity,
        precursor_mz=250.0,
        target=target,
    )
    assert control.attenuated == target.attenuated
    assert control.boosted == target.boosted
    assert np.allclose(np.sort(control.factors), np.sort(target.factors))


def test_pairwise_evidence_is_scale_invariant_and_antisymmetric() -> None:
    bins = 100
    mz = np.asarray([100.0, 200.0, 300.0])
    index = peak_bins(mz, bins)
    prediction = np.full((4, bins), 1e-4, dtype=np.float32)
    prediction[0, index] = [1.0, 0.2, 0.4]
    prediction[1, index] = [0.2, 1.0, 0.4]
    prediction[2, index] = [0.1, 0.8, 0.4]
    prediction[3, index] = [0.2, 0.9, 0.4]
    profile = pairwise_consensus_evidence(
        prediction,
        true_position=0,
        boundary_position=1,
        negative_positions=np.asarray([1, 2, 3]),
        observed_mz=mz,
        minimum_prediction=0.0,
    )
    rescaled = prediction * np.asarray([7.0, 0.5, 3.0, 11.0])[:, None]
    second = pairwise_consensus_evidence(
        rescaled,
        true_position=0,
        boundary_position=1,
        negative_positions=np.asarray([1, 2, 3]),
        observed_mz=mz,
        minimum_prediction=0.0,
    )
    assert profile.signed[0, 1] > 0
    assert np.allclose(profile.signed, -profile.signed.T)
    assert np.allclose(profile.signed, second.signed)


def test_pair_logratio_action_and_inverse_have_exact_direction_contract() -> None:
    signed = np.asarray(
        [
            [0.0, 1.2, 0.8, 0.0, 0.0],
            [-1.2, 0.0, 0.0, -0.7, 0.0],
            [-0.8, 0.0, 0.0, 0.6, 0.0],
            [0.0, 0.7, -0.6, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    profile = PairEvidenceProfile(
        signed=signed,
        agreement=np.where(signed != 0, 1.0, 0.0).astype(np.float32),
        amplitude=np.ones_like(signed),
    )
    mz = np.asarray([50.0, 75.0, 100.0, 125.0, 150.0])
    intensity = np.asarray([0.7, 0.65, 0.6, 0.55, 1.0])
    target = build_pair_logratio_action_plan(
        profile,
        mz,
        intensity,
        precursor_mz=500.0,
        log_ratio_dose=0.5,
        top_pairs=2,
        minimum_abs_evidence=0.5,
    )
    assert target.attenuated == target.boosted == 2
    assert len(np.unique(target.positions)) == 4
    assert np.isclose(np.sum(np.log(target.factors)), 0.0, atol=1e-6)
    inverse = invert_action_plan(target)
    assert np.array_equal(inverse.positions, target.positions)
    assert np.array_equal(inverse.roles, -target.roles)
    assert np.allclose(inverse.factors, 1.0 / target.factors)


def test_pair_control_matches_count_and_log_dose() -> None:
    signed = np.asarray(
        [
            [0.0, 1.0, 0.8, 0.6, 0.0],
            [-1.0, 0.0, 0.7, 0.5, 0.0],
            [-0.8, -0.7, 0.0, 0.4, 0.0],
            [-0.6, -0.5, -0.4, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    profile = PairEvidenceProfile(
        signed=signed,
        agreement=np.where(signed != 0, 1.0, 0.0).astype(np.float32),
        amplitude=np.ones_like(signed),
    )
    mz = np.asarray([50.0, 75.0, 100.0, 125.0, 150.0])
    intensity = np.asarray([0.7, 0.65, 0.6, 0.55, 1.0])
    target = build_pair_logratio_action_plan(
        profile, mz, intensity, 500.0, 0.25, 2, 0.1
    )
    permuted = intensity_rank_permuted_pair_profile(
        profile, intensity, seed=19, groups=2
    )
    control = capacity_matched_pair_logratio_plan(
        permuted, mz, intensity, 500.0, target
    )
    assert control.attenuated == target.attenuated
    assert control.boosted == target.boosted
    assert np.allclose(np.sort(control.factors), np.sort(target.factors))


def test_action_changes_only_selected_intensities() -> None:
    mz = np.asarray([50.0, 75.0, 100.0, 125.0, 0.0])
    intensity = np.asarray([0.4, 1.0, 0.6, 0.3, 0.0])
    profile = EvidenceProfile(
        signed=np.asarray([-0.4, 0.0, 0.5, 0.0, 0.0], dtype=np.float32),
        agreement=np.ones(5, dtype=np.float32),
        amplitude=np.ones(5, dtype=np.float32),
    )
    plan = build_peak_action_plan(
        profile,
        mz,
        intensity,
        precursor_mz=500.0,
        mode="bidirectional_sharpen",
        strength=0.5,
        top_k=1,
        minimum_abs_evidence=0.1,
    )
    changed_mz, changed_intensity = apply_action_to_arrays(mz, intensity, plan)
    assert np.array_equal(changed_mz, mz)
    assert 1 not in plan.positions
    assert int(np.argmax(changed_intensity)) == int(np.argmax(intensity))
    assert np.max(changed_intensity) == np.max(intensity) == 1.0
    assert changed_intensity[0] < intensity[0]
    assert changed_intensity[2] > intensity[2]
    # Unselected observed peaks retain their exact intensity ratio after the
    # common max normalization.
    assert np.isclose(
        changed_intensity[1] / changed_intensity[3], intensity[1] / intensity[3]
    )
    assert changed_intensity[4] == intensity[4]


def test_torch_wrapper_matches_numpy_when_torch_is_available() -> None:
    try:
        import torch
    except ModuleNotFoundError:
        return
    clean = torch.tensor(
        [
            [500.0, 1.1],
            [50.0, 0.7],
            [75.0, 1.0],
            [100.0, 0.6],
            [0.0, 0.0],
        ]
    )
    profile = EvidenceProfile(
        signed=np.asarray([-0.5, 0.0, 0.4, 0.0], dtype=np.float32),
        agreement=np.ones(4, dtype=np.float32),
        amplitude=np.ones(4, dtype=np.float32),
    )
    plan = build_peak_action_plan(
        profile,
        clean[1:, 0].numpy(),
        clean[1:, 1].numpy(),
        500.0,
        "bidirectional_sharpen",
        0.25,
        1,
        0.1,
    )
    mz, intensity = apply_action_to_arrays(
        clean[1:, 0].numpy(),
        clean[1:, 1].numpy(),
        plan,
    )
    wrapped = apply_action_plan(clean, plan)
    assert np.array_equal(wrapped[1:, 0].numpy(), mz)
    assert np.allclose(wrapped[1:, 1].numpy(), intensity)
    assert torch.equal(wrapped[0], clean[0])


def main() -> None:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print(f"PASS: {len(tests)} ChemAware boundary-consensus action contracts")


if __name__ == "__main__":
    main()
