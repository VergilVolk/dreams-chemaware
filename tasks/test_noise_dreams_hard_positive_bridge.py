"""Focused contracts for the native Noise hard-positive bridge."""
from __future__ import annotations

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import train_noise_dreams_native as native_train

from build_noise_dreams_native_triplets import (
    NATIVE_ACTION_PREPROCESSOR,
    align_action_tensors,
    audit_same_identity_action_views,
    build_pools,
    pool_report,
    select_actions,
)
from dreams.models.heads.heads import ContrastiveHead
from test_noise_dreams_native import action_tensor, fixture
from train_noise_dreams_native import native_query_disjoint_one_pass_batches


ROOT = Path(__file__).resolve().parents[1]


def test_official_slim_is_reconstructed_before_native_head_creation() -> None:
    observed: dict[str, object] = {}

    class SourceHead:
        def state_dict(self):
            return {"weight": torch.tensor([1.0])}

    class Initialized:
        backbone = object()
        head = SourceHead()

    class DestinationHead:
        def load_state_dict(self, state, strict):
            observed["state"] = state
            observed["strict"] = strict

    class FakeContrastiveHead:
        def __init__(self, backbone, lr, weight_decay, triplet_loss_margin):
            observed["constructor"] = (
                backbone, lr, weight_decay, triplet_loss_margin,
            )
            self.head = DestinationHead()
            self.unfreeze_backbone_at_epoch = None

    def fake_load_base_model(path, architecture, device, n_highest_peaks):
        observed["base_loader"] = (
            path, architecture, device.type, n_highest_peaks,
        )
        return Initialized(), "official_embedding_slim"

    original_loader = native_train.load_base_model
    original_head = native_train.ContrastiveHead
    native_train.load_base_model = fake_load_base_model
    native_train.ContrastiveHead = FakeContrastiveHead
    try:
        model = native_train.load_trusted_native_contrastive_head(
            Path("official_embedding_slim.pt"),
            architecture_checkpoint=Path("ssl_model_server.pt"),
            map_location=torch.device("cpu"), strict=True,
            lr=5e-6, weight_decay=0.0, triplet_loss_margin=0.1,
        )
    finally:
        native_train.load_base_model = original_loader
        native_train.ContrastiveHead = original_head
    assert observed["base_loader"][3] == 100
    assert observed["constructor"][1:] == (5e-6, 0.0, 0.1)
    assert observed["strict"] is True
    assert model.unfreeze_backbone_at_epoch == 0


def test_same_query_registered_control_is_not_cross_query_tensor_shuffle() -> None:
    _, _, actions = fixture()
    targeted = np.stack([action_tensor(i) for i in range(len(actions))])
    control = targeted.copy()
    control[:, 2, 1] *= 0.5
    selected_targeted, selected_control, report = align_action_tensors(
        actions,
        {
            "action_ids": actions["action_id"].astype(str).to_numpy(),
            "action_spectra": targeted,
            "control_spectra": control,
        },
        shuffle_seed=20260925,
    )
    assert np.array_equal(selected_targeted, targeted)
    assert np.array_equal(selected_control, control)
    assert report["same_query_by_construction"] is True
    assert report["cross_query_tensor_donors"] == 0


def test_each_representable_action_has_one_hard_positive_triplet() -> None:
    graph, cache, actions = fixture()
    selected = select_actions(
        actions, margin_floor=5e-6, outer_fold=0,
        formula_fold_seed=20260825, expected_actions=7,
    )
    selected["native_action_view_representable"] = True
    train, _, _ = build_pools(
        graph, cache, selected,
        outer_fold=0, formula_fold_seed=20260825,
        validation_folds=10, validation_fold=0, validation_seed=20260920,
        hard_negative_molecules=2, max_positive_pool=32,
        max_negative_pool=32,
    )
    report = pool_report(train, epochs=1)
    assert report["action_units"] == 7
    assert report["action_measured_positive_hard_triplets"] == 7
    assert report["forbidden_clean_to_action_triplets"] == 0
    assert report["action_clean_boundary_triplets"] == 0
    for action_index in range(7):
        events = np.flatnonzero(train["event_action_index"] == action_index)
        assert len(events) == 1
        assert set(map(int, train["event_kind"][events])) == {2}
        hard_positive = int(events[0])
        hard_anchor = int(train["anchor_idx"][hard_positive])
        hard_positive_member = int(
            train["positive_idx"][train["positive_ptr"][hard_positive]]
        )
        assert int(train["registry_kind"][hard_anchor]) == 1
        assert int(train["registry_kind"][hard_positive_member]) == 0


def test_query_disjoint_schedule_exposes_every_action_once() -> None:
    graph, cache, actions = fixture()
    copies = []
    for suffix, negative_row in (("b", 3), ("c", 4)):
        row = actions.iloc[0].copy()
        row["action_id"] = f"a0{suffix}"
        row["recipe_id"] = f"recipe_0{suffix}"
        row["action_hard_negative_row"] = negative_row
        copies.append(row)
    actions = pd.concat([actions, pd.DataFrame(copies)], ignore_index=True)
    selected = select_actions(
        actions, margin_floor=5e-6, outer_fold=0,
        formula_fold_seed=20260825, expected_actions=9,
    )
    selected["native_action_view_representable"] = True
    train, _, _ = build_pools(
        graph, cache, selected,
        outer_fold=0, formula_fold_seed=20260825,
        validation_folds=10, validation_fold=0, validation_seed=20260920,
        hard_negative_molecules=2, max_positive_pool=32,
        max_negative_pool=32,
    )
    dataset_indices = np.arange(1000, 1000 + len(train["event_kind"]), dtype=np.int64)
    batches, fillers, audit = native_query_disjoint_one_pass_batches(
        train, dataset_indices, batch_size=4, seed=17,
    )
    position = {int(value): index for index, value in enumerate(dataset_indices)}
    counts = np.zeros(len(dataset_indices), dtype=np.int64)
    for batch in batches:
        positions = [position[int(value)] for value in batch]
        assert len(set(map(int, train["event_query"][positions]))) == len(batch)
        for index in positions:
            counts[index] += 1
    action_events = np.flatnonzero(train["event_kind"] > 0)
    assert np.all(counts[action_events] == 1)
    assert audit["same_query_events_never_share_an_optimizer_batch"] is True
    assert audit["hard_positive_action_events"] == 9
    assert audit["forbidden_clean_to_action_events"] == 0
    assert audit["minimum_action_events_per_semantic_unit"] == 1
    assert audit["maximum_action_events_per_semantic_unit"] == 1
    assert audit["synthetic_query_equalization_events"] == 0
    assert np.all(train["event_kind"][fillers] == 0)


def test_clean_and_hard_positive_steps_reach_all_four_roles() -> None:
    class IdentityHead:
        triplet_loss_margin = 0.1

        def __call__(self, value: torch.Tensor, charge=None) -> torch.Tensor:
            del charge
            return value[:, 0, :]

    clean = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
    action = torch.tensor([[[0.8, 0.2]]], requires_grad=True)
    positive = torch.tensor([[[0.6, 0.4]]], requires_grad=True)
    negative = torch.tensor([[[0.95, 0.05]]], requires_grad=True)
    _, clean_identity = ContrastiveHead.step(
        IdentityHead(),
        {"spec": clean, "pos_specs": positive[:, None], "neg_specs": negative[:, None]},
        0,
    )
    _, hard_positive = ContrastiveHead.step(
        IdentityHead(),
        {"spec": action, "pos_specs": positive[:, None], "neg_specs": negative[:, None]},
        0,
    )
    (clean_identity + hard_positive).backward()
    for tensor in (clean, action, positive, negative):
        assert tensor.grad is not None
        assert torch.count_nonzero(tensor.grad) > 0


def test_identity_audit_keeps_registered_counterfactual_peak_payload() -> None:
    _, _, actions = fixture()
    actions = actions.iloc[:2].copy().reset_index(drop=True)
    with tempfile.TemporaryDirectory() as temporary:
        data = Path(temporary) / "spectra.hdf5"
        rows = int(actions["query_row"].max()) + 1
        raw = np.zeros((rows, 2, 128), dtype=np.float32)
        precursor = np.full(rows, 500.0, dtype=np.float32)
        for row in actions["query_row"].to_numpy(np.int64):
            raw[row, :, 0] = [100.0, 1.0]
            raw[row, :, 1] = [200.0, 0.5]
        with h5py.File(data, "w") as handle:
            handle.create_dataset("spectrum", data=raw)
            handle.create_dataset("precursor_mz", data=precursor)
        tensors = []
        for row in actions["query_row"].to_numpy(np.int64):
            keep = (raw[row, 0] > 0) & (raw[row, 1] > 0)
            tensor = NATIVE_ACTION_PREPROCESSOR(
                raw[row][:, keep], prec_mz=500.0, high_form=False,
            )
            tensor[2, 1] *= 0.5
            tensors.append(tensor)
        bank = np.stack(tensors).astype(np.float32)
        report = audit_same_identity_action_views(
            actions, bank, bank.copy(), data=data,
        )
        assert report["both_arms_keep_registered_query_provenance"] is True
        counterfactual = bank.copy()
        counterfactual[0, 1, 0] = 777.0
        report = audit_same_identity_action_views(
            actions, counterfactual, bank, data=data,
        )
        assert report["targeted"][
            "rows_with_introduced_or_shifted_fragment_mz"
        ] == 1
        wrong_precursor = bank.copy()
        wrong_precursor[0, 0, 0] = 501.0
        try:
            audit_same_identity_action_views(
                actions, wrong_precursor, bank, data=data,
            )
        except RuntimeError as error:
            assert "violates frozen query provenance" in str(error)
        else:
            raise AssertionError("wrong precursor passed the query-provenance gate")


def test_two_gpu_entrypoint_runs_targeted_against_same_query_control() -> None:
    script = (
        ROOT / "tasks/run_noise_dreams_hard_positive_native_2gpu.sbatch"
    ).read_text(encoding="utf-8")
    assert "#SBATCH --gpus=2" in script
    assert "#SBATCH --mem" not in script
    assert "tasks/build_noise_dreams_native_triplets.py" in script
    assert "tasks/train_noise_dreams_native.py" in script
    assert 'run_arm "${ALLOCATED_GPUS[0]}" targeted &' in script
    assert 'run_arm "${ALLOCATED_GPUS[1]}" control &' in script
    assert "--max-epochs 1" in script
    assert "--query-equalize-action-dose" not in script
    assert "--minimum-delta-recall1-pp 2.0" in script
    assert "build_noise_reference_aligned_native_triplets.py" not in script
    assert "train_noise_reference_native.py" not in script
    builder = (ROOT / "tasks/build_noise_dreams_native_triplets.py").read_text(
        encoding="utf-8"
    )
    assert "source_family_shuffled_action_bank(" not in builder
    assert '"control_action_spectra": control' in builder
    assert "kind=2" in builder
    assert "kind=3" not in builder
    trainer = (ROOT / "tasks/train_noise_dreams_native.py").read_text(
        encoding="utf-8"
    )
    assert 'NATIVE_SCHEDULE_VERSION = "native_hard_positive_only_query_disjoint_v8"' in trainer
    assert "NATIVE_MAX_OPTIMIZER_STEPS = 9000" in trainer
    replay = (ROOT / "tasks/audit_noise_dreams_native_official_replay.py").read_text(
        encoding="utf-8"
    )
    assert "targeted_hard_positive_relation_beats_same_query_control" in replay


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"[test_noise_dreams_hard_positive_bridge] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
