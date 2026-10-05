"""Local synthetic tests for the Noise-only native DreaMS boundary."""
from __future__ import annotations

import random
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

from audit_noise_dreams_native_official_replay import (
    compare_selection_and_native_geometry,
    row_embeddings,
    weighted_mean,
)
from build_noise_dreams_native_triplets import (
    REGISTERED_SOURCES,
    audit_pickle_free_arrays,
    align_action_tensors,
    build_pools,
    canonicalize_effective_native_triplets,
    canonicalize_semantic_action_aliases,
    fixed_unicode,
    native_action_semantic_sha256,
    pool_report,
    select_actions,
    stable_fold,
)
from dreams.models.heads.heads import ContrastiveHead
from dreams.utils.data import SpectrumPreprocessor
from dreams.utils.dformats import DataFormatA
from train_noise_dreams_native import (
    native_dataset,
    native_one_pass_training_indices,
    trusted_torch_load_scope,
)
from noise_dreams_native_spectrum import (
    action_fragment_profile,
    make_action_spectrum,
    native_action_model_input as adapt_native_action_model_input,
)


ROOT = Path(__file__).resolve().parents[1]


def eligible_formula(prefix: str, *, validation: bool) -> str:
    for index in range(100000):
        value = f"{prefix}{index}H{index + 2}"
        if stable_fold(value, 5, 20260825) == 0:
            continue
        if (stable_fold(value, 10, 20260920) == 0) == validation:
            return value
    raise AssertionError("could not synthesize a formula fold")


def fixture() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame]:
    action_formulas = [
        eligible_formula(f"C{10 + index}", validation=False) for index in range(7)
    ]
    clean_formulas = [
        eligible_formula(f"N{20 + index}", validation=index < 2) for index in range(5)
    ]
    formulas = action_formulas + clean_formulas
    query_rows: list[int] = []
    candidate_rows: list[int] = []
    query_ptr = [0]
    molecule_ptr = [0]
    molecule_label: list[int] = []
    molecule_formula: list[str] = []
    molecule_ik14: list[str] = []
    row = 0
    action_positive: list[int] = []
    action_negative: list[int] = []
    for query, formula in enumerate(formulas):
        query_rows.append(row)
        positive = [row, row + 1]
        negative_a = [row + 2, row + 3]
        negative_b = [row + 4]
        action_positive.append(row + 1)
        action_negative.append(row + 2)
        for label, rows, suffix in (
            (1, positive, "P"),
            (0, negative_a, "A"),
            (0, negative_b, "B"),
        ):
            candidate_rows.extend(rows)
            molecule_ptr.append(len(candidate_rows))
            molecule_label.append(label)
            molecule_formula.append(formula if label else f"X{query}{suffix}")
            molecule_ik14.append(f"IK{query:012d}" if label else f"NK{query:011d}{suffix}")
        query_ptr.append(len(molecule_label))
        row += 5
    graph = {
        "query_ptr": np.asarray(query_ptr, dtype=np.int64),
        "molecule_ptr": np.asarray(molecule_ptr, dtype=np.int64),
        "molecule_label": np.asarray(molecule_label, dtype=np.int8),
        "molecule_formula": np.asarray(molecule_formula),
        "molecule_ik14": np.asarray(molecule_ik14),
        "pair_candidate_row": np.asarray(candidate_rows, dtype=np.int64),
        "query_row": np.asarray(query_rows, dtype=np.int64),
        "query_formula": np.asarray(formulas),
        "query_ik14": np.asarray([f"IK{q:012d}" for q in range(len(formulas))]),
    }
    rng = np.random.default_rng(4)
    embeddings = rng.normal(size=(row, 8)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    cache = {
        "rows": np.arange(row, dtype=np.int64),
        "embeddings": embeddings,
    }
    records: list[dict[str, object]] = []
    for query, source in enumerate(sorted(REGISTERED_SOURCES)):
        formula = formulas[query]
        records.append({
            "action_id": f"a{query}",
            "query_index": query,
            "query_row": query_rows[query],
            "query_ik14": f"IK{query:012d}",
            "query_formula": formula,
            "formula_fold": stable_fold(formula, 5, 20260825),
            "source": source,
            "family": f"family_{source}",
            "recipe_id": f"recipe_{source}",
            "supervision_kind": "corrective",
            "clean_rank": 2,
            "action_rank": 1,
            "action_margin": 0.2,
            "action_tensor_index": query,
            "action_positive_row": action_positive[query],
            "action_hard_negative_row": action_negative[query],
            "control_positive_row": action_positive[query],
            "control_hard_negative_row": action_negative[query],
        })
    return graph, cache, pd.DataFrame(records)


def action_tensor(offset: float) -> np.ndarray:
    body = np.zeros((101, 2), dtype=np.float32)
    body[0] = [500.0 + offset, 1.1]
    body[1] = [100.0 + offset, 1.0]
    body[2] = [200.0 + offset, 0.5]
    return body


def test_all_zero_action_is_routed_before_unmodified_native_preprocessing() -> None:
    preprocessor = SpectrumPreprocessor(
        DataFormatA(), prec_intens=1.1, n_highest_peaks=100,
        spec_entropy_cleaning=False, precision=32,
        mz_shift_aug_p=0, mz_shift_aug_max=0,
    )
    all_zero = np.zeros((101, 2), dtype=np.float32)
    all_zero[0] = [500.0, 1.1]
    all_zero[1:100, 0] = np.arange(1, 100, dtype=np.float32)
    profile = action_fragment_profile(all_zero)
    assert profile["native_action_view_representable"] is False
    assert profile["real_fragment_tokens"] == 99
    try:
        adapt_native_action_model_input(all_zero, preprocessor)
    except RuntimeError as error:
        assert "not representable" in str(error)
    else:
        raise AssertionError("all-zero action reached unmodified native preprocessing")


def test_partial_zero_action_preserves_official_preprocessing_bitwise() -> None:
    tensor = action_tensor(0)
    tensor[2, 1] = 0
    spectrum = make_action_spectrum(tensor)
    official = SpectrumPreprocessor(
        DataFormatA(), prec_intens=1.1, n_highest_peaks=100,
        spec_entropy_cleaning=False, precision=32,
        mz_shift_aug_p=0, mz_shift_aug_max=0,
    )
    args = (spectrum.get_peak_list(),)
    kwargs = {"prec_mz": spectrum.get_precursor_mz(), "high_form": False}
    replay = adapt_native_action_model_input(tensor, official)
    source_tokens = tensor[tensor[:, 0] > 0]
    replay_tokens = replay[replay[:, 0] > 0]
    assert np.array_equal(replay_tokens, source_tokens)
    assert replay_tokens[2, 1] == 0.0


def test_formal_npz_strings_are_pickle_free_unicode() -> None:
    values = fixed_unicode(["A4_exact", "E12B", "μ-action"])
    assert values.dtype.kind == "U"
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "archive.npz"
        arrays = {
            "action_ids": values,
            "query_index": np.asarray([1, 2, 3], dtype=np.int64),
        }
        audit_pickle_free_arrays(arrays, label="synthetic archive")
        np.savez_compressed(path, **arrays)
        with np.load(path, allow_pickle=False) as body:
            replayed = {key: np.asarray(body[key]) for key in body.files}
        audit_pickle_free_arrays(replayed, label="synthetic replay")
        assert replayed["action_ids"].dtype.kind == "U"
        assert replayed["action_ids"].tolist() == values.tolist()
    try:
        audit_pickle_free_arrays(
            {"bad": np.asarray(["x"], dtype=object)}, label="bad archive"
        )
    except RuntimeError as error:
        assert "pickle-requiring object arrays" in str(error)
    else:
        raise AssertionError("object dtype escaped the formal NPZ gate")


def test_native_replay_never_uses_legacy_selection_embeddings_for_roles() -> None:
    rows = np.arange(64, dtype=np.int64)
    selection = np.zeros((64, 1024), dtype=np.float32)
    native = np.zeros((64, 1024), dtype=np.float32)
    selection[:, 0] = 1.0
    native[:, 1] = 1.0
    selection_cache = {"rows": rows, "embeddings": selection}
    native_cache = {"rows": rows, "embeddings": native}
    diagnostic = compare_selection_and_native_geometry(
        selection_cache, rows, native
    )
    assert diagnostic["numerically_identical"] is False
    requested = np.asarray([3, 17, 63], dtype=np.int64)
    replayed = row_embeddings(native_cache, requested)
    assert np.array_equal(replayed, native[requested])
    assert not np.array_equal(replayed, selection[requested])


def test_action_selection_and_lossless_coverage() -> None:
    graph, cache, frame = fixture()
    selected = select_actions(
        frame,
        margin_floor=5e-6,
        outer_fold=0,
        formula_fold_seed=20260825,
        expected_actions=7,
    )
    bank = {
        "action_ids": selected["action_id"].astype(str).to_numpy(),
        "action_spectra": np.stack([action_tensor(i) for i in range(7)]),
        "control_spectra": np.stack([action_tensor(50 + i) for i in range(7)]),
    }
    targeted, shuffled, shuffle_report = align_action_tensors(
        selected, bank, shuffle_seed=3
    )
    assert targeted.shape == shuffled.shape == (7, 101, 2)
    assert shuffle_report["rows"] == 7
    assert not np.array_equal(targeted, shuffled)
    selected["native_action_view_representable"] = True
    train, validation, split = build_pools(
        graph,
        cache,
        selected,
        outer_fold=0,
        formula_fold_seed=20260825,
        validation_folds=10,
        validation_fold=0,
        validation_seed=20260920,
        hard_negative_molecules=2,
        max_positive_pool=32,
        max_negative_pool=32,
    )
    action_events = train["event_action_index"]
    action_events = action_events[action_events >= 0]
    assert np.array_equal(np.sort(action_events), np.arange(7))
    assert split["all_selected_actions_in_optimization"] is True
    assert split["validation_formulas_have_no_selected_actions"] is True
    assert not set(train["event_formula"].astype(str)) & set(
        validation["event_formula"].astype(str)
    )
    report = pool_report(train, epochs=1)
    # ``pool_report`` calls these rows action events.  Keep the test on the
    # canonical report field instead of the removed historical alias
    # ``action_anchors``.
    assert report["action_events"] == 7
    assert report["action_units"] == 7
    assert report["action_clean_boundary_triplets"] == 0
    assert report["action_measured_positive_hard_triplets"] == 7
    assert report["clean_anchors"] == split["train_queries"]
    assert report["possible_triplets"] > report["anchors"]
    assert report["all_action_events_are_singleton_triplets"] is True
    action_events = np.flatnonzero(train["event_kind"] > 0)
    for event in action_events:
        p0, p1 = train["positive_ptr"][event:event + 2]
        n0, n1 = train["negative_ptr"][event:event + 2]
        assert int(p1 - p0) == int(n1 - n0) == 1
        action_index = int(train["event_action_index"][event])
        positive_registry = int(train["positive_idx"][int(p0)])
        negative_registry = int(train["negative_idx"][int(n0)])
        assert int(train["registry_kind"][negative_registry]) == 0
        assert int(train["registry_source_index"][negative_registry]) == int(
            selected.at[action_index, "action_hard_negative_row"]
        )
        if int(train["event_kind"][event]) == 1:
            assert int(train["registry_kind"][positive_registry]) == 0
            assert int(train["registry_source_index"][positive_registry]) == int(
                selected.at[action_index, "action_positive_row"]
            )
            anchor_registry = int(train["anchor_idx"][event])
            assert int(train["registry_kind"][anchor_registry]) == 2
            assert int(train["registry_source_index"][anchor_registry]) == int(
                selected.at[action_index, "query_row"]
            )
        else:
            assert int(train["registry_kind"][positive_registry]) == 0
            assert int(train["registry_source_index"][positive_registry]) == int(
                selected.at[action_index, "action_positive_row"]
            )


def test_native_builder_uses_registered_same_query_matched_controls() -> None:
    actions = pd.DataFrame([
        {
            "query_index": query,
            "action_id": f"{family}|q{query}",
            "source": "A4_exact",
            "family": family,
            # Formal A4 stores peak role/rank in family, not recipe_id.
            "recipe_id": "token=137|dose=0.25",
            "supervision_kind": "corrective",
        }
        for family, query in (
            ("exact_peak_shared|rank=1", 1),
            ("exact_peak_shared|rank=1", 2),
            ("exact_peak_unmatched|rank=1", 3),
            ("exact_peak_unmatched|rank=1", 4),
        )
    ])
    spectra = np.stack([
        np.full((101, 2), index + 1, dtype=np.float32)
        for index in range(len(actions))
    ])
    targeted, control, report = align_action_tensors(
        actions,
        {
            "action_ids": actions["action_id"].astype(str).to_numpy(),
            "action_spectra": spectra,
            "control_spectra": -spectra,
        },
        shuffle_seed=53,
    )
    assert np.array_equal(targeted, spectra)
    assert np.array_equal(control, -spectra)
    assert report["version"] == "same_query_registered_control_spectrum_v1"
    assert report["same_query_by_construction"] is True
    assert report["cross_query_tensor_donors"] == 0
    assert report["targeted_control_distinct_rows"] == len(actions)
    assert report["targeted_control_distinct_fraction"] == 1.0


def test_exact_semantic_aliases_become_one_optimizer_unit_with_full_provenance() -> None:
    graph, cache, frame = fixture()
    first = frame.iloc[0].copy()
    alias = first.copy()
    alias["action_id"] = "independent-source-alias"
    alias["source"] = "E12B"
    alias["family"] = "independent-family"
    qualified = pd.DataFrame([first, alias]).reset_index(drop=True)
    tensor = action_tensor(0)
    padded_alias = tensor.copy()
    padded_alias[3] = padded_alias[2]
    padded_alias[2] = 0
    assert not np.array_equal(tensor, padded_alias)
    assert (
        native_action_semantic_sha256(tensor)
        == native_action_semantic_sha256(padded_alias)
    )
    canonical, canonical_targeted, aliases, report = (
        canonicalize_semantic_action_aliases(
            qualified, np.stack([tensor, padded_alias]),
        )
    )
    assert len(qualified) == len(aliases) == 2
    assert len(canonical) == len(canonical_targeted) == 1
    assert report["qualified_provenance_rows"] == 2
    assert report["unique_semantic_training_units"] == 1
    assert report["duplicate_alias_rows"] == 1
    assert aliases["canonical_action_index"].tolist() == [0, 0]
    assert set(canonical.iloc[0]["qualified_alias_sources"].split("|")) == {
        str(first["source"]), "E12B",
    }
    canonical["native_action_view_representable"] = True
    train, _, _ = build_pools(
        graph,
        cache,
        canonical,
        outer_fold=0,
        formula_fold_seed=20260825,
        validation_folds=10,
        validation_fold=0,
        validation_seed=20260920,
        hard_negative_molecules=2,
        max_positive_pool=32,
        max_negative_pool=32,
    )
    indices, fillers, audit = native_one_pass_training_indices(
        train,
        np.arange(len(train["event_kind"]), dtype=np.int64),
        batch_size=4,
        seed=17,
    )
    assert audit["semantic_action_units"] == 1
    assert audit["every_action_event_exposed_exactly_once"] is True
    assert audit["query_balanced_oversampling"] is False
    assert len(indices) % 4 == 0
    assert np.all(train["event_kind"][fillers] == 0)


def test_replay_uses_exact_schedule_weights_not_unit_mean() -> None:
    values = np.asarray([0.0, 1.0], dtype=np.float64)
    exposures = np.asarray([1, 3], dtype=np.int64)
    assert weighted_mean(values, exposures) == 0.75
    assert float(np.mean(values)) == 0.5
    try:
        weighted_mean(values, np.zeros(2, dtype=np.int64))
    except RuntimeError:
        pass
    else:
        raise AssertionError("zero-dose replay weights were accepted")


def test_unrepresentable_actions_collapse_by_effective_clean_triplet() -> None:
    graph, cache, frame = fixture()
    first = frame.iloc[0].copy()
    second = first.copy()
    second["action_id"] = "different-invisible-zero-action"
    second["source"] = "E12B"
    qualified = pd.DataFrame([first, second]).reset_index(drop=True)
    zero_a = np.zeros((101, 2), dtype=np.float32)
    zero_b = np.zeros((101, 2), dtype=np.float32)
    zero_a[0] = zero_b[0] = [500.0, 1.1]
    zero_a[1, 0] = 100.0
    zero_b[1, 0] = 200.0
    canonical, targeted, aliases, _ = canonicalize_semantic_action_aliases(
        qualified, np.stack([zero_a, zero_b]),
    )
    assert len(canonical) == 2
    effective, effective_targeted, effective_shuffled, effective_aliases, report = (
        canonicalize_effective_native_triplets(
            canonical, targeted, targeted.copy(), aliases,
        )
    )
    assert len(effective) == len(effective_targeted) == len(effective_shuffled) == 1
    assert effective["native_action_view_representable"].tolist() == [False]
    assert effective_aliases["canonical_action_index"].tolist() == [0, 0]
    assert report["clean_only_duplicates_removed"] == 1
    assert report["clean_boundary_only_units"] == 1
    train, _, _ = build_pools(
        graph, cache, effective,
        outer_fold=0, formula_fold_seed=20260825,
        validation_folds=10, validation_fold=0, validation_seed=20260920,
        hard_negative_molecules=2, max_positive_pool=32, max_negative_pool=32,
    )
    assert np.sum(train["event_kind"] == 1) == 1
    assert np.sum(train["event_kind"] == 2) == 0


def test_native_one_pass_uses_every_unique_action_event_once() -> None:
    graph, cache, frame = fixture()
    copies = []
    for suffix in ("b", "c"):
        row = frame.iloc[0].copy()
        row["action_id"] = f"a0{suffix}"
        row["recipe_id"] = f"recipe_0{suffix}"
        copies.append(row)
    frame = pd.concat([frame, pd.DataFrame(copies)], ignore_index=True)
    selected = select_actions(
        frame,
        margin_floor=5e-6,
        outer_fold=0,
        formula_fold_seed=20260825,
        expected_actions=9,
    )
    selected["native_action_view_representable"] = True
    train, _, split = build_pools(
        graph,
        cache,
        selected,
        outer_fold=0,
        formula_fold_seed=20260825,
        validation_folds=10,
        validation_fold=0,
        validation_seed=20260920,
        hard_negative_molecules=2,
        max_positive_pool=32,
        max_negative_pool=32,
    )
    assert split["maximum_actions_per_query"] == 3
    event_indices = np.arange(len(train["event_kind"]), dtype=np.int64)
    scheduled, fillers, audit = native_one_pass_training_indices(
        train,
        event_indices,
        batch_size=4,
        seed=17,
    )
    assert audit["scheduler"] == "official_native_shuffle_drop_last_one_triplet_v5"
    assert audit["semantic_action_units"] == 9
    assert audit["every_action_event_exposed_exactly_once"] is True
    assert audit["query_balanced_oversampling"] is False
    assert audit["maximum_distinct_actions_per_query"] == 3
    assert len(scheduled) % 4 == 0
    base_counts = np.bincount(scheduled, minlength=len(event_indices))
    action_events = np.flatnonzero(train["event_kind"] > 0)
    assert np.all(base_counts[action_events] == 1)
    assert np.all(train["event_kind"][fillers] == 0)
    action_queries = set(map(int, train["event_query"][action_events]))
    assert all(int(train["event_query"][event]) not in action_queries for event in fillers)
    assert int(np.sum(base_counts) - len(event_indices)) == len(fillers)


def test_native_dataset_roundtrip_and_dynamic_one_one() -> None:
    graph, cache, frame = fixture()
    copies = []
    for suffix, negative_row in (("b", 3), ("c", 4)):
        row = frame.iloc[0].copy()
        row["action_id"] = f"a0{suffix}"
        row["recipe_id"] = f"recipe_0{suffix}"
        row["action_hard_negative_row"] = negative_row
        row["control_hard_negative_row"] = negative_row
        copies.append(row)
    frame = pd.concat([frame, pd.DataFrame(copies)], ignore_index=True)
    selected = select_actions(
        frame,
        margin_floor=5e-6,
        outer_fold=0,
        formula_fold_seed=20260825,
        expected_actions=9,
    )
    selected["native_action_view_representable"] = True
    train, _, _ = build_pools(
        graph,
        cache,
        selected,
        outer_fold=0,
        formula_fold_seed=20260825,
        validation_folds=10,
        validation_fold=0,
        validation_seed=20260920,
        hard_negative_molecules=2,
        max_positive_pool=32,
        max_negative_pool=32,
    )
    actions = np.stack([action_tensor(i) for i in range(9)])
    preprocessor = SpectrumPreprocessor(
        DataFormatA(), prec_intens=1.1, n_highest_peaks=100, precision=32
    )
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "data.hdf5"
        rows = int(np.max(graph["pair_candidate_row"])) + 1
        raw = np.zeros((rows, 2, 128), dtype=np.float32)
        precursor = np.zeros(rows, dtype=np.float32)
        for row in range(rows):
            raw[row, :, 0] = [100.0 + row / 1000.0, 1.0]
            raw[row, :, 1] = [200.0 + row / 1000.0, 0.5]
            precursor[row] = 500.0 + row / 1000.0
        with h5py.File(path, "w") as handle:
            handle.create_dataset("spectrum", data=raw)
            handle.create_dataset("precursor_mz", data=precursor)
        dataset, indices, audit = native_dataset(
            train, path, actions, preprocessor
        )
        assert audit["action_native_preprocessor_max_abs_error"] == 0.0
        assert audit["event_memberships_are_independent"] is True
        assert audit["event_anchor_rows"] == len(train["anchor_idx"])
        assert audit["dataset_rows"] == (
            audit["registry_spectra"] + audit["event_anchor_rows"]
        )
        assert np.array_equal(
            indices,
            np.arange(
                audit["registry_spectra"],
                audit["registry_spectra"] + audit["event_anchor_rows"],
                dtype=np.int64,
            ),
        )
        for event, dataset_index in enumerate(indices):
            p0, p1 = map(int, train["positive_ptr"][event:event + 2])
            n0, n1 = map(int, train["negative_ptr"][event:event + 2])
            assert dataset.df.at[int(dataset_index), "pos_idx"] == list(
                map(int, train["positive_idx"][p0:p1])
            )
            assert dataset.df.at[int(dataset_index), "neg_idx"] == list(
                map(int, train["negative_idx"][n0:n1])
            )
        query_hard_positive_events = np.flatnonzero(
            (train["event_query"] == 0) & (train["event_kind"] == 2)
        )
        assert len(query_hard_positive_events) == 3
        assert {
            tuple(dataset.df.at[int(indices[event]), "neg_idx"])
            for event in query_hard_positive_events
        } == {
            tuple(map(int, train["negative_idx"][
                int(train["negative_ptr"][event]):
                int(train["negative_ptr"][event + 1])
            ]))
            for event in query_hard_positive_events
        }
        assert len({
            tuple(dataset.df.at[int(indices[event]), "neg_idx"])
            for event in query_hard_positive_events
        }) == 3
        action_event = int(np.flatnonzero(train["event_kind"] == 2)[0])
        item = dataset[int(indices[action_event])]
        assert item["spec"].shape == (101, 2)
        assert item["pos_specs"].shape == (1, 101, 2)
        assert item["neg_specs"].shape == (1, 101, 2)
        random.seed(1)
        negatives = {
            dataset[int(indices[0])]["neg_specs"].tobytes() for _ in range(20)
        }
        assert len(negatives) >= 2


class IdentityHead:
    triplet_loss_margin = 0.1

    def __call__(self, value: torch.Tensor, charge=None) -> torch.Tensor:
        del charge
        return value[:, 0, :]


def test_native_contrastive_loss_reaches_all_three_roles() -> None:
    anchor = torch.tensor(
        [[[1.0, 0.0]], [[0.0, 1.0]]], requires_grad=True
    )
    positive = torch.tensor(
        [[[[0.8, 0.2]]], [[[0.2, 0.8]]]], requires_grad=True
    )
    negative = torch.tensor(
        [[[[0.9, 0.1]]], [[[0.1, 0.9]]]], requires_grad=True
    )
    _, loss = ContrastiveHead.step(
        IdentityHead(),
        {"spec": anchor, "pos_specs": positive, "neg_specs": negative},
        0,
    )
    expected = torch.clamp_min(
        0.1
        - torch.nn.functional.cosine_similarity(anchor, positive[:, 0], dim=-1)
        + torch.nn.functional.cosine_similarity(anchor, negative[:, 0], dim=-1),
        0,
    ).mean()
    assert torch.allclose(loss, expected, atol=0, rtol=0)
    loss.backward()
    for value in (anchor, positive, negative):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad) > 0


def test_two_sequential_native_triplets_reach_all_chain_roles() -> None:
    clean = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
    action = torch.tensor([[[0.8, 0.2]]], requires_grad=True)
    positive = torch.tensor([[[0.6, 0.4]]], requires_grad=True)
    negative = torch.tensor([[[0.95, 0.05]]], requires_grad=True)
    _, clean_boundary = ContrastiveHead.step(
        IdentityHead(),
        {
            "spec": clean,
            "pos_specs": positive[:, None],
            "neg_specs": negative[:, None],
        },
        0,
    )
    _, boundary = ContrastiveHead.step(
        IdentityHead(),
        {
            "spec": action,
            "pos_specs": positive[:, None],
            "neg_specs": negative[:, None],
        },
        0,
    )
    assert float(clean_boundary.detach()) > 0
    clean_boundary.backward()
    for value in (clean, positive, negative):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad) > 0
    assert action.grad is None
    for value in (action, positive, negative):
        value.grad = None
    assert float(boundary.detach()) > 0
    boundary.backward()
    for value in (action, positive, negative):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad) > 0


def test_trusted_load_scope_overrides_missing_and_explicit_none() -> None:
    original_torch_load = torch.load
    observed: list[object] = []

    def recording_load(path, *, weights_only=None, **kwargs):
        observed.append(weights_only)
        return weights_only

    torch.load = recording_load
    try:
        with trusted_torch_load_scope():
            assert torch.load("missing") is False
            assert torch.load("explicit-none", weights_only=None) is False
            assert torch.load("explicit-true", weights_only=True) is True
        assert torch.load is recording_load
    finally:
        torch.load = original_torch_load
    assert observed == [False, False, True]


def test_production_source_contains_no_custom_training_semantics() -> None:
    trainer = (ROOT / "tasks/train_noise_dreams_native.py").read_text(
        encoding="utf-8"
    )
    assert "from dreams.models.heads.heads import ContrastiveHead" in trainer
    assert "ContrastiveSpectraDataset, SpectrumPreprocessor" in trainer
    assert "from dreams.utils.dformats import DataFormatA" in trainer
    assert "def load_trusted_native_contrastive_head(" in trainer
    assert 'NATIVE_LOADER_COMPAT_VERSION = "chemaware_slim_to_native_contrastive_v5"' in trainer
    assert 'kwargs["weights_only"] = False' in trainer
    assert "initialized, kind = load_base_model(" in trainer
    assert "model = ContrastiveHead(" in trainer
    assert "model.head.load_state_dict(initialized.head.state_dict(), strict=True)" in trainer
    assert "ContrastiveHead.load_from_checkpoint" not in trainer
    assert 'kwargs.get("weights_only") is None' in trainer
    assert "model.configure_optimizers()" in trainer
    assert trainer.count("native_query_disjoint_one_pass_batches(") >= 2
    assert 'NATIVE_SCHEDULE_VERSION = "native_hard_positive_only_query_disjoint_v8"' in trainer
    assert "NATIVE_MAX_EPOCHS = 1" in trainer
    assert "batch_sampler=train_sampler" not in trainer
    assert "shuffle=True" in trainer
    assert "drop_last=True" in trainer
    assert "save_top_k=0" in trainer
    for forbidden in (
        "AdamW",
        "softplus",
        "SeparatedE4",
        "injector",
        "preservation",
        "safety_loss",
        "margin_floor",
        "autocast",
        "GradScaler",
        "final_shared_encoder.pt",
    ):
        assert forbidden not in trainer, forbidden
    assert "torch.optim.Adam(" not in trainer
    assert "clamp_min(" not in trainer
    adapter = (ROOT / "tasks/noise_dreams_native_spectrum.py").read_text(
        encoding="utf-8"
    )
    assert "class NativeCanonicalPaddingSpectrumPreprocessor" not in adapter
    assert "output[padding] = 0" not in adapter
    assert "all-zero action is not representable" in adapter
    assert "spec_preproc = SpectrumPreprocessor(" in trainer
    builder = (ROOT / "tasks/build_noise_dreams_native_triplets.py").read_text(
        encoding="utf-8"
    )
    assert 'MATCHED_CONTROL_VERSION = "same_query_registered_control_spectrum_v1"' in builder
    assert "from noise_corrected_shuffled_control_v3 import" not in builder
    assert "from noise_corrected_action_routing_v3 import" not in builder
    assert (
        'NATIVE_TRIPLET_BUILDER_VERSION = "native_hard_positive_only_v16"'
        in builder
    )
    assert "each effective action must contribute exactly one native triplet" in builder
    assert "exact hard negative" in builder
    assert "canonicalize_semantic_action_aliases(" in builder
    assert '"builder_version": NATIVE_TRIPLET_BUILDER_VERSION' in builder
    assert '"matched_control_version": MATCHED_CONTROL_VERSION' in builder
    assert "fixed_unicode(actions[\"action_id\"])" in builder
    assert "Reload every formal archive" in builder
    assert 'staging / "qualified_actions.csv.gz"' in builder
    assert 'staging / "action_aliases.csv.gz"' in builder
    assert "clean_positive_order" not in builder
    replay = (
        ROOT / "tasks/audit_noise_dreams_native_official_replay.py"
    ).read_text(encoding="utf-8")
    assert "targeted_schedule_weighted_mean_margin_beats_matched_shuffled" not in replay
    assert "margin_comparisons_are_diagnostic_not_training_gates" in replay
    assert "native_query_disjoint_one_pass_batches(" in replay
    assert '"exact_native_training_schedule"' in replay
    assert 'train_pool_path = args.triplet_dir / "train_pool.npz"' in replay
    assert '"action_unit_exposure_sha256": schedule_exposure_sha256' in replay
    assert '"seed": 3407' in replay
    assert "all_32114_qualified_rows_preserved_in_alias_ledger" in replay
    assert "all_effective_training_units_replayed" in replay
    assert "actual_action_gradient_audit(" in replay
    assert "native_action_model_input(source_tensor, preprocessor)" in replay
    assert (
        'NATIVE_REPLAY_GEOMETRY_VERSION = "native_hard_positive_only_official_v16"'
        in replay
    )
    assert "positive = row_embeddings(\n        native_cache" in replay
    assert "negative = row_embeddings(\n        native_cache" in replay
    assert "clean = row_embeddings(\n        native_cache" in replay
    assert '"native_checkpoint_matches_frozen_official_cache"' not in replay
    assert '"clean_action_measured_positive_negative_share_native_checkpoint_geometry"' in replay
    assert "schedule_weighted_clean_query_relation_active_fraction" in replay
    assert "clean_and_hard_positive_native_paths_have_hinge_signal" in replay
    assert "capable_actions = actions.loc[action_capable]" in replay
    assert '"seed": 3407' in trainer
    assert '!= replay_schedule.get("action_unit_exposure_sha256")' in trainer
    evaluator = (ROOT / "tasks/evaluate_noise_dreams_native.py").read_text(
        encoding="utf-8"
    )
    metric_core = (
        ROOT / "tasks/noise_corrected_fullgraph_evaluation.py"
    ).read_text(encoding="utf-8")
    assert "full_metrics(" in evaluator
    assert "mature_e8_checkpoint" in evaluator
    assert "candidate_vs_mature_e8" in evaluator
    assert (
        'NATIVE_EVALUATION_BASELINE_VERSION = "official_slim_reencoded_v3"'
        in evaluator
    )
    assert "baseline_scores = score_embeddings(graph, rows, official_embeddings)" in evaluator
    assert "--official-slim-checkpoint" in evaluator
    summary = (ROOT / "tasks/summarize_noise_dreams_native.py").read_text(
        encoding="utf-8"
    )
    assert "targeted_recall1_gain_vs_mature_e8_at_least_4pp" in summary
    assert "targeted_action_curriculum_gain_vs_control_at_least_0_5pp" in summary
    combined_evaluation_source = evaluator + metric_core
    assert "for cutoff in (1, 2, 3, 5, 10, 20)" in metric_core
    for required in (
        "near_subset",
        "risk_net_lambda2",
        "massspecgym_10ppm_pooled_pairwise",
        "massspecgym_mh_10ppm_pooled_pairwise",
    ):
        assert required in combined_evaluation_source, required


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
    print(f"[test_noise_dreams_native] PASS tests={len(tests)}")


if __name__ == "__main__":
    main()
