from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import h5py
import pytest

from tasks.chemaware_shared_v2_core import (
    MOLFORMER_STATUS, MORGAN_STATUS, TOKEN_STATUS, MoleculeTeacherStore, formula_folds,
)
from tasks.analyze_chemaware_shared_v3_frozen_probe_reachability import (
    main as analyze_frozen_probe,
)
from tasks.encode_chemaware_shared_v2_molformer import (
    collision_audit, graph_molecule_records, hdf5_strings,
)
from tasks.encode_chemaware_shared_v3_morgan import main as encode_morgan
from tasks.noise_final_core import CandidateGraph
from tasks.summarize_chemaware_shared_v2_g1 import main as summarize_g1
from tasks.summarize_chemaware_shared_v2_g2 import ARMS, main as summarize_g2
from tasks.summarize_chemaware_shared_v3_g1 import main as summarize_v3_g1
from tasks.summarize_chemaware_shared_v3_g2 import main as summarize_v3_g2
from tasks.train_chemaware_shared_v2 import main as train_g1
from tasks.train_chemaware_shared_v3_peft import targeted_probe_query_audit
from tasks.validate_chemaware_shared_v3_g2_pilot import main as validate_v3_g2_pilot


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def formulas_covering_folds(count: int = 5, seed: int = 1701) -> list[str]:
    output: dict[int, str] = {}
    index = 1
    while len(output) < count:
        value = f"C{index}H{index + 2}O{index % 4 + 1}"
        fold = int(formula_folds(np.asarray([value]), count, seed)[0])
        output.setdefault(fold, value)
        index += 1
    return [output[fold] for fold in range(count)]


def write_synthetic_graph_and_cache(root: Path) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(41)
    dimension = 1024
    formulas = formulas_covering_folds()
    n_queries = 10
    rows: list[int] = []
    embeddings: list[np.ndarray] = []
    query_row: list[int] = []
    query_formula: list[str] = []
    query_ik14: list[str] = []
    pair_candidate_row: list[int] = []
    features: list[list[float]] = []
    molecule_label: list[int] = []
    molecule_ik14: list[str] = []
    molecule_formula: list[str] = []
    molecule_grade: list[int] = []
    query_ptr = [0]
    molecule_ptr = [0]

    def unit(value: np.ndarray) -> np.ndarray:
        return (value / np.linalg.norm(value)).astype(np.float32)

    row = 0
    for query in range(n_queries):
        fold = query % 5
        q = unit(rng.normal(size=dimension).astype(np.float32))
        qrow = row
        rows.append(row)
        embeddings.append(q)
        row += 1
        query_row.append(qrow)
        query_formula.append(formulas[fold])
        query_ik14.append(f"Q{query:013d}")
        candidate_formulas = [
            formulas[fold], formulas[(fold + 1) % 5], formulas[(fold + 2) % 5]
        ]
        for molecule, candidate_formula in enumerate(candidate_formulas):
            # Positive is close, first negative deliberately hard, second easier.
            strength = (0.88, 0.82, 0.55)[molecule]
            candidate = unit(strength * q + (1.0 - strength) * unit(rng.normal(size=dimension)))
            rows.append(row)
            embeddings.append(candidate)
            pair_candidate_row.append(row)
            features.append([float(q @ candidate)])
            row += 1
            molecule_label.append(1 if molecule == 0 else 0)
            molecule_ik14.append(f"M{query:010d}{molecule:03d}")
            molecule_formula.append(candidate_formula)
            molecule_grade.append(-2 if molecule == 0 else molecule)
            molecule_ptr.append(molecule_ptr[-1] + 1)
        query_ptr.append(query_ptr[-1] + 3)

    graph_path = root / "graph.npz"
    np.savez_compressed(
        graph_path,
        feature_names=np.asarray(["dreams_similarity"], dtype=object),
        features=np.asarray(features, dtype=np.float32),
        pair_candidate_row=np.asarray(pair_candidate_row, dtype=np.int64),
        query_ptr=np.asarray(query_ptr, dtype=np.int64),
        molecule_ptr=np.asarray(molecule_ptr, dtype=np.int64),
        molecule_label=np.asarray(molecule_label, dtype=np.int8),
        molecule_ik14=np.asarray(molecule_ik14, dtype=object),
        molecule_formula=np.asarray(molecule_formula, dtype=object),
        molecule_mces_grade=np.asarray(molecule_grade, dtype=np.int8),
        query_row=np.asarray(query_row, dtype=np.int64),
        query_ik14=np.asarray(query_ik14, dtype=object),
        query_formula=np.asarray(query_formula, dtype=object),
        query_has_near=np.asarray([(index % 2) == 0 for index in range(n_queries)]),
    )
    official_path = root / "official.pt"
    official_path.write_bytes(b"synthetic official checkpoint identity")
    cache = root / "cache"
    cache.mkdir()
    rows_array = np.asarray(rows, dtype=np.int64)
    order = np.argsort(rows_array)
    rows_array = rows_array[order]
    embedding_array = np.asarray(embeddings, dtype=np.float32)[order]
    np.save(cache / "rows.npy", rows_array)
    np.save(cache / "tokens_f16.npy", rng.normal(size=(len(rows), 3, dimension)).astype(np.float16))
    mz = np.tile(np.asarray([50.0, 90.0, 130.0], dtype=np.float32), (len(rows), 1))
    np.save(cache / "mz_f32.npy", mz)
    np.save(cache / "intensity_f32.npy", np.tile([0.2, 1.0, 0.4], (len(rows), 1)).astype(np.float32))
    np.save(cache / "valid.npy", np.ones((len(rows), 3), dtype=bool))
    np.save(cache / "precursor_mz_f32.npy", np.full(len(rows), 300.0, dtype=np.float32))
    np.save(cache / "official_embeddings_f32.npy", embedding_array)
    (cache / "report.json").write_text(json.dumps({
        "status": TOKEN_STATUS,
        "formal": False,
        "spectra": len(rows),
        "provenance": {
            "graph_sha256": sha256(graph_path),
            "official_checkpoint_sha256": sha256(official_path),
        },
    }), encoding="utf-8")
    return graph_path, cache, official_path


def write_synthetic_teacher(root: Path, graph_path: Path) -> Path:
    graph = CandidateGraph(graph_path)
    teacher = root / "teacher"
    teacher.mkdir()
    ik14, first = np.unique(graph.molecule_ik14.astype(str), return_index=True)
    formula = graph.molecule_formula[first].astype(str)
    rng = np.random.default_rng(97)
    embeddings = rng.normal(size=(len(ik14), 16)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    np.save(teacher / "molecule_ik14.npy", ik14.astype("U14"))
    np.save(teacher / "molecule_formula.npy", formula.astype("U64"))
    np.save(teacher / "molformer_smiles.npy", np.asarray([
        "C" * (index % 8 + 1) for index in range(len(ik14))
    ], dtype=str))
    np.save(teacher / "embeddings_f32.npy", embeddings)
    (teacher / "report.json").write_text(json.dumps({
        "status": MOLFORMER_STATUS,
        "formal": False,
        "training_only": True,
        "provenance": {"graph_sha256": sha256(graph_path)},
    }), encoding="utf-8")
    return teacher


def test_g1_end_to_end_smoke(tmp_path: Path):
    graph, cache, official = write_synthetic_graph_and_cache(tmp_path)
    output = tmp_path / "output"
    argv = [
        "train_chemaware_shared_v2.py",
        "--graph", str(graph),
        "--token-dir", str(cache),
        "--official-checkpoint", str(official),
        "--output-root", str(output),
        "--outer-fold", "0",
        "--seed", "7",
        "--epochs", "1",
        "--batch-queries", "2",
        "--eval-batch-size", "32",
        "--hidden-dim", "16",
        "--device", "cpu",
        "--max-train-queries", "3",
        "--max-eval-queries", "1",
    ]
    with patch.object(sys, "argv", argv):
        train_g1()
    run = output / "seed_7" / "fold_0"
    decision = json.loads((run / "decision.json").read_text(encoding="utf-8"))
    assert decision["status"] == "chemaware_shared_v2_g1_fold_complete"
    assert decision["formal"] is False
    assert decision["chemical_supervision"] is False
    assert decision["preflight"]["zero_init_exact_rank_reproduction"] is True
    assert decision["preflight"]["heldout_formula_molecules_excluded_from_training"] is True
    assert (run / "adapter.pt").exists()
    assert (run / "outer_predictions.npz").exists()


def test_cache_provenance_fails_closed(tmp_path: Path):
    graph, cache, official = write_synthetic_graph_and_cache(tmp_path)
    official.write_bytes(b"changed checkpoint")
    output = tmp_path / "output"
    argv = [
        "train_chemaware_shared_v2.py",
        "--graph", str(graph), "--token-dir", str(cache),
        "--official-checkpoint", str(official), "--output-root", str(output),
        "--outer-fold", "0", "--seed", "7", "--device", "cpu",
        "--max-train-queries", "1", "--max-eval-queries", "1",
    ]
    with patch.object(sys, "argv", argv):
        try:
            train_g1()
        except RuntimeError as error:
            assert "provenance mismatch" in str(error)
        else:
            raise AssertionError("checkpoint mismatch must fail closed")


def test_g2_correct_teacher_and_pseudoteachers_are_graph_aligned(tmp_path: Path):
    graph_path, cache, official = write_synthetic_graph_and_cache(tmp_path)
    graph = CandidateGraph(graph_path)
    teacher_dir = write_synthetic_teacher(tmp_path, graph_path)
    store = MoleculeTeacherStore(teacher_dir, graph_path, graph, require_formal=False)
    allowed = np.ones(len(graph.molecule_ik14), dtype=bool)
    correct, correct_mask, correct_audit = store.graph_embeddings(graph, allowed, "correct", 11)
    permuted, permuted_mask, permuted_audit = store.graph_embeddings(graph, allowed, "identity_permuted", 11)
    random, random_mask, random_audit = store.graph_embeddings(graph, allowed, "random_marginal", 11)
    assert correct.shape == permuted.shape == random.shape
    assert np.all(correct_mask) and np.all(permuted_mask) and np.all(random_mask)
    assert correct_audit["fixed_points"] == len(np.unique(graph.molecule_ik14))
    assert permuted_audit["fixed_points"] == 0
    assert random_audit["coordinate_marginal_preserved"] is True
    assert not np.allclose(correct, permuted)
    assert not np.allclose(correct, random)
    identity_correct, identity_mask, identity_audit = store.identity_targets(
        graph, allowed, "correct", 11
    )
    assert np.all(identity_mask)
    assert identity_audit["assigned_unique_identities"] == len(store.ik14)
    assert np.allclose(correct, identity_correct[store.graph_index])
    scoped, scoped_mask, scoped_audit = store.graph_embeddings(
        graph, allowed, "correct_same_formula_scope", 11
    )
    mismatched, mismatch_mask, mismatch_audit = store.graph_embeddings(
        graph, allowed, "same_formula_mismatched", 11
    )
    assert np.array_equal(scoped_mask, mismatch_mask)
    assert scoped_audit["same_formula_only"] and mismatch_audit["same_formula_only"]
    assert np.allclose(np.linalg.norm(scoped[scoped_mask], axis=1), 1.0)
    assert not np.allclose(scoped[scoped_mask], mismatched[mismatch_mask])
    queries = np.arange(graph.n_queries, dtype=np.int64)
    correct_selection = targeted_probe_query_audit(
        graph, queries, allowed, correct_mask, margin_threshold=1.0
    )
    permuted_selection = targeted_probe_query_audit(
        graph, queries, allowed, permuted_mask, margin_threshold=1.0
    )
    scoped_selection = targeted_probe_query_audit(
        graph, queries, allowed, scoped_mask, margin_threshold=1.0
    )
    mismatch_selection = targeted_probe_query_audit(
        graph, queries, allowed, mismatch_mask, margin_threshold=1.0
    )
    assert correct_selection["selection_query_ledger_sha256"] == (
        permuted_selection["selection_query_ledger_sha256"]
    )
    assert scoped_selection["selection_query_ledger_sha256"] == (
        mismatch_selection["selection_query_ledger_sha256"]
    )

    output = tmp_path / "g2_output"
    argv = [
        "train_chemaware_shared_v2.py",
        "--graph", str(graph_path), "--token-dir", str(cache),
        "--official-checkpoint", str(official), "--output-root", str(output),
        "--molecule-teacher-dir", str(teacher_dir), "--teacher-control", "correct",
        "--outer-fold", "0", "--seed", "7", "--epochs", "1",
        "--batch-queries", "2", "--eval-batch-size", "32",
        "--hidden-dim", "16",
        "--device", "cpu", "--max-train-queries", "3", "--max-eval-queries", "1",
    ]
    with patch.object(sys, "argv", argv):
        train_g1()
    run = output / "correct" / "seed_7" / "fold_0"
    decision = json.loads((run / "decision.json").read_text(encoding="utf-8"))
    assert decision["status"] == "chemaware_shared_v2_g2_fold_complete"
    assert decision["chemical_supervision"] is True
    assert decision["teacher_control"] == "correct"
    assert decision["training_only_projector_used"] is False
    assert decision["teacher_control_audit"]["chemical_effect_queries"] > 0
    assert decision["history"][1]["train"]["molecule"] > 0


def test_g1_summary_requires_complete_paired_multiseed_ledger(tmp_path: Path):
    graph_path, _, _ = write_synthetic_graph_and_cache(tmp_path)
    graph = CandidateGraph(graph_path)
    root = tmp_path / "runs"
    for seed in (17, 41, 73):
        for fold in range(5):
            query = np.flatnonzero(np.arange(graph.n_queries) % 5 == fold)
            run = root / f"seed_{seed}" / f"fold_{fold}"
            run.mkdir(parents=True)
            old_rank = np.full(len(query), 2, dtype=np.int32)
            new_rank = np.ones(len(query), dtype=np.int32)
            np.savez_compressed(
                run / "outer_predictions.npz",
                query=query, old_rank=old_rank, new_rank=new_rank,
            )
            (run / "decision.json").write_text(json.dumps({
                "status": "chemaware_shared_v2_g1_fold_complete",
                "formal": True,
                "chemical_supervision": False,
                "seed": seed,
                "outer_fold": fold,
                "best_epoch": 1,
            }), encoding="utf-8")
    output = tmp_path / "summary.json"
    argv = [
        "summarize_chemaware_shared_v2_g1.py",
        "--root", str(root), "--graph", str(graph_path),
        "--output", str(output), "--bootstrap-resamples", "50",
    ]
    with patch.object(sys, "argv", argv):
        summarize_g1()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "chemaware_shared_v2_g1_multifold_summary_complete"
    assert report["matched_control_ready_for_G2"] is True
    assert report["across_seed"]["delta_recall1_mean"] == 1.0
    assert report["gates"]["all_seed_formula_ci_positive"] is True


def test_v3_g1_summary_requires_peft_gradient_and_capacity_contract(tmp_path: Path):
    graph_path, _, _ = write_synthetic_graph_and_cache(tmp_path)
    graph = CandidateGraph(graph_path)
    root = tmp_path / "v3_runs"
    capacity = {
        "config": {"last_blocks": 1, "rank": 8, "alpha": 8.0},
        "trainable_parameters": 1024,
    }
    for seed in (17, 41, 73):
        for fold in range(5):
            query = np.flatnonzero(np.arange(graph.n_queries) % 5 == fold)
            run = root / f"seed_{seed}" / f"fold_{fold}"
            run.mkdir(parents=True)
            np.savez_compressed(
                run / "outer_predictions.npz",
                query=query,
                old_rank=np.full(len(query), 2, dtype=np.int32),
                new_rank=np.ones(len(query), dtype=np.int32),
            )
            (run / "peft.pt").write_bytes(b"synthetic peft")
            (run / "decision.json").write_text(json.dumps({
                "status": "chemaware_shared_v3_g1_peft_fold_complete",
                "formal": True,
                "chemical_supervision": False,
                "query_reference_encoder_shared": True,
                "candidate_inputs_at_inference": False,
                "P2b_used": False,
                "seed": seed,
                "outer_fold": fold,
                "best_epoch": 1,
                "capacity": capacity,
                "first_step_audit": {
                    "gradient_l2": 0.1,
                    "parameter_update_l2": 0.01,
                    "changed_parameter_tensors": 4,
                },
                "outer": {"preservation_min": 0.999},
                "preflight": {"complete_split_eligible_candidate_groups": True},
            }), encoding="utf-8")
    output = tmp_path / "v3_summary.json"
    argv = [
        "summarize_chemaware_shared_v3_g1.py",
        "--root", str(root), "--graph", str(graph_path),
        "--output", str(output), "--bootstrap-resamples", "50",
    ]
    with patch.object(sys, "argv", argv):
        summarize_v3_g1()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "chemaware_shared_v3_g1_peft_multifold_summary_complete"
    assert report["capacity"] == capacity
    assert report["across_seed"]["delta_recall1_mean"] == 1.0
    assert all(report["gates"].values())


def test_molformer_records_reconcile_graph_hdf5_and_audit_stereo_collapse(tmp_path: Path):
    graph_path, _, _ = write_synthetic_graph_and_cache(tmp_path)
    graph = CandidateGraph(graph_path)
    data = tmp_path / "structures.hdf5"
    total_rows = int(max(np.max(graph.query_row), np.max(graph.pair_candidate_row))) + 1
    inchikey = np.asarray(["UNUSED-REST"] * total_rows, dtype="S32")
    smiles = np.asarray(["CC"] * total_rows, dtype="S128")
    pair_molecule = np.repeat(np.arange(len(graph.molecule_ik14)), np.diff(graph.molecule_ptr))
    for pair, molecule in enumerate(pair_molecule):
        row = int(graph.pair_candidate_row[pair])
        inchikey[row] = f"{graph.molecule_ik14[molecule]}-REST".encode()
        smiles[row] = ("C" * (molecule % 6 + 1)).encode()
    with h5py.File(data, "w") as handle:
        handle.create_dataset("INCHIKEY", data=inchikey)
        handle.create_dataset("smiles", data=smiles)
    records = graph_molecule_records(graph, data)
    assert len(records) == len(np.unique(graph.molecule_ik14))
    audit = collision_audit([
        {
            "ik14": "AAAAAAAAAAAAAA",
            "molformer_smiles": "CC(O)F",
            "canonical_isomeric_smiles": "C[C@H](O)F",
            "stereochemistry_removed": True,
        },
        {
            "ik14": "BBBBBBBBBBBBBB",
            "molformer_smiles": "CC(O)F",
            "canonical_isomeric_smiles": "C[C@@H](O)F",
            "stereochemistry_removed": True,
        },
    ])
    assert audit["cross_identity_collapsed_smiles"] == 1
    assert audit["cross_identity_collapsed_molecules"] == 2


def test_molformer_hdf5_strings_restores_duplicate_candidate_rows(tmp_path: Path):
    data = tmp_path / "duplicates.hdf5"
    with h5py.File(data, "w") as handle:
        handle.create_dataset(
            "INCHIKEY",
            data=np.asarray(["ROW0", "ROW1", "ROW2", "ROW3"], dtype="S8"),
        )
    requested = np.asarray([3, 1, 3, 0, 1], dtype=np.int64)
    with h5py.File(data, "r") as handle:
        observed = hdf5_strings(handle, "INCHIKEY", requested)
    assert observed.tolist() == ["ROW3", "ROW1", "ROW3", "ROW0", "ROW1"]


def test_morgan_teacher_reuses_audited_identity_smiles_ledger(tmp_path: Path):
    graph_path, _, _ = write_synthetic_graph_and_cache(tmp_path)
    source = write_synthetic_teacher(tmp_path, graph_path)
    output = tmp_path / "morgan"
    argv = [
        "encode_chemaware_shared_v3_morgan.py",
        "--graph", str(graph_path), "--source-dir", str(source),
        "--output-dir", str(output), "--dimensions", "128",
    ]
    with patch.object(sys, "argv", argv):
        encode_morgan()
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == MORGAN_STATUS
    assert report["teacher_kind"] == "morgan_binary_connectivity"
    assert report["fingerprint"]["include_chirality"] is False
    assert report["identity_smiles_ledger"]["kind"] == "legacy_audited_molformer_cache"
    graph = CandidateGraph(graph_path)
    store = MoleculeTeacherStore(output, graph_path, graph, require_formal=False)
    values = np.asarray(store.embeddings)
    assert values.shape == (len(np.unique(graph.molecule_ik14)), 128)
    assert np.allclose(np.linalg.norm(values, axis=1), 1.0)


def test_morgan_teacher_builds_directly_from_graph_hdf5_without_molformer(tmp_path: Path):
    graph_path, _, _ = write_synthetic_graph_and_cache(tmp_path)
    graph = CandidateGraph(graph_path)
    data = tmp_path / "morgan_structures.hdf5"
    total_rows = int(max(np.max(graph.query_row), np.max(graph.pair_candidate_row))) + 1
    inchikey = np.asarray(["UNUSED-REST"] * total_rows, dtype="S32")
    smiles = np.asarray(["CC"] * total_rows, dtype="S256")
    pair_molecule = np.repeat(np.arange(len(graph.molecule_ik14)), np.diff(graph.molecule_ptr))
    for pair, molecule in enumerate(pair_molecule):
        row = int(graph.pair_candidate_row[pair])
        inchikey[row] = f"{graph.molecule_ik14[molecule]}-REST".encode()
        smiles[row] = ("C" * (molecule + 1)).encode()
    with h5py.File(data, "w") as handle:
        handle.create_dataset("INCHIKEY", data=inchikey)
        handle.create_dataset("smiles", data=smiles)
    output = tmp_path / "morgan_direct"
    argv = [
        "encode_chemaware_shared_v3_morgan.py",
        "--graph", str(graph_path),
        "--data", str(data),
        "--output-dir", str(output),
        "--dimensions", "128",
        "--max-molecules", "999",
    ]
    with patch.object(sys, "argv", argv):
        encode_morgan()
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["formal"] is False
    assert report["identity_smiles_ledger"]["kind"] == (
        "direct_frozen_graph_hdf5_identity_smiles"
    )
    assert report["provenance"]["source_report_sha256"] is None
    assert report["provenance"]["hdf5_sha256"] == sha256(data)
    assert (output / "connectivity_smiles.npy").is_file()
    store = MoleculeTeacherStore(output, graph_path, graph, require_formal=False)
    assert len(store.ik14) == len(np.unique(graph.molecule_ik14))
    preflight = tmp_path / "bad_preflight.json"
    preflight.write_text(json.dumps({
        "status": "chemaware_shared_v2_preflight_passed",
        "formal": True,
        "hashes": {"graph_sha256": "wrong", "hdf5_sha256": sha256(data)},
    }), encoding="utf-8")
    formal_argv = [
        "encode_chemaware_shared_v3_morgan.py",
        "--graph", str(graph_path),
        "--data", str(data),
        "--preflight", str(preflight),
        "--output-dir", str(tmp_path / "morgan_formal_rejected"),
        "--dimensions", "128",
    ]
    with patch.object(sys, "argv", formal_argv), pytest.raises(
        RuntimeError, match="differs from frozen preflight"
    ):
        encode_morgan()


def test_frozen_probe_reachability_uses_nested_formula_folds(tmp_path: Path):
    graph_path, cache, official = write_synthetic_graph_and_cache(tmp_path)
    teacher = write_synthetic_teacher(tmp_path, graph_path)
    output = tmp_path / "probe_reachability.json"
    argv = [
        "analyze_chemaware_shared_v3_frozen_probe_reachability.py",
        "--graph", str(graph_path),
        "--token-dir", str(cache),
        "--official-checkpoint", str(official),
        "--teacher-dir", str(teacher),
        "--output", str(output),
        "--alphas", "1.0",
        "--permutation-seeds", "17",
        "--bootstrap-resamples", "20",
    ]
    with patch.object(sys, "argv", argv):
        analyze_frozen_probe()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "chemaware_shared_v3_frozen_probe_reachability_complete"
    assert report["queries"] == CandidateGraph(graph_path).n_queries
    assert len(report["correct_teacher"]["folds"]) == 5
    assert len(report["identity_permuted_controls"]) == 1
    assert report["claim_limit"].startswith("Formula-held-out linear decodability")


def test_g2_summary_enforces_matched_controls_and_paired_direction(tmp_path: Path):
    graph_path, _, _ = write_synthetic_graph_and_cache(tmp_path)
    graph = CandidateGraph(graph_path)
    g1_root = tmp_path / "g1"
    g2_root = tmp_path / "g2"
    common = {
        "training_query_ledger_sha256": "train-ledger",
        "allowed_molecule_ledger_sha256": "molecule-ledger",
        "initial_adapter_sha256": "initial-adapter",
        "training_contract": {"epochs": 5, "batch_queries": 8},
        "train_queries": 6,
        "train_identities": 6,
    }
    for seed in (17, 41, 73):
        for fold in range(5):
            query = np.flatnonzero(np.arange(graph.n_queries) % 5 == fold)
            old = np.full(len(query), 2, dtype=np.int32)
            clean = np.full(len(query), 2, dtype=np.int32)
            g1_run = g1_root / f"seed_{seed}" / f"fold_{fold}"
            g1_run.mkdir(parents=True)
            np.savez_compressed(g1_run / "outer_predictions.npz", query=query, old_rank=old, new_rank=clean)
            (g1_run / "decision.json").write_text(json.dumps({
                "status": "chemaware_shared_v2_g1_fold_complete",
                "formal": True, "chemical_supervision": False,
                "seed": seed, "outer_fold": fold, **common,
            }), encoding="utf-8")
            for arm in ARMS:
                run = g2_root / arm / f"seed_{seed}" / f"fold_{fold}"
                run.mkdir(parents=True)
                improved = arm in {"correct", "correct_same_formula_scope"}
                final = np.ones(len(query), dtype=np.int32) if improved else clean.copy()
                np.savez_compressed(
                    run / "outer_predictions.npz", query=query, old_rank=old, new_rank=final
                )
                (run / "decision.json").write_text(json.dumps({
                    "status": "chemaware_shared_v2_g2_fold_complete",
                    "formal": True, "chemical_supervision": True,
                    "teacher_control": arm, "seed": seed, "outer_fold": fold,
                    "chemical_objective": "frozen_teacher_candidate_hardness_reweighting",
                    "training_only_projector_used": False,
                    "chemical_contract": {
                        "lambda_molecule": 0.25,
                        "chemical_hardness_beta": 4.0,
                    },
                    "teacher_control_audit": {"chemical_effect_queries": 6},
                    "history": [
                        {"epoch": 0},
                        {"epoch": 1, "gradient_audit": {
                            "chemical_minus_clean_gradient_norm": 0.1,
                            "chemical_delta_nonzero_parameter_tensors": 2,
                        }},
                    ],
                    **common,
                }), encoding="utf-8")
    output = tmp_path / "g2_summary.json"
    argv = [
        "summarize_chemaware_shared_v2_g2.py",
        "--g1-root", str(g1_root), "--g2-root", str(g2_root),
        "--graph", str(graph_path), "--output", str(output),
        "--bootstrap-resamples", "50",
    ]
    with patch.object(sys, "argv", argv):
        summarize_g2()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "chemaware_shared_v2_g2_multifold_summary_complete"
    assert report["pass_to_G3"] is True
    assert all(report["gates"].values())


def test_v3_g2_summary_enforces_matched_peft_and_chemical_gradient(tmp_path: Path):
    graph_path, _, _ = write_synthetic_graph_and_cache(tmp_path)
    graph = CandidateGraph(graph_path)
    g1_root = tmp_path / "v3_g1"
    g2_root = tmp_path / "v3_g2"
    common = {
        "training_query_ledger_sha256": "train-ledger",
        "allowed_molecule_ledger_sha256": "molecule-ledger",
        "initial_peft_state_sha256": "initial-peft",
        "capacity": {"config": {"rank": 8, "last_blocks": 1}},
        "training_contract": {"epochs": 5, "batch_queries": 4},
        "train_queries": 6,
        "train_identities": 6,
        "query_reference_encoder_shared": True,
        "candidate_inputs_at_inference": False,
        "P2b_used": False,
        "first_step_audit": {
            "gradient_l2": 0.1,
            "parameter_update_l2": 0.01,
            "changed_parameter_tensors": 4,
        },
        "preflight": {"complete_split_eligible_candidate_groups": True},
    }
    for seed in (17, 41, 73):
        for fold in range(5):
            query = np.flatnonzero(np.arange(graph.n_queries) % 5 == fold)
            old = np.full(len(query), 2, dtype=np.int32)
            clean = old.copy()
            g1_run = g1_root / f"seed_{seed}" / f"fold_{fold}"
            g1_run.mkdir(parents=True)
            np.savez_compressed(
                g1_run / "outer_predictions.npz",
                query=query, old_rank=old, new_rank=clean,
            )
            (g1_run / "peft.pt").write_bytes(b"g1 peft")
            (g1_run / "decision.json").write_text(json.dumps({
                "status": "chemaware_shared_v3_g1_peft_fold_complete",
                "formal": True, "chemical_supervision": False,
                "seed": seed, "outer_fold": fold, **common,
            }), encoding="utf-8")
            for arm in ARMS:
                run = g2_root / arm / f"seed_{seed}" / f"fold_{fold}"
                run.mkdir(parents=True)
                improved = arm in {"correct", "correct_same_formula_scope"}
                final = np.ones(len(query), dtype=np.int32) if improved else clean.copy()
                np.savez_compressed(
                    run / "outer_predictions.npz",
                    query=query, old_rank=old, new_rank=final,
                )
                (run / "peft.pt").write_bytes(b"g2 peft")
                (run / "decision.json").write_text(json.dumps({
                    "status": "chemaware_shared_v3_g2_peft_fold_complete",
                    "formal": True, "chemical_supervision": True,
                    "teacher_control": arm,
                    "seed": seed, "outer_fold": fold,
                    "chemical_objective": "frozen_morgan_binary_connectivity_candidate_hardness_absolute_bounded",
                    "teacher_kind": "morgan_binary_connectivity",
                    "training_only_projector_used": False,
                    "chemical_contract": {
                        "lambda_molecule": 0.25,
                        "chemical_weighting": "absolute_bounded",
                    },
                    "teacher_control_audit": {
                        "chemical_effect_queries": 6,
                        "teacher_observable_mask_sha256": (
                            "scoped-mask"
                            if arm in {"correct_same_formula_scope", "same_formula_mismatched"}
                            else "global-mask"
                        ),
                    },
                    "history": [
                        {"epoch": 0},
                        {"epoch": 1, "chemical_gradient_audit": {
                            "chemical_minus_clean_gradient_norm": 0.1,
                            "chemical_delta_nonzero_parameter_tensors": 4,
                            "chemical_delta_gradient_signature": (
                                [1.0] + [0.0] * 127
                                if arm in {"correct", "correct_same_formula_scope"}
                                else [0.0, 1.0] + [0.0] * 126
                            ),
                        }},
                    ],
                    **common,
                }), encoding="utf-8")
    output = tmp_path / "v3_g2_summary.json"
    argv = [
        "summarize_chemaware_shared_v3_g2.py",
        "--g1-root", str(g1_root), "--g2-root", str(g2_root),
        "--graph", str(graph_path), "--output", str(output),
        "--bootstrap-resamples", "50",
    ]
    with patch.object(sys, "argv", argv):
        summarize_v3_g2()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "chemaware_shared_v3_g2_peft_multifold_summary_complete"
    assert report["pass_to_G3"] is True
    assert all(report["gates"].values())

    # The targeted frozen-probe branch must pass the same multifold attribution
    # audit and additionally pin identical query-selection ledgers per control
    # scope.  Reuse the paired fixture so only the objective contract changes.
    for seed in (17, 41, 73):
        for fold in range(5):
            for arm in ARMS:
                decision_path = (
                    g2_root / arm / f"seed_{seed}" / f"fold_{fold}"
                    / "decision.json"
                )
                decision = json.loads(decision_path.read_text(encoding="utf-8"))
                scoped = arm in {
                    "correct_same_formula_scope", "same_formula_mismatched"
                }
                decision.update({
                    "chemical_objective": "frozen_morgan_ridge_probe_targeted_listwise",
                    "training_only_frozen_probe_used": True,
                    "chemical_gradient_absorber_trainable": False,
                    "frozen_probe_fit_audit": {
                        "trainable_parameters": 0,
                        "discarded_at_inference": True,
                    },
                    "chemical_contract": {
                        "lambda_probe": 0.05,
                        "probe_margin_threshold": 0.01,
                    },
                })
                decision["teacher_control_audit"][
                    "selection_query_ledger_sha256"
                ] = "scoped-selection" if scoped else "global-selection"
                decision_path.write_text(json.dumps(decision), encoding="utf-8")
    g2b_output = tmp_path / "v3_g2b_summary.json"
    g2b_argv = [
        "summarize_chemaware_shared_v3_g2.py",
        "--g1-root", str(g1_root), "--g2-root", str(g2_root),
        "--graph", str(graph_path), "--output", str(g2b_output),
        "--chemical-objective", "frozen_probe_targeted",
        "--bootstrap-resamples", "50",
    ]
    with patch.object(sys, "argv", g2b_argv):
        summarize_v3_g2()
    g2b_report = json.loads(g2b_output.read_text(encoding="utf-8"))
    assert g2b_report["chemical_objective"] == "frozen_probe_targeted"
    assert g2b_report["pass_to_G3"] is True

    selection_path = (
        g2_root / "identity_permuted" / "seed_17" / "fold_0" / "decision.json"
    )
    selection_decision = json.loads(selection_path.read_text(encoding="utf-8"))
    selection_decision["teacher_control_audit"][
        "selection_query_ledger_sha256"
    ] = "leaked-selection"
    selection_path.write_text(json.dumps(selection_decision), encoding="utf-8")
    g2b_argv[g2b_argv.index(str(g2b_output))] = str(
        tmp_path / "v3_g2b_summary_rejected.json"
    )
    with patch.object(sys, "argv", g2b_argv), pytest.raises(
        RuntimeError, match="different query selection"
    ):
        summarize_v3_g2()
    selection_decision["teacher_control_audit"][
        "selection_query_ledger_sha256"
    ] = "global-selection"
    selection_path.write_text(json.dumps(selection_decision), encoding="utf-8")

    drift_path = g1_root / "seed_41" / "fold_0" / "decision.json"
    drift = json.loads(drift_path.read_text(encoding="utf-8"))
    drift["capacity"] = {"config": {"rank": 4, "last_blocks": 1}}
    drift_path.write_text(json.dumps(drift), encoding="utf-8")
    g2b_argv[g2b_argv.index(str(tmp_path / "v3_g2b_summary_rejected.json"))] = str(
        tmp_path / "v3_g2b_capacity_rejected.json"
    )
    with patch.object(sys, "argv", g2b_argv), pytest.raises(
        RuntimeError, match="changes capacity or training contract across folds"
    ):
        summarize_v3_g2()


def test_v3_g2_pilot_gate_requires_matched_masks_and_positive_attribution(tmp_path: Path):
    graph_path, _, _ = write_synthetic_graph_and_cache(tmp_path)
    graph = CandidateGraph(graph_path)
    seed, fold = 17, 0
    query = np.flatnonzero(formula_folds(graph.query_formula, 5, 1701) == fold)
    assert len(query) > 0
    old = np.full(len(query), 2, dtype=np.int32)
    common = {
        "training_query_ledger_sha256": "train-ledger",
        "allowed_molecule_ledger_sha256": "molecule-ledger",
        "initial_peft_state_sha256": "initial-peft",
        "capacity": {"config": {"rank": 8, "last_blocks": 1}},
        "training_contract": {"epochs": 5, "batch_queries": 4},
        "train_queries": 6,
        "train_identities": 6,
        "query_reference_encoder_shared": True,
        "candidate_inputs_at_inference": False,
        "P2b_used": False,
        "first_step_audit": {
            "gradient_l2": 0.1,
            "parameter_update_l2": 0.01,
            "changed_parameter_tensors": 4,
        },
        "preflight": {"complete_split_eligible_candidate_groups": True},
    }
    g1_root = tmp_path / "pilot_g1"
    g1_run = g1_root / f"seed_{seed}" / f"fold_{fold}"
    g1_run.mkdir(parents=True)
    np.savez_compressed(
        g1_run / "outer_predictions.npz", query=query, old_rank=old, new_rank=old,
    )
    (g1_run / "peft.pt").write_bytes(b"g1")
    (g1_run / "decision.json").write_text(json.dumps({
        "status": "chemaware_shared_v3_g1_peft_fold_complete",
        "formal": True,
        "chemical_supervision": False,
        "seed": seed,
        "outer_fold": fold,
        **common,
    }), encoding="utf-8")
    g1_summary = tmp_path / "g1_summary.json"
    g1_summary.write_text(json.dumps({
        "status": "chemaware_shared_v3_g1_peft_multifold_summary_complete",
        "formal": True,
        "matched_capacity_control_ready_for_G2": True,
        "provenance": {"graph_sha256": sha256(graph_path)},
    }), encoding="utf-8")

    g2_root = tmp_path / "pilot_g2"
    signature_index = {
        "correct": 0,
        "identity_permuted": 1,
        "random_marginal": 2,
        "correct_same_formula_scope": 3,
        "same_formula_mismatched": 4,
    }
    for arm in ARMS:
        run = g2_root / arm / f"seed_{seed}" / f"fold_{fold}"
        run.mkdir(parents=True)
        improved = arm in {"correct", "correct_same_formula_scope"}
        np.savez_compressed(
            run / "outer_predictions.npz",
            query=query,
            old_rank=old,
            new_rank=np.ones(len(query), dtype=np.int32) if improved else old,
        )
        (run / "peft.pt").write_bytes(arm.encode("utf-8"))
        signature = [0.0] * 128
        signature[signature_index[arm]] = 1.0
        (run / "decision.json").write_text(json.dumps({
            "status": "chemaware_shared_v3_g2_peft_fold_complete",
            "formal": True,
            "chemical_supervision": True,
            "teacher_control": arm,
            "seed": seed,
            "outer_fold": fold,
            "chemical_objective": (
                "frozen_morgan_binary_connectivity_candidate_hardness_absolute_bounded"
            ),
            "teacher_kind": "morgan_binary_connectivity",
            "training_only_projector_used": False,
            "chemical_contract": {
                "lambda_molecule": 0.25,
                "chemical_weighting": "absolute_bounded",
            },
            "teacher_control_audit": {
                "chemical_effect_queries": 6,
                "teacher_observable_mask_sha256": (
                    "scoped-mask"
                    if arm in {"correct_same_formula_scope", "same_formula_mismatched"}
                    else "global-mask"
                ),
            },
            "history": [
                {"epoch": 0},
                {"epoch": 1, "chemical_gradient_audit": {
                    "chemical_minus_clean_gradient_norm": 0.1,
                    "chemical_delta_nonzero_parameter_tensors": 4,
                    "chemical_delta_gradient_signature": signature,
                }},
            ],
            **common,
        }), encoding="utf-8")

    output = tmp_path / "pilot_decision.json"
    argv = [
        "validate_chemaware_shared_v3_g2_pilot.py",
        "--g1-root", str(g1_root),
        "--g2-root", str(g2_root),
        "--g1-summary", str(g1_summary),
        "--graph", str(graph_path),
        "--output", str(output),
        "--bootstrap-resamples", "50",
    ]
    with patch.object(sys, "argv", argv):
        validate_v3_g2_pilot()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["pass_to_full_matrix"] is True
    assert all(report["gates"].values())
    with patch.object(sys, "argv", argv + ["--verify-only"]):
        validate_v3_g2_pilot()

    mismatch_path = (
        g2_root / "same_formula_mismatched" / f"seed_{seed}" / f"fold_{fold}"
        / "decision.json"
    )
    mismatch = json.loads(mismatch_path.read_text(encoding="utf-8"))
    mismatch["teacher_control_audit"]["teacher_observable_mask_sha256"] = "wrong-mask"
    mismatch_path.write_text(json.dumps(mismatch), encoding="utf-8")
    with patch.object(sys, "argv", argv + ["--verify-only"]), pytest.raises(
        RuntimeError, match="no longer pins same_formula_mismatched"
    ):
        validate_v3_g2_pilot()
    argv[argv.index(str(output))] = str(tmp_path / "pilot_decision_rejected.json")
    with patch.object(sys, "argv", argv), pytest.raises(
        RuntimeError, match="unmatched observable masks"
    ):
        validate_v3_g2_pilot()


def test_v3_g2b_pilot_requires_matched_targeted_query_selection(tmp_path: Path):
    graph_path, _, _ = write_synthetic_graph_and_cache(tmp_path)
    graph = CandidateGraph(graph_path)
    seed, fold = 17, 0
    query = np.flatnonzero(formula_folds(graph.query_formula, 5, 1701) == fold)
    old = np.full(len(query), 2, dtype=np.int32)
    common = {
        "training_query_ledger_sha256": "train-ledger",
        "allowed_molecule_ledger_sha256": "molecule-ledger",
        "initial_peft_state_sha256": "initial-peft",
        "capacity": {"config": {"rank": 8, "last_blocks": 1}},
        "training_contract": {"epochs": 5, "batch_queries": 4},
        "train_queries": 6,
        "train_identities": 6,
        "query_reference_encoder_shared": True,
        "candidate_inputs_at_inference": False,
        "P2b_used": False,
        "first_step_audit": {
            "gradient_l2": 0.1,
            "parameter_update_l2": 0.01,
            "changed_parameter_tensors": 4,
        },
        "preflight": {"complete_split_eligible_candidate_groups": True},
    }
    g1_root = tmp_path / "pilot_g1"
    g1_run = g1_root / f"seed_{seed}" / f"fold_{fold}"
    g1_run.mkdir(parents=True)
    np.savez_compressed(
        g1_run / "outer_predictions.npz", query=query, old_rank=old, new_rank=old,
    )
    (g1_run / "peft.pt").write_bytes(b"g1")
    (g1_run / "decision.json").write_text(json.dumps({
        "status": "chemaware_shared_v3_g1_peft_fold_complete",
        "formal": True,
        "chemical_supervision": False,
        "seed": seed,
        "outer_fold": fold,
        **common,
    }), encoding="utf-8")
    g1_summary = tmp_path / "g1_summary.json"
    g1_summary.write_text(json.dumps({
        "status": "chemaware_shared_v3_g1_peft_multifold_summary_complete",
        "formal": True,
        "matched_capacity_control_ready_for_G2": True,
        "provenance": {"graph_sha256": sha256(graph_path)},
    }), encoding="utf-8")

    g2_root = tmp_path / "pilot_g2b"
    signature_index = {arm: index for index, arm in enumerate(ARMS)}
    for arm in ARMS:
        run = g2_root / arm / f"seed_{seed}" / f"fold_{fold}"
        run.mkdir(parents=True)
        improved = arm in {"correct", "correct_same_formula_scope"}
        np.savez_compressed(
            run / "outer_predictions.npz",
            query=query,
            old_rank=old,
            new_rank=np.ones(len(query), dtype=np.int32) if improved else old,
        )
        (run / "peft.pt").write_bytes(arm.encode("utf-8"))
        signature = [0.0] * 128
        signature[signature_index[arm]] = 1.0
        scoped = arm in {"correct_same_formula_scope", "same_formula_mismatched"}
        (run / "decision.json").write_text(json.dumps({
            "status": "chemaware_shared_v3_g2_peft_fold_complete",
            "formal": True,
            "chemical_supervision": True,
            "teacher_control": arm,
            "seed": seed,
            "outer_fold": fold,
            "chemical_objective": "frozen_morgan_ridge_probe_targeted_listwise",
            "teacher_kind": "morgan_binary_connectivity",
            "training_only_projector_used": False,
            "training_only_frozen_probe_used": True,
            "chemical_gradient_absorber_trainable": False,
            "frozen_probe_fit_audit": {
                "trainable_parameters": 0,
                "discarded_at_inference": True,
            },
            "chemical_contract": {
                "lambda_probe": 0.05,
                "probe_margin_threshold": 0.01,
            },
            "teacher_control_audit": {
                "chemical_effect_queries": 6,
                "teacher_observable_mask_sha256": (
                    "scoped-mask" if scoped else "global-mask"
                ),
                "selection_query_ledger_sha256": (
                    "scoped-selection" if scoped else "global-selection"
                ),
            },
            "history": [
                {"epoch": 0},
                {"epoch": 1, "chemical_gradient_audit": {
                    "chemical_minus_clean_gradient_norm": 0.1,
                    "chemical_delta_nonzero_parameter_tensors": 4,
                    "chemical_delta_gradient_signature": signature,
                }},
            ],
            **common,
        }), encoding="utf-8")

    output = tmp_path / "pilot_g2b_decision.json"
    argv = [
        "validate_chemaware_shared_v3_g2_pilot.py",
        "--g1-root", str(g1_root),
        "--g2-root", str(g2_root),
        "--g1-summary", str(g1_summary),
        "--graph", str(graph_path),
        "--output", str(output),
        "--chemical-objective", "frozen_probe_targeted",
        "--bootstrap-resamples", "50",
    ]
    with patch.object(sys, "argv", argv):
        validate_v3_g2_pilot()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["chemical_objective"] == "frozen_probe_targeted"
    assert report["pass_to_full_matrix"] is True

    mismatch_path = (
        g2_root / "identity_permuted" / f"seed_{seed}" / f"fold_{fold}"
        / "decision.json"
    )
    mismatch = json.loads(mismatch_path.read_text(encoding="utf-8"))
    mismatch["teacher_control_audit"]["selection_query_ledger_sha256"] = "leaked"
    mismatch_path.write_text(json.dumps(mismatch), encoding="utf-8")
    rejected_output = tmp_path / "pilot_g2b_rejected.json"
    argv[argv.index(str(output))] = str(rejected_output)
    with patch.object(sys, "argv", argv), pytest.raises(
        RuntimeError, match="unmatched query selection"
    ):
        validate_v3_g2_pilot()
