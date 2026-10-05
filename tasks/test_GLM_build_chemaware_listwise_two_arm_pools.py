"""Contracts for GLM_build_chemaware_listwise_two_arm_pools.

Synthetic world (6 manifest queries, 21 reference edges, 26 embedding rows):
- q0/q1/q2/q3 form the frozen training panel; q4 is the role-2 selection
  panel; q5 is the role-3 confirmation panel; formulas are pairwise disjoint.
- One tier-C source ledger with two families (pi, nl) realises every verdict
  path: unanimous main (q0 c1, q3 c1), contested_dominant (q0 c2),
  dominant_opposition_only rejection (q3 c2), and a main relation dropped by
  the candidate cap (q2 c3).
- Frozen Phase-A geometry scores make every qualified relation hinge-active
  under the sealed margin-0.1 / 1-1 filter.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import GLM_build_chemaware_listwise_two_arm_pools as builder  # noqa: E402
from build_chemaware_dynamic_reference_native_triplets import (  # noqa: E402
    array_sha256, file_sha256,
)

BASIS_SEED = 11
DIMENSION = 8
COSINES = {
    0: {1: 0.90, 2: 0.80, 3: 0.85, 4: 0.70, 5: 0.85, 6: 0.50},
    7: {8: 0.90, 9: 0.88},
    10: {11: 0.90, 12: 0.89, 13: 0.88, 14: 0.82, 25: 0.50},
    15: {16: 0.90, 17: 0.85, 18: 0.83},
    19: {20: 0.90, 21: 0.86},
    22: {23: 0.90, 24: 0.86},
}
QUERY_ROWS = [0, 7, 10, 15, 19, 22]
GROUP_REF_ROWS = [
    [1, 2], [3, 4], [5, 6],          # q0: truth, n1, n2
    [8], [9],                          # q1
    [11], [12], [13], [14, 25],       # q2: truth, n1, n2, n3(ledger, capped)
    [16], [17], [18],                  # q3
    [20], [21],                        # q4 (selection)
    [23], [24],                        # q5 (confirmation)
]
QUERY_FORMULAS = ["F0", "F1", "F2", "F3", "F4", "F5"]
LEDGER_ROWS = [
    # (query, candidate, family, score)
    (0, 0, "pi", 0.50), (0, 0, "nl", 0.60),
    (0, 1, "pi", 0.30), (0, 1, "nl", 0.40),
    (0, 2, "pi", 0.55), (0, 2, "nl", 0.40),
    (2, 0, "pi", 0.50), (2, 3, "pi", 0.30),
    (3, 0, "pi", 0.50), (3, 1, "pi", 0.30), (3, 2, "pi", 0.60),
]
EXPECTED = {
    "qualified": 4, "admitted": 3, "uncontested": 3, "recovered": 0,
    "contested": 1,
}


def unit_vectors() -> np.ndarray:
    rng = np.random.default_rng(BASIS_SEED)
    matrix = rng.standard_normal((DIMENSION, DIMENSION))
    basis, _ = np.linalg.qr(matrix)
    return np.asarray(basis, dtype=np.float64)


def embedding_for(basis: np.ndarray, row: int, cosine: float) -> np.ndarray:
    perpendicular = basis[1 + (row % (DIMENSION - 1))]
    vector = cosine * basis[0] + float(np.sqrt(1.0 - cosine ** 2)) * perpendicular
    return vector / float(np.linalg.norm(vector))


def molecule_ik14() -> list[str]:
    names: list[str] = []
    molecule_counts = [3, 2, 4, 3, 2, 2]
    for query, count in enumerate(molecule_counts):
        for local in range(count):
            names.append(
                f"IKQ{query}T" if local == 0 else f"IKQ{query}N{local}"
            )
    return names


def build_world(tmp_path: Path) -> dict[str, Path]:
    root = tmp_path / "world"
    root.mkdir()
    basis = unit_vectors()

    rows_used = sorted(
        {row for anchor in COSINES for row in [anchor, *COSINES[anchor].keys()]}
    )
    rows_registry = np.asarray(rows_used, dtype=np.int64)
    embeddings = np.zeros((len(rows_registry), DIMENSION), dtype=np.float32)
    for anchor, scores in COSINES.items():
        anchor_index = int(np.flatnonzero(rows_registry == anchor)[0])
        embeddings[anchor_index] = embedding_for(basis, anchor, 1.0)
        for row, cosine in scores.items():
            index = int(np.flatnonzero(rows_registry == row)[0])
            embeddings[index] = embedding_for(basis, row, cosine)

    embedding_dir = root / "phasea_embeddings"
    embedding_dir.mkdir()
    np.save(embedding_dir / "rows.npy", rows_registry)
    np.save(embedding_dir / "embeddings_f32.npy", embeddings)
    checkpoint = root / "geometry.ckpt"
    checkpoint.write_bytes(b"glm-listwise-synthetic-geometry")

    # Manifest
    molecule_counts = [3, 2, 4, 3, 2, 2]
    query_ptr = [0]
    molecule_ptr = [0]
    pair_candidate_row: list[int] = []
    molecule_label: list[int] = []
    molecule_formula: list[str] = []
    molecule_query: list[int] = []
    names = molecule_ik14()
    molecule_offset = 0
    for query, count in enumerate(molecule_counts):
        for local in range(count):
            pair_candidate_row.extend(GROUP_REF_ROWS[molecule_offset])
            molecule_ptr.append(len(pair_candidate_row))
            molecule_label.append(int(local == 0))
            molecule_query.append(query)
            molecule_formula.append(QUERY_FORMULAS[query])
            molecule_offset += 1
        query_ptr.append(len(molecule_query))
    manifest_path = root / "manifest.npz"
    np.savez_compressed(
        manifest_path,
        query_row=np.asarray(QUERY_ROWS, dtype=np.int64),
        query_ik14=np.asarray(
            [f"IKQ{q}T" for q in range(6)], dtype="U14",
        ),
        query_formula=np.asarray(QUERY_FORMULAS, dtype=str),
        query_adduct=np.asarray(["[M+H]+"] * 6, dtype=str),
        query_simulation_challenge_membership=np.asarray(["none"] * 6, dtype=str),
        query_ptr=np.asarray(query_ptr, dtype=np.int64),
        molecule_ptr=np.asarray(molecule_ptr, dtype=np.int64),
        molecule_query=np.asarray(
            [q for q, refs in enumerate(GROUP_REF_ROWS) for _ in refs], dtype=np.int64,
        ),
        molecule_label=np.asarray(molecule_label, dtype=np.int8),
        molecule_ik14=np.asarray(names, dtype="U14"),
        molecule_formula=np.asarray(molecule_formula, dtype=str),
        pair_candidate_row=np.asarray(pair_candidate_row, dtype=np.int64),
    )

    # Panels
    def panel(name: str, queries: list[int]) -> Path:
        path = root / name
        np.savez_compressed(
            path,
            query=np.asarray(queries, dtype=np.int64),
            formula=np.asarray([QUERY_FORMULAS[q] for q in queries], dtype=str),
            identity=np.asarray([f"IKQ{q}T" for q in queries], dtype=str),
            baseline_rank=np.ones(len(queries), dtype=np.int32),
        )
        return path

    train_evidence = panel("train_evidence.npz", [0, 1, 2, 3])
    selection_evidence = panel("selection_evidence.npz", [4])
    confirmation_evidence = panel("confirmation_evidence.npz", [5])

    # Source ledger
    ledger_dir = root / "ledger"
    ledger_dir.mkdir()
    table = ledger_dir / "candidate_scores.tsv"
    header = (
        "manifest_query\tlocal_candidate\tsource_family\tik14\tformula\tscope\t"
        "source_score\tcontrols_available\tcontrol_shuf_score\tcontrol_perm_score\n"
    )
    lines = [header]
    molecule_base = {}
    cursor = 0
    for query, count in enumerate([3, 2, 4, 3, 2, 2]):
        molecule_base[query] = cursor
        cursor += count
    for query, candidate, family, score in LEDGER_ROWS:
        molecule = molecule_base[query] + candidate
        lines.append(
            f"{query}\t{candidate}\t{family}\t{names[molecule]}\t"
            f"{QUERY_FORMULAS[query]}\tall_candidates\t{score:.6f}\t1\t"
            "0.500000\t0.500000\n"
        )
    table.write_text("".join(lines), encoding="utf-8")
    ledger_report = {
        "status": "CHEMAWARE_CANDIDATE_SOURCE_LEDGER_COMPLETE",
        "truth_fields_exported": False,
        "candidate_scores_sha256": file_sha256(table),
        "source_families": {
            "pi": {
                "scope": "all_candidates", "larger_is_better": True,
                "specificity_gate_passed": True,
                "confidence_tier": "C_calibration_required",
                "matched_controls": ["shuf", "perm"],
            },
            "nl": {
                "scope": "all_candidates", "larger_is_better": True,
                "specificity_gate_passed": True,
                "confidence_tier": "C_calibration_required",
                "matched_controls": ["shuf", "perm"],
            },
        },
    }
    (ledger_dir / "report.json").write_text(
        json.dumps(ledger_report), encoding="utf-8",
    )

    # Diagnostic gate
    diagnostic = {
        "status": "GLM_CHEMAWARE_EVIDENCE_CLASS_DIAGNOSTIC_COMPLETE",
        "sanity_anchors": {"all_match": True},
        "decision_summary": {"decision_eligible": True},
        "error_boundary_counts": {
            "vetoed_by_subsignificant_oppose": 25, "contested_dominant": 15,
        },
        "provenance": {
            "manifest_sha256": file_sha256(manifest_path),
            "ledgers": [{
                "report_sha256": file_sha256(ledger_dir / "report.json"),
                "tsv_sha256": file_sha256(table),
            }],
        },
    }
    diagnostic_path = root / "diagnostic.json"
    diagnostic_path.write_text(json.dumps(diagnostic), encoding="utf-8")

    # Embedding report (provenance contract)
    embedding_report = {
        "status": "CHEMAWARE_CHECKPOINT_MANIFEST_EMBEDDINGS_COMPLETE",
        "checkpoint_sha256": file_sha256(checkpoint),
        "manifest_sha256": file_sha256(manifest_path),
        "rows_array_sha256": array_sha256(np.load(embedding_dir / "rows.npy", mmap_mode="r")),
        "embeddings_array_sha256": array_sha256(
            np.load(embedding_dir / "embeddings_f32.npy", mmap_mode="r")
        ),
        "formula_role_4_accessed": False,
    }
    (embedding_dir / "report.json").write_text(
        json.dumps(embedding_report), encoding="utf-8",
    )

    data_path = root / "data.hdf5"
    with h5py.File(data_path, "w") as handle:
        handle.create_dataset("INCHIKEY", data=np.asarray(
            [f"SYN{row:04d}" for row in range(30)], dtype="S10",
        ))

    return {
        "root": root,
        "manifest": manifest_path,
        "data": data_path,
        "train": train_evidence,
        "selection": selection_evidence,
        "confirmation": confirmation_evidence,
        "ledger": ledger_dir,
        "diagnostic": diagnostic_path,
        "checkpoint": checkpoint,
        "embedding_dir": embedding_dir,
    }


def base_args(world: dict[str, Path], output: Path) -> list[str]:
    return [
        "builder",
        "--manifest", str(world["manifest"]),
        "--data", str(world["data"]),
        "--train-evidence", str(world["train"]),
        "--selection-evidence", str(world["selection"]),
        "--confirmation-evidence", str(world["confirmation"]),
        "--source-ledger", str(world["ledger"]),
        "--embedding-rows", str(world["embedding_dir"] / "rows.npy"),
        "--phasea-embeddings", str(world["embedding_dir"] / "embeddings_f32.npy"),
        "--embedding-report", str(world["embedding_dir"] / "report.json"),
        "--geometry-checkpoint", str(world["checkpoint"]),
        "--evidence-class-diagnostic", str(world["diagnostic"]),
        "--delta-main", "0.05",
        "--delta-contested", "0.025",
        "--maximum-candidates-per-query", "3",
        "--maximum-reference-spectra-per-molecule", "8",
        "--expected-qualified-relations", str(EXPECTED["qualified"]),
        "--expected-admitted-relations", str(EXPECTED["admitted"]),
        "--expected-uncontested-relations", str(EXPECTED["uncontested"]),
        "--expected-recovered-relations", str(EXPECTED["recovered"]),
        "--expected-contested-relations", str(EXPECTED["contested"]),
        "--output", str(output),
    ]


def run_builder(world: dict[str, Path], output: Path) -> None:
    import sys as _sys
    original = _sys.argv
    _sys.argv = base_args(world, output)
    try:
        builder.main()
    finally:
        _sys.argv = original


def load_pool(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def test_full_two_arm_pool_contract(tmp_path: Path) -> None:
    world = build_world(tmp_path)
    output = tmp_path / "pools" / "groups.npz"
    run_builder(world, output)
    assert output.is_file()
    report = json.loads(
        output.with_suffix(".json").read_text(encoding="utf-8"),
    )
    assert report["status"] == "GLM_LISTWISE_TWO_ARM_POOL_BUILT"
    assert report["panels"] == {
        "train_panel_queries": 4,
        "selection_panel_queries": 1,
        "confirmation_panel_queries": 1,
    }
    assert report["sealed_v16_relation_recompute"][
        "qualified_candidate_relations"
    ] == EXPECTED["qualified"]
    assert report["sealed_v16_relation_recompute"][
        "relations_routed_to_contested_arm"
    ] == EXPECTED["contested"]

    pool = load_pool(output)
    queries = pool["query_row"]
    assert queries.tolist() == [0, 7, 10, 15]
    groups = np.diff(pool["group_ptr"])
    assert groups.tolist() == [3, 2, 3, 3]  # q2 capped from 4 molecules
    assert report["caps"]["groups_where_candidate_cap_applied"] == 1
    assert report["margins"]["relations_dropped_by_candidate_cap"] == {"main": 1, "contested": 0}

    labels = pool["molecule_label"]
    for left, right in zip(pool["group_ptr"][:-1], pool["group_ptr"][1:]):
        window = labels[int(left):int(right)]
        assert window[0] == 1 and int(window.sum()) == 1 and len(window) >= 2
    assert np.all(pool["arm1_margin"] == 0.0)
    arm2 = pool["arm2_margin"]
    kinds = pool["arm2_margin_kind"].astype(str)
    assert np.all(arm2[labels == 1] == 0.0)
    # Pool molecule order: q0 (truth, neg1, neg2), q1 (truth, neg),
    # q2 (truth, neg, neg), q3 (truth, neg1, neg2).
    # q0 neg1 -> main, q0 neg2 -> contested, q3 neg1 -> main.
    assert arm2[1] == pytest.approx(0.05) and kinds[1] == "main"
    assert arm2[2] == pytest.approx(0.025) and kinds[2] == "contested"
    assert arm2[3] == 0.0 and kinds[3] == "none"
    assert arm2[9] == pytest.approx(0.05) and kinds[9] == "main"
    assert int((arm2 > 0).sum()) == 3
    assert report["margins"]["arm2_molecules_with_main_margin"] == 2
    assert report["margins"]["arm2_molecules_with_contested_margin"] == 1

    # Reference ordering: highest frozen cosine first within each molecule.
    ref_ptr = pool["molecule_ref_ptr"]
    refs = pool["ref_row"]
    assert refs[int(ref_ptr[0]):int(ref_ptr[1])].tolist() == [1, 2]
    assert refs[int(ref_ptr[1]):int(ref_ptr[2])].tolist() == [3, 4]
    assert refs[int(ref_ptr[2]):int(ref_ptr[3])].tolist() == [5, 6]

    # Query rows never appear inside their own group references.
    for index, query_row in enumerate(queries):
        left, right = map(int, pool["group_ptr"][index:index + 2])
        rows = refs[int(ref_ptr[left]):int(ref_ptr[right])]
        assert query_row not in set(int(r) for r in rows)

    expected_val = np.asarray(
        [builder.is_validation_query(int(row)) for row in queries], dtype=bool,
    )
    assert np.array_equal(pool["val_query_mask"], expected_val)
    assert report["counts"]["training_queries"] == 4
    assert report["margins"]["arm1_margin_is_identically_zero"] is True
    assert "shared_arrays_sha256" in report


def test_anchor_drift_fails_closed(tmp_path: Path) -> None:
    world = build_world(tmp_path)
    output = tmp_path / "pools" / "groups.npz"
    args = base_args(world, output)
    args[args.index("--expected-admitted-relations") + 1] = "99"
    original = sys.argv
    sys.argv = args
    try:
        with pytest.raises(RuntimeError, match="anchor drifted"):
            builder.main()
    finally:
        sys.argv = original
    assert not output.exists()


def test_panel_leakage_fails_closed(tmp_path: Path) -> None:
    world = build_world(tmp_path)
    np.savez_compressed(
        world["train"],
        query=np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
        formula=np.asarray([QUERY_FORMULAS[q] for q in [0, 1, 2, 3, 4]], dtype=str),
        identity=np.asarray([f"IKQ{q}T" for q in [0, 1, 2, 3, 4]], dtype=str),
        baseline_rank=np.ones(5, dtype=np.int32),
    )
    output = tmp_path / "pools" / "groups.npz"
    original = sys.argv
    sys.argv = base_args(world, output)
    try:
        with pytest.raises(RuntimeError, match="intersects the role-2"):
            builder.main()
    finally:
        sys.argv = original
    assert not output.exists()


def test_determinism_and_shared_identity(tmp_path: Path) -> None:
    world = build_world(tmp_path)
    first = tmp_path / "pools_a" / "groups.npz"
    second = tmp_path / "pools_b" / "groups.npz"
    run_builder(world, first)
    run_builder(world, second)
    pool_a = load_pool(first)
    pool_b = load_pool(second)
    assert set(pool_a) == set(pool_b)
    for key in sorted(pool_a):
        assert np.array_equal(pool_a[key], pool_b[key]), key
    report_a = json.loads(first.with_suffix(".json").read_text(encoding="utf-8"))
    report_b = json.loads(second.with_suffix(".json").read_text(encoding="utf-8"))
    assert report_a["shared_arrays_sha256"] == report_b["shared_arrays_sha256"]


def test_refuses_overwrite(tmp_path: Path) -> None:
    world = build_world(tmp_path)
    output = tmp_path / "pools" / "groups.npz"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        run_builder(world, output)


def test_validation_split_is_deterministic_and_bounded() -> None:
    rows = list(range(200))
    flags = [builder.is_validation_query(row) for row in rows]
    assert flags == [builder.is_validation_query(row) for row in rows]
    fraction = sum(flags) / len(flags)
    assert 0.02 < fraction < 0.25
