#!/usr/bin/env python3
"""One CPU-only, self-contained release test for frozen CALMS v1."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from calms import (  # noqa: E402
    CALMS,
    CORE_SHA256,
    ENCODER_NAMES,
    EXPECTED_ENCODER_CHECKPOINT_SHA256,
    EXPECTED_UPSTREAM_SHA256,
    METHOD_NAME,
    REPRESENTATION_SCHEMA,
    REQUIRED_SCORE_NAMES,
    V2_UNAVAILABLE_CONTRACT,
    _load_representation,
    sha256_file,
)
from unified_additive_release import CHEMICAL_KEYS, validate_axis  # noqa: E402


EXPECTED_PARENT_SHA = "7f1309e2a89ef3640d9b6871a821df6866f2528e629547075d5305ab6707d717"
EXPECTED_READOUT_SHA = "6603013a04fdc26946445a6ba1ed566d768bac49423046ce3922daac79fe637e"
EXPECTED_GOLDEN_SIGNATURE = "d4236ddb2da67902aae7668d017d0f04a10a167e63a410d2c78d9a414cea71df"


def expect_failure(function, message: str) -> None:  # type: ignore[no-untyped-def]
    try:
        function()
    except (RuntimeError, ValueError):
        return
    raise AssertionError(message)


def rounded_signature(values: np.ndarray) -> str:
    stable = np.round(np.asarray(values, dtype=np.float64), 6).astype("<f8")
    return hashlib.sha256(stable.tobytes()).hexdigest()


def validate_package_checksums() -> int:
    checksum_path = ROOT / "checksums.sha256"
    registered: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        registered[relative] = digest
    actual = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file()
        and path.name != "checksums.sha256"
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    }
    if set(registered) != actual:
        raise AssertionError("release checksum inventory is incomplete or contains stale files")
    for relative, expected in registered.items():
        if sha256_file(ROOT / relative) != expected:
            raise AssertionError(f"release file hash drifted: {relative}")
    return len(registered)


def write_fixture(folder: Path) -> dict:
    query_ptr = np.asarray([0, 2, 4, 4], dtype=np.int64)
    query_ids = np.asarray(["q0", "q1", "q2"])
    candidate_query_ids = np.asarray(["q0", "q0", "q1", "q1"])
    candidate_ids = np.asarray(["c0", "c1", "c2", "c3"])
    pair_ids = np.asarray(["q0:c0", "q0:c1", "q1:c2", "q1:c3"])
    evidence_arrays: dict[str, np.ndarray] = {
        "query_ptr": query_ptr,
        "query_ids": query_ids,
        "candidate_query_ids": candidate_query_ids,
        "candidate_ids": candidate_ids,
        "candidate_pair_ids": pair_ids,
    }
    for index, name in enumerate(REQUIRED_SCORE_NAMES):
        evidence_arrays[name] = np.asarray(
            [0.83, 0.21, 0.68, 0.31], dtype=np.float32
        ) + np.float32(index * 0.013)
    evidence_path = folder / "candidate_evidence.npz"
    np.savez_compressed(evidence_path, **evidence_arrays)
    upstream = {}
    for index, name in enumerate(("candidate_axis", *REQUIRED_SCORE_NAMES)):
        upstream[name] = {
            "sha256": EXPECTED_UPSTREAM_SHA256.get(name, f"{index + 1:064x}")
        }
    upstream["chemaware_v2_continuous_null"] = {
        "mode": "unavailable",
        "sha256": CORE_SHA256,
        "contract": V2_UNAVAILABLE_CONTRACT,
    }
    evidence_manifest = {
        "schema": "unified_additive_candidate_evidence_v1",
        "evidence_sha256": sha256_file(evidence_path),
        "required_score_names": list(REQUIRED_SCORE_NAMES),
        "upstream_artifacts": upstream,
        "claim_boundary": "synthetic release contract only",
    }
    evidence_manifest_path = folder / "candidate_evidence.json"
    evidence_manifest_path.write_text(
        json.dumps(evidence_manifest, indent=2) + "\n", encoding="utf-8"
    )

    rng = np.random.default_rng(20261103)
    bank_paths = {}
    bank_hashes = {}
    banks = {}
    for encoder_index, name in enumerate(ENCODER_NAMES):
        values = rng.normal(
            loc=encoder_index * 0.01, size=(12, 1024)
        ).astype(np.float32)
        path = folder / f"{name}.npz"
        np.savez_compressed(
            path,
            rows=np.arange(len(values), dtype=np.int64),
            embeddings=values,
        )
        bank_paths[name] = path
        bank_hashes[name] = sha256_file(path)
        banks[name] = values

    query_positions = np.asarray(
        [[0, 0, 0, 0], [0, 0, 0, 0], [-1, -1, -1, -1], [-1, -1, -1, -1]],
        dtype=np.int64,
    )
    reference_positions = np.empty((4, len(ENCODER_NAMES), 3), dtype=np.int64)
    for row in range(4):
        reference_positions[row] = np.asarray(
            [[1 + row, 5 + row, 8 + row]] * len(ENCODER_NAMES), dtype=np.int64
        )
    reference_mask = np.ones_like(reference_positions, dtype=bool)
    representation_path = folder / "candidate_references.npz"
    np.savez_compressed(
        representation_path,
        candidate_pair_ids=pair_ids,
        query_positions=query_positions,
        reference_positions=reference_positions,
        reference_mask=reference_mask,
    )
    representation_manifest = {
        "schema": REPRESENTATION_SCHEMA,
        "representation_sha256": sha256_file(representation_path),
        "candidate_evidence_sha256": sha256_file(evidence_path),
        "embedding_banks": {
            name: {
                "sha256": bank_hashes[name],
                "encoder_checkpoint_sha256": EXPECTED_ENCODER_CHECKPOINT_SHA256[name],
            }
            for name in ENCODER_NAMES
        },
        "claim_boundary": "synthetic release contract only",
    }
    representation_manifest_path = folder / "candidate_references.json"
    representation_manifest_path.write_text(
        json.dumps(representation_manifest, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "axis": validate_axis(
            query_ptr, query_ids, candidate_query_ids, candidate_ids, pair_ids
        ),
        "evidence": {name: evidence_arrays[name] for name in REQUIRED_SCORE_NAMES},
        "evidence_path": evidence_path,
        "evidence_manifest_path": evidence_manifest_path,
        "representation_path": representation_path,
        "representation_manifest_path": representation_manifest_path,
        "pair_ids": pair_ids,
        "query_positions": query_positions,
        "reference_positions": reference_positions,
        "reference_mask": reference_mask,
        "banks": banks,
        "bank_paths": bank_paths,
        "bank_hashes": bank_hashes,
    }


def main() -> None:
    file_count = validate_package_checksums()
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["public_identity"]["method_name"] == METHOD_NAME == "CALMS"
    assert sha256_file(ROOT / "assets" / "calms_candidate_evidence.pt") == EXPECTED_PARENT_SHA
    assert sha256_file(ROOT / "assets" / "calms_embedding_corrector.pt") == EXPECTED_READOUT_SHA

    with tempfile.TemporaryDirectory(prefix="calms_release_") as raw:
        temp = Path(raw)
        fixture = write_fixture(temp)
        axis = fixture["axis"]
        evidence = fixture["evidence"]

        scorer_small = CALMS(device="cpu", batch_queries=1)
        scorer_large = CALMS(device="cpu", batch_queries=64)
        args = (
            axis,
            evidence,
            fixture["pair_ids"],
            fixture["query_positions"],
            fixture["reference_positions"],
            fixture["reference_mask"],
            fixture["banks"],
            fixture["bank_hashes"],
        )
        first, first_report = scorer_small.score(*args)
        repeated, _ = scorer_small.score(*args)
        large_batch, _ = scorer_large.score(*args)
        assert np.array_equal(first, repeated)
        assert np.allclose(first, large_batch, rtol=0.0, atol=1e-7)
        assert first_report["residual_scored_queries"] == 1
        assert first_report["parent_fallback_nonempty_queries"] == 1

        parent, _ = scorer_small.parent.score(axis, evidence)
        assert np.array_equal(first[2:], parent[2:])
        chemical_unavailable = {
            "chem_valid": np.zeros(axis.candidates, dtype=bool),
            **{
                name: np.zeros(axis.candidates, dtype=np.float32)
                for name in CHEMICAL_KEYS[1:]
            },
        }
        explicit_unavailable, _ = scorer_small.score(*args, chemical_unavailable)
        assert np.array_equal(first, explicit_unavailable)

        signature = rounded_signature(first)
        if EXPECTED_GOLDEN_SIGNATURE == "TO_BE_FILLED":
            raise AssertionError(f"fill golden signature with {signature}")
        assert signature == EXPECTED_GOLDEN_SIGNATURE

        drifted_pairs = fixture["pair_ids"].copy()
        drifted_pairs[0] = "wrong"
        expect_failure(
            lambda: scorer_small.score(
                axis, evidence, drifted_pairs, *args[3:]
            ),
            "pair-axis drift was accepted",
        )
        changed_query = fixture["query_positions"].copy()
        changed_query[1, 0] = 1
        expect_failure(
            lambda: scorer_small.score(
                axis,
                evidence,
                fixture["pair_ids"],
                changed_query,
                fixture["reference_positions"],
                fixture["reference_mask"],
                fixture["banks"],
                fixture["bank_hashes"],
            ),
            "within-query representation drift was accepted",
        )
        malformed_banks = dict(fixture["banks"])
        malformed_banks[ENCODER_NAMES[0]] = np.zeros((12, 16), dtype=np.float32)
        expect_failure(
            lambda: scorer_small.score(
                axis,
                evidence,
                fixture["pair_ids"],
                fixture["query_positions"],
                fixture["reference_positions"],
                fixture["reference_mask"],
                malformed_banks,
                fixture["bank_hashes"],
            ),
            "wrong embedding dimension was accepted",
        )
        expect_failure(
            lambda: validate_axis(
                np.asarray([0, 2]),
                np.asarray(["q"]),
                np.asarray(["q", "q"]),
                np.asarray(["c", "c"]),
                np.asarray(["p0", "p1"]),
            ),
            "duplicate candidate identity was accepted",
        )

        forged_manifest = json.loads(
            fixture["representation_manifest_path"].read_text(encoding="utf-8")
        )
        forged_manifest["embedding_banks"][ENCODER_NAMES[0]]["sha256"] = "f" * 64
        forged_path = temp / "forged_representation.json"
        forged_path.write_text(json.dumps(forged_manifest), encoding="utf-8")
        expect_failure(
            lambda: _load_representation(
                fixture["representation_path"],
                forged_path,
                fixture["bank_paths"],
                sha256_file(fixture["evidence_path"]),
            ),
            "forged embedding-bank hash was accepted",
        )

        cli_output = temp / "cli_output"
        command = [
            sys.executable,
            str(ROOT / "calms.py"),
            "--evidence",
            str(fixture["evidence_path"]),
            "--evidence-manifest",
            str(fixture["evidence_manifest_path"]),
            "--representation",
            str(fixture["representation_path"]),
            "--representation-manifest",
            str(fixture["representation_manifest_path"]),
        ]
        for name in ENCODER_NAMES:
            command.extend(["--embedding-bank", f"{name}={fixture['bank_paths'][name]}"])
        command.extend(["--output", str(cli_output), "--device", "cpu"])
        completed = subprocess.run(
            command,
            cwd=temp,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "CALMS_INFERENCE_COMPLETE" in completed.stdout
        with np.load(cli_output / "candidate_scores.npz", allow_pickle=False) as body:
            assert np.allclose(body["calms_score"], first, rtol=0.0, atol=1e-7)
        with np.load(cli_output / "query_registry.npz", allow_pickle=False) as body:
            assert body["query_status"].tolist() == [
                "scored",
                "scored",
                "no_candidates",
            ]

    print(json.dumps({
        "status": "CALMS_FROZEN_RELEASE_PASS",
        "package_files": file_count,
        "parent_checkpoint_sha256": EXPECTED_PARENT_SHA,
        "embedding_corrector_sha256": EXPECTED_READOUT_SHA,
        "golden_score_sha256_rounded_6dp": EXPECTED_GOLDEN_SIGNATURE,
        "test_queries": 3,
        "test_candidates": 4,
    }, indent=2))


if __name__ == "__main__":
    main()
