#!/usr/bin/env python3
"""Contracts for the thin Full23 Additive release wrapper."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

from enveda_unified_candidate_integrator_core import predict_from_payload
from unified_additive_release import (
    CHECKPOINT_SHA256,
    CONTROL_ARM,
    CORE_SHA256,
    DEFAULT_CHECKPOINT,
    EXPECTED_FEATURE_NAMES,
    EXPECTED_UPSTREAM_SHA256,
    FrozenUnifiedAdditive,
    REQUIRED_SCORE_NAMES,
    V2_POLICY_SHA256,
    V2_UNAVAILABLE_CONTRACT,
    sha256_file,
    validate_axis,
)


def main() -> None:
    checkpoint = Path(DEFAULT_CHECKPOINT)
    assert checkpoint.is_file()
    assert sha256_file(checkpoint) == CHECKPOINT_SHA256
    ranker = FrozenUnifiedAdditive(checkpoint, "cpu")
    assert ranker.payload["control_arm"] == CONTROL_ARM
    assert tuple(ranker.payload["scalar_feature_names"]) == EXPECTED_FEATURE_NAMES

    axis = validate_axis(
        np.asarray([0, 0, 1, 4, 6], dtype=np.int64),
        np.asarray(["q0", "q1", "q2", "q3"]),
        np.asarray(["q1", "q2", "q2", "q2", "q3", "q3"]),
        np.asarray(["c1", "c2", "c3", "c4", "c5", "c6"]),
        np.asarray(["p1", "p2", "p3", "p4", "p5", "p6"]),
    )
    base = np.asarray([0.8, 0.9, 0.6, 0.2, 0.7, 0.3], dtype=np.float64)
    evidence = {
        "noise_v1": base,
        "weighted_spectral_entropy": base + np.asarray([0.01, -0.02, 0.03, 0.0, 0.02, -0.01]),
        "p2b_noise_v1_frozen": base + np.asarray([0.02, 0.01, -0.02, 0.01, 0.0, 0.02]),
        "noise_msg_fusion_stack_v1": base + np.asarray([0.00, -0.03, 0.02, 0.04, 0.01, -0.02]),
        "chemaware_stage1": base + np.asarray([0.01, 0.02, -0.01, 0.03, -0.01, 0.0]),
        "chemaware_phasea": base + np.asarray([-0.01, 0.01, 0.04, -0.02, 0.03, 0.0]),
        "noise_chemaware_fullmsg": base + np.asarray([0.03, -0.01, 0.01, 0.02, 0.0, -0.03]),
    }
    # One missing required candidate abstains the complete q3 block.
    evidence["chemaware_stage1"][-1] = np.nan
    chemical = {
        "chem_valid": np.asarray([False, True, True, False, True, True]),
        "candidate_utility_correct": np.asarray([0.0, 0.8, 0.2, 0.0, 0.4, 0.3]),
        "candidate_utility_zero": np.asarray([0.0, 0.1, 0.1, 0.0, 0.1, 0.1]),
        "candidate_utility_reversed": np.asarray([0.0, 0.2, 0.2, 0.0, 0.0, 0.2]),
        "candidate_utility_rotated": np.asarray([0.0, 0.3, 0.1, 0.0, 0.2, 0.0]),
    }
    real, report = ranker.score(axis, evidence, chemical)
    unavailable, unavailable_report = ranker.score(axis, evidence, None)
    assert np.all(np.isfinite(real[:4])) and np.all(np.isnan(real[4:]))
    assert np.all(np.isfinite(unavailable[:4])) and np.all(np.isnan(unavailable[4:]))
    assert report["zero_candidate_queries"] == 1
    assert report["singleton_queries"] == 1
    assert report["scored_queries"] == 2
    assert report["abstained_nonempty_queries"] == 1
    assert report["v2_available_candidates"] == 4
    assert unavailable_report["v2_available_candidates"] == 0
    assert not np.allclose(real[:4], unavailable[:4])

    # The wrapper must remain a transparent call into the frozen original core.
    expected_real = np.asarray(
        [10.286002159118652, 13.403972625732422, 8.073853492736816, 0.43832075595855713],
        dtype=np.float32,
    )
    assert np.allclose(real[:4], expected_real, rtol=0, atol=1e-7)

    swapped = list(EXPECTED_FEATURE_NAMES)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    try:
        predict_from_payload(
            ranker.payload,
            swapped,
            np.zeros((2, 23), dtype=np.float32),
            arm=CONTROL_ARM,
            device="cpu",
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("feature-order drift was not rejected")

    # Exercise the public CLI, provenance sidecar and output registries for
    # both legal V2 modes.  This is deliberately the same fixture used above,
    # so wrapper drift cannot hide behind a separate smoke-test construction.
    with tempfile.TemporaryDirectory(prefix="full23_release_") as raw:
        root = Path(raw)
        script = Path(__file__).with_name("unified_additive_release.py")

        def digest(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()

        def manifest_for(path: Path, mode: str) -> dict:
            upstream = {
                "candidate_axis": {"sha256": "1" * 64},
                "weighted_spectral_entropy": {"sha256": "2" * 64},
                "p2b_noise_v1_frozen": {"sha256": "3" * 64},
            }
            for name in (
                "noise_v1",
                "chemaware_stage1",
                "chemaware_phasea",
                "noise_chemaware_fullmsg",
                "noise_msg_fusion_stack_v1",
            ):
                upstream[name] = {"sha256": EXPECTED_UPSTREAM_SHA256[name]}
            upstream["chemaware_v2_continuous_null"] = (
                {
                    "mode": "real",
                    "sha256": "4" * 64,
                    "policy_sha256": V2_POLICY_SHA256,
                }
                if mode == "real"
                else {
                    "mode": "unavailable",
                    "sha256": CORE_SHA256,
                    "contract": V2_UNAVAILABLE_CONTRACT,
                }
            )
            return {
                "schema": "unified_additive_candidate_evidence_v1",
                "evidence_sha256": digest(path),
                "required_score_names": list(REQUIRED_SCORE_NAMES),
                "upstream_artifacts": upstream,
            }

        base_arrays = {
            "query_ptr": axis.query_ptr,
            "query_ids": axis.query_ids,
            "candidate_query_ids": axis.candidate_query_ids,
            "candidate_ids": axis.candidate_ids,
            "candidate_pair_ids": axis.candidate_pair_ids,
            **evidence,
        }
        for mode, expected in (("real", real), ("unavailable", unavailable)):
            evidence_path = root / f"evidence_{mode}.npz"
            arrays = dict(base_arrays)
            if mode == "real":
                arrays.update(chemical)
            np.savez_compressed(evidence_path, **arrays)
            manifest_path = root / f"evidence_{mode}.json"
            manifest_path.write_text(
                json.dumps(manifest_for(evidence_path, mode), indent=2) + "\n",
                encoding="utf-8",
            )
            output = root / f"output_{mode}"
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    str(script),
                    "--evidence",
                    str(evidence_path),
                    "--evidence-manifest",
                    str(manifest_path),
                    "--output",
                    str(output),
                ],
                check=True,
            )
            with np.load(output / "candidate_scores.npz", allow_pickle=False) as body:
                assert np.allclose(
                    body["unified_candidate_ranker_score"][:4],
                    expected[:4],
                    rtol=0,
                    atol=1e-7,
                )
                assert np.all(np.isnan(body["unified_candidate_ranker_score"][4:]))
            with np.load(output / "query_registry.npz", allow_pickle=False) as body:
                assert body["query_status"].tolist() == [
                    "no_candidates",
                    "singleton_hypothesis",
                    "scored",
                    "missing_required_evidence",
                ]

        tampered = manifest_for(root / "evidence_unavailable.npz", "unavailable")
        tampered["evidence_sha256"] = "0" * 64
        tampered_path = root / "tampered.json"
        tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
        failed = subprocess.run(
            [
                sys.executable,
                "-u",
                str(script),
                "--evidence",
                str(root / "evidence_unavailable.npz"),
                "--evidence-manifest",
                str(tampered_path),
                "--output",
                str(root / "tampered_output"),
            ],
            capture_output=True,
        )
        assert failed.returncode != 0

    print("PASS: unified Additive Full23 release contracts")
    print("fixture_real_scores", real[:4].tolist())


if __name__ == "__main__":
    main()
