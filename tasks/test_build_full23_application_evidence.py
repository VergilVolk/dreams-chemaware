#!/usr/bin/env python3
"""Synthetic contract test for the Full23 application evidence adapter."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from unified_additive_release import (
    CORE_SHA256,
    EXPECTED_UPSTREAM_SHA256,
    FrozenUnifiedAdditive,
    REQUIRED_SCORE_NAMES,
    V2_UNAVAILABLE_CONTRACT,
    validate_axis,
)


ENCODERS = (
    "noise_v1",
    "chemaware_stage1",
    "chemaware_phasea",
    "noise_chemaware_fullmsg",
)
PRODUCTION_QUERY_COUNT = 38_628
PRODUCTION_CANDIDATE_COUNT = 22_046


def digest(path: Path) -> str:
    body = hashlib.sha256()
    body.update(path.read_bytes())
    return body.hexdigest()


def write_fixture(root: Path) -> tuple[Path, list[str]]:
    scoring = root / "scoring"
    caches = root / "caches"
    scoring.mkdir()
    caches.mkdir()
    compression = {"method": "gzip", "mtime": 0}

    counts = np.zeros(PRODUCTION_QUERY_COUNT, dtype=np.int64)
    counts[1:4] = [4, 1, 4]
    counts[4:2_563] = 4
    counts[2_563:4_338] = 3
    counts[4_338:10_814] = 1
    assert int(counts.sum()) == PRODUCTION_CANDIDATE_COUNT
    assert (
        int(np.sum(counts == 0)), int(np.sum(counts == 1)), int(np.sum(counts > 1))
    ) == (27_815, 6_477, 4_336)
    query_id = np.asarray([f"q{index}" for index in range(PRODUCTION_QUERY_COUNT)])
    study_id = np.asarray([f"s{index}" for index in range(PRODUCTION_QUERY_COUNT)])
    query_block_id = np.asarray([
        f"{query}|{study}" for query, study in zip(query_id, study_id, strict=True)
    ])
    queries = pd.DataFrame({
        "query_id": query_id,
        "candidate_study_id": study_id,
        "query_block_id": query_block_id,
        "candidate_count": counts,
        "query_status": np.select(
            [counts == 0, counts == 1],
            ["no_mass_compatible_candidate", "single_mass_compatible_candidate"],
            default="multiple_mass_compatible_candidates",
        ),
    })
    candidate_query_blocks = np.repeat(query_block_id, counts)
    candidate_queries = np.repeat(query_id, counts)
    base = np.linspace(.1, .9, PRODUCTION_CANDIDATE_COUNT, dtype=np.float64)
    wse_available = np.ones(PRODUCTION_CANDIDATE_COUNT, dtype=bool)
    wse_available[8] = False
    pairs = pd.DataFrame({
        "candidate_pair_id": [f"p{index + 1}" for index in range(PRODUCTION_CANDIDATE_COUNT)],
        "query_block_id": candidate_query_blocks,
        "query_id": candidate_queries,
        "candidate_id": [f"c{index + 1}" for index in range(PRODUCTION_CANDIDATE_COUNT)],
        "WSE_score": base,
        # A finite source value with false availability must become NaN so the
        # downstream release abstains the whole four-candidate q3 block.
        "WSE_available": wse_available,
        "P2b_score": base + .01,
        "noise_v1_score": base + .02,
        "chemaware_stage1_score": base + .03,
        "chemaware_phasea_score": base + .04,
        "noise_chemaware_fullmsg_score": base + .05,
        "auxiliary__noise_msg_fusion_stack_v1_score": base + .06,
    })
    query_ptr = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(counts)))
    axis = validate_axis(
        query_ptr,
        query_block_id,
        candidate_query_blocks,
        pairs["candidate_id"].to_numpy(dtype="U"),
        pairs["candidate_pair_id"].to_numpy(dtype="U"),
    )
    fixture_evidence = {
        "noise_v1": pairs["noise_v1_score"].to_numpy(np.float64),
        "weighted_spectral_entropy": pairs["WSE_score"].to_numpy(np.float64),
        "p2b_noise_v1_frozen": pairs["P2b_score"].to_numpy(np.float64),
        "noise_msg_fusion_stack_v1": pairs[
            "auxiliary__noise_msg_fusion_stack_v1_score"
        ].to_numpy(np.float64),
        "chemaware_stage1": pairs["chemaware_stage1_score"].to_numpy(np.float64),
        "chemaware_phasea": pairs["chemaware_phasea_score"].to_numpy(np.float64),
        "noise_chemaware_fullmsg": pairs[
            "noise_chemaware_fullmsg_score"
        ].to_numpy(np.float64),
    }
    fixture_evidence["weighted_spectral_entropy"] = fixture_evidence[
        "weighted_spectral_entropy"
    ].copy()
    fixture_evidence["weighted_spectral_entropy"][~wse_available] = np.nan
    scorer_replay, _ = FrozenUnifiedAdditive(device="cpu").score(
        axis, fixture_evidence, None,
    )
    pairs["unified_candidate_ranker_score"] = scorer_replay
    candidate_path = scoring / "identity_candidate_scores.tsv.gz"
    query_path = scoring / "directional_query_registry.tsv.gz"
    pairs.to_csv(candidate_path, sep="\t", index=False, compression=compression)
    queries.to_csv(query_path, sep="\t", index=False, compression=compression)

    sidecar_args: list[str] = []
    embedding_inputs: dict[str, dict[str, str]] = {}
    for index, name in enumerate(ENCODERS):
        cache_sha = f"{index + 1:064x}"
        embedding_inputs[name] = {
            "path": str(caches / f"{name}.npz"),
            "sha256": cache_sha,
        }
        sidecar = {
            "status": "CROSSSTUDY_FROZEN_ENCODER_EMBEDDINGS_COMPLETE",
            "encoder_name": "chemaware_phase_a" if name == "chemaware_phasea" else name,
            "checkpoint_kind": "official_embedding_slim",
            "labels_or_responses_accessed": False,
            "embedding_dimension": 1024,
            "inputs": {
                "checkpoint": {
                    "path": f"frozen/{name}.pt",
                    "sha256": EXPECTED_UPSTREAM_SHA256[name],
                }
            },
            "output": {
                "path": str(caches / f"{name}.npz"),
                "sha256": cache_sha,
            },
        }
        sidecar_path = caches / f"{name}.npz.json"
        sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
        sidecar_args.extend(["--encoder-sidecar", f"{name}={sidecar_path}"])

    report = {
        "status": "CROSSSTUDY_IDENTITY_MULTIENCODER_COMPLETE",
        "labels_or_responses_accessed": False,
        "candidate_admission": (
            "frozen complete-identity handoff only; no application-side candidate generation"
        ),
        "methods": [*REQUIRED_SCORE_NAMES, "official_dreams", "unified_candidate_ranker"],
        "weighted_entropy_backend": "canonical_handoff_wse_score_no_recomputation",
        "parameters": {"fragment_da_for_p2b_only": 0.02},
        "denominators": {
            "directional_queries": PRODUCTION_QUERY_COUNT,
            "queries_with_identity_candidates": 10_813,
            "queries_without_identity_candidates": 27_815,
            "queries_with_one_identity_candidate": 6_477,
            "queries_with_multiple_identity_candidates": 4_336,
            "identity_candidate_pairs": PRODUCTION_CANDIDATE_COUNT,
        },
        "inputs": {
            "embeddings": embedding_inputs,
            "noise_msg_composite": {
                "family": "noise_msg_fusion_stack_v1",
                "model_path": "frozen/fusion_model.joblib",
                "model_sha256": EXPECTED_UPSTREAM_SHA256["noise_msg_fusion_stack_v1"],
                "report_path": "frozen/report.json",
                "report_sha256": "a" * 64,
            },
        },
        "outputs": {
            "candidate_scores": {"path": str(candidate_path), "sha256": digest(candidate_path)},
            "query_registry": {"path": str(query_path), "sha256": digest(query_path)},
        },
    }
    (scoring / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    return scoring, sidecar_args


def main() -> None:
    script = Path(__file__).with_name("build_full23_application_evidence.py")
    release = Path(__file__).with_name("unified_additive_release.py")
    verifier = Path(__file__).with_name("verify_full23_application_replay.py")
    with tempfile.TemporaryDirectory(prefix="full23_application_adapter_") as raw:
        root = Path(raw)
        scoring, sidecar_args = write_fixture(root)
        adapted = root / "adapted"
        subprocess.run(
            [
                sys.executable, "-u", str(script),
                "--scoring-dir", str(scoring),
                *sidecar_args,
                "--output-dir", str(adapted),
            ],
            check=True,
        )

        evidence_path = adapted / "candidate_evidence.npz"
        manifest_path = adapted / "candidate_evidence.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["schema"] == "unified_additive_candidate_evidence_v1"
        assert manifest["evidence_sha256"] == digest(evidence_path)
        assert tuple(manifest["required_score_names"]) == REQUIRED_SCORE_NAMES
        assert manifest["v2_mode"] == "unavailable"
        v2 = manifest["upstream_artifacts"]["chemaware_v2_continuous_null"]
        assert v2 == {
            "mode": "unavailable",
            "sha256": CORE_SHA256,
            "contract": V2_UNAVAILABLE_CONTRACT,
        }
        assert manifest["axis"] == {
            "queries": PRODUCTION_QUERY_COUNT,
            "candidates": PRODUCTION_CANDIDATE_COUNT,
            "zero_candidate_queries": 27_815,
            "singleton_queries": 6_477,
            "multiple_candidate_queries": 4_336,
            "semantic_sha256": manifest["axis"]["semantic_sha256"],
        }

        with np.load(evidence_path, allow_pickle=False) as body:
            assert set(body.files) == {
                "query_ptr", "query_ids", "candidate_query_ids", "candidate_ids",
                "candidate_pair_ids", *REQUIRED_SCORE_NAMES,
            }
            assert body["query_ptr"][:5].tolist() == [0, 0, 4, 5, 9]
            assert body["query_ptr"].shape == (PRODUCTION_QUERY_COUNT + 1,)
            assert body["query_ids"][:4].tolist() == ["q0|s0", "q1|s1", "q2|s2", "q3|s3"]
            assert body["query_ids"].shape == (PRODUCTION_QUERY_COUNT,)
            assert body["candidate_query_ids"][:5].tolist() == [
                "q1|s1", "q1|s1", "q1|s1", "q1|s1", "q2|s2",
            ]
            assert body["candidate_pair_ids"][:5].tolist() == ["p1", "p2", "p3", "p4", "p5"]
            assert body["candidate_pair_ids"].shape == (PRODUCTION_CANDIDATE_COUNT,)
            assert np.isnan(body["weighted_spectral_entropy"][8])
            assert not any(
                name in {
                    "chem_valid", "candidate_utility_correct", "candidate_utility_zero",
                    "candidate_utility_reversed", "candidate_utility_rotated",
                }
                for name in body.files
            )
            validate_axis(
                body["query_ptr"], body["query_ids"], body["candidate_query_ids"],
                body["candidate_ids"], body["candidate_pair_ids"],
            )

        # The real frozen release must accept the adapter output unchanged.
        release_output = root / "release_output"
        subprocess.run(
            [
                sys.executable, "-u", str(release),
                "--evidence", str(evidence_path),
                "--evidence-manifest", str(manifest_path),
                "--output", str(release_output),
                "--device", "cpu",
            ],
            check=True,
        )
        release_report = json.loads(
            (release_output / "report.json").read_text(encoding="utf-8")
        )
        assert release_report["zero_candidate_queries"] == 27_815
        assert release_report["singleton_queries"] == 6_477
        assert release_report["multiple_candidate_queries"] == 4_336
        assert release_report["scored_queries"] == 10_812
        assert release_report["scored_candidates"] == 22_042
        assert release_report["abstained_nonempty_queries"] == 1
        assert release_report["v2_mode"] == "unavailable"
        with np.load(release_output / "query_registry.npz", allow_pickle=False) as registry:
            assert registry["query_status"][:4].tolist() == [
                "no_candidates", "scored", "singleton_hypothesis",
                "missing_required_evidence",
            ]
        scorer_table = pd.read_csv(
            scoring / "identity_candidate_scores.tsv.gz",
            sep="\t", dtype={"candidate_pair_id": "string"},
        )
        with np.load(release_output / "candidate_scores.npz", allow_pickle=False) as released:
            assert released["candidate_pair_ids"].tolist() == scorer_table[
                "candidate_pair_id"
            ].astype(str).tolist()
            observed = released["unified_candidate_ranker_score"]
            expected = pd.to_numeric(
                scorer_table["unified_candidate_ranker_score"], errors="coerce",
            ).to_numpy(np.float64)
            assert np.array_equal(np.isnan(observed), np.isnan(expected))
            finite = np.isfinite(expected)
            maximum_replay_drift = float(np.max(np.abs(
                observed[finite].astype(np.float64) - expected[finite]
            )))
            assert maximum_replay_drift <= 2e-6, maximum_replay_drift
        replay_report = root / "portable_replay.json"
        subprocess.run(
            [
                sys.executable, "-u", str(verifier),
                "--scorer-tsv", str(scoring / "identity_candidate_scores.tsv.gz"),
                "--release-scores", str(release_output / "candidate_scores.npz"),
                "--output", str(replay_report),
            ],
            check=True,
        )
        replay = json.loads(replay_report.read_text(encoding="utf-8"))
        assert replay["status"] == "FULL23_APPLICATION_PORTABLE_REPLAY_PASS"
        assert replay["candidates"] == PRODUCTION_CANDIDATE_COUNT
        assert replay["maximum_absolute_score_drift"] <= 2e-6

        # A cache identity mismatch must fail before any output directory is made.
        broken_sidecar = root / "caches" / "noise_v1.npz.json"
        broken = json.loads(broken_sidecar.read_text(encoding="utf-8"))
        broken["output"]["sha256"] = "f" * 64
        broken_sidecar.write_text(json.dumps(broken), encoding="utf-8")
        failed_output = root / "failed"
        failed = subprocess.run(
            [
                sys.executable, "-u", str(script),
                "--scoring-dir", str(scoring),
                *sidecar_args,
                "--output-dir", str(failed_output),
            ],
            capture_output=True,
        )
        assert failed.returncode != 0
        assert not failed_output.exists()

    print("PASS: Full23 application evidence adapter contracts")


if __name__ == "__main__":
    main()
