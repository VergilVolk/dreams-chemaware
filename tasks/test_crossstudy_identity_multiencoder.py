#!/usr/bin/env python3
"""Small numerical contracts for complete identity-candidate scoring."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import joblib
import json
from pathlib import Path
import tempfile

from enveda_unified_candidate_integrator_core import (
    CHECKPOINT_SCHEMA as ENVEDA_INTEGRATOR_SCHEMA,
    PRIMARY_ARM as ENVEDA_PRIMARY_ARM,
    CandidateEvidenceIntegrator, IntegratorSettings,
)
from massspecgym_embedding_readout_core import (
    ARM_PROJECTED, ARM_SCALAR, REQUIRED_ENCODERS, FrozenEmbeddingReadout,
    ReadoutSettings,
)
from noise_msg_pair_evidence_core import MOLECULE_FEATURES
from score_crossstudy_identity_multiencoder import (
    MSG_FAMILY, apply_unified_model, build_enveda_application_features,
    frozen_p2b_block, load_auxiliary_candidate_scores, load_fixed_identity_axis,
    matched_peaks, minmax, peak_rows, prepare_spectrum,
    score_frozen_noise_msg, valid_precursor_mz,
)


class DummyProbabilityModel:
    """Pickleable two-class stand-in for the frozen MSG classifier."""

    classes_ = np.asarray([0, 1])

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        score = 1.0 / (1.0 + np.exp(-np.asarray(values)[:, 0]))
        return np.column_stack((1.0 - score, score))


def main() -> None:
    assert np.allclose(minmax(np.asarray([2.0, 4.0, 3.0])), [0.0, 1.0, 0.5])
    assert np.allclose(minmax(np.asarray([7.0, 7.0])), [0.0, 0.0])
    left = np.asarray([[100.000, 1.0], [150.000, 2.0], [200.000, 3.0]])
    right = np.asarray([[99.991, 1.0], [150.019, 2.0], [175.000, 3.0]])
    assert matched_peaks(left, right, 0.02) == 2
    empty_rows = peak_rows(np.asarray([]))
    assert empty_rows.shape == (0, 2)
    assert prepare_spectrum(empty_rows.T).shape == (2, 0)
    assert matched_peaks(empty_rows, right, 0.02) == 0
    assert valid_precursor_mz(100.0)
    assert not valid_precursor_mz(0.0)
    assert not valid_precursor_mz(float("nan"))
    p2b = frozen_p2b_block(
        np.asarray([0.7, 0.6]), np.asarray([0.2, 0.9]),
        np.asarray([0.3, 0.1]), np.asarray([0.1, 0.8]),
    )
    assert p2b.shape == (2,) and np.isfinite(p2b).all()

    # The fixed handoff includes a zero-candidate block, two candidate-study
    # blocks for the same query_id, and a complete singleton block.  One of the
    # two q_shared blocks is deliberately incomplete on one of two rows: the
    # entire local block must abstain without affecting the other study.
    pair_rows = pd.DataFrame({
        "candidate_pair_id": ["p1", "p2", "p3", "p5", "p4"],
        "query_id": ["q_shared", "q_shared", "q_shared", "q_shared", "q_single"],
        "candidate_id": ["c1", "c2", "c3", "c5", "c4"],
        "query_study_id": ["query_study"] * 5,
        "query_raw_entity_id": ["raw_q"] * 4 + ["raw_single"],
        "candidate_study_id": ["study_a", "study_a", "study_b", "study_b", "study_c"],
        "candidate_raw_entity_id": ["raw_c1", "raw_c2", "raw_c3", "raw_c5", "raw_c4"],
        "query_precursor_mz": [100.0] * 5,
        "candidate_precursor_mz": [100.0] * 5,
        "wse_score": [.8, .2, .5, .4, .6],
        "wse_matched_fragments": [8, 2, 5, 4, 6],
    })
    query_rows = pd.DataFrame({
        "query_id": ["q_zero", "q_shared", "q_shared", "q_single"],
        "query_study_id": ["query_study"] * 4,
        "query_raw_entity_id": ["raw_zero", "raw_q", "raw_q", "raw_single"],
        "candidate_study_id": ["study_zero", "study_a", "study_b", "study_c"],
        "candidate_count": [0, 2, 2, 1],
        "query_status": [
            "no_mass_compatible_candidate", "multiple_mass_compatible_candidates",
            "multiple_mass_compatible_candidates", "single_mass_compatible_candidate",
        ],
    })
    query_ptr = np.asarray([0, 0, 2, 4, 5], dtype=np.int64)
    query_ids = np.asarray([
        "q_zero|study_zero", "q_shared|study_a", "q_shared|study_b",
        "q_single|study_c",
    ])
    with tempfile.TemporaryDirectory() as raw:
        handoff = Path(raw) / "handoff"
        handoff.mkdir()
        compression = {"method": "gzip", "mtime": 0}
        # TSV order is deliberately non-canonical.  The NPZ order below is the
        # sole deployed order and the loader must restore it exactly.
        pair_rows.iloc[[2, 0, 4, 1, 3]].to_csv(
            handoff / "complete_identity_candidate_pairs.tsv.gz",
            sep="\t", index=False, compression=compression,
        )
        query_rows.to_csv(
            handoff / "directional_query_registry.tsv.gz",
            sep="\t", index=False, compression=compression,
        )
        np.savez_compressed(
            handoff / "wse_identity_inference_axis.npz",
            schema=np.asarray("msv000095331_complete_identity_candidate_axis_v1"),
            query_ptr=query_ptr, query_ids=query_ids,
            candidate_ids=pair_rows["candidate_id"].to_numpy(dtype="U"),
            candidate_pair_ids=pair_rows["candidate_pair_id"].to_numpy(dtype="U"),
            wse_score=pair_rows["wse_score"].to_numpy(np.float32),
            wse_availability=np.ones(len(pair_rows), dtype=bool),
        )
        pairs, queries, loaded_ptr, loaded_ids, _ = load_fixed_identity_axis(
            handoff, expected_queries=4, expected_candidates=5,
        )
        assert np.array_equal(loaded_ptr, query_ptr)
        assert np.array_equal(loaded_ids, query_ids)
        assert queries["candidate_cardinality"].tolist() == [
            "zero", "multiple", "multiple", "single",
        ]
        assert queries["query_block_id"].tolist() == query_ids.tolist()
        assert pairs["candidate_pair_id"].tolist() == pair_rows["candidate_pair_id"].tolist()
        assert pairs["query_block_id"].tolist() == [
            "q_shared|study_a", "q_shared|study_a", "q_shared|study_b",
            "q_shared|study_b",
            "q_single|study_c",
        ]
        assert np.allclose(pairs["WSE_score"], [.8, .2, .5, .4, .6])
        _, _, _, _, axis_report = load_fixed_identity_axis(handoff)
        assert (
            axis_report["zero_candidate_blocks"],
            axis_report["single_candidate_blocks"],
            axis_report["multiple_candidate_blocks"],
        ) == (1, 1, 2)

        # A convenience score file may cover a superset, but the out-of-axis
        # value is removed before any query-relative feature is constructed.
        auxiliary_path = Path(raw) / "auxiliary.tsv.gz"
        pd.DataFrame({
            "candidate_pair_id": ["p4", "outside", "p2", "p5", "p1", "p3"],
            "official_score": [.4, 999.0, .2, .35, .1, .3],
            "evidence_available": [True] * 6,
        }).to_csv(auxiliary_path, sep="\t", index=False, compression=compression)
        auxiliary, auxiliary_report = load_auxiliary_candidate_scores(
            [f"official_dreams={auxiliary_path}::official_score"],
            pairs["candidate_pair_id"].to_numpy(dtype="U"),
        )
        assert np.allclose(auxiliary["official_dreams"], [.1, .2, .3, .35, .4])
        assert auxiliary_report["official_dreams"]["outside_axis_rows_discarded"] == 1

    pairs["P2b_sqrt_cosine"] = [.72, .28, .52, .42, .62]
    pairs["P2b_entropy"] = [.68, .32, .48, .38, .58]
    pairs["P2b_neutral_loss"] = [.65, .35, .45, .35, .55]
    pairs["P2b_score"] = [.7, .3, .5, .4, .6]
    pairs["noise_v1_score"] = [.9, .1, .6, .5, .7]
    # One missing official score invalidates only q_shared|study_b for the MSG
    # composite. The singleton remains a legitimate scored candidate.
    pairs["official_dreams_score"] = [.82, .18, np.nan, .44, .61]
    pairs["chemaware_stage1_score"] = [.8, .2, .4, .3, .6]
    pairs["chemaware_phasea_score"] = [.75, .25, np.nan, .35, .55]
    pairs["noise_chemaware_fullmsg_score"] = [.85, .15, .55, .45, .65]
    with tempfile.TemporaryDirectory() as raw:
        msg_dir = Path(raw) / "fusion_model"
        msg_dir.mkdir()
        joblib.dump(
            {"model": DummyProbabilityModel(), "feature_names": list(MOLECULE_FEATURES)},
            msg_dir / "fusion_model.joblib",
        )
        (msg_dir / "report.json").write_text(
            json.dumps({"status": "NOISE_MSG_FUSION_FROZEN"}), encoding="utf-8",
        )
        msg_scores, msg_report = score_frozen_noise_msg(pairs, query_ptr, msg_dir)
        assert np.isfinite(msg_scores[[0, 1, 4]]).all()
        assert np.isnan(msg_scores[[2, 3]]).all()
        assert msg_report["scored_queries"] == 2
        assert msg_report["raw_channels_passed_to_unified_integrator"] is False

        # Build a schema-faithful Enveda scalar payload from exactly the same
        # fixed-axis evidence. V2 is represented by its registered exact no-op.
        selected = pairs.iloc[[0, 1, 4]].reset_index(drop=True)
        scalar, feature_names = build_enveda_application_features(
            selected, np.asarray([0, 2, 3]), msg_scores[[0, 1, 4]],
        )
        integrator_settings = IntegratorSettings(
            scalar_features=scalar.shape[1],
            anchor_index=feature_names.index("weighted_spectral_entropy"),
            hidden=4,
        )
        integrator = CandidateEvidenceIntegrator(
            ENVEDA_PRIMARY_ARM, integrator_settings,
        )
        enveda_payload = {
            "schema": ENVEDA_INTEGRATOR_SCHEMA,
            "selection_class": "candidate_evidence_integration",
            "primary_arm": ENVEDA_PRIMARY_ARM,
            "encoder_names": REQUIRED_ENCODERS,
            "scalar_feature_names": feature_names,
            "scalar_mean": np.zeros(scalar.shape[1], dtype=np.float32),
            "scalar_scale": np.ones(scalar.shape[1], dtype=np.float32),
            "anchor_index": integrator_settings.anchor_index,
            "hidden": integrator_settings.hidden,
            "arms": {ENVEDA_PRIMARY_ARM: integrator.state_dict()},
        }
        enveda_path = Path(raw) / "enveda_integrator.pt"
        torch.save(enveda_payload, enveda_path)
        enveda_scores, enveda_report = apply_unified_model(
            pairs, {}, enveda_path, torch.device("cpu"), query_ptr, query_ids,
            auxiliary_scores={MSG_FAMILY: msg_scores},
        )
        assert np.isfinite(enveda_scores[[0, 1, 4]]).all()
        assert np.isnan(enveda_scores[[2, 3]]).all()
        assert enveda_report["schema"] == ENVEDA_INTEGRATOR_SCHEMA
        assert enveda_report["scored_queries"] == 2
        assert "availability_0_continuous_0" in enveda_report["chemaware_v2"]
        assert enveda_report["bioaware"].startswith("not_supplied")
        try:
            apply_unified_model(
                pairs, {}, enveda_path, torch.device("cpu"), query_ptr, query_ids,
                auxiliary_scores={},
            )
        except RuntimeError as error:
            assert "exactly one frozen auxiliary family" in str(error)
        else:
            raise AssertionError("new integrator silently fabricated missing MSG")
    settings = ReadoutSettings(
        scalar_features=30, encoder_names=REQUIRED_ENCODERS,
        embedding_dimension=4, projection_dimension=2,
    )
    model = FrozenEmbeddingReadout(ARM_PROJECTED, settings)
    names = [
        "noise_v1", "noise_v1_percentile", "weighted_spectral_entropy",
        "weighted_spectral_entropy_percentile", "p2b_noise_v1_frozen",
        "p2b_noise_v1_frozen_percentile", "query_molecules", "query_edges",
        "log_query_edges", "noise_v1_top_gap", "weighted_spectral_entropy_top_gap",
        "p2b_noise_v1_frozen_top_gap", "noise_v1_weighted_spectral_entropy_top1_agree",
        "noise_v1_p2b_noise_v1_frozen_top1_agree",
        "weighted_spectral_entropy_p2b_noise_v1_frozen_top1_agree",
        "chemaware_stage1_cosine", "chemaware_stage1_percentile", "chemaware_stage1_top_gap",
        "chemaware_phasea_cosine", "chemaware_phasea_percentile", "chemaware_phasea_top_gap",
        "noise_chemaware_fullmsg_cosine", "noise_chemaware_fullmsg_percentile",
        "noise_chemaware_fullmsg_top_gap",
        "chemaware_v2_available", "chemaware_v2_correct_minus_zero",
        "chemaware_v2_correct_minus_worst_null", "chemaware_v2_null_spread",
        "chemaware_v2_delta_percentile", "chemaware_v2_delta_top_gap",
    ]
    rng = np.random.default_rng(4)
    vectors = {entity: value for entity, value in zip(
        ["q_shared", "q_single", "c1", "c2", "c4"],
        rng.normal(size=(5, 4)).astype(np.float32), strict=True,
    )}
    caches = {name: vectors for name in settings.encoder_names}
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / "model.pt"
        payload = {
            "arm": ARM_PROJECTED, "state_dict": model.state_dict(),
            "selection_class": "full_embedding_integration",
            "non_promotable_control": ARM_SCALAR,
            "representation_increment_verdict": "TEST_ONLY",
            "settings": settings.__dict__, "scalar_mean": np.zeros(30, np.float32),
            "scalar_scale": np.ones(30, np.float32), "scalar_feature_names": names,
            "encoder_names": settings.encoder_names,
        }
        torch.save(payload, path)
        scores, provenance = apply_unified_model(
            pairs, caches, path, torch.device("cpu"), query_ptr, query_ids,
        )
        scores_with_unused_msg, legacy_extra_report = apply_unified_model(
            pairs, caches, path, torch.device("cpu"), query_ptr, query_ids,
            auxiliary_scores={MSG_FAMILY: msg_scores},
        )
        assert np.allclose(scores, scores_with_unused_msg, equal_nan=True)
        assert legacy_extra_report["unused_supplied_auxiliary_family_names"] == [MSG_FAMILY]
        # q_shared|study_a and q_single|study_c are scored.  A single missing
        # row makes both rows of q_shared|study_b abstain, while grouping by
        # bare query_id would have incorrectly removed study_a too.
        assert np.isfinite(scores[[0, 1, 4]]).all()
        assert np.isnan(scores[[2, 3]]).all()
        assert provenance["scored_queries"] == 2
        assert provenance["zero_candidate_queries"] == 1
        assert provenance["incomplete_nonempty_queries"] == 1

        # A new checkpoint that declares auxiliary families must receive their
        # real aligned values; absence is never replaced by zeros.
        payload["auxiliary_family_names"] = ("official_dreams",)
        torch.save(payload, path)
        try:
            apply_unified_model(
                pairs, caches, path, torch.device("cpu"), query_ptr, query_ids,
            )
        except RuntimeError as error:
            assert "auxiliary-family contract" in str(error)
        else:
            raise AssertionError("missing auxiliary evidence was silently fabricated")
        payload.pop("auxiliary_family_names")
        payload["arm"] = ARM_SCALAR
        torch.save(payload, path)
        try:
            apply_unified_model(
                pairs, caches, path, torch.device("cpu"), query_ptr, query_ids,
            )
        except RuntimeError as error:
            assert "full-embedding integration arm" in str(error)
        else:
            raise AssertionError("scalar control was accepted as unified deployment")
    assert scores.shape == (5,)
    assert provenance["chemaware_v2"].startswith("unavailable")
    print("PASS: cross-study identity multi-encoder contracts", flush=True)


if __name__ == "__main__":
    main()
