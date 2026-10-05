"""Contracts for the Phase-A current-error-winner native curriculum."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))

import build_chemaware_phasea_error_winner_native_triplets as target  # noqa: E402
from build_chemaware_sirius_native_triplets import load_npz  # noqa: E402


def test_frozen_legacy_specific_registry_is_recovered() -> None:
    evidence = load_npz(
        ROOT / "data/validation/chemaware_high_coverage_native/run_2340524"
        / "evidence/train_triplet_evidence.npz"
    )
    manifest = load_npz(
        ROOT / "data/validation/chemaware_corrected_candidate_manifest_v1/manifest.npz"
    )
    recovered = target.legacy_specific_winners(evidence, manifest)
    assert set(recovered) == set(map(int, np.asarray(evidence["query"])))
    assert all(all(candidate >= 0 for candidate in values) for values in recovered.values())
    assert any(values for values in recovered.values())


def test_builder_is_native_error_winner_only() -> None:
    text = Path(target.__file__).read_text(encoding="utf-8")
    assert "copy_phasea(phasea, writer)" in text
    assert "verify_phasea_prefix(phasea, output)" in text
    assert "false_score < true_score" in text
    assert "active_reference_events(" in text
    assert "one_event_per_added_query" in text
    assert '"loss": "unchanged DreaMS cosine triplet margin"' in text
    assert "listwise" not in text.lower()
    assert "distill" not in text.lower()


def test_error_winner_role_fits_serialized_dtype() -> None:
    assert 0 <= target.ERROR_WINNER_ROLE <= np.iinfo(np.int8).max

