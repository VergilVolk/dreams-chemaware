import pytest

from append_contrast_atlas_frozen_collection import expected_hashes


def test_expected_hashes_supports_corrected_and_builder_reports():
    assert expected_hashes({"output_sha256": {"a": "1"}}) == {"a": "1"}
    assert expected_hashes({"outputs_sha256": {"b": "2"}}) == {"b": "2"}


def test_expected_hashes_rejects_unverifiable_report():
    with pytest.raises(RuntimeError):
        expected_hashes({"status": "complete"})
