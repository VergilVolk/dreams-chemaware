import pytest

from tasks.audit_bioaware_metdna3_negative_source_loso import biological_source


def test_biological_source_strips_only_chromatography_suffix() -> None:
    assert biological_source("Mouse_brain__hilic") == "Mouse_brain"
    assert biological_source("NIST_plasma__rplc") == "NIST_plasma"
    with pytest.raises(ValueError):
        biological_source("invalid")
