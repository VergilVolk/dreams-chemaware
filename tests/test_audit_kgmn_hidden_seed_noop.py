from pathlib import Path


def test_noop_audit_covers_all_identity_bearing_outputs() -> None:
    text = Path("tasks/audit_kgmn_hidden_seed_noop.py").read_text(encoding="utf-8")
    assert "03_annotation_credential/annontation_credential_long.csv" in text
    assert "00_annotation_table/table1_identification.csv" in text
    assert "00_annotation_table/table3_identification_pair.csv" in text
    assert "compare_csv_multisets" in text
