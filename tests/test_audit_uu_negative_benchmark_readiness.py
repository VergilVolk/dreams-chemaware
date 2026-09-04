from pathlib import Path

from tasks.audit_uu_negative_benchmark_readiness import chemical_record, parse_mgf


def test_parse_mgf_and_approve_exact_m_h(tmp_path: Path) -> None:
    path = tmp_path / "tiny.mgf"
    path.write_text(
        """BEGIN IONS
SPECTRUMID=x1
NAME=acetate
FORMULA=C2H4O2
INCHIKEY=QTBSBXVTEAMEQO-UHFFFAOYSA-N
SMILES=CC(=O)O
ADDUCT=[M-H]-
PEPMASS=59.013853
CHARGE=1-
IONMODE=Negative
COLLISION_ENERGY=30
15.0 2.0
30.0 1.0
END IONS
""",
        encoding="utf-8",
    )
    records = list(parse_mgf(path))
    assert len(records) == 1
    audited = chemical_record(records[0], ppm=10.0)
    assert audited["approved_exact_m_h"]
    assert audited["truth_ik14"] == "QTBSBXVTEAMEQO"


def test_wrong_adduct_is_not_approved(tmp_path: Path) -> None:
    path = tmp_path / "tiny.mgf"
    path.write_text(
        """BEGIN IONS
SPECTRUMID=x1
FORMULA=C2H4O2
INCHIKEY=QTBSBXVTEAMEQO-UHFFFAOYSA-N
SMILES=CC(=O)O
ADDUCT=[2M-H]-
PEPMASS=119.0350
CHARGE=1-
IONMODE=Negative
15.0 2.0
END IONS
""",
        encoding="utf-8",
    )
    audited = chemical_record(next(parse_mgf(path)), ppm=10.0)
    assert not audited["approved_exact_m_h"]
