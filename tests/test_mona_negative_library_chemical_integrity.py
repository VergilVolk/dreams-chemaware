import pandas as pd

from tasks.audit_mona_negative_library_chemical_integrity import audit_manifest


def test_integrity_accepts_true_m_h_and_rejects_in_source_loss() -> None:
    manifest = pd.DataFrame({
        "smiles": ["O=C(O)CCCC(=O)O", "OCC1OC(O)C(O)C1O"],
        "inchikey": [
            "JFCQEDHGNNZCLN-UHFFFAOYSA-N",
            "HMFHBZSHGGEWLO-SOOFDHNKSA-N",
        ],
        "name": ["glutaric acid", "ribose in-source loss"],
        "precursor_mz": [131.0349827, 131.0337010],
    })
    audited = audit_manifest(manifest, ppm=10.0)
    assert bool(audited.iloc[0]["approved_m_h_reference"])
    assert not bool(audited.iloc[1]["approved_m_h_reference"])


def test_integrity_rejects_smiles_inchikey_mismatch() -> None:
    manifest = pd.DataFrame({
        "smiles": ["O=C(O)CCCC(=O)O"],
        "inchikey": ["HMFHBZSHGGEWLO-SOOFDHNKSA-N"],
        "name": ["bad metadata"],
        "precursor_mz": [131.0349827],
    })
    audited = audit_manifest(manifest, ppm=10.0)
    assert not bool(audited.iloc[0]["structure_identity_consistent"])
    assert not bool(audited.iloc[0]["approved_m_h_reference"])
