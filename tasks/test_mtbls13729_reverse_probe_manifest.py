from __future__ import annotations

import tempfile
from pathlib import Path

from build_mtbls13729_reverse_probe_manifest import classify_family, iter_mgf_metadata, sample_fields


def main() -> None:
    assert classify_family("Palmitoylcarnitine", "", 400.0) == "long_chain_acylcarnitine_parent"
    assert classify_family("L-carnitine", "", 162.0) == ""
    assert classify_family("N1,N8-diacetylspermidine [M+H]+", "", 286.0) == "acetylated_polyamine_parent"
    assert classify_family("1-Methylguanosine [M+H]+", "", 298.0) == "methylated_guanosine_parent"
    assert classify_family("Guanosine", "", 284.0) == ""
    assert classify_family("N-acetylneuraminic acid - 40 eV M-H", "", 310.0) == "free_sialic_acid"
    assert classify_family(
        "Dopamine_suberic acid_Acetyl-L-Carnitine (known isomers: 0; isobaric peaks: 0) M+H",
        "", 500.0,
    ) == ""
    assert classify_family(
        "N-Acetylneuraminic acid_octylamine (known isomers: 0; isobaric peaks: 0) M+H",
        "", 500.0,
    ) == ""
    fields = sample_fields("pos_rp", Path("P21-Rmu.hdf5"))
    assert fields["patient_id"] == "P21" and fields["histology"] == "mucinous"
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tiny.mgf"
        path.write_text(
            "BEGIN IONS\nNAME=x\nINCHIKEY=ABCDEFGHIJKLMN-UHFFFAOYSA-N\n"
            "PEPMASS=123.4\nADDUCT=[M+H]+\n50 10\n60 20\nEND IONS\n",
            encoding="utf-8",
        )
        rows = list(iter_mgf_metadata(path))
        assert len(rows) == 1 and rows[0][0] == 0 and rows[0][3] == 2
    print("[test_mtbls13729_reverse_probe_manifest] PASS")


if __name__ == "__main__":
    main()
