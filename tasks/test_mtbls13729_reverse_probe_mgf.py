from __future__ import annotations

import tempfile
from pathlib import Path

from extract_mtbls13729_reverse_probe_mgf import extract_blocks


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source, output = root / "source.mgf", root / "out.mgf"
        source.write_text(
            "BEGIN IONS\nNAME=a\nPEPMASS=1\n1 2\nEND IONS\n"
            "BEGIN IONS\nNAME=b\nPEPMASS=2\n2 3\nEND IONS\n",
            encoding="utf-8",
        )
        rows = extract_blocks(source, {1: "R000001"}, output)
        assert rows == [{"subset_row": 0, "source_mgf_record": 1, "reference_spectrum_id": "R000001"}]
        text = output.read_text(encoding="utf-8")
        assert "NAME=b" in text and "NAME=a" not in text and "REVERSE_PROBE_ID=R000001" in text
    print("[test_mtbls13729_reverse_probe_mgf] PASS")


if __name__ == "__main__":
    main()
