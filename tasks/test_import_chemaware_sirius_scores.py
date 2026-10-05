"""CPU contracts for SIRIUS score normalization."""
from __future__ import annotations

import tempfile
from pathlib import Path

from import_chemaware_sirius_scores import (
    formula_scores,
    ik14,
    normalized_name,
    resolve_column,
    structure_scores,
)


def main() -> None:
    assert normalized_name("CSI:FingerIDScore") == "csifingeridscore"
    assert resolve_column(
        ["mappingFeatureId", "molecularFormula", "TreeScore"],
        ["featureId", "mappingFeatureId"],
    ) == "mappingFeatureId"
    assert ik14("ABCDEFGHIJKLMN-UHFFFAOYSA-N") == "ABCDEFGHIJKLMN"
    try:
        resolve_column(["rank"], ["TreeScore"])
    except KeyError:
        pass
    else:
        raise AssertionError("missing SIRIUS score column was not rejected")
    with tempfile.TemporaryDirectory(prefix="chem_sirius_import_") as directory:
        root = Path(directory)
        (root / "formula.tsv").write_text(
            "mappingFeatureId\tmolecularFormula\tTreeScore\n"
            "q1_f00\tC10H12O4\t3.25\n",
            encoding="utf-8",
        )
        (root / "structure.tsv").write_text(
            "mappingFeatureId\tmolecularFormula\tInChIkey2D\tCSI:FingerIDScore\n"
            "q1_f00\tC10H12O4\tABCDEFGHIJKLMN-UHFFFAOYSA-N\t7.5\n",
            encoding="utf-8",
        )
        (root / "unrelated.tsv").write_text("rank\tname\n1\tx\n", encoding="utf-8")
        assert formula_scores([root]) == {("q1_f00", "C10H12O4"): 3.25}
        assert structure_scores([root]) == {
            ("q1_f00", "C10H12O4", "ABCDEFGHIJKLMN"): 7.5
        }
    print("PASS: ChemAware SIRIUS score-import contracts", flush=True)


if __name__ == "__main__":
    main()
