"""Deterministic unit contracts for contrast-asset metadata parsing."""

from audit_contrast_application_assets import parse_factors, sample_role


def main() -> None:
    assert parse_factors("Sample source:iPSC cells | GROUP:KO,Basal+DHEAS") == {
        "Sample source": "iPSC cells",
        "GROUP": "KO,Basal+DHEAS",
    }
    assert sample_role({"sample_source": "iPSC cells", "raw_data": "KO_1.mzML"}) == "study"
    assert sample_role({"sample_source": "Pool_QC", "raw_data": "Pool_QC_DDA_1.mzML"}) == "pooled_qc"
    assert sample_role({"sample_source": "Process_Blank"}) == "blank"
    assert sample_role({"sample_source": "Qstd"}) == "standard_or_reagent"
    print("PASS: contrast application asset contracts")


if __name__ == "__main__":
    main()
