#!/usr/bin/env python
"""Synthetic contracts for the GNPS2 Standard mega-benchmark preflight."""
from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path

import h5py

from preflight_gnps2_standard_mega_benchmark import (
    apply_overlap,
    classify_identity_panel_status,
    forecast_pairwise_verification,
    load_slim_metadata,
    preflight,
)


def write_record(
    handle,
    *,
    spectrum_id: str,
    precursor: float,
    smiles: str,
    inchikey: str,
    formula: str,
    adduct: str = "[M+H]+",
    mode: str = "Positive",
    quality: int = 1,
    shift: float = 0.0,
) -> None:
    handle.write("BEGIN IONS\n")
    handle.write(f"TITLE={spectrum_id}\nSPECTRUMID={spectrum_id}\n")
    handle.write(f"PEPMASS={precursor:.7f}\nMSLEVEL=2\n")
    handle.write(f"IONMODE={mode}\nADDUCT={adduct}\nLIBRARYQUALITY={quality}\n")
    handle.write(f"SMILES={smiles}\nINCHIAUX={inchikey}\nFORMULA={formula}\n")
    for index in range(12):
        handle.write(f"{40 + index * 5 + shift:.5f} {100 + index:.3f}\n")
    handle.write("END IONS\n")


def main() -> None:
    sparse_status, sparse_fraction = classify_identity_panel_status(Counter({
        "raw_records": 30_574,
        "eligible_before_overlap": 2,
        "no_verifiable_identity": 30_425,
    }))
    assert sparse_status == "annotation_only_sparse_verifiable_ik14"
    assert 0.0 < sparse_fraction < 0.01

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source_a = root / "A.mgf"
        source_b = root / "B.mgf"
        # Ethanol and dimethyl ether share formula and lie in the same 10-ppm
        # precursor window. Each has two non-identical replicate spectra.
        ethanol_key = "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"
        ether_key = "LCGLNKUTAGEVQW-UHFFFAOYSA-N"
        with source_a.open("w", encoding="utf-8") as handle:
            write_record(handle, spectrum_id="A1", precursor=47.049141, smiles="CCO", inchikey=ethanol_key, formula="C2H6O", shift=0.0)
            write_record(handle, spectrum_id="A2", precursor=47.049141, smiles="CCO", inchikey=ethanol_key, formula="C2H6O", shift=0.01)
            # Q3 is intentionally retained only in separate/expanded strata.
            write_record(handle, spectrum_id="A3", precursor=47.049141, smiles="COC", inchikey=ether_key, formula="C2H6O", quality=3, shift=0.02)
        with source_b.open("w", encoding="utf-8") as handle:
            write_record(handle, spectrum_id="B1", precursor=47.049141, smiles="COC", inchikey=ether_key, formula="C2H6O", quality=3, shift=0.03)

        # A separate source demonstrates an MGF -> official slim JSON identity
        # join. The MGF intentionally lacks all structure fields.
        source_c = root / "C.mgf"
        with source_c.open("w", encoding="utf-8") as handle:
            for spectrum_id, shift in (("C1", 0.04), ("C2", 0.05)):
                handle.write("BEGIN IONS\n")
                handle.write(f"TITLE={spectrum_id}\nSPECTRUMID={spectrum_id}\n")
                handle.write("PEPMASS=47.049141\nMSLEVEL=2\nIONMODE=Positive\n")
                handle.write("NAME=ethanol [M+H]+\nLIBRARYQUALITY=1\n")
                for index in range(12):
                    handle.write(f"{40 + index * 5 + shift:.5f} {100 + index:.3f}\n")
                handle.write("END IONS\n")
        slim = root / "C.slim.json"
        slim.write_text(json.dumps([
            {"SpectrumID": "C1", "Smiles": "CCO", "INCHI_AUX": ethanol_key, "Formula_smiles": "C2H6O", "Adduct": "M+H", "Ion_Mode": "Positive", "Library_Class": "1"},
            {"SpectrumID": "C2", "Smiles": "CCO", "INCHI_AUX": ethanol_key, "Formula_smiles": "C2H6O", "Adduct": "M+H", "Ion_Mode": "Positive", "Library_Class": "1"},
        ]), encoding="utf-8")

        manifest = root / "sources.json"
        manifest.write_text(json.dumps({
            "schema": "gnps2_standard_mega_source_manifest_v1",
            "dataset_id": "synthetic_gnps2_standard",
            "sources": [
                {"source_id": "A", "mgf": "A.mgf", "required": True},
                {"source_id": "B", "mgf": "B.mgf", "required": True},
                {"source_id": "C", "mgf": "C.mgf", "slim_metadata": "C.slim.json", "required": True},
            ],
        }), encoding="utf-8")

        # Minimal but fully present exclusion registry. Formula overlap is
        # deliberately absent so the synthetic formula-disjoint panel survives.
        consumed = root / "consumed.h5"
        with h5py.File(consumed, "w") as handle:
            handle.create_dataset("INCHIKEY", data=[b"AAAAAAAAAAAAAA-UHFFFAOYSA-N"])
            handle.create_dataset("FORMULA", data=[b"C9H9O9"])
        registry = root / "registry.json"
        registry.write_text(json.dumps({
            "schema": "unified_consumed_source_registry_v1",
            "sources": [{"name": "consumed", "kind": "hdf5", "path": str(consumed), "required": True}],
        }), encoding="utf-8")

        report, inventory = preflight(manifest, registry, candidate_ppm=10.0, seed=7)
        assert report["status"] == "GNPS2_STANDARD_MEGA_PREFLIGHT_READY"
        assert report["scores_accessed"] is False
        assert report["panel_build_ready"] is True
        assert set(inventory.quality_tier) == {"Q12_PRIMARY", "Q3_SEPARATE"}
        assert int((inventory.identity_basis == "official_slim_json:structure").sum()) == 2
        q123 = report["panel_forecasts"]["q123_expanded.identity_disjoint"]
        assert q123["N_Q_labelled_queries"] > 0
        assert q123["N_E_query_candidate_molecule_edges"] >= 2 * q123["N_Q_labelled_queries"]
        assert q123["N_P_query_reference_spectrum_comparisons"] >= q123["N_E_query_candidate_molecule_edges"]
        assert q123["reaches_10M_comparisons"] is False
        assert "not labelled queries" in q123["unit_warning"]
        assert report["panel_forecasts"]["q12_primary.identity_disjoint"]["N_Q_labelled_queries"] == 0
        assert report["panel_forecasts"]["q3_separate.identity_disjoint"]["N_Q_labelled_queries"] == 0
        pair_report, pair_arrays = forecast_pairwise_verification(
            inventory,
            seed=7,
            hard_abs_da=0.05,
            outer_abs_da=2.0,
            positive_cap=2,
            negative_cap=4,
            return_indices=True,
        )
        assert pair_report["N_Q_labelled_queries"] > 0
        assert pair_report["N_pairs_positive"] > 0
        assert pair_report["N_pairs_negative"] > 0
        assert pair_report["reaches_10M_pairs"] is False
        assert len(pair_arrays["query_row"]) == pair_report["N_pairs_total"]
        assert len(set(zip(pair_arrays["query_row"].tolist(), pair_arrays["reference_row"].tolist()))) == pair_report["N_pairs_total"]

        # Identity, formula, and normalized exact-spectrum signatures are all
        # independent fail-closed exclusion channels.
        first = inventory.iloc[0]
        overlap = apply_overlap(
            inventory,
            {str(first.ik14)},
            {str(first.formula)},
            {
                "primary_sha256": {str(first.spectrum_hash)},
                "secondary_blake2b": set(),
            },
        )
        selected = overlap.loc[overlap.global_row == first.global_row].iloc[0]
        assert bool(selected.consumed_identity_overlap)
        assert bool(selected.consumed_formula_overlap)
        assert bool(selected.consumed_spectrum_overlap)

        # Duplicate slim join keys are a whole-source contract failure.
        duplicate = root / "duplicate.json"
        duplicate.write_text(json.dumps([
            {"SpectrumID": "X", "Smiles": "CCO"},
            {"SpectrumID": "X", "Smiles": "COC"},
        ]), encoding="utf-8")
        try:
            load_slim_metadata(duplicate)
        except RuntimeError as error:
            assert "not one-to-one" in str(error)
        else:
            raise AssertionError("duplicate slim key was not rejected")

        # Missing required consumed sources do not produce an optimistic panel.
        missing_registry = root / "missing_registry.json"
        missing_registry.write_text(json.dumps({
            "schema": "unified_consumed_source_registry_v1",
            "sources": [{"name": "missing", "kind": "mgf", "path": str(root / "missing.mgf"), "required": True}],
        }), encoding="utf-8")
        missing_report, _ = preflight(manifest, missing_registry, candidate_ppm=10.0, seed=7)
        assert missing_report["status"] == "GNPS2_STANDARD_MEGA_PREFLIGHT_BLOCKED_NOT_EXTERNAL"
        assert missing_report["panel_build_ready"] is False
        assert missing_report["panel_forecasts"] == {}
        assert missing_report["pairwise_verification_panel_forecasts"] == {}
        assert missing_report["score_blind_available_exclusion_forecasts"]
        assert missing_report["score_blind_available_exclusion_pairwise_verification_forecasts"]
        assert missing_report["score_blind_pre_exclusion_forecasts"]["q123_expanded"]["N_Q_labelled_queries"] > 0
        assert missing_report["score_blind_pre_exclusion_pairwise_verification_forecasts"]["q123_expanded"]["N_pairs_total"] > 0
        assert missing_report["overlap_exclusion"]["missing_required_sources"] == ["missing"]

    print("PASS: GNPS2 Standard mega-benchmark preflight contracts")


if __name__ == "__main__":
    main()
