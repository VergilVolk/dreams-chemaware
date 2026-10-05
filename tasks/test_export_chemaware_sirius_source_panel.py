"""CPU contracts for the SIRIUS ChemAware source-panel exporter."""
from __future__ import annotations

import numpy as np

from export_chemaware_sirius_source_panel import (
    charge_from_adduct,
    choose_structure_representative,
    controlled_spectrum,
    connectivity_smiles,
    sirius_ms_text,
    sirius_instrument_profile,
    spectrum_peaks,
)


def main() -> None:
    spectrum = np.asarray([
        [120.0, 50.0, 0.0, 80.0],
        [0.2, 1.0, 0.0, 0.4],
    ])
    mass, intensity = spectrum_peaks(spectrum)
    assert mass.tolist() == [50.0, 80.0, 120.0]
    assert intensity.tolist() == [1.0, 0.4, 0.2]
    body = sirius_ms_text(
        "q7_r11", 201.1234, "[M+H]+", "C10H16O4",
        "Orbitrap", 30.0, spectrum,
    )
    assert ">compound q7_r11" in body
    assert ">feature_id q7_r11" in body
    assert ">parentmass 201.12340000" in body
    assert ">ionization [M+H]+" in body
    assert ">formula C10H16O4" in body
    assert ">collision 30\n" in body
    blind_body = sirius_ms_text(
        "q7_r11", 201.1234, "[M+H]+", None,
        "Orbitrap", 30.0, spectrum,
    )
    assert ">formula" not in blind_body
    unknown_body = sirius_ms_text(
        "q7_r11", 201.1234, "[M+H]+", None, "nan", 30.0, spectrum,
    )
    assert ">instrumentation Unknown (LCMS)" in unknown_body
    assert ">formula C10H16O4" in body
    assert connectivity_smiles("C[C@H](O)C") == "CC(C)O"
    assert choose_structure_representative({"C[NH3+]", "CN"}) == "CN"
    assert body.index("50.00000000") < body.index("120.00000000")
    assert charge_from_adduct("[M+Na]+") == "1+"
    assert charge_from_adduct("[M-H]-") == "1-"
    assert sirius_instrument_profile("Orbitrap") == "orbitrap"
    assert sirius_instrument_profile("LC-ESI-QTOF") == "qtof"
    assert sirius_instrument_profile("nan") == "default"
    permuted = controlled_spectrum(spectrum, 7, "intensity_rank_permuted")
    assert sorted(permuted[1, permuted[0] > 0].tolist()) == sorted(spectrum[1, spectrum[0] > 0].tolist())
    assert not np.array_equal(permuted[1], spectrum[1])
    shifted = controlled_spectrum(spectrum, 7, "mass_shifted")
    assert np.array_equal(shifted[1], spectrum[1])
    assert not np.array_equal(shifted[0], spectrum[0])
    try:
        charge_from_adduct("[M+2H]2+")
    except ValueError:
        pass
    else:
        raise AssertionError("multiply charged SIRIUS input was not rejected")
    print("PASS: ChemAware SIRIUS source-panel export contracts", flush=True)


if __name__ == "__main__":
    main()
