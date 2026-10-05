#!/usr/bin/env python
"""Fast dependency-free unit checks for BioAware B38-M0 helpers."""
from __future__ import annotations

from audit_bioaware_b38_m0_explicit_path_coverage import (
    direction_status,
    missingness_class,
    normalise_reaction_id,
    parse_kgmn_query,
    parse_prefixed_query,
)


def main() -> None:
    assert normalise_reaction_id(20689.0) == "20689"
    assert normalise_reaction_id("20689;23204") == "20689;23204"
    assert direction_status("reactome_consensus_lr", {"left"}, {"right"}) == "supported"
    assert direction_status("reactome_consensus_lr", {"right"}, {"left"}) == "conflicted"
    assert direction_status("reaction_direction_unknown", {"left"}, {"right"}) == "direction_unknown"
    assert direction_status("reactome_consensus_lr", {"left", "right"}, {"right"}) == "ambiguous_participant_side"
    assert missingness_class(0, False, False, False) == "no_path"
    assert missingness_class(1, False, False, False) == "path_unobservable"
    assert missingness_class(1, True, False, False) == "complete_nonspecific_screen"
    assert missingness_class(1, False, True, True) == "complete_specific_screen"
    assert parse_prefixed_query("ST001154_same_formula_10ppm::q1", "ST001154_same_formula_10ppm") == "q1"
    assert parse_kgmn_query("KGMN200STD_hidden_seed::M122T364::repeat=7") == ("M122T364", 7)
    print("[test_bioaware_b38_m0_explicit_path_coverage] PASS", flush=True)


if __name__ == "__main__":
    main()
