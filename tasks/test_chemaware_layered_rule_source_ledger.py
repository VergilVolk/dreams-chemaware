"""CPU contracts for layered rule scoring and confidence separation."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from build_chemaware_layered_rule_source_ledger import (
    executable_rules,
    observation_strength,
    read_query_rows,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    corpus = json.loads((
        ROOT / "data/validation/chemaware_layered_rule_corpus_v5_20260928/rule_corpus.json"
    ).read_text(encoding="utf-8"))
    families, structure = executable_rules(corpus)
    assert {name: len(rules) for name, rules in families.items()} == {
        "layered_curated_formula_observations": 84,
        "msfinder_recurrent_formula_observations": 105,
        "msfinder_longtail_formula_observations": 1976,
    }
    assert len(structure) == 2
    query_rows = read_query_rows(
        ROOT / "data/validation/chemaware_sirius_source_panel_v9_profiled_collision_20260928/query_registry.tsv"
    )
    assert len(query_rows) == 4032
    assert any(query != spectrum_row for query, spectrum_row in query_rows.items())
    strength = observation_strength(
        mz=np.asarray([18.01056, 50.0, 82.0]),
        intensity=np.asarray([0.8, 0.5, 0.3]),
        precursor=100.0,
        targets=np.asarray([18.01056, 50.0, 18.0]),
        neutral=np.asarray([False, False, True]),
        ppm=20.0,
        floor_da=0.01,
    )
    assert np.allclose(strength, [0.8, 0.5, 0.3])
    print("PASS: ChemAware layered rule-source contracts", flush=True)


if __name__ == "__main__":
    main()
