#!/usr/bin/env python
"""Unit checks for B44 query/reference isolation."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tasks"))
from build_bioaware_b44_massbank_blind_panel import build_panel  # noqa: E402


def main() -> None:
    rows = []
    for index, (identity, formula) in enumerate((("A", "F"), ("B", "F"), ("C", "G"))):
        for repeat in range(2):
            rows.append({
                "hdf5_row": index * 2 + repeat, "metadata_row": index * 2 + repeat,
                "record_id": f"x{index}{repeat}", "full_inchikey": identity,
                "ik14": identity, "formula": formula,
                "MS_ION_MODE": "POSITIVE", "PRECURSOR_TYPE_ADDUCT": "[M+H]+",
                "PRECURSOR_MZ": 100.0, "peak_count": 5,
            })
    queries, candidates, library, report = build_panel(pd.DataFrame(rows), set(), set(), 7)
    assert len(queries) == 2
    assert report["candidate_identities_per_query"]["minimum"] == 2
    assert not (set(queries["query_hdf5_row"]) & set(library["hdf5_row"]))
    assert candidates.groupby("query_id")["is_positive"].sum().eq(1).all()
    print("[test_bioaware_b44_massbank_blind_panel] PASS", flush=True)


if __name__ == "__main__":
    main()
