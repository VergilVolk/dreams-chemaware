#!/usr/bin/env python
"""Require two GNPS panel files to describe the identical candidate edge axis."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


AXIS_FIELDS = (
    "query_row", "query_ik14", "query_formula", "query_ptr",
    "molecule_ptr", "molecule_ik14", "molecule_formula", "molecule_label",
    "candidate_row", "independent_positive", "near_query",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.left, allow_pickle=False) as left, np.load(args.right, allow_pickle=False) as right:
        missing = [name for name in AXIS_FIELDS if name not in left or name not in right]
        if missing:
            raise RuntimeError(f"GNPS_PANEL_AXIS_MISSING_FIELDS: {missing}")
        drift = [name for name in AXIS_FIELDS if not np.array_equal(left[name], right[name])]
    if drift:
        raise RuntimeError(f"GNPS_PANEL_CANDIDATE_AXIS_DRIFT: {drift}")
    print(f"GNPS_PANEL_CANDIDATE_AXIS_IDENTICAL: {args.left} == {args.right}")


if __name__ == "__main__":
    main()
