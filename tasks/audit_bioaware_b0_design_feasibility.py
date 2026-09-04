#!/usr/bin/env python
"""Audit B0 fold/control feasibility without encoding a single spectrum."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tasks.audit_bioaware_b0_reaction_embedding_signal import (  # noqa: E402
    assign_formula_community_folds,
    build_matched_pairs,
    fast_reaction_edges,
    molecular_table,
    standardised_imbalance,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows", type=Path,
        default=ROOT / "data/validation/bioaware_embedding_relation_manifest_v2_20260830/rows.csv.gz",
    )
    parser.add_argument(
        "--hdf5", type=Path,
        default=ROOT / "data/models/MassSpecGym_MurckoHist_split.hdf5",
    )
    parser.add_argument(
        "--participants", type=Path,
        default=ROOT / "data/reference/bioaware_rhea_reactome_direction_20260830/rhea_participants.csv.gz",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--community-resolution", type=float, default=2.0)
    parser.add_argument("--controls-per-edge", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260901)
    args = parser.parse_args()
    for path in (args.rows, args.hdf5, args.participants):
        if not path.is_file():
            raise FileNotFoundError(path)
    row_table = pd.read_csv(args.rows, usecols=["row"])
    rows = np.sort(row_table["row"].to_numpy(np.int64))
    dummy = np.ones((len(rows), 1), dtype=np.float32)
    molecules, _prototypes, fingerprints = molecular_table(args.hdf5, rows, dummy)
    identities = set(molecules["ik14"].astype(str))
    edges = fast_reaction_edges(pd.read_csv(args.participants), identities)
    formula = molecules.set_index("ik14")["formula"].astype(str).to_dict()
    fold_by_identity, partition = assign_formula_community_folds(
        sorted(identities), formula, edges, args.folds, args.seed,
        args.community_resolution,
    )
    pairs = build_matched_pairs(
        molecules, fingerprints, edges, fold_by_identity, args.controls_per_edge,
        args.seed + 1,
    )
    per_fold = {
        str(fold): int(pairs.loc[pairs.fold == fold, "group_id"].nunique())
        for fold in range(args.folds)
    }
    report = {
        "status": "bioaware_b0_design_feasibility_complete",
        "spectra": int(len(rows)),
        "identities": int(len(identities)),
        "reaction_edges": int(len(edges)),
        "matched_groups": int(pairs.group_id.nunique()),
        "matched_rows": int(len(pairs)),
        "matched_groups_per_fold": per_fold,
        "matching_standardised_imbalance": standardised_imbalance(pairs),
        "formula_community_partition": partition,
        "gates": {
            "matched_groups_ge_100": bool(pairs.group_id.nunique() >= 100),
            "every_fold_ge_10": bool(all(value >= 10 for value in per_fold.values())),
            "target_degree_exact": bool(
                standardised_imbalance(pairs)["target_log_degree"] < 1e-12
            ),
        },
        "claim_limit": "Design-only audit; contains no DreaMS performance result.",
    }
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
