#!/usr/bin/env python
"""Audit whether exact library-supported structures can anchor Atlas truth.

This is a read-only feasibility audit.  It never uses unresolved spectral edges
and it does not assign biological truth.  It counts response-blind, best-library
entities per full InChIKey and the resulting cross-source overlap available for
a future entity-disjoint signed-response test.
"""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import pandas as pd


def read_tsv(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        return pd.read_csv(handle, sep="\t", low_memory=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities", type=Path, required=True)
    parser.add_argument("--library-evidence", type=Path, required=True)
    parser.add_argument("--min-common", type=int, default=3)
    args = parser.parse_args()

    entities = read_tsv(args.entities)
    library = read_tsv(args.library_evidence)

    library = library.loc[
        library["library_supported"].astype(str).str.lower().eq("true")
        & library["library_inchikey"].fillna("").astype(str).str.match(
            r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$"
        )
    ].copy()
    library["wse_sort"] = pd.to_numeric(library["wse"], errors="coerce").fillna(-1)
    library["mp_sort"] = pd.to_numeric(
        library["matched_peaks"], errors="coerce"
    ).fillna(-1)

    # One response-blind observed ion per exact structure and study.  Ranking
    # uses only library evidence and a deterministic entity-id tie break.
    library = library.sort_values(
        ["study_id", "library_inchikey", "wse_sort", "mp_sort", "entity_id"],
        ascending=[True, True, False, False, True],
        kind="mergesort",
    ).drop_duplicates(["study_id", "library_inchikey"], keep="first")

    joined = entities.merge(
        library[["study_id", "entity_id", "library_inchikey", "wse", "matched_peaks"]],
        on=["study_id", "entity_id"],
        how="inner",
        validate="many_to_one",
    )
    studies = sorted(joined["study_id"].unique())
    rows: list[dict[str, object]] = []
    for i, left in enumerate(studies):
        left_keys = set(
            joined.loc[joined["study_id"].eq(left), "library_inchikey"].unique()
        )
        for right in studies[i + 1 :]:
            right_keys = set(
                joined.loc[joined["study_id"].eq(right), "library_inchikey"].unique()
            )
            common = left_keys & right_keys
            if len(common) < args.min_common:
                continue
            left_contrasts = joined.loc[
                joined["study_id"].eq(left), "contrast_id"
            ].nunique()
            right_contrasts = joined.loc[
                joined["study_id"].eq(right), "contrast_id"
            ].nunique()
            rows.append(
                {
                    "left_study": left,
                    "right_study": right,
                    "common_exact_inchikeys": len(common),
                    "left_contrasts": int(left_contrasts),
                    "right_contrasts": int(right_contrasts),
                    "contrast_pair_opportunities": int(left_contrasts * right_contrasts),
                }
            )

    per_study = (
        joined.groupby("study_id")
        .agg(
            exact_inchikeys=("library_inchikey", "nunique"),
            contrasts=("contrast_id", "nunique"),
            response_rows=("entity_id", "size"),
        )
        .reset_index()
        .sort_values(["exact_inchikeys", "study_id"], ascending=[False, True])
    )
    pairs = pd.DataFrame(rows)
    if not pairs.empty:
        pairs = pairs.sort_values(
            ["common_exact_inchikeys", "left_study", "right_study"],
            ascending=[False, True, True],
        )

    print(
        "SUMMARY\t"
        f"library_supported_entity_study_keys={len(library)}\t"
        f"joined_response_rows={len(joined)}\t"
        f"studies={joined['study_id'].nunique()}\t"
        f"exact_inchikeys={joined['library_inchikey'].nunique()}\t"
        f"study_pairs_ge_{args.min_common}={len(pairs)}"
    )
    print("\nPER_STUDY")
    print(per_study.to_csv(sep="\t", index=False), end="")
    print("\nCROSS_STUDY_PAIRS")
    print(pairs.to_csv(sep="\t", index=False), end="")


if __name__ == "__main__":
    main()
