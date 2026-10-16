#!/usr/bin/env python3
"""Query study-level signed spectral-response sketches at repository scale."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


WORLDS = ("all", "known", "unresolved")


def dot_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", a, b)


def score_query(
    manifest: pd.DataFrame,
    sketches: dict[str, np.ndarray],
    query_row: int,
    allow_same_source: bool,
    minimum_entities: int = 12,
) -> pd.DataFrame:
    query = manifest.iloc[query_row]
    eligible = np.ones(len(manifest), dtype=bool)
    eligible[query_row] = False
    if not allow_same_source:
        eligible &= manifest["source_family"].astype(str).to_numpy() != str(query["source_family"])
    target_rows = np.flatnonzero(eligible)
    output: list[pd.DataFrame] = []
    for world in WORLDS:
        count_column = f"n_entities_{world}"
        if count_column not in manifest:
            raise ValueError(f"manifest lacks {count_column}")
        pos = sketches[f"positive_{world}"]
        neg = sketches[f"negative_{world}"]
        query_count = int(manifest.iloc[query_row][count_column])
        query_has_signal = bool(
            query_count >= minimum_entities
            and (np.linalg.norm(pos[query_row]) > 0 or np.linalg.norm(neg[query_row]) > 0)
        )
        qpos = np.repeat(pos[[query_row]], len(target_rows), axis=0)
        qneg = np.repeat(neg[[query_row]], len(target_rows), axis=0)
        tpos, tneg = pos[target_rows], neg[target_rows]
        pp = dot_rows(qpos, tpos)
        nn = dot_rows(qneg, tneg)
        pn = dot_rows(qpos, tneg)
        np_ = dot_rows(qneg, tpos)
        block = manifest.iloc[target_rows].copy()
        block.insert(0, "query_sketch_row", int(query_row))
        block.insert(1, "query_study_id", str(query["study_id"]))
        block.insert(2, "query_contrast_id", str(query["contrast_id"]))
        block.insert(3, "evidence_world", world)
        block["positive_positive"] = pp
        block["negative_negative"] = nn
        block["positive_negative"] = pn
        block["negative_positive"] = np_
        block["signed_response_score"] = 0.5 * ((pp + nn) - (pn + np_))
        block["unsigned_response_score"] = 0.5 * ((pp + nn) + (pn + np_))
        target_has_signal = (
            block[count_column].to_numpy(int) >= minimum_entities
        ) & ((np.linalg.norm(tpos, axis=1) > 0) | (np.linalg.norm(tneg, axis=1) > 0))
        block["answerable"] = query_has_signal & target_has_signal
        block["abstention_reason"] = np.where(
            block["answerable"], "",
            "QUERY_INSUFFICIENT_EVIDENCE" if not query_has_signal else "TARGET_INSUFFICIENT_EVIDENCE",
        )
        block.loc[~block["answerable"], ["signed_response_score", "unsigned_response_score"]] = np.nan
        block["rank"] = np.nan
        answerable = block["answerable"]
        block.loc[answerable, "rank"] = block.loc[answerable, "signed_response_score"].rank(
            ascending=False, method="min"
        )
        output.append(block)
    return pd.concat(output, ignore_index=True) if output else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sketches", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--query-study")
    parser.add_argument("--query-contrast")
    parser.add_argument("--all-queries", action="store_true")
    parser.add_argument("--allow-same-source", action="store_true")
    parser.add_argument("--minimum-entities", type=int, default=12)
    args = parser.parse_args()
    if args.all_queries == bool(args.query_study or args.query_contrast):
        raise ValueError("use --all-queries or both --query-study and --query-contrast")
    if not args.all_queries and not (args.query_study and args.query_contrast):
        raise ValueError("both query identifiers are required")

    manifest = pd.read_csv(args.manifest, sep="\t")
    loaded = np.load(args.sketches)
    sketches = {name: loaded[name] for name in loaded.files}
    expected = {f"{direction}_{world}" for direction in ("positive", "negative") for world in WORLDS}
    if set(sketches) != expected:
        raise ValueError(f"sketch archive has unexpected arrays: {sorted(sketches)}")
    if any(len(values) != len(manifest) for values in sketches.values()):
        raise ValueError("sketch arrays and manifest have different row counts")
    if args.all_queries:
        query_rows = list(range(len(manifest)))
    else:
        mask = manifest["study_id"].astype(str).eq(args.query_study) & manifest["contrast_id"].astype(str).eq(args.query_contrast)
        query_rows = list(np.flatnonzero(mask.to_numpy()))
        if len(query_rows) != 1:
            raise ValueError(f"query matched {len(query_rows)} manifest rows")

    scores = pd.concat(
        [score_query(
            manifest, sketches, row, args.allow_same_source, args.minimum_entities,
        ) for row in query_rows],
        ignore_index=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_dir / "retrieval_scores.tsv", sep="\t", index=False)
    report = {
        "schema": "signed_spectral_measure_retrieval_v1",
        "query_count": len(query_rows),
        "candidate_contrasts": len(manifest),
        "scored_rows": len(scores),
        "same_source_allowed": args.allow_same_source,
        "minimum_entities_per_query_and_target": args.minimum_entities,
        "answerable_rows": int(scores["answerable"].sum()),
        "abstained_rows": int((~scores["answerable"]).sum()),
        "truth_opened": False,
        "claim_boundary": "Ranks are pre-truth retrieval outputs, not validated biological associations.",
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
