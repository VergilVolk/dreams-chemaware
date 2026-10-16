#!/usr/bin/env python3
"""Truth-blind all-query retrieval for factorial HFD response objects."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


WORLDS = ("all", "known", "unresolved")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sketches", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--minimum-entities", type=int, default=12)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    manifest = pd.read_csv(args.manifest, sep="\t")
    with np.load(args.sketches, allow_pickle=False) as packed:
        vectors = {name: np.asarray(packed[name], dtype=np.float32) for name in packed.files}
    expected = {f"{kind}_{world}" for kind in ("positive", "negative", "chemistry") for world in WORLDS}
    if set(vectors) != expected:
        raise ValueError(f"unexpected sketch arrays: {sorted(vectors)}")
    if any(len(value) != len(manifest) for value in vectors.values()):
        raise ValueError("manifest/sketch row mismatch")

    rows: list[dict[str, object]] = []
    for query_row, query in manifest.iterrows():
        target_rows = np.flatnonzero(manifest.study_id.astype(str).ne(str(query.study_id)).to_numpy())
        for world in WORLDS:
            q_count = int(query[f"n_entities_{world}"])
            qpos, qneg = vectors[f"positive_{world}"][query_row], vectors[f"negative_{world}"][query_row]
            qchem = vectors[f"chemistry_{world}"][query_row]
            q_answerable = q_count >= args.minimum_entities and (
                np.linalg.norm(qpos) > 0 or np.linalg.norm(qneg) > 0
            )
            for target_row in target_rows:
                target = manifest.iloc[target_row]
                tpos = vectors[f"positive_{world}"][target_row]
                tneg = vectors[f"negative_{world}"][target_row]
                tchem = vectors[f"chemistry_{world}"][target_row]
                answerable = q_answerable and int(target[f"n_entities_{world}"]) >= args.minimum_entities and (
                    np.linalg.norm(tpos) > 0 or np.linalg.norm(tneg) > 0
                )
                signed = float((qpos - qneg) @ (tpos - tneg)) if answerable else np.nan
                magnitude = float((qpos + qneg) @ (tpos + tneg)) if answerable else np.nan
                chemistry = float(qchem @ tchem) if answerable else np.nan
                rows.append({
                    "query_sketch_row": int(query_row),
                    "query_study_id": str(query.study_id),
                    "query_contrast_id": str(query.contrast_id),
                    "target_sketch_row": int(target_row),
                    "target_study_id": str(target.study_id),
                    "target_contrast_id": str(target.contrast_id),
                    "evidence_world": world,
                    "answerable": answerable,
                    "signed_response_score": signed,
                    "magnitude_only_score": magnitude,
                    "chemistry_only_score": chemistry,
                })
    scores = pd.DataFrame(rows)
    group = ["query_sketch_row", "evidence_world"]
    for score in ("signed_response_score", "magnitude_only_score", "chemistry_only_score"):
        scores[f"rank_{score}"] = scores.groupby(group)[score].rank(ascending=False, method="min")
    args.output_dir.mkdir(parents=True)
    scores.to_csv(args.output_dir / "retrieval_scores.tsv.gz", sep="\t", index=False, compression="gzip")
    report = {
        "schema": "factorial_hfd_truth_blind_retrieval_v1",
        "status": "pretruth_scores_frozen",
        "denominators": {
            "query_objects": int(manifest.shape[0]),
            "candidate_rows": len(scores),
            "studies": int(manifest.study_id.nunique()),
            "evidence_worlds": len(WORLDS),
        },
        "score_definitions": {
            "signed_response_score": "inner product of signed spectral response measures",
            "magnitude_only_score": "inner product after discarding response direction",
            "chemistry_only_score": "effect-independent spectral composition similarity",
        },
        "truth_opened": False,
        "claim_boundary": "Ranks are frozen retrieval outputs; biological relationship has not been adjudicated.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
