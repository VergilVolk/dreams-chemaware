#!/usr/bin/env python3
"""Package the complete-axis WSE result without duplicating frozen axis IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/validation/contrast_atlas_complete_identity_wse_v2_20261029"),
    )
    parser.add_argument(
        "--axis-dir",
        type=Path,
        default=Path("data/validation/contrast_atlas_model_independent_identity_axis_v5_20261027"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/validation/contrast_atlas_complete_identity_wse_release_v1_20261029"),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    source_report_path = args.source_dir / "report.json"
    pair_path = args.axis_dir / "directed_complete_candidate_pairs.tsv.gz"
    block_path = args.axis_dir / "query_target_source_blocks.tsv.gz"
    score_path = args.source_dir / "candidate_scores.tsv.gz"
    decision_path = args.source_dir / "block_decisions.tsv.gz"
    for path in (source_report_path, pair_path, block_path, score_path, decision_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    if source_report.get("status") != "DEVELOPMENT_CONSUMED_CONTRAST_ATLAS_COMPLETE_IDENTITY_WSE_SCORED":
        raise RuntimeError("source is not a complete-axis WSE result")
    for filename, path in (
        ("candidate_scores.tsv.gz", score_path),
        ("block_decisions.tsv.gz", decision_path),
    ):
        if sha256(path) != source_report["outputs"][filename]:
            raise RuntimeError(f"source output hash mismatch: {filename}")

    pair_ids = pd.read_csv(pair_path, sep="\t", usecols=["directed_pair_id"])[
        "directed_pair_id"
    ].astype(str).to_numpy()
    scores = pd.read_csv(
        score_path,
        sep="\t",
        usecols=[
            "directed_pair_id",
            "wse_score",
            "wse_matched_fragments",
            "wse_available",
            "wse_rank_min",
            "wse_rank_stable",
        ],
    )
    if not np.array_equal(scores["directed_pair_id"].astype(str).to_numpy(), pair_ids):
        raise RuntimeError("WSE rows are not aligned to the frozen directed-pair axis")
    decisions = pd.read_csv(decision_path, sep="\t", low_memory=False)
    nonempty = decisions.loc[decisions["candidate_count"].gt(0)].copy()
    expected = source_report["denominators"]
    if len(scores) != int(expected["directed_candidate_pairs"]):
        raise RuntimeError("pair denominator drifted")
    if len(decisions) != int(expected["query_blocks"]):
        raise RuntimeError("block denominator drifted")
    if len(nonempty) != int(expected["singleton_blocks"]) + int(expected["multiple_candidate_blocks"]):
        raise RuntimeError("nonempty block denominator drifted")

    release_denominators = dict(expected)
    legacy_answered = release_denominators.pop("answered_blocks", None)
    positive_answered = int(
        (
            decisions["wse_answered"].astype(bool)
            & decisions["wse_top1_score"].fillna(0).gt(0)
            & decisions["wse_top1_matched_fragments"].fillna(0).gt(0)
        ).sum()
    )
    if "answered_with_positive_spectral_evidence" in release_denominators:
        if int(release_denominators["answered_with_positive_spectral_evidence"]) != positive_answered:
            raise RuntimeError("positive-evidence answer denominator drifted")
    release_denominators["answered_with_positive_spectral_evidence"] = positive_answered
    release_denominators["legacy_nominal_unique_top1_blocks"] = (
        None if legacy_answered is None else int(legacy_answered)
    )
    release_denominators["zero_evidence_singleton_blocks"] = int(
        (
            decisions["candidate_count"].eq(1)
            & (
                decisions["wse_top1_score"].fillna(0).le(0)
                | decisions["wse_top1_matched_fragments"].fillna(0).le(0)
            )
        ).sum()
    )

    blocks = pd.read_csv(
        block_path,
        sep="\t",
        usecols=[
            "candidate_pair_offset_start",
            "candidate_pair_offset_end",
        ],
    )
    starts = blocks["candidate_pair_offset_start"].to_numpy(np.int64)
    stops = blocks["candidate_pair_offset_end"].to_numpy(np.int64)
    query_ptr = np.concatenate((starts[:1], stops)).astype(np.int64)
    if query_ptr[0] != 0 or query_ptr[-1] != len(scores) or np.any(np.diff(query_ptr) < 0):
        raise RuntimeError("frozen block offsets are invalid")

    output = args.output_dir
    output.mkdir(parents=True)
    compact_path = output / "wse_scores_aligned.npz"
    np.savez_compressed(
        compact_path,
        schema=np.asarray("contrast_atlas_complete_identity_wse_aligned_v1"),
        directed_pair_axis_sha256=np.asarray(sha256(pair_path)),
        query_block_axis_sha256=np.asarray(sha256(block_path)),
        query_ptr=query_ptr,
        wse_score=scores["wse_score"].to_numpy(np.float32),
        wse_matched_fragments=scores["wse_matched_fragments"].to_numpy(np.int16),
        wse_availability=scores["wse_available"].to_numpy(bool),
        wse_rank_min=scores["wse_rank_min"].to_numpy(np.int32),
        wse_rank_stable=scores["wse_rank_stable"].to_numpy(np.int32),
    )
    nonempty_path = output / "nonempty_block_decisions.tsv.gz"
    nonempty.to_csv(
        nonempty_path,
        sep="\t",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    report = {
        "schema": "contrast_atlas_complete_identity_wse_release_v1",
        "status": "DEVELOPMENT_CONSUMED_CONTRAST_ATLAS_COMPLETE_IDENTITY_WSE_RELEASED",
        "data_role": "development_consumed",
        "alignment_contract": {
            "score_rows": "exact row order of directed_complete_candidate_pairs.tsv.gz",
            "block_pointer": "exact half-open offsets of query_target_source_blocks.tsv.gz",
            "pair_axis_sha256": sha256(pair_path),
            "block_axis_sha256": sha256(block_path),
            "ids_not_duplicated": True,
        },
        "denominators": release_denominators,
        "score_summary": source_report["score_summary"],
        "truth_blinding": source_report["truth_blinding"],
        "method": source_report["method"],
        "claim_boundary": source_report["claim_boundary"] + [
            "Only blocks with positive WSE and at least one matched fragment count as spectral answers; zero-evidence singleton blocks remain explicit abstentions."
        ],
        "provenance": {
            "source_report": {"path": str(source_report_path), "sha256": sha256(source_report_path)},
            "source_candidate_scores": {"path": str(score_path), "sha256": sha256(score_path)},
            "source_block_decisions": {"path": str(decision_path), "sha256": sha256(decision_path)},
            "directed_pair_axis": {"path": str(pair_path), "sha256": sha256(pair_path)},
            "query_block_axis": {"path": str(block_path), "sha256": sha256(block_path)},
            "implementation": {"path": str(Path(__file__)), "sha256": sha256(Path(__file__))},
        },
        "outputs": {
            "wse_scores_aligned.npz": sha256(compact_path),
            "nonempty_block_decisions.tsv.gz": sha256(nonempty_path),
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output / "checksums.sha256").write_text(
        "\n".join(f"{digest}  {name}" for name, digest in report["outputs"].items()) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"denominators": release_denominators, "outputs": report["outputs"]}, indent=2))


if __name__ == "__main__":
    main()
