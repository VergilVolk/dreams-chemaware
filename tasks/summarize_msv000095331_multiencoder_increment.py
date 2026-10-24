#!/usr/bin/env python3
"""Summarize fixed-axis multi-encoder identity and response increments.

This is a post-scoring analysis.  It never chooses a method from biological
responses.  Top-1 identity hypotheses are compared first; the already frozen
response endpoint is joined only to describe application consequences.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


STATUS = "MSV000095331_MULTIENCODER_INCREMENT_SUMMARY_COMPLETE"
BASELINE = "noise_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--increment-dir", type=Path,
        default=Path(
            "data/validation/"
            "msv000095331_entity_method_increment_multiencoder_run_2404230"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(
            "data/validation/"
            "msv000095331_multiencoder_increment_summary_v1_20261024"
        ),
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    decision_path = args.increment_dir / "entity_method_decisions.tsv.gz"
    overlap_path = args.increment_dir / "prespecified_overlap_results.tsv"
    report_path = args.increment_dir / "report.json"
    decisions = pd.read_csv(decision_path, sep="\t", dtype=str)
    overlaps = pd.read_csv(overlap_path, sep="\t")
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    if source_report.get("dataset_role") != "development_consumed_not_external_validation":
        raise RuntimeError("unexpected dataset role")
    if BASELINE not in set(decisions["method_name"]):
        raise RuntimeError(f"missing baseline {BASELINE}")

    for column in [
        "candidate_count", "selected_method_score", "selected_score_margin",
        "ranked_candidate_count",
    ]:
        decisions[column] = pd.to_numeric(decisions[column], errors="coerce")
    decisions["mutual_top1"] = (
        decisions["mutual_top1"].astype(str).str.lower().eq("true")
    )
    selected = decisions["decision_status"].eq("selected_identity_hypothesis")
    decisions["selected"] = selected
    decisions["multi_candidate_selected"] = (
        selected & decisions["ranked_candidate_count"].ge(2)
    )

    coverage_rows: list[dict] = []
    for method, block in decisions.groupby("method_name", sort=True):
        multi = block["multi_candidate_selected"]
        margins = block.loc[multi, "selected_score_margin"]
        coverage_rows.append({
            "method_name": method,
            "fixed_directional_queries": int(len(block)),
            "selected_queries": int(block["selected"].sum()),
            "abstained_with_mass_candidates": int(
                block["decision_status"].eq(
                    "mass_candidates_but_method_abstained"
                ).sum()
            ),
            "multi_candidate_selected_queries": int(multi.sum()),
            "mutual_top1_directions": int(block["mutual_top1"].sum()),
            "zero_margin_ties": int(margins.eq(0).sum()),
            "margin_q25": float(margins.quantile(0.25)) if len(margins) else None,
            "margin_median": float(margins.median()) if len(margins) else None,
        })
    coverage = pd.DataFrame(coverage_rows)

    keys = ["query_id", "candidate_study_id"]
    candidate = decisions.pivot(
        index=keys, columns="method_name", values="selected_candidate_id"
    )
    methods = sorted(candidate.columns)
    agreement_rows: list[dict] = []
    for index, left in enumerate(methods):
        for right in methods[index + 1:]:
            common = candidate[left].notna() & candidate[right].notna()
            agree = candidate.loc[common, left].eq(candidate.loc[common, right])
            agreement_rows.append({
                "method_left": left,
                "method_right": right,
                "common_answered_queries": int(common.sum()),
                "same_top1": int(agree.sum()),
                "different_top1": int((~agree).sum()),
                "top1_agreement_fraction": float(agree.mean()),
            })
    agreement = pd.DataFrame(agreement_rows)

    baseline = decisions.loc[decisions["method_name"].eq(BASELINE)].set_index(keys)
    delta_rows: list[dict] = []
    changed_blocks: list[pd.DataFrame] = []
    for method in methods:
        if method == BASELINE:
            continue
        current = decisions.loc[decisions["method_name"].eq(method)].set_index(keys)
        joined = baseline[[
            "query_study_id", "decision_status", "selected_candidate_id",
            "selected_candidate_pair_id", "mutual_top1", "selected_score_margin",
        ]].join(
            current[[
                "decision_status", "selected_candidate_id",
                "selected_candidate_pair_id", "mutual_top1",
                "selected_score_margin",
            ]],
            lsuffix="_noise", rsuffix="_method", how="inner",
        )
        common = (
            joined["decision_status_noise"].eq("selected_identity_hypothesis")
            & joined["decision_status_method"].eq("selected_identity_hypothesis")
        )
        changed = common & joined["selected_candidate_id_noise"].ne(
            joined["selected_candidate_id_method"]
        )
        forward = joined["query_study_id"].eq("MSV000095331")
        gained = forward & ~joined["mutual_top1_noise"] & joined["mutual_top1_method"]
        lost = forward & joined["mutual_top1_noise"] & ~joined["mutual_top1_method"]
        delta_rows.append({
            "method_name": method,
            "common_answered_queries": int(common.sum()),
            "different_top1_vs_noise": int(changed.sum()),
            "different_top1_fraction": float(changed.sum() / common.sum()),
            "forward_different_top1_vs_noise": int((changed & forward).sum()),
            "forward_mutual_top1_gained": int(gained.sum()),
            "forward_mutual_top1_lost": int(lost.sum()),
            "forward_mutual_top1_net": int(gained.sum() - lost.sum()),
        })
        changed_block = joined.loc[changed].reset_index()
        changed_block.insert(0, "method_name", method)
        changed_blocks.append(changed_block)
    deltas = pd.DataFrame(delta_rows)
    changed_queries = pd.concat(changed_blocks, ignore_index=True)

    overlap_key = [
        "query_contrast_id", "candidate_study_id", "candidate_contrast_id"
    ]
    noise_overlap = overlaps.loc[
        overlaps["method_name"].eq(BASELINE)
    ].set_index(overlap_key)
    overlap_rows: list[dict] = []
    comparison_blocks: list[pd.DataFrame] = []
    for method in sorted(set(overlaps["method_name"]) - {BASELINE}):
        current = overlaps.loc[overlaps["method_name"].eq(method)].set_index(overlap_key)
        joined = noise_overlap[[
            "candidate_rank", "coverage_sensitive_response_score",
            "matched_entity_pairs",
        ]].join(
            current[[
                "candidate_rank", "coverage_sensitive_response_score",
                "matched_entity_pairs",
            ]],
            lsuffix="_noise", rsuffix="_method", how="inner",
        )
        rank_delta = joined["candidate_rank_noise"] - joined["candidate_rank_method"]
        overlap_rows.append({
            "method_name": method,
            "registered_contexts": int(len(joined)),
            "rank_improved_vs_noise": int(rank_delta.gt(0).sum()),
            "rank_worsened_vs_noise": int(rank_delta.lt(0).sum()),
            "rank_unchanged_vs_noise": int(rank_delta.eq(0).sum()),
            "top1_gained_vs_noise": int((
                joined["candidate_rank_method"].eq(1)
                & ~joined["candidate_rank_noise"].eq(1)
            ).sum()),
            "top1_lost_vs_noise": int((
                joined["candidate_rank_noise"].eq(1)
                & ~joined["candidate_rank_method"].eq(1)
            ).sum()),
            "median_rank_delta_vs_noise": float(rank_delta.median()),
            "mean_response_score_delta_vs_noise": float((
                joined["coverage_sensitive_response_score_method"]
                - joined["coverage_sensitive_response_score_noise"]
            ).mean()),
        })
        comparison = joined.assign(
            method_name=method, rank_delta_vs_noise=rank_delta
        ).reset_index()
        comparison_blocks.append(comparison)
    overlap_deltas = pd.DataFrame(overlap_rows)
    overlap_comparison = pd.concat(comparison_blocks, ignore_index=True)

    args.output_dir.mkdir(parents=True)
    compression = {"method": "gzip", "mtime": 0}
    outputs = {
        "method_coverage": args.output_dir / "method_coverage.tsv",
        "pairwise_top1_agreement": args.output_dir / "pairwise_top1_agreement.tsv",
        "deltas_vs_noise": args.output_dir / "deltas_vs_noise.tsv",
        "changed_queries_vs_noise": args.output_dir / "changed_queries_vs_noise.tsv.gz",
        "prespecified_rank_deltas": args.output_dir / "prespecified_rank_deltas_vs_noise.tsv",
        "prespecified_rank_comparison": args.output_dir / "prespecified_rank_comparison.tsv",
    }
    coverage.to_csv(outputs["method_coverage"], sep="\t", index=False)
    agreement.to_csv(outputs["pairwise_top1_agreement"], sep="\t", index=False)
    deltas.to_csv(outputs["deltas_vs_noise"], sep="\t", index=False)
    changed_queries.to_csv(
        outputs["changed_queries_vs_noise"], sep="\t", index=False,
        compression=compression,
    )
    overlap_deltas.to_csv(outputs["prespecified_rank_deltas"], sep="\t", index=False)
    overlap_comparison.to_csv(
        outputs["prespecified_rank_comparison"], sep="\t", index=False
    )

    fullmsg = overlap_deltas.loc[
        overlap_deltas["method_name"].eq("noise_chemaware_fullmsg")
    ].iloc[0]
    report = {
        "status": STATUS,
        "dataset_role": "development_consumed_not_external_validation",
        "selection_policy": (
            "identity methods were fixed before biological response outcomes; "
            "response results are application evaluation only"
        ),
        "fixed_denominators": source_report["denominators"],
        "key_result": {
            "fullmsg_vs_noise_forward_top1_changes": int(
                deltas.loc[
                    deltas["method_name"].eq("noise_chemaware_fullmsg"),
                    "forward_different_top1_vs_noise",
                ].iloc[0]
            ),
            "fullmsg_vs_noise_forward_mutual_net": int(
                deltas.loc[
                    deltas["method_name"].eq("noise_chemaware_fullmsg"),
                    "forward_mutual_top1_net",
                ].iloc[0]
            ),
            "fullmsg_prespecified_top1_gained_vs_noise": int(
                fullmsg["top1_gained_vs_noise"]
            ),
            "fullmsg_prespecified_top1_lost_vs_noise": int(
                fullmsg["top1_lost_vs_noise"]
            ),
            "interpretation": (
                "FullMSG changes real entity decisions and improves some registered "
                "response ranks, but the observed Top-1 gains are repeated contexts "
                "within one mixture and do not establish broad biological transfer."
            ),
        },
        "inputs": {
            "increment_report": {"path": str(report_path), "sha256": sha256(report_path)},
            "decisions": {"path": str(decision_path), "sha256": sha256(decision_path)},
            "prespecified_overlap": {
                "path": str(overlap_path), "sha256": sha256(overlap_path),
            },
        },
        "outputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in outputs.items()
        },
        "claim_boundary": [
            "Candidate matches are mass-compatible identity hypotheses, not confirmed identities.",
            "No identity truth exists on this application panel; changes are not called corrections.",
            "Repeated microbial strata are contexts, not independent studies.",
            "This run contains WSE and encoder scores, not B, P2b, V2, MSG rerankers or BioAware.",
            "Simple query-minmax averaging is an untrained comparator, not a selected final model.",
        ],
    }
    output_report = args.output_dir / "report.json"
    output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
