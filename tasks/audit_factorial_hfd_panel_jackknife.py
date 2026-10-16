#!/usr/bin/env python3
"""Truth-blind single-panel-deletion retrieval audit for every query."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-manifest", required=True, type=Path)
    parser.add_argument("--panel-sketches", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--world", default="all", choices=("all", "known", "unresolved"))
    parser.add_argument("--minimum-entities", type=int, default=12)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    panel = pd.read_csv(args.panel_manifest, sep="\t")
    with np.load(args.panel_sketches, allow_pickle=False) as packed:
        signed = (
            np.asarray(packed[f"positive_{args.world}"], dtype=np.float32)
            - np.asarray(packed[f"negative_{args.world}"], dtype=np.float32)
        )
    panel = panel.copy()
    panel["vector_row"] = np.arange(len(panel))
    groups: dict[tuple[str, str], np.ndarray] = {
        (str(study), str(contrast)): block.vector_row.to_numpy(int)
        for (study, contrast), block in panel.groupby(["study_id", "contrast_id"], sort=True)
    }
    full = {key: signed[rows].mean(axis=0) for key, rows in groups.items()}
    study_panels = {
        str(study): sorted(block.panel_id.astype(str).unique())
        for study, block in panel.groupby("study_id")
    }
    objects = sorted(groups)
    rows_out: list[dict[str, object]] = []

    for query_key in objects:
        query_study, query_contrast = query_key
        candidates = [key for key in objects if key[0] != query_study]
        scenarios: list[tuple[str, str, str]] = []
        scenarios += [("query_panel_deleted", query_study, panel_id) for panel_id in study_panels[query_study]]
        for target_study in sorted({key[0] for key in candidates}):
            scenarios += [("target_study_panel_deleted", target_study, panel_id) for panel_id in study_panels[target_study]]
        for scenario_type, deleted_study, deleted_panel in scenarios:
            q_rows = groups[query_key]
            if scenario_type == "query_panel_deleted":
                keep = panel.iloc[q_rows].panel_id.astype(str).ne(deleted_panel).to_numpy()
                if not keep.any():
                    continue
                selected_q_rows = q_rows[keep]
                query_vector = signed[selected_q_rows].mean(axis=0)
            else:
                selected_q_rows = q_rows
                query_vector = full[query_key]
            scenario_records: list[dict[str, object]] = []
            for candidate in candidates:
                candidate_rows = groups[candidate]
                if scenario_type == "target_study_panel_deleted" and candidate[0] == deleted_study:
                    keep = panel.iloc[candidate_rows].panel_id.astype(str).ne(deleted_panel).to_numpy()
                    if not keep.any():
                        continue
                    selected_target_rows = candidate_rows[keep]
                    target_vector = signed[selected_target_rows].mean(axis=0)
                else:
                    selected_target_rows = candidate_rows
                    target_vector = full[candidate]
                q_counts = int(panel.iloc[selected_q_rows][f"n_entities_{args.world}"].sum())
                t_counts = int(panel.iloc[selected_target_rows][f"n_entities_{args.world}"].sum())
                answerable = q_counts >= args.minimum_entities and t_counts >= args.minimum_entities
                scenario_records.append({
                    "query_study_id": query_study,
                    "query_contrast_id": query_contrast,
                    "target_study_id": candidate[0],
                    "target_contrast_id": candidate[1],
                    "evidence_world": args.world,
                    "scenario_type": scenario_type,
                    "deleted_study_id": deleted_study,
                    "deleted_panel_id": deleted_panel,
                    "answerable": answerable,
                    "signed_response_score": float(query_vector @ target_vector) if answerable else np.nan,
                })
            scenario_frame = pd.DataFrame(scenario_records)
            scenario_frame["rank_signed_response_score"] = scenario_frame.signed_response_score.rank(
                ascending=False, method="min"
            )
            rows_out.extend(scenario_frame.to_dict("records"))

    output = pd.DataFrame(rows_out)
    args.output_dir.mkdir(parents=True)
    output.to_csv(args.output_dir / "single_panel_deletion_scores.tsv.gz", sep="\t", index=False, compression="gzip")
    report = {
        "schema": "factorial_hfd_panel_jackknife_v1",
        "status": "pretruth_single_panel_deletion_scores_frozen",
        "evidence_world": args.world,
        "response_objects": len(objects),
        "scored_rows": len(output),
        "truth_opened": False,
        "claim_boundary": "All deletion scenarios were generated before selecting the primary relationship.",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
